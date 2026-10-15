import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss, training_loss, utility_supervision


def sample_masks(templates, searches, config):
    batch = searches[0].shape[0]
    device = searches[0].device
    states = torch.tensor([[1., 1.], [0., 1.], [1., 0.]], device=device)
    probability = torch.tensor(config['complete_rgb_missing_tir_missing_probability'], device=device)
    available = [torch.ones(batch, 2, device=device)]
    available += [states[torch.multinomial(probability, batch, replacement=True)]
                  for _ in range(len(templates) - 1 + len(searches))]
    patch = [torch.rand(batch, 256, 2, device=device).ge(config['patch_mask_probability']).float()
             for _ in searches]
    return available, patch


def fix_bn(model, freeze_all=False):
    target = model if freeze_all else model.baseline.box_head
    for module in target.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    assert config['stage'] in ['E0-FT', 'E2', 'E3', 'E4', 'E5', 'A']
    is_utility = config['stage'] == 'A'
    if is_utility:
        assert not config['repair'] and config['freeze_all_bn']
        assert config['utility_probe_interval'] >= 1
        assert config['utility_weight'] >= 0 and config['task_utility'] is not None
    if config['stage'] == 'E0-FT':
        assert not config['repair'] and config['forward_weight'] == config['reverse_weight'] == 0
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, config['checkpoint'], repair=config['repair'],
        width=config['repair_width'], repair_layer=config['repair_layer'],
        task_utility=config.get('task_utility')).cuda().train()
    fix_bn(model, config.get('freeze_all_bn', False))
    # Repair construction consumes RNG; restore the shared data/mask/drop-path seed.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    loader = make_loader(cfg, config['dataset'], 'train', config['microbatch'],
                         config['samples_per_epoch'], config['workers'])
    groups = [dict(params=[p for n, p in model.named_parameters() if p.requires_grad and 'backbone' in n],
                   lr=config['learning_rate'] * config['backbone_multiplier']),
              dict(params=[p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n],
                   lr=config['learning_rate'])]
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_drop_epoch'], gamma=.1)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ['miss_rgbt', 'lib'] for p in sorted(Path(folder).rglob('*.py'))}
    (output / 'run_config.json').write_text(json.dumps(config, indent=2))
    (output / 'source_sha256.json').write_text(json.dumps(source, indent=2))
    (output / 'source_version.txt').write_text(Path(config['source_version_file']).read_text())
    start = time.time()
    updates = 0
    first_update_inputs = []
    if is_utility:
        probe_generator = torch.Generator().manual_seed(seed)
        utility_records, utility_calls, update_loss_sum = [], 0, 0.
    assert len(loader) % config['accumulation'] == 0
    with (output / 'steps.jsonl').open('w') as log:
        for epoch in range(1, config['epochs'] + 1):
            optimizer.zero_grad(set_to_none=True)
            for index, data in enumerate(loader):
                templates = [image.cuda() for image in data['template_images']]
                searches = [image.cuda() for image in data['search_images']]
                boxes = data['search_anno'].cuda()
                first_input_sha256 = hashlib.sha256(data['search_images'].numpy().tobytes()).hexdigest() if updates == 0 else None
                if updates == 0:
                    first_update_inputs.append(first_input_sha256)
                if config['stage'] == 'E0-FT':
                    complete = model(templates, searches)
                    loss = task_loss(complete, boxes, cfg).mean()
                    status = dict(task_c=float(loss.detach()))
                else:
                    available, patch = sample_masks(templates, searches, config)
                    probe, probe_view = None, None
                    microstep = (epoch - 1) * len(loader) + index
                    if is_utility and microstep % config['utility_probe_interval'] == 0:
                        probe_id = microstep // config['utility_probe_interval']
                        probe_view = 'C' if probe_id % 2 == 0 else 'M'
                        regions = torch.randint(model.utility_repair.region_tokens.shape[0],
                            (searches[0].shape[0], 1), generator=probe_generator).to(searches[0].device)
                        probe = dict(search_index=(probe_id // 2) % len(searches), regions=regions)
                    complete = model(templates, searches, utility_probe=probe if probe_view == 'C' else None)
                    missing = model(templates, searches, available, patch,
                                    utility_probe=probe if probe_view == 'M' else None)
                    loss, status = training_loss(complete, missing, boxes, cfg,
                        config['forward_weight'], config['reverse_weight'], config['reverse_margin'])
                    if probe is not None:
                        auxiliary, record = utility_supervision(
                            complete if probe_view == 'C' else missing, boxes, cfg)
                        loss = loss + config['utility_weight'] * auxiliary
                        record['view'] = probe_view
                        utility_records.append(record)
                        utility_calls += 1
                    if is_utility:
                        update_loss_sum += float(loss.detach())
                assert torch.isfinite(loss), status
                (loss / config['accumulation']).backward()
                if (index + 1) % config['accumulation'] == 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                    assert torch.isfinite(norm), float(norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    status.update(epoch=epoch, update=updates, loss=float(loss.detach()),
                        seconds=time.time() - start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        microbatch=config['microbatch'], accumulation=config['accumulation'])
                    if updates == 1:
                        status['input_sha256'] = first_input_sha256
                        status['input_sha256_microbatches'] = first_update_inputs
                    if is_utility:
                        status.update(utility_probes=utility_records, utility_probe_calls=utility_calls,
                            update_loss_mean=update_loss_sum / config['accumulation'],
                            selected_tokens_per_search=complete[0]['utility_tokens'].shape[1])
                        utility_records, update_loss_sum = [], 0.
                    log.write(json.dumps(status) + '\n')
                    log.flush()
                    if updates % 25 == 0 or updates == 1:
                        print(json.dumps(status), flush=True)
                    if args.max_steps is not None and updates >= args.max_steps:
                        torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                            updates=updates, optimizer=optimizer.state_dict()), str(output / 'smoke.pth.tar'))
                        print('SMOKE_COMPLETE ' + str(updates), flush=True)
                        return
            scheduler.step()
            if epoch in config['checkpoint_epochs']:
                torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                    updates=updates, optimizer=optimizer.state_dict()), str(output / ('epoch%02d.pth.tar' % epoch)))
    (output / 'completed.json').write_text(json.dumps(dict(epochs=config['epochs'], updates=updates,
                                                          seconds=time.time() - start)))


if __name__ == '__main__':
    main()
