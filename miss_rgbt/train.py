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
from miss_rgbt.objectives import training_loss


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


def fix_bn(model):
    for module in model.baseline.box_head.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    assert config['stage'] in ['E2', 'E3', 'E4', 'E5']
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, config['checkpoint'], repair=config['repair'],
        width=config['repair_width'], repair_layer=config['repair_layer']).cuda().train()
    fix_bn(model)
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
    start = time.time()
    updates = 0
    first_update_inputs = []
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
                available, patch = sample_masks(templates, searches, config)
                complete = model(templates, searches)
                missing = model(templates, searches, available, patch)
                loss, status = training_loss(complete, missing, boxes, cfg,
                    config['forward_weight'], config['reverse_weight'], config['reverse_margin'])
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
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                updates=updates, optimizer=optimizer.state_dict()), str(output / ('epoch%02d.pth.tar' % epoch)))
    (output / 'completed.json').write_text(json.dumps(dict(epochs=config['epochs'], updates=updates,
                                                          seconds=time.time() - start)))


if __name__ == '__main__':
    main()
