"""E3 dual-branch training from OSTrack; fixed twenty-epoch protocol."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import torch

from research.protocol_r.common import (
    REPORT, SNAPSHOT, read, save, sha, seed, tensor_sha, build, check_sources,
)
from lib.config.cadtrack.config import cfg
from miss_rgbt.data import make_loader
from miss_rgbt.objectives import task_loss, training_loss
from miss_rgbt.train import fix_bn, sample_masks


def validate(model, batches):
    model.eval()
    totals = dict.fromkeys(['11', '01', '10'], 0.)
    count = 0
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for data in batches:
            templates = [x.cuda() for x in data['template_images']]
            searches = [x.cuda() for x in data['search_images']]
            boxes = data['search_anno'].cuda()
            size = searches[0].shape[0]
            for label, flags in [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]:
                available = [searches[0].new_ones(size, 2) for _ in templates]
                available += [searches[0].new_tensor(flags).expand(size, -1)
                              for _ in searches]
                loss = task_loss(model(templates, searches, available), boxes, cfg)
                assert torch.isfinite(loss).all()
                totals[label] += float(loss.sum())
            count += size
    assert count == 128
    return dict(samples=count, task_loss={k: v/count for k, v in totals.items()},
                mean_task_loss=sum(totals.values())/(3*count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    config = read(args.config)
    check_sources()
    assert config['init'] == 'ostrack' and config['epochs'] == 20
    assert sha(config['development_cache']) == config['development_cache_sha256']
    output = Path(config['output'] + ('_smoke' if args.max_steps else ''))
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'run_config.json', config)
    save(output / 'source_sha256.json', read(REPORT / 'source_sha256.json'))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model, initialization = build(config)
    assert initialization['initializer_sha256'] == read(REPORT / 'registration.json')['initializer_sha256']
    save(output / 'initial_state.json', initialization)
    model.cuda().train()
    fix_bn(model)
    seed(config['seed'])
    loader = make_loader(cfg, config['dataset'], 'train', config['microbatch'],
                         config['samples_per_epoch'], config['workers'])
    assert len(loader) % config['accumulation'] == 0
    groups = [dict(params=[p for n, p in model.named_parameters() if 'backbone' in n],
                   lr=config['learning_rate']*config['backbone_multiplier']),
              dict(params=[p for n, p in model.named_parameters() if 'backbone' not in n],
                   lr=config['learning_rate'])]
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    base_rates = [group['lr'] for group in groups]
    batches = torch.load(config['development_cache'], map_location='cpu')
    started, updates, history = time.time(), 0, []
    first_inputs = []
    with (output / 'steps.jsonl').open('x') as log:
        for epoch in range(1, 21):
            model.train()
            fix_bn(model)
            optimizer.zero_grad(set_to_none=True)
            for index, data in enumerate(loader):
                if updates == 0:
                    first_inputs.append({key: tensor_sha(data[key]) for key in
                        ['template_images', 'search_images', 'search_anno']})
                warmup = min((updates+1)/config['warmup_updates'], 1.)
                decay = .1 if epoch > config['lr_drop_epoch'] else 1.
                for group, rate in zip(optimizer.param_groups, base_rates):
                    group['lr'] = rate*warmup*decay
                templates = [x.cuda() for x in data['template_images']]
                searches = [x.cuda() for x in data['search_images']]
                boxes = data['search_anno'].cuda()
                availability, patch = sample_masks(templates, searches, config)
                complete = model(templates, searches)
                missing = model(templates, searches, availability, patch)
                loss, status = training_loss(complete, missing, boxes, cfg, .1, 0., .1)
                assert torch.isfinite(loss), status
                (loss/config['accumulation']).backward()
                if (index+1) % config['accumulation'] != 0:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
                assert torch.isfinite(norm), float(norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                status.update(epoch=epoch, update=updates,
                    microstep=(epoch-1)*len(loader)+index+1,
                    loss=float(loss.detach()), gradient_norm=float(norm),
                    seconds=time.time()-started,
                    learning_rates=[group['lr'] for group in optimizer.param_groups],
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                    effective_batch=config['microbatch']*config['accumulation'])
                if updates == 1:
                    status['first_update_input_sha256'] = first_inputs
                log.write(json.dumps(status)+'\n')
                log.flush()
                if updates == 1 or updates % 25 == 0:
                    print(json.dumps(status), flush=True)
                if args.max_steps and updates >= args.max_steps:
                    path = output / 'smoke.pth.tar'
                    torch.save(dict(model=model.state_dict(), config=config,
                        epoch=epoch, updates=updates), str(path))
                    model.eval()
                    with torch.no_grad():
                        before = model(templates, searches)
                        model.load_state_dict(torch.load(str(path), map_location='cpu')['model'], strict=True)
                        after = model(templates, searches)
                    assert all(torch.equal(a[key], b[key]) for a, b in zip(before, after)
                               for key in ['score_map', 'size_map', 'offset_map', 'pred_boxes'])
                    save(output / 'smoke_verified.json', dict(status='passed',
                        updates=updates, strict_reload_bitwise_equal=True,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        checkpoint_sha256=sha(path), scope='Smoke only; no benchmark score'))
                    print('SMOKE_VERIFIED', flush=True)
                    return
            result = validate(model, batches)
            result.update(epoch=epoch, updates=updates, seconds=time.time()-started)
            history.append(result)
            save(output / 'development_history.json', history)
            if epoch in config['checkpoint_epochs']:
                torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                    updates=updates, optimizer=optimizer.state_dict()),
                    str(output / ('epoch%02d.pth.tar' % epoch)))
            print('DEVELOPMENT_LOSS '+json.dumps(result), flush=True)
    check_sources()
    final = history[-5:]
    selected = min(final, key=lambda x: (x['mean_task_loss'], x['epoch']))
    save(output / 'completed.json', dict(status='complete', epochs=20, updates=updates,
        seconds=time.time()-started, pid=os.getpid(),
        final_five=[dict(epoch=x['epoch'],
            path=str(output / ('epoch%02d.pth.tar' % x['epoch'])),
            sha256=sha(output / ('epoch%02d.pth.tar' % x['epoch']))) for x in final],
        selected_epoch=selected['epoch'], selection_rule=config['development_selection'],
        all_checkpoints_retained=True, official_test_selection_used=False))


if __name__ == '__main__':
    main()
