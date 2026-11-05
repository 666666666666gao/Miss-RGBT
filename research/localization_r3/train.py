"""R3 paired objectives on native fulltrain882 tuples; no resume or dispatch."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.model import MissingTrackerModel
from miss_rgbt.train import fix_bn
from research.curriculum_p.protocol_train import provenance_loader, record_batch
from research.localization_r3.objective import ARMS, forward_objective
from research.localization_r3.prepare import check_config, check_evidence
from research.protocol_r.common import MISSING, read, save, sha, seed, tensor_sha
from research.protocol_r.train import validate


def bn_state(model):
    return {name: tensor_sha(value) for name, value in model.named_buffers()
            if any(key in name for key in
                   ('running_mean', 'running_var', 'num_batches_tracked'))}


def build(config):
    seed(config['seed'])
    core = build_cadtrack(cfg, training=False)
    result = core.load_state_dict(
        torch.load(config['checkpoint'], map_location='cpu')['net'], strict=False)
    assert result.missing_keys == MISSING and not result.unexpected_keys, result
    model = MissingTrackerModel(core, repair=False, width=config['repair_width'],
                                repair_layer=config['repair_layer'])
    initial = dict(initializer_sha256=config['initializer_sha256'],
        missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys,
        parameters_sha256={name: tensor_sha(value)
                           for name, value in model.named_parameters()},
        bn_sha256=bn_state(model))
    return model, initial


def checkpoint(path, model, optimizer, config, epoch, updates, microsteps):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        config=config, epoch=epoch, updates=updates, microsteps=microsteps,
        resume_supported=False), str(path))


def check_smokes(config):
    common = None
    for arm in ARMS:
        candidate = read(Path(config['source_manifest']).parent / (arm + '.json'))
        check_evidence(candidate)
        assert {key: value for key, value in candidate.items()
                if key not in ('arm', 'output')} == {
                    key: value for key, value in config.items()
                    if key not in ('arm', 'output')}
        root = Path(candidate['output'] + '_smoke')
        receipt = read(root / 'smoke_verified.json')
        assert read(root / 'run_config.json') == candidate
        assert receipt['status'] == 'passed' and receipt['updates'] == 4
        assert receipt['strict_reload_bitwise_equal'] and receipt['bn_unchanged']
        assert sha(root / 'smoke.pth.tar') == receipt['checkpoint_sha256']
        assert sha(root / 'input_manifest.jsonl') == receipt['input_manifest_sha256']
        paired = dict(initial=read(root / 'initial_state.json'),
                      inputs=receipt['input_manifest_sha256'],
                      masks=receipt['candidate_masks_sha256'])
        if common is None:
            common = paired
        assert paired == common


def train(config, max_steps=None):
    check_config(config)
    check_evidence(config)
    if max_steps is None:
        check_smokes(config)
    root = Path(config['output'] + ('_smoke' if max_steps else ''))
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'run_config.json', config)
    save(root / 'source_sha256.json', read(config['source_manifest']))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(Path(config['snapshot']) / 'experiments/cadtrack/cadtrack.yaml'))
    model, initial = build(config)
    save(root / 'initial_state.json', initial)
    model.cuda().train()
    fix_bn(model, True)
    seed(config['seed'])
    loader, names = provenance_loader(config)
    assert names == (Path(config['snapshot']) /
                     'lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(loader) * config['microbatch'] == config['samples_per_epoch']
    assert len(loader) % config['accumulation'] == 0
    save(root / 'data_boundary.json', dict(split='train882', train_names=names,
        accepted_tuple_identities=True, rejected_crop_reads_recorded=False,
        development_samples=128, train_names_sha256=config['train_names_sha256']))
    groups = [dict(params=[value for name, value in model.named_parameters()
                           if 'backbone' in name],
                   lr=config['learning_rate'] * config['backbone_multiplier']),
              dict(params=[value for name, value in model.named_parameters()
                           if 'backbone' not in name], lr=config['learning_rate'])]
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    rates = [group['lr'] for group in groups]
    development = None if max_steps else torch.load(config['development_cache'], map_location='cpu')
    updates, history, started, candidate_masks = 0, [], time.time(), []
    with (root / 'input_manifest.jsonl').open('x') as inputs, \
            (root / 'steps.jsonl').open('x') as steps:
        for epoch in range(1, 21):
            model.train()
            fix_bn(model, True)
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.
            for index, data in enumerate(loader):
                record = record_batch(data, names, epoch, index, inputs)
                templates = [value.cuda() for value in data['template_images']]
                searches = [value.cuda() for value in data['search_images']]
                boxes = data['search_anno'].cuda()
                for group, rate in zip(optimizer.param_groups, rates):
                    group['lr'] = rate * min((updates + 1) / config['warmup_updates'], 1.)
                    group['lr'] *= .1 if epoch > config['lr_drop_epoch'] else 1.
                microstep = (epoch - 1) * len(loader) + index
                loss, status, _, _, availability, patch = forward_objective(
                    model, templates, searches, boxes, cfg, config['arm'],
                    microstep, config['stochastic_seed'])
                (loss / config['accumulation']).backward()
                update_loss += float(loss.detach()) / config['accumulation']
                masks = [tensor_sha(value) for value in availability + patch]
                status.update(epoch=epoch, loss=float(loss.detach()),
                    input_sha256=record['tensors_sha256'], candidate_masks_sha256=masks)
                if max_steps:
                    candidate_masks.append(masks)
                if (index + 1) % config['accumulation'] == 0:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['clip_grad_norm'])
                    assert torch.isfinite(norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    status.update(update=updates, update_loss_mean=update_loss,
                        gradient_norm=float(norm),
                        learning_rates=[group['lr'] for group in optimizer.param_groups],
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        seconds=time.time()-started)
                    update_loss = 0.
                    if updates == 1 or updates % 25 == 0:
                        print(json.dumps(status), flush=True)
                steps.write(json.dumps(status) + '\n')
                steps.flush()
                if max_steps and updates == max_steps:
                    assert bn_state(model) == initial['bn_sha256']
                    path = root / 'smoke.pth.tar'
                    checkpoint(path, model, optimizer, config, epoch, updates, microstep + 1)
                    model.eval()
                    with torch.no_grad():
                        before = model(templates, searches)
                        model.load_state_dict(
                            torch.load(str(path), map_location='cpu')['model'], strict=True)
                        after = model(templates, searches)
                    assert all(torch.equal(first[key], second[key])
                               for first, second in zip(before, after)
                               for key in ('score_map', 'size_map', 'offset_map', 'pred_boxes'))
                    save(root / 'smoke_verified.json', dict(status='passed', updates=updates,
                        strict_reload_bitwise_equal=True, bn_unchanged=True,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        seconds=time.time()-started, candidate_masks_sha256=candidate_masks,
                        checkpoint_sha256=sha(path),
                        input_manifest_sha256=sha(root / 'input_manifest.jsonl'),
                        scope='Four native optimizer updates; not efficacy or full-run proof'))
                    return
            assert bn_state(model) == initial['bn_sha256']
            result = validate(model, development)
            result.update(epoch=epoch, updates=updates, seconds=time.time()-started)
            history.append(result)
            save(root / 'development_history.json', history)
            if epoch in config['checkpoint_epochs']:
                checkpoint(root / ('epoch%02d.pth.tar' % epoch), model, optimizer,
                           config, epoch, updates, epoch * len(loader))
            print('DEVELOPMENT_LOSS ' + json.dumps(result), flush=True)
    check_evidence(config)
    assert updates == 50000
    selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
    save(root / 'completed.json', dict(status='complete', epochs=20, updates=updates,
        pid=os.getpid(), selected_epoch=selected['epoch'],
        primary_development_epoch20_complete_loss=history[-1]['task_loss']['11'],
        final_five=[dict(epoch=epoch, path=str(root / ('epoch%02d.pth.tar' % epoch)),
                        sha256=sha(root / ('epoch%02d.pth.tar' % epoch)))
                    for epoch in range(16, 21)],
        selection_rule=config['development_selection'], early_stopped=False,
        all_checkpoints_retained=True, official_test_selection_used=False,
        input_manifest_sha256=sha(root / 'input_manifest.jsonl'), resume_supported=False,
        scope='One R3 arm training only; twenty full pair evaluations still required'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-steps', type=int, choices=[4])
    args = parser.parse_args()
    train(read(args.config), args.max_steps)


if __name__ == '__main__':
    main()
