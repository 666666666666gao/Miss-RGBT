"""Isolated full20 OSTrack Utility trainer; no changes to active A/R runtimes."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.data import make_loader
from miss_rgbt.model import MissingTrackerModel
from miss_rgbt.objectives import utility_supervision
from miss_rgbt.train import fix_bn
from research.curriculum_p.engine import ARMS, CurriculumState, forward_microbatch
from research.curriculum_p.protocol_prepare import make_config, REPORT
from research.protocol_r.common import MISSING, read, save, sha, seed, tensor_sha
from research.protocol_r.train import validate


def check_config(config):
    assert config['curriculum_arm'] in ARMS
    assert config['stage'] == 'A' and config['init'] == 'ostrack'
    assert config['epochs'] == 20 and not config['early_stop']
    assert config['checkpoint_epochs'] == [10, 16, 17, 18, 19, 20]
    assert config['freeze_all_bn'] and not config['repair']
    assert config['task_utility']['selection'] == 'utility'
    assert config['utility_probe_interval'] == 4 and config['utility_weight'] == .1
    assert config['patch_mask_probability'] == .1
    assert config['forward_weight'] == .1 and config['reverse_weight'] == 0.
    assert config['samples_per_epoch'] == 20000 and config['workers'] == 2
    assert config['samples_per_epoch'] % (
        config['microbatch'] * config['accumulation']) == 0
    assert config['warmup_updates'] == 500
    assert not config['official_test_selection_used'] and not config['resume_supported']


def check_evidence(config):
    assert Path(config['snapshot']) == Path(__file__).resolve().parents[2]
    assert sha(config['source_manifest']) == config['source_manifest_sha256']
    for path, expected in read(config['source_manifest']).items():
        assert sha(path) == expected, path
    assert sha(config['gate']) == config['gate_sha256']
    gate = read(config['gate'])
    assert gate['ready'] and gate['protocol_arm'] == 'r1'
    for path, expected in gate['parents_sha256'].items():
        assert sha(path) == expected, path
    expected_config = make_config(config['curriculum_arm'], read(REPORT / 'r1.json'))
    for key, value in expected_config.items():
        if key not in ('gate_sha256', 'source_manifest_sha256'):
            assert config[key] == value, key
    assert sha(config['checkpoint']) == config['initializer_sha256']
    assert sha(config['development_cache']) == config['development_cache_sha256']


def provenance_loader(config):
    """Record identities of accepted native tuples without changing RNG draws."""
    loader = make_loader(cfg, config['dataset'], 'train', config['microbatch'],
                         config['samples_per_epoch'], config['workers'])
    dataset = loader.dataset.datasets[0]
    names = list(dataset.sequence_list)
    assert len(names) == len(set(names)) == 882
    calls = []
    original_frames, original_item = dataset.get_frames, loader.dataset.getitem

    def frames(index, ids, anno=None):
        calls.append((index, [int(frame) for frame in ids]))
        return original_frames(index, ids, anno)

    def item():
        calls.clear()
        data = original_item()
        template, search = calls[-2:]
        assert template[0] == search[0]
        assert len(template[1]) == 2 and len(search[1]) == 3
        data['frame_identity'] = torch.tensor([template[0]] + template[1] + search[1])
        return data

    dataset.get_frames, loader.dataset.getitem = frames, item
    return loader, names


def record_batch(data, names, epoch, index, stream):
    record = dict(epoch=epoch, index=index, identities=[
        dict(sequence=names[row[0]], template_frames=row[1:3], search_frames=row[3:])
        for row in data['frame_identity'].t().tolist()],
        tensors_sha256={key: tensor_sha(data[key]) for key in
                       ('template_images', 'search_images', 'search_anno')})
    stream.write(json.dumps(record) + '\n')
    stream.flush()
    return record


def rng_state(probe):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all(),
                utility_probe=probe.get_state())


def build(config):
    seed(config['seed'])
    core = build_cadtrack(cfg, training=False)
    result = core.load_state_dict(
        torch.load(config['checkpoint'], map_location='cpu')['net'], strict=False)
    assert result.missing_keys == MISSING and not result.unexpected_keys, result
    model = MissingTrackerModel(core, repair=False, repair_layer=6,
                                task_utility=config['task_utility'])
    return model, dict(missing_keys=result.missing_keys,
                       unexpected_keys=result.unexpected_keys,
                       initializer_sha256=config['initializer_sha256'],
                       parameters_sha256={name: tensor_sha(value)
                                          for name, value in model.named_parameters()},
                       bn_sha256={name: tensor_sha(value)
                                  for name, value in model.named_buffers()
                                  if any(x in name for x in
                                         ('running_mean', 'running_var', 'num_batches_tracked'))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-steps', type=int, choices=[4])
    args = parser.parse_args()
    config = read(args.config)
    check_config(config)
    check_evidence(config)
    if not args.max_steps:
        check_smokes(config)
    root = Path(config['output'] + ('_smoke' if args.max_steps else ''))
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'run_config.json', config)
    save(root / 'source_sha256.json', read(config['source_manifest']))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(Path(config['snapshot']) / 'experiments/cadtrack/cadtrack.yaml'))
    model, initialization = build(config)
    save(root / 'initial_state.json', initialization)
    model.cuda().train()
    fix_bn(model, True)
    # Match initial data and native mask seeds in all arms after Utility construction.
    seed(config['seed'])
    loader, names = provenance_loader(config)
    names_path = Path(config['snapshot']) / 'lib/train/data_specs/lasher_train.txt'
    assert sha(names_path) == config['train_names_sha256']
    assert names == names_path.read_text().splitlines()
    assert len(loader) * config['microbatch'] == config['samples_per_epoch']
    assert len(loader) % config['accumulation'] == 0
    save(root / 'data_boundary.json', dict(split='train882', train_names=names,
        development_split='val97', development_samples=128,
        accepted_tuple_identities=True, rejected_crop_reads_recorded=False,
        train_names_sha256=config['train_names_sha256']))
    groups = [dict(params=[p for n, p in model.named_parameters() if 'backbone' in n],
                   lr=config['learning_rate'] * config['backbone_multiplier']),
              dict(params=[p for n, p in model.named_parameters() if 'backbone' not in n],
                   lr=config['learning_rate'])]
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    rates = [group['lr'] for group in groups]
    state = CurriculumState(config['curriculum_arm'], next(model.parameters()).device,
                            config['seed'])
    probe_rng = torch.Generator().manual_seed(config['seed'])
    development = torch.load(config['development_cache'], map_location='cpu')
    updates, history, started = 0, [], time.time()
    with (root / 'input_manifest.jsonl').open('x') as inputs, \
            (root / 'steps.jsonl').open('x') as steps:
        for epoch in range(1, 21):
            model.train()
            fix_bn(model, True)
            optimizer.zero_grad(set_to_none=True)
            for index, data in enumerate(loader):
                record = record_batch(data, names, epoch, index, inputs)
                templates = [value.cuda() for value in data['template_images']]
                searches = [value.cuda() for value in data['search_images']]
                boxes = data['search_anno'].cuda()
                for group, rate in zip(optimizer.param_groups, rates):
                    group['lr'] = rate * min((updates + 1) / config['warmup_updates'], 1.)
                    group['lr'] *= .1 if epoch > config['lr_drop_epoch'] else 1.
                microstep = state.microsteps
                probe, view = None, None
                if microstep % config['utility_probe_interval'] == 0:
                    probe_id = microstep // config['utility_probe_interval']
                    view = 'C' if probe_id % 2 == 0 else 'M'
                    regions = torch.randint(model.utility_repair.region_tokens.shape[0],
                        (searches[0].shape[0], 1), generator=probe_rng).cuda()
                    probe = dict(search_index=(probe_id // 2) % len(searches), regions=regions)
                loss, status, complete, missing, available, patch = forward_microbatch(
                    model, templates, searches, boxes, cfg, config, epoch, state, probe, view)
                if probe is not None:
                    auxiliary, utility = utility_supervision(
                        complete if view == 'C' else missing, boxes, cfg)
                    loss = loss + config['utility_weight'] * auxiliary
                    status.update(utility=utility, probe_view=view)
                assert torch.isfinite(loss), status
                (loss / config['accumulation']).backward()
                status.update(epoch=epoch, microstep=state.microsteps,
                    loss=float(loss.detach()), input_sha256=record['tensors_sha256'],
                    masks_sha256=[tensor_sha(value) for value in available + patch])
                if (index + 1) % config['accumulation'] == 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                    assert torch.isfinite(norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    status.update(update=updates, gradient_norm=float(norm),
                        learning_rates=[group['lr'] for group in optimizer.param_groups],
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(), seconds=time.time()-started)
                    if updates == 1 or updates % 25 == 0:
                        print(json.dumps(status), flush=True)
                steps.write(json.dumps(status) + '\n')
                steps.flush()
                if args.max_steps and updates == args.max_steps:
                    path = root / 'smoke.pth.tar'
                    checkpoint(path, model, optimizer, config, epoch, updates,
                               state, probe_rng, index + 1, len(loader))
                    model.eval()
                    with torch.no_grad():
                        before = model(templates, searches)
                        saved = torch.load(str(path), map_location='cpu')
                        model.load_state_dict(saved['model'], strict=True)
                        state.load_state_dict(saved['curriculum'])
                        after = model(templates, searches)
                    assert all(torch.equal(a[key], b[key]) for a, b in zip(before, after)
                               for key in ('score_map', 'size_map', 'offset_map', 'pred_boxes'))
                    save(root / 'smoke_verified.json', dict(status='passed', updates=updates,
                        strict_reload_bitwise_equal=True, checkpoint_sha256=sha(path),
                        input_manifest_sha256=sha(root / 'input_manifest.jsonl'),
                        scope='Four real optimizer updates; no official score or full-run proof'))
                    return
            result = validate(model, development)
            result.update(epoch=epoch, updates=updates, seconds=time.time()-started)
            history.append(result)
            save(root / 'development_history.json', history)
            check_evidence(config)
            if epoch in config['checkpoint_epochs']:
                checkpoint(root / ('epoch%02d.pth.tar' % epoch), model, optimizer,
                           config, epoch, updates, state, probe_rng, len(loader), len(loader))
            print('DEVELOPMENT_LOSS ' + json.dumps(result), flush=True)
    selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
    save(root / 'completed.json', dict(status='complete', epochs=20, updates=updates,
        pid=os.getpid(), selected_epoch=selected['epoch'],
        final_five=[dict(epoch=epoch, path=str(root / ('epoch%02d.pth.tar' % epoch)),
                        sha256=sha(root / ('epoch%02d.pth.tar' % epoch)))
                    for epoch in range(16, 21)],
        selection_rule=config['development_selection'], early_stopped=False,
        all_checkpoints_retained=True, official_test_selection_used=False,
        input_manifest_sha256=sha(root / 'input_manifest.jsonl'),
        resume_supported=False, scope='Training only; sixty full evaluations still required'))


def check_smokes(config):
    common_inputs, common_initialization = None, None
    for arm in ARMS:
        candidate = read(Path(config['gate']).parent / (arm + '.json'))
        check_config(candidate)
        check_evidence(candidate)
        assert {k: v for k, v in candidate.items() if k not in ('curriculum_arm', 'output')} == {
            k: v for k, v in config.items() if k not in ('curriculum_arm', 'output')}
        smoke = Path(candidate['output'] + '_smoke')
        receipt = read(smoke / 'smoke_verified.json')
        assert read(smoke / 'run_config.json') == candidate
        assert receipt['status'] == 'passed' and receipt['updates'] == 4
        assert receipt['strict_reload_bitwise_equal']
        assert sha(smoke / 'smoke.pth.tar') == receipt['checkpoint_sha256']
        inputs = sha(smoke / 'input_manifest.jsonl')
        assert inputs == receipt['input_manifest_sha256']
        initial = read(smoke / 'initial_state.json')
        if common_inputs is None:
            common_inputs, common_initialization = inputs, initial
        assert inputs == common_inputs and initial == common_initialization


def checkpoint(path, model, optimizer, config, epoch, updates, state,
               probe_rng, consumed, batches):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        config=config, epoch=epoch, updates=updates, curriculum=state.state_dict(),
        rng=rng_state(probe_rng), loader_position=dict(epoch=epoch,
            consumed_batches=consumed, batches_per_epoch=batches),
        resume_supported=False,
        resume_limitation='Worker RNG/prefetch queues are not captured; no resume CLI'),
        str(path))


if __name__ == '__main__':
    main()
