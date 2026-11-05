"""OSTrack F training loop; the caller must authorize parents and all smokes."""
import json
import os
from pathlib import Path
import time

import cv2
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.objectives import utility_supervision
from miss_rgbt.train import fix_bn
from research.curriculum_p.engine import CurriculumState
from research.curriculum_p.protocol_train import (
    check_config as check_native_config, checkpoint, provenance_loader, record_batch,
)
from research.feedback_f.protocol_training import ARMS, build, forward_microbatch
from research.feedback_f.protocol_validation import validate
from research.protocol_r.common import read, save, sha, seed, tensor_sha


def check_config(config):
    check_native_config(config)
    assert config['feedback'] == dict(mode=config['feedback']['mode'], width=128)
    assert config['feedback']['mode'] in ARMS
    assert config['microbatch'] == config['accumulation'] == 2
    assert config['learning_rate'] == 5e-5 and config['backbone_multiplier'] == .1
    assert isinstance(config['mask_seed'], int) and config['mask_seed'] >= 0
    assert config['development_samples'] == 128
    assert config['lr_drop_epoch'] == 10 and config['clip_grad_norm'] == .1


def set_learning_rates(optimizer, config, epoch, next_update):
    assert 1 <= epoch <= 20 and next_update >= 1
    factor = min(next_update / config['warmup_updates'], 1.)
    factor *= .1 if epoch > config['lr_drop_epoch'] else 1.
    rates = [config['learning_rate'] * config['backbone_multiplier'],
             config['learning_rate']]
    assert len(optimizer.param_groups) == 2
    for group, rate in zip(optimizer.param_groups, rates):
        group['lr'] = rate * factor


def step(model, optimizer, config, epoch, index, data, state, probe_rng, updates):
    """One native microbatch; gradients update only at accumulation boundaries."""
    device = next(model.parameters()).device
    templates = [value.to(device) for value in data['template_images']]
    searches = [value.to(device) for value in data['search_images']]
    boxes = data['search_anno'].to(device)
    set_learning_rates(optimizer, config, epoch, updates + 1)
    microstep = state.microsteps
    probe, view = None, None
    if microstep % config['utility_probe_interval'] == 0:
        probe_id = microstep // config['utility_probe_interval']
        view = 'C' if probe_id % 2 == 0 else 'M'
        regions = torch.randint(model.utility_repair.region_tokens.shape[0],
            (searches[0].shape[0], 1), generator=probe_rng).to(device)
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
    status.update(epoch=epoch, microstep=state.microsteps, loss=float(loss.detach()),
                  masks_sha256=[tensor_sha(value) for value in available + patch])
    if (index + 1) % config['accumulation'] == 0:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
        assert torch.isfinite(norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        status.update(update=updates, gradient_norm=float(norm),
                      learning_rates=[group['lr'] for group in optimizer.param_groups])
    return status, updates


def prediction_snapshot(outputs):
    """Retain native prediction tensors from every executed round for reload QA."""
    result = []
    for output in outputs:
        rounds = [output['round1'], output] if 'round1' in output else [output]
        result.append([{key: row[key].detach().cpu().clone() for key in
                        ('score_map', 'size_map', 'offset_map', 'pred_boxes')}
                       for row in rounds])
    return result


def check_reload(model, config, path, data, state):
    saved = torch.load(str(path), map_location='cpu')
    assert saved['config'] == config
    # Fresh CPU construction checks the architecture independently of the live model.
    reloaded, _ = build(config)
    reloaded.load_state_dict(saved['model'], strict=True)
    assert all(torch.equal(value, reloaded.state_dict()[name])
               for name, value in saved['model'].items())
    del reloaded
    device = next(model.parameters()).device
    templates = [value.to(device) for value in data['template_images']]
    searches = [value.to(device) for value in data['search_images']]
    model.eval()
    model.utility_repair.reset_selection()
    with torch.no_grad():
        before = prediction_snapshot(model(templates, searches))
        model.load_state_dict(saved['model'], strict=True)
        state.load_state_dict(saved['curriculum'])
        model.utility_repair.reset_selection()
        after = prediction_snapshot(model(templates, searches))
    assert len(before) == len(after) == 3
    for left, right in zip(before, after):
        assert len(left) == len(right)
        assert all(torch.equal(a[key], b[key])
                   for a, b in zip(left, right) for key in a)


def run(config, verify_evidence, verify_smokes, smoke=False):
    """Run only after caller-owned parent/source checks; not a standalone launcher.

    The supplied checks must validate the chosen P curriculum, M1 decision and
    frozen F snapshot. Formal runs additionally require all five paired smokes.
    No existing P receipt is accepted here as an F authorization.
    """
    check_config(config)
    verify_evidence(config)
    if not smoke:
        verify_smokes(config)
    root = Path(config['output'] + ('_smoke' if smoke else ''))
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'run_config.json', config)
    save(root / 'source_sha256.json', read(config['source_manifest']))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(Path(config['snapshot']) / 'experiments/cadtrack/cadtrack.yaml'))
    model, initialization = build(config)
    save(root / 'initial_state.json', initialization)
    native_names = set(initialization['parameters_sha256'])
    save(root / 'feedback_initial_state.json', {
        name: dict(shape=list(value.shape), sha256=tensor_sha(value))
        for name, value in model.named_parameters() if name not in native_names})
    model.cuda().train()
    fix_bn(model, True)
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
    state = CurriculumState(config['curriculum_arm'], next(model.parameters()).device,
                            config['seed'])
    probe_rng = torch.Generator().manual_seed(config['seed'])
    development = None if smoke else torch.load(config['development_cache'], map_location='cpu')
    if not smoke:
        (root / 'round_diagnostics').mkdir()
    updates, history, started = 0, [], time.time()
    with (root / 'input_manifest.jsonl').open('x') as inputs, \
            (root / 'steps.jsonl').open('x') as steps:
        for epoch in range(1, 21):
            model.train()
            fix_bn(model, True)
            optimizer.zero_grad(set_to_none=True)
            for index, data in enumerate(loader):
                record = record_batch(data, names, epoch, index, inputs)
                status, updates = step(
                    model, optimizer, config, epoch, index, data, state, probe_rng, updates)
                status['input_sha256'] = record['tensors_sha256']
                if 'update' in status:
                    status.update(peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                  seconds=time.time() - started)
                    if updates == 1 or updates % 25 == 0:
                        print(json.dumps(status), flush=True)
                steps.write(json.dumps(status) + '\n')
                steps.flush()
                if smoke and updates == 4:
                    path = root / 'smoke.pth.tar'
                    checkpoint(path, model, optimizer, config, epoch, updates,
                               state, probe_rng, index + 1, len(loader))
                    check_reload(model, config, path, data, state)
                    verify_evidence(config)
                    save(root / 'smoke_verified.json', dict(status='passed', updates=updates,
                        fresh_cpu_strict_load=True, all_rounds_reload_bitwise_equal=True,
                        checkpoint_sha256=sha(path),
                        input_manifest_sha256=sha(root / 'input_manifest.jsonl'),
                        scope='Four real optimizer updates only; no official or full-run proof'))
                    return
            result = validate(model, development, cfg, config['feedback']['mode'])
            result.update(epoch=epoch, updates=updates, seconds=time.time() - started)
            diagnostic = root / 'round_diagnostics' / ('epoch%02d.json' % epoch)
            save(diagnostic, result)
            summary = {key: value for key, value in result.items() if key != 'round_samples'}
            summary.update(diagnostic_path=str(diagnostic), diagnostic_sha256=sha(diagnostic))
            history.append(summary)
            save(root / 'development_history.json', history)
            verify_evidence(config)
            if epoch in config['checkpoint_epochs']:
                checkpoint(root / ('epoch%02d.pth.tar' % epoch), model, optimizer,
                           config, epoch, updates, state, probe_rng, len(loader), len(loader))
            print('DEVELOPMENT_LOSS ' + json.dumps(summary), flush=True)
    selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
    save(root / 'completed.json', dict(status='complete', epochs=20, updates=updates,
        pid=os.getpid(), selected_epoch=selected['epoch'],
        final_five=[dict(epoch=epoch, path=str(root / ('epoch%02d.pth.tar' % epoch)),
                        sha256=sha(root / ('epoch%02d.pth.tar' % epoch)))
                    for epoch in range(16, 21)],
        selection_rule=config['development_selection'], early_stopped=False,
        all_checkpoints_retained=True, official_test_selection_used=False,
        input_manifest_sha256=sha(root / 'input_manifest.jsonl'),
        resume_supported=False,
        scope='Training only; ten full official evaluations per F arm still required'))
