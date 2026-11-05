"""Audit the new OSTrack six-arm saved evidence, without opening test images."""
import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

import torch

from research.curriculum_p.engine import ARMS
from research.curriculum_p.pacing import schedule
from research.curriculum_p.protocol_prepare import CAMPAIGN
from research.curriculum_p.protocol_train import check_config, check_evidence, check_smokes
from research.protocol_r.common import MISSING, read, save, sha, tensor_sha


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def check_history(config, history, done):
    assert done['status'] == 'complete'
    assert done['epochs'] == config['epochs'] == len(history) == 20
    assert not done['early_stopped'] and not done['official_test_selection_used']
    assert done['all_checkpoints_retained'] and not done['resume_supported']
    assert done['selection_rule'] == config['development_selection']
    updates = config['samples_per_epoch'] // (config['microbatch'] * config['accumulation'])
    for epoch, row in enumerate(history, 1):
        assert row['epoch'] == epoch and row['updates'] == epoch * updates
        assert row['samples'] == 128 and set(row['task_loss']) == {'11', '01', '10'}
        assert all(math.isfinite(value) for value in row['task_loss'].values())
        average = sum(row['task_loss'].values()) / 3.
        assert math.isfinite(row['mean_task_loss'])
        assert math.isclose(average, row['mean_task_loss'], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isfinite(row['seconds']) and row['seconds'] >= 0
    selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
    assert selected['epoch'] == done['selected_epoch']
    assert done['updates'] == 20 * updates
    assert [row['epoch'] for row in done['final_five']] == list(range(16, 21))
    return selected


def check_step(config, index, row):
    batches = config['samples_per_epoch'] // config['microbatch']
    epoch = index // batches + 1
    assert row['epoch'] == epoch and row['microstep'] == row['curriculum_microstep'] == index + 1
    arm = config['curriculum_arm']
    assert row['curriculum_arm'] == arm and row['state_norm'] == (arm == 'paced+norm')
    probability = schedule('paced' if arm == 'paced+norm' else arm)[epoch - 1]
    low = max(.2, probability - .3) if arm.startswith('paced') else probability
    assert row['p_epoch'] == probability and row['p_hard'] == low
    assert 0 <= row['hard_fraction'] <= 1
    expected = probability + row['hard_fraction'] * (low - probability)
    assert abs(row['mean_missing_probability'] - expected) < 1e-6
    assert 0 <= row['actual_search_missing_fraction'] <= 1
    assert len(row['masks_sha256']) == 8
    keys = ['loss', 'task_c', 'task_m', 'kd_forward', 'kd_reverse', 'ema_c', 'reverse_fraction']
    if arm == 'paced+norm':
        keys += ['task_m_normalized', 'ema_m_11', 'ema_m_01', 'ema_m_10']
        assert all(row[key] > 0 for key in ['ema_m_11', 'ema_m_01', 'ema_m_10'])
    assert all(math.isfinite(row[key]) for key in keys)
    if index % config['utility_probe_interval'] == 0:
        assert row['probe_view'] == ('C' if index // config['utility_probe_interval'] % 2 == 0 else 'M')
        probe = row['utility']
        assert probe['search_index'] == (index // config['utility_probe_interval'] // 2) % 3
        assert len(probe['regions']) == config['microbatch']
        assert all(isinstance(region, int) and 0 <= region < 16 for region in probe['regions'])
        assert all(math.isfinite(probe[key]) for key in (
            'loss_off', 'loss_on', 'target_mean', 'beneficial_fraction',
            'prediction_mean', 'utility_loss'))
    else:
        assert 'probe_view' not in row and 'utility' not in row
    fields = {'update', 'gradient_norm', 'learning_rates', 'peak_gpu_bytes', 'seconds'}
    if (index + 1) % config['accumulation'] == 0:
        update = (index + 1) // config['accumulation']
        assert row['update'] == update and math.isfinite(row['gradient_norm'])
        assert row['gradient_norm'] >= 0
        factor = min(update / config['warmup_updates'], 1.)
        factor *= .1 if epoch > config['lr_drop_epoch'] else 1.
        expected_rates = [config['learning_rate'] * config['backbone_multiplier'] * factor,
                          config['learning_rate'] * factor]
        assert len(row['learning_rates']) == 2
        assert all(abs(a - b) < 1e-15 for a, b in zip(row['learning_rates'], expected_rates))
        assert row['peak_gpu_bytes'] > 0 and math.isfinite(row['seconds']) and row['seconds'] >= 0
    else:
        assert not fields.intersection(row)


def check_pair(left, right):
    assert left['microstep'] == right['microstep'] and left['epoch'] == right['epoch']
    assert left['input_sha256'] == right['input_sha256']
    assert left['masks_sha256'][-3:] == right['masks_sha256'][-3:]
    if (not left['curriculum_arm'].startswith('paced') and
            not right['curriculum_arm'].startswith('paced') and left['p_epoch'] == right['p_epoch']):
        assert left['masks_sha256'] == right['masks_sha256']


def check_checkpoint(saved, config, initial, epoch, last_step):
    batches = config['samples_per_epoch'] // config['microbatch']
    updates = epoch * batches // config['accumulation']
    assert saved['config'] == config and saved['epoch'] == epoch and saved['updates'] == updates
    assert saved['resume_supported'] is False
    assert saved['loader_position'] == dict(epoch=epoch, consumed_batches=batches, batches_per_epoch=batches)
    state = saved['curriculum']
    assert state['arm'] == config['curriculum_arm'] and state['microsteps'] == epoch * batches
    assert state['complete_momentum'] == state['missing_momentum'] == .99
    assert float(state['ema_c']) == last_step['ema_c']
    assert state['ema_m'].shape == (3,) and torch.isfinite(state['ema_m']).all()
    if config['curriculum_arm'] == 'paced+norm':
        assert state['ema_m'].tolist() == [last_step['ema_m_' + key] for key in ('11', '01', '10')]
    else:
        assert torch.equal(state['ema_m'], torch.ones(3))
    assert state['generator'].dtype == torch.uint8 and state['generator'].numel() > 0
    rng = saved['rng']
    assert set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'utility_probe'}
    assert len(rng['python']) == 3 and rng['numpy'][0] == 'MT19937'
    assert rng['torch_cuda'] and all(value.dtype == torch.uint8 for value in rng['torch_cuda'])
    assert all(rng[key].dtype == torch.uint8 and rng[key].numel() > 0
               for key in ('torch_cpu', 'utility_probe'))
    optimizer = saved['optimizer']
    assert len(optimizer['param_groups']) == 2 and optimizer['state']
    for group, rate in zip(optimizer['param_groups'], last_step['learning_rates']):
        assert group['lr'] == rate and group['weight_decay'] == config['weight_decay']
    for value in optimizer['state'].values():
        # Utility selector parameters receive gradients only on probe updates.
        assert 0 < int(value['step']) <= updates
        assert all(torch.isfinite(value[key]).all() for key in ('exp_avg', 'exp_avg_sq'))
    assert max(int(value['step']) for value in optimizer['state'].values()) == updates
    model = saved['model']
    assert all(torch.isfinite(value).all() for value in model.values())
    assert initial['bn_sha256']
    assert all(tensor_sha(model[name]) == digest for name, digest in initial['bn_sha256'].items())
    changed = [name for name, digest in initial['parameters_sha256'].items()
               if tensor_sha(model[name]) != digest]
    assert all(any(token in name for name in changed)
               for token in ('backbone', 'box_head', 'utility_repair'))
    return len(changed)


def verify_arm(arm):
    config_path = CAMPAIGN / (arm + '.json')
    config = read(config_path)
    check_config(config)
    check_evidence(config)
    check_smokes(config)
    root = Path(config['output'])
    assert read(root / 'run_config.json') == config
    assert read(root / 'source_sha256.json') == read(config['source_manifest'])
    boundary = read(root / 'data_boundary.json')
    names_path = Path(config['snapshot']) / 'lib/train/data_specs/lasher_train.txt'
    names = names_path.read_text().splitlines()
    assert boundary == dict(split='train882', train_names=names,
        development_split='val97', development_samples=128,
        accepted_tuple_identities=True, rejected_crop_reads_recorded=False,
        train_names_sha256=config['train_names_sha256'])
    assert len(names) == len(set(names)) == 882 and sha(names_path) == config['train_names_sha256']
    names = set(names)
    done = read(root / 'completed.json')
    selected = check_history(config, read(root / 'development_history.json'), done)
    assert sha(root / 'input_manifest.jsonl') == done['input_manifest_sha256']
    batches = config['samples_per_epoch'] // config['microbatch']
    total, last_steps = 0, {}
    for index, (identity, step) in enumerate(zip_longest(
            records(root / 'input_manifest.jsonl'), records(root / 'steps.jsonl'))):
        assert identity is not None and step is not None
        check_step(config, index, step)
        assert identity['epoch'] == index // batches + 1 and identity['index'] == index % batches
        assert identity['tensors_sha256'] == step['input_sha256']
        assert set(step['input_sha256']) == {'template_images', 'search_images', 'search_anno'}
        assert len(identity['identities']) == config['microbatch']
        for item in identity['identities']:
            assert item['sequence'] in names
            assert len(item['template_frames']) == 2 and len(item['search_frames']) == 3
            assert all(isinstance(frame, int) and frame >= 0
                       for frame in item['template_frames'] + item['search_frames'])
        if (index + 1) % batches == 0:
            last_steps[step['epoch']] = step
        total += 1
    assert total == 20 * batches
    initial = read(root / 'initial_state.json')
    assert initial['missing_keys'] == MISSING and not initial['unexpected_keys']
    assert initial['initializer_sha256'] == config['initializer_sha256']
    assert initial == read(Path(config['output'] + '_smoke') / 'initial_state.json')
    assert sorted(path.name for path in root.glob('epoch*.pth.tar')) == [
        'epoch%02d.pth.tar' % epoch for epoch in config['checkpoint_epochs']]
    for item in done['final_five']:
        assert Path(item['path']) == root / ('epoch%02d.pth.tar' % item['epoch'])
        assert sha(item['path']) == item['sha256']
    changed = {}
    for epoch in config['checkpoint_epochs']:
        saved = torch.load(str(root / ('epoch%02d.pth.tar' % epoch)), map_location='cpu')
        changed[str(epoch)] = check_checkpoint(saved, config, initial, epoch, last_steps[epoch])
        del saved
    evidence = {str(root / name): sha(root / name) for name in (
        'completed.json', 'run_config.json', 'source_sha256.json', 'data_boundary.json',
        'initial_state.json', 'input_manifest.jsonl', 'steps.jsonl', 'development_history.json')}
    evidence[str(config_path)] = sha(config_path)
    return dict(status='passed', arm=arm, completed_sha256=sha(root / 'completed.json'),
        selected_epoch=selected['epoch'], changed_parameters=changed,
        microsteps=total, evidence_sha256=evidence,
        verifier_sha256=sha(Path(__file__)), resume_supported=False,
        scope='Saved records/weights only; not independent image/optimizer replay or module efficacy')


def verify_pairs():
    roots = [Path(read(CAMPAIGN / (arm + '.json'))['output']) for arm in ARMS]
    for name in ('initial_state.json', 'data_boundary.json', 'input_manifest.jsonl'):
        assert len({sha(root / name) for root in roots}) == 1, name
    total = 0
    for rows in zip_longest(*(records(root / 'steps.jsonl') for root in roots)):
        assert all(row is not None for row in rows)
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                check_pair(left, right)
        total += 1
    return dict(microsteps=total, initial_and_inputs_exact=True, patch_masks_exact=True,
                equal_probability_control_masks_exact=True,
                paced_availability_masks_may_differ=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.arm:
        result = verify_arm(args.arm)
        path = CAMPAIGN / (args.arm + '_training_verified.json')
    else:
        arms = {arm: verify_arm(arm) for arm in ARMS}
        result = dict(status='passed', arms=arms, pairing=verify_pairs())
        path = CAMPAIGN / 'paired_training_verified.json'
    assert not path.exists()
    save(path, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
