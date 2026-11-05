"""Verify all P evidence and final-five weights without opening test datasets."""
import argparse
import hashlib
from itertools import combinations, zip_longest
import json
import math
from pathlib import Path

from prepare import PROJECT, sha
from schedule import ARMS, epoch_config, schedule


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def check_history(config, history, completed):
    assert len(history) == completed['epochs'] == config['epochs'] == 20
    assert completed['early_stopped'] is False
    assert completed['official_test_selection_used'] is False
    assert completed['all_checkpoints_retained'] is True
    assert completed['curriculum_arm'] == config['curriculum_arm']
    assert completed['schedule'] == schedule(config['curriculum_arm'])
    updates_per_epoch = config['samples_per_epoch']//config['microbatch']//config['accumulation']
    for epoch, row in enumerate(history, 1):
        assert row['epoch'] == epoch and row['samples'] == config['development_samples']
        assert row['updates'] == epoch*updates_per_epoch
        assert row['missing_probability'] == schedule(config['curriculum_arm'])[epoch-1]
        assert set(row['task_loss']) == {'11', '01', '10'}
        assert all(math.isfinite(value) for value in row['task_loss'].values())
        average = sum(row['task_loss'].values())/3.
        assert math.isfinite(average) and abs(average-row['mean_task_loss']) < 1e-12
    selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
    assert completed['selected_epoch'] == selected['epoch']
    assert [row['epoch'] for row in completed['final_five']] == list(range(16, 21))
    assert completed['updates'] == 20*updates_per_epoch
    return selected


def check_step(config, index, step):
    batches = config['samples_per_epoch']//config['microbatch']
    epoch = index//batches+1
    assert step['microstep'] == index and step['epoch'] == epoch
    local = epoch_config(config, epoch)
    assert step['availability_probability'] == local['complete_rgb_missing_tir_missing_probability']
    assert step['missing_probability'] == schedule(config['curriculum_arm'])[epoch-1]
    assert step['masks_sha256'][-3:] == step['patch_sha256'] and len(step['patch_sha256']) == 3
    for key in ['loss', 'task_c', 'task_m', 'kd_forward', 'kd_reverse']:
        assert math.isfinite(step[key]), (index, key)
    if (index % batches+1) % config['accumulation'] == 0:
        assert step['update'] == (index+1)//config['accumulation']
        assert math.isfinite(step['grad_norm'])
        factor = .1**((epoch-1)//config['lr_drop_epoch'])
        expected = [config['learning_rate']*config['backbone_multiplier']*factor,
                    config['learning_rate']*factor]
        assert all(abs(a-b) < 1e-15 for a, b in zip(step['learning_rates'], expected))
        assert len(step['learning_rates']) == 2


def check_pair(left, right):
    assert left['input_sha256'] == right['input_sha256']
    assert left['patch_sha256'] == right['patch_sha256']
    assert left['microstep'] == right['microstep'] and left['epoch'] == right['epoch']
    if left['missing_probability'] == right['missing_probability']:
        assert left['masks_sha256'] == right['masks_sha256']


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def check_weights(root, config, completed, initial):
    import torch
    torch.set_num_threads(1)
    assert sorted(path.name for path in root.glob('epoch*.pth.tar')) == [
        'epoch%02d.pth.tar' % epoch for epoch in range(1, 21)]
    assert initial['bn_sha256']
    first = torch.load(str(root/'epoch01.pth.tar'), map_location='cpu')['model']
    assert all(tensor_hash(first[name]) == digest for name, digest in initial['bn_sha256'].items())
    changes = {}
    for row in completed['final_five']:
        path = root/('epoch%02d.pth.tar' % row['epoch'])
        assert str(path) == row['path'] and sha(path) == row['sha256']
        saved = torch.load(str(path), map_location='cpu')
        assert saved['config'] == config and saved['epoch'] == row['epoch']
        assert saved['updates'] == row['epoch']*config['samples_per_epoch']//(
            config['microbatch']*config['accumulation'])
        model = saved['model']
        assert set(model) == set(first)
        assert all(torch.isfinite(value).all() for value in model.values())
        assert all(tensor_hash(model[name]) == digest for name, digest in initial['bn_sha256'].items())
        changed = [name for name in first if not torch.equal(first[name], model[name])]
        assert any('backbone' in name for name in changed)
        assert any('box_head' in name for name in changed)
        assert any('utility_repair' in name and 'selection_rng' not in name for name in changed)
        changes[str(row['epoch'])] = len(changed)
        del model, saved
    assert str(root/('epoch%02d.pth.tar' % completed['selected_epoch'])) == completed['selected_checkpoint']
    assert sha(completed['selected_checkpoint']) == completed['selected_sha256']
    return dict(final_five_changed_state_tensors=changes, frozen_bn_buffers=len(initial['bn_sha256']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['smoke', 'formal'])
    args = parser.parse_args()
    campaign = PROJECT/'reports'/('server28_curriculum_p_20261102_'+args.phase)
    prepared = read(campaign/'prepared.json')
    assert [row['arm'] for row in prepared['arms']] == list(ARMS)
    names_path = PROJECT/'runtime/task_utility_20261018/lib/train/data_specs/lasher_train.txt'
    names = names_path.read_text().splitlines()
    expected = dict(train=[name for index, name in enumerate(names) if index % 5],
                    development=names[::5][:88], audit=names[::5][88:])
    assert len(names) == len(set(names)) == 882
    assert [len(expected[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    roots, initials, boundaries, configs, results = [], [], [], [], {}
    for entry in prepared['arms']:
        assert sha(entry['config']) == entry['config_sha256']
        config = read(entry['config'])
        assert config['phase'] == args.phase and config['curriculum_arm'] == entry['arm']
        assert entry['missing_frequency'] == schedule(entry['arm'])
        assert config['samples_per_epoch'] == (20 if args.phase == 'smoke' else 20000)
        assert config['development_samples'] == (8 if args.phase == 'smoke' else 128)
        assert config['seed'] == (65201 if args.phase == 'smoke' else 2029)
        assert config['development_seed'] == (65202 if args.phase == 'smoke' else 65102)
        assert config['microbatch'] == 5 and config['accumulation'] == 2
        root = Path(config['output'])
        assert config == read(root/'run_config.json')
        launch = read(campaign/('launch_'+entry['arm']+'.json'))
        assert launch['config_sha256'] == entry['config_sha256']
        assert launch['config'] == entry['config']
        assert sha(launch['command'][2]) == launch['trainer_sha256']
        assert launch['protocol_sha256'] == sha(campaign/'protocol.md')
        for key in ['parent_completion_sha256', 'exclude_manifest_sha256']:
            assert config[key] == prepared[key]
            assert all(sha(path) == digest for path, digest in config[key].items())
        for path, digest in read(root/'source_sha256.json').items():
            assert sha(path) == digest, path
        boundary = read(root/'data_boundary.json')
        assert boundary['partitions'] == expected and boundary['train_list_sha256'] == sha(names_path)
        for name, digest in boundary['prior_sha256'].items():
            path = Path(name) if Path(name).is_absolute() else PROJECT/'reports'/name
            if not path.suffix:
                path = path/'report.json'
            assert sha(path) == digest, str(path)
        excluded = {name: set(values) for name, values in boundary['excluded_frames'].items()}
        counts = dict(train=0, development=0)
        steps = records(root/'steps.jsonl')
        step_count = 0
        for row in records(root/'input_manifest.jsonl'):
            phase = row['phase']
            assert phase in counts
            for identity in row['identities']:
                name = identity['sequence']
                assert name in expected[phase]
                frames = identity['template_frames']+identity['search_frames']
                assert len(frames) == 5 and not set(frames).intersection(excluded.get(name, set()))
            counts[phase] += len(row['identities'])
            if phase == 'train':
                step = next(steps)
                check_step(config, step_count, step)
                assert step['input_sha256'] == row['tensors_sha256']
                assert len(row['identities']) == config['microbatch']
                batches = config['samples_per_epoch']//config['microbatch']
                assert row['epoch'] == step_count//batches+1 and row['microstep'] == step_count % batches
                step_count += 1
        assert next(steps, None) is None
        assert counts == dict(train=20*config['samples_per_epoch'], development=config['development_samples'])
        assert step_count == counts['train']//config['microbatch']
        completed = read(root/'completed.json')
        assert completed['status'] == 'complete'
        selected = check_history(config, read(root/'development_history.json'), completed)
        initial = read(root/'initial_state.json')
        weights = check_weights(root, config, completed, initial)
        results[entry['arm']] = dict(weights, counts=counts, microsteps=step_count,
            selected_epoch=selected['epoch'], completed_sha256=sha(root/'completed.json'),
            input_manifest_sha256=sha(root/'input_manifest.jsonl'))
        roots.append(root)
        initials.append(initial)
        boundaries.append(boundary)
        configs.append({key: value for key, value in config.items() if key not in ['curriculum_arm', 'output']})
    assert all(value == initials[0] for value in initials)
    assert all(value == boundaries[0] for value in boundaries)
    assert all(value == configs[0] for value in configs)
    for root in roots[1:]:
        for left, right in zip_longest(records(roots[0]/'input_manifest.jsonl'),
                                      records(root/'input_manifest.jsonl')):
            assert left is not None and right is not None and left == right
    for left_root, right_root in combinations(roots, 2):
        for left, right in zip_longest(records(left_root/'steps.jsonl'), records(right_root/'steps.jsonl')):
            assert left is not None and right is not None
            check_pair(left, right)
    result = dict(status='PASS', phase=args.phase, arms=results, all_twenty_epochs=True,
        initial_parameters_and_bn_exact=True, inputs_and_patch_masks_paired=True,
        all_checkpoints_retained=True, official_test_selection_used=False,
        prepared_sha256=sha(campaign/'prepared.json'),
        source_sha256={str(path.resolve()): sha(path) for path in [
            Path(__file__), Path(__file__).with_name('prepare.py'),
            Path(__file__).with_name('schedule.py')]},
        scope='Saved P evidence and weights; not independent image/optimizer replay or method efficacy')
    with (campaign/'verification.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
