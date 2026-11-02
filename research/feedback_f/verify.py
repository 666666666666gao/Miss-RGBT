"""Verify saved F image-run evidence; never evaluate official test images."""
import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

from prepare import PROJECT, sha
import train as trainer


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def check_history(config, history, completed):
    assert completed['status'] == 'complete'
    assert len(history) == completed['epochs'] == config['epochs'] == 20
    assert completed['early_stopped'] is False
    assert completed['official_test_selection_used'] is False
    assert completed['all_checkpoints_retained'] is True
    assert completed['feedback'] == config['feedback']
    assert completed['curriculum_arm'] == config['curriculum_arm']
    per_epoch = config['samples_per_epoch']//(config['microbatch']*config['accumulation'])
    for epoch, row in enumerate(history, 1):
        assert row['epoch'] == epoch and row['updates'] == epoch*per_epoch
        assert row['samples'] == config['development_samples']
        assert set(row['task_loss']) == {'11', '01', '10'}
        assert all(math.isfinite(value) for value in row['task_loss'].values())
        assert abs(sum(row['task_loss'].values())/3.-row['mean_task_loss']) < 1e-12
    selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
    assert completed['updates'] == 20*per_epoch
    assert completed['selected_epoch'] == selected['epoch']
    assert [row['epoch'] for row in completed['final_five']] == list(range(16, 21))
    return selected


def check_step(config, index, step):
    batches = config['samples_per_epoch']//config['microbatch']
    epoch = index//batches+1
    assert step['microstep'] == index and step['epoch'] == epoch
    assert step['mask_seed'] == config['mask_seed']+index
    local = trainer.epoch_config(config, epoch)
    assert step['availability_probability'] == local['complete_rgb_missing_tir_missing_probability']
    assert len(step['masks_sha256']) == 8
    assert math.isfinite(step['loss'])
    rounds = [step] if config['feedback']['mode'] == 'single' else [
        step['round1'], step['round2']]
    for status in rounds:
        for key in ['task_c', 'task_m', 'kd_forward', 'kd_reverse']:
            assert math.isfinite(status[key]), (index, key)
    if (index % batches+1) % config['accumulation'] == 0:
        assert step['update'] == (index+1)//config['accumulation']
        assert math.isfinite(step['grad_norm'])
        factor = .1**((epoch-1)//config['lr_drop_epoch'])
        expected = [config['learning_rate']*config['backbone_multiplier']*factor,
                    config['learning_rate']*factor]
        assert len(step['learning_rates']) == 2
        assert all(abs(left-right) < 1e-15
                   for left, right in zip(step['learning_rates'], expected))


def check_rounds(root, config, history):
    if config['feedback']['mode'] == 'single':
        assert all('round_comparison' not in row for row in history)
        return
    for row in history:
        raw = read(root/('development_round_samples_epoch%02d.json' % row['epoch']))
        assert set(raw) == {'11', '01', '10'}
        for label in raw:
            before = [value for batch in raw[label] for value in batch['round1']]
            after = [value for batch in raw[label] for value in batch['round2']]
            assert [batch['batch_index'] for batch in raw[label]] == list(range(
                (config['development_samples']+3)//4))
            assert len(before) == len(after) == config['development_samples']
            assert all(math.isfinite(value) for value in before+after)
            summary = row['round_comparison'][label]
            count = len(before)
            lower = sum(right < left for left, right in zip(before, after))
            higher = sum(right > left for left, right in zip(before, after))
            assert summary['samples'] == count
            for key, number in [('lower_loss', lower), ('higher_loss', higher),
                                ('equal_loss', count-lower-higher)]:
                assert summary[key] == number
                assert summary[key+'_fraction'] == number/count
            for key, value in [('round1_mean', sum(before)/count),
                               ('round2_mean', sum(after)/count),
                               ('mean_delta', (sum(after)-sum(before))/count)]:
                assert abs(summary[key]-value) < 1e-12
            assert abs(row['task_loss'][label]-sum(after)/count) < 1e-5


def check_weights(root, config, completed, initial):
    import torch
    torch.set_num_threads(1)
    assert sorted(path.name for path in root.glob('epoch*.pth.tar')) == [
        'epoch%02d.pth.tar' % epoch for epoch in range(1, 21)]
    first = torch.load(str(root/'epoch01.pth.tar'), map_location='cpu')['model']
    assert initial['bn_sha256']
    assert all(trainer.base.tensor_hash(first[name]) == digest
               for name, digest in initial['bn_sha256'].items())
    changes = {}
    for entry in completed['final_five']:
        path = root/('epoch%02d.pth.tar' % entry['epoch'])
        assert str(path) == entry['path'] and sha(path) == entry['sha256']
        saved = torch.load(str(path), map_location='cpu')
        assert saved['config'] == config and saved['epoch'] == entry['epoch']
        assert saved['updates'] == entry['epoch']*config['samples_per_epoch']//(
            config['microbatch']*config['accumulation'])
        model = saved['model']
        assert set(model) == set(first)
        assert all(torch.isfinite(value).all() for value in model.values())
        assert all(trainer.base.tensor_hash(model[name]) == digest
                   for name, digest in initial['bn_sha256'].items())
        changed = [name for name in first if not torch.equal(first[name], model[name])]
        for component in ['backbone', 'box_head', 'utility_repair']:
            assert any(component in name and 'selection_rng' not in name for name in changed)
        if config['feedback']['mode'] in ['shallow', 'deep']:
            assert any(name.startswith('feedback.') for name in changed)
        changes[str(entry['epoch'])] = len(changed)
        del model, saved
    assert completed['selected_checkpoint'] == str(root/(
        'epoch%02d.pth.tar' % completed['selected_epoch']))
    assert sha(completed['selected_checkpoint']) == completed['selected_sha256']
    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    campaign = Path(args.campaign)
    prepared = read(campaign/'prepared.json')
    assert prepared['phase'] in ['smoke', 'formal']
    assert [row['arm'] for row in prepared['arms']] == list(trainer.ARMS)
    assert sha(prepared['source_config']) == prepared['source_config_sha256']
    for path, digest in prepared['source_sha256'].items():
        assert sha(path) == digest
    names_path = PROJECT/'runtime/task_utility_20261018/lib/train/data_specs/lasher_train.txt'
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    expected = dict(train=[name for index, name in enumerate(names) if index % 5],
                    development=names[::5][:88], audit=names[::5][88:])
    roots, initials, boundaries, configs, results = [], [], [], [], {}
    for entry in prepared['arms']:
        assert sha(entry['config']) == entry['config_sha256']
        config = read(entry['config'])
        assert config['feedback']['mode'] == entry['arm']
        assert config['curriculum_arm'] == prepared['curriculum_arm']
        assert config['phase'] == prepared['phase']
        trainer.validate_parents(config)
        root = Path(config['output'])
        assert read(root/'run_config.json') == config
        for path, digest in read(root/'source_sha256.json').items():
            assert sha(path) == digest
        boundary = read(root/'data_boundary.json')
        assert boundary['partitions'] == expected
        assert boundary['train_list_sha256'] == sha(names_path)
        for name, digest in boundary['prior_sha256'].items():
            path = Path(name) if Path(name).is_absolute() else PROJECT/'reports'/name
            if not path.suffix:
                path = path/'report.json'
            assert sha(path) == digest
        excluded = {name: set(values) for name, values in boundary['excluded_frames'].items()}
        counts = dict(train=0, development=0)
        steps = records(root/'steps.jsonl')
        index = 0
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
                check_step(config, index, step)
                assert step['input_sha256'] == row['tensors_sha256']
                assert len(row['identities']) == config['microbatch']
                batches = config['samples_per_epoch']//config['microbatch']
                assert row['epoch'] == index//batches+1 and row['microstep'] == index % batches
                index += 1
        assert next(steps, None) is None
        assert counts == dict(train=20*config['samples_per_epoch'],
                              development=config['development_samples'])
        completed = read(root/'completed.json')
        history = read(root/'development_history.json')
        selected = check_history(config, history, completed)
        check_rounds(root, config, history)
        initial = read(root/'initial_state.json')
        changes = check_weights(root, config, completed, initial)
        results[entry['arm']] = dict(counts=counts, microsteps=index,
            selected_epoch=selected['epoch'], final_five_changed_tensors=changes,
            completed_sha256=sha(root/'completed.json'),
            input_manifest_sha256=sha(root/'input_manifest.jsonl'))
        roots.append(root)
        initials.append((initial['shared_parameters_sha256'], initial['bn_sha256']))
        boundaries.append(boundary)
        configs.append({key: value for key, value in config.items()
                        if key not in ['feedback', 'output']})
    assert all(value == initials[0] for value in initials)
    assert all(value == boundaries[0] for value in boundaries)
    assert all(value == configs[0] for value in configs)
    for root in roots[1:]:
        for left, right in zip_longest(records(roots[0]/'input_manifest.jsonl'),
                                      records(root/'input_manifest.jsonl')):
            assert left is not None and left == right
        for left, right in zip_longest(records(roots[0]/'steps.jsonl'), records(root/'steps.jsonl')):
            assert left is not None and right is not None
            for key in ['microstep', 'epoch', 'input_sha256', 'mask_seed', 'masks_sha256']:
                assert left[key] == right[key]
    result = dict(status='PASS', phase=prepared['phase'], arms=results,
        curriculum_arm=prepared['curriculum_arm'], prepared_sha256=sha(campaign/'prepared.json'),
        source_sha256={str(path.resolve()): sha(path) for path in [
            Path(__file__), Path(__file__).with_name('prepare.py'),
            Path(__file__).with_name('train.py')]},
        scope='Saved F image-run evidence; not optimizer replay or official efficacy')
    with (campaign/'verification.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
