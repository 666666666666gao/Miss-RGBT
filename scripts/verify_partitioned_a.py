"""Recompute paired input boundaries, checkpoint selection and weight witnesses."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

PROJECT = Path('/home/gaob/Miss-RGBT')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['smoke', 'formal'])
    args = parser.parse_args()
    stem = 'server28_partitioned_a_20261102_'+args.phase
    campaign = PROJECT/'reports'/stem
    launch = read(campaign/'launch.json')
    names_path = PROJECT/'runtime/task_utility_20261018/lib/train/data_specs/lasher_train.txt'
    names = names_path.read_text().splitlines()
    expected = dict(train=[name for index, name in enumerate(names) if index % 5],
                    development=names[::5][:88], audit=names[::5][88:])
    assert not set(expected['train']).intersection(expected['development']+expected['audit'])
    assert not set(expected['development']).intersection(expected['audit'])
    paired_inputs, paired_steps, initials, boundaries, results = [], [], [], [], {}
    for record in launch:
        selector = record['selector']
        root = Path(record['config']['output'])
        config, completed = read(root/'run_config.json'), read(root/'completed.json')
        assert completed['status'] == 'complete' and completed['all_checkpoints_retained']
        assert completed['official_test_selection_used'] is False
        assert config == record['config']
        assert sha(record['command'][2]) == record['script_sha256']
        assert sha(campaign/'preregistered_protocol.md') == record['protocol_sha256']
        for path, digest in read(root/'source_sha256.json').items():
            assert sha(path) == digest, path
        boundary = read(root/'data_boundary.json')
        assert boundary['partitions'] == expected
        assert boundary['train_list_sha256'] == sha(names_path)
        for path, digest in boundary['prior_sha256'].items():
            candidate = Path(path) if Path(path).is_absolute() else PROJECT/'reports'/path
            if not candidate.suffix:
                candidate = candidate/'report.json'
            assert sha(candidate) == digest, str(candidate)
        boundaries.append(boundary)
        inputs = lines(root/'input_manifest.jsonl')
        counts = dict(train=0, development=0)
        for row in inputs:
            phase = row['phase']
            assert phase in counts
            for identity in row['identities']:
                name = identity['sequence']
                assert name in expected[phase]
                frames = identity['template_frames']+identity['search_frames']
                assert len(frames) == 5
                assert not set(frames).intersection(boundary['excluded_frames'].get(name, []))
                counts[phase] += 1
        assert counts['development'] == config['development_samples']
        assert counts['train'] == completed['epochs']*config['samples_per_epoch']
        steps = lines(root/'steps.jsonl')
        assert len(steps) == counts['train']//config['microbatch']
        assert completed['updates'] == len(steps)//config['accumulation']
        for index, row in enumerate(steps):
            assert row['microstep'] == index
            for key in ['loss', 'task_c', 'task_m', 'kd_forward', 'kd_reverse']:
                assert math.isfinite(row[key]), (index, key)
        train_inputs = [row for row in inputs if row['phase'] == 'train']
        assert all(left['tensors_sha256'] == right['input_sha256']
                   for left, right in zip(train_inputs, steps))
        history = read(root/'development_history.json')
        assert len(history) == completed['epochs']
        reference, stale, stop = float('inf'), 0, False
        for epoch, row in enumerate(history, 1):
            assert row['epoch'] == epoch and row['samples'] == config['development_samples']
            average = sum(row['task_loss'].values())/3
            assert math.isfinite(average) and abs(average-row['mean_task_loss']) < 1e-12
            if epoch == config['lr_drop_epoch']+1:
                reference, stale = float('inf'), 0
            if average < reference*(1-config['early_stop_relative_delta']):
                reference, stale = average, 0
            elif epoch > config['lr_drop_epoch']:
                stale += 1
            stop = epoch >= config['early_stop_min_epochs'] and stale >= config['early_stop_patience']
            assert not stop or epoch == len(history)
        assert stop == completed['early_stopped']
        selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
        assert selected['epoch'] == completed['selected_epoch']
        assert len(completed['final_five']) == 5
        assert [row['epoch'] for row in completed['final_five']] == list(range(completed['epochs']-4, completed['epochs']+1))
        for row in completed['final_five']:
            assert sha(row['path']) == row['sha256']
            saved = torch.load(row['path'], map_location='cpu')
            assert saved['epoch'] == row['epoch'] and saved['config'] == config
            assert all(torch.isfinite(value).all() for value in saved['model'].values())
            del saved
        assert sha(completed['selected_checkpoint']) == completed['selected_sha256']
        first = torch.load(str(root/'epoch01.pth.tar'), map_location='cpu')['model']
        last = torch.load(str(root/('epoch%02d.pth.tar' % completed['epochs'])), map_location='cpu')['model']
        changed = [name for name in first if not torch.equal(first[name], last[name])]
        assert any('backbone' in name for name in changed)
        assert any('box_head' in name for name in changed)
        assert any('utility_repair' in name and 'selection_rng' not in name for name in changed)
        buffers = [name for name in first if any(key in name for key in
                   ['running_mean', 'running_var', 'num_batches_tracked'])]
        assert buffers and all(torch.equal(first[name], last[name]) for name in buffers)
        del first, last
        initials.append(read(root/'initial_state.json'))
        paired_inputs.append(inputs)
        paired_steps.append([(row['input_sha256'], row['masks_sha256']) for row in steps])
        results[selector] = dict(epochs=completed['epochs'], updates=completed['updates'],
            selected_epoch=selected['epoch'], finite_final_five=True,
            changed_state_tensors=len(changed), frozen_bn_buffers=len(buffers),
            counts=counts, peak_gpu_bytes=max(row.get('peak_gpu_bytes', 0) for row in steps),
            completed_sha256=sha(root/'completed.json'), input_manifest_sha256=sha(root/'input_manifest.jsonl'))
    assert initials[0] == initials[1]
    assert boundaries[0] == boundaries[1]
    assert paired_inputs[0] == paired_inputs[1]
    assert paired_steps[0] == paired_steps[1]
    if args.phase == 'smoke':
        assert all(value['updates'] == 12 and value['epochs'] == 6 for value in results.values())
    result = dict(status='passed', phase=args.phase, arms=results,
        paired_inputs_and_masks_exact=True, initial_parameters_exact=True,
        scope='Saved evidence verification; shared trainer sources, not independent image/gradient replay')
    destination = campaign/'verification.json'
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
