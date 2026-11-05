"""Read-only audit of an already-written, fixed prefix of the live training pair."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import partitioned_a_official_queue as queue


def prefix(path, count):
    with path.open() as stream:
        text = list(itertools.islice(stream, count))
    assert len(text) == count and all(line.endswith('\n') for line in text)
    return [json.loads(line) for line in text], hashlib.sha256(''.join(text).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--microsteps', type=int, default=800)
    args = parser.parse_args()
    assert args.microsteps >= 200 and args.microsteps % 2 == 0
    queue.check_sources()
    launch = queue.read(queue.CAMPAIGN/'launch.json')
    assert all(queue.training_alive(item) for item in launch)
    results, paired_steps, paired_inputs, boundaries = {}, [], [], []
    for item in launch:
        selector = item['selector']
        root = Path(item['config']['output'])
        config, boundary = queue.read(root/'run_config.json'), queue.read(root/'data_boundary.json')
        for path, digest in queue.read(root/'source_sha256.json').items():
            assert queue.sha(path) == digest, path
        steps, step_sha = prefix(root/'steps.jsonl', args.microsteps)
        dev_batches = config['development_samples']//4
        inputs, input_sha = prefix(root/'input_manifest.jsonl', dev_batches+args.microsteps)
        assert all(row['phase'] == 'development' for row in inputs[:dev_batches])
        assert all(row['phase'] == 'train' for row in inputs[dev_batches:])
        for row in inputs:
            for identity in row['identities']:
                name = identity['sequence']
                assert name in boundary['partitions'][row['phase']]
                frames = identity['template_frames']+identity['search_frames']
                assert not set(frames).intersection(boundary['excluded_frames'].get(name, []))
                assert min(identity['search_frames']) > max(identity['template_frames'])
        errors, probe_ids = [], []
        for index, (step, data) in enumerate(zip(steps, inputs[dev_batches:])):
            assert step['microstep'] == index
            assert step['epoch'] == index//(config['samples_per_epoch']//config['microbatch'])+1
            assert step['input_sha256'] == data['tensors_sha256']
            assert len(step['masks_sha256']) == 8
            assert all(math.isfinite(step[key]) for key in
                       ['loss', 'task_c', 'task_m', 'kd_forward', 'kd_reverse', 'reverse_fraction'])
            expected = step['task_c']+step['task_m']+config['forward_weight']*step['kd_forward']
            if index % config['utility_probe_interval'] == 0:
                probe_id = index//config['utility_probe_interval']
                assert step['probe_view'] == ('C' if probe_id % 2 == 0 else 'M')
                assert step['utility']['search_index'] == (probe_id//2) % 3
                regions = step['utility']['regions']
                assert len(regions) == config['microbatch'] and all(0 <= region < 16 for region in regions)
                expected += config['utility_weight']*step['utility']['utility_loss']
                probe_ids.append((index, regions, step['probe_view'], step['utility']['search_index']))
            else:
                assert 'utility' not in step and 'probe_view' not in step
            error = abs(expected-step['loss'])
            assert math.isclose(expected, step['loss'], rel_tol=2e-6, abs_tol=2e-6)
            errors.append(error)
            if (index+1) % config['accumulation'] == 0:
                assert step['update'] == (index+1)//config['accumulation']
        updates = [row for row in steps if 'update' in row]
        seconds_per_update = (updates[-1]['seconds']-updates[0]['seconds'])/(len(updates)-1)
        results[selector] = dict(pid=item['pid'], microsteps=len(steps), updates=len(updates),
            epoch=steps[-1]['epoch'], fitting_tuples=args.microsteps*config['microbatch'],
            fitting_sequences=len({identity['sequence'] for row in inputs[dev_batches:]
                                   for identity in row['identities']}),
            development_tuples=config['development_samples'],
            utility_probes=len(probe_ids), maximum_loss_reconstruction_error=max(errors),
            loss_min=min(row['loss'] for row in steps), loss_max=max(row['loss'] for row in steps),
            mean_loss=statistics.mean(row['loss'] for row in steps),
            median_loss=statistics.median(row['loss'] for row in steps),
            seconds_per_update=seconds_per_update,
            estimated_minutes_per_epoch=seconds_per_update*config['samples_per_epoch']/
                                        (config['microbatch']*config['accumulation']*60),
            steps_prefix_sha256=step_sha, inputs_prefix_sha256=input_sha,
            source_snapshot_sha256=queue.sha(root/'source_sha256.json'))
        paired_steps.append([(row['input_sha256'], row['masks_sha256']) for row in steps]+probe_ids)
        paired_inputs.append(inputs)
        boundaries.append(boundary)
    assert paired_steps[0] == paired_steps[1] and paired_inputs[0] == paired_inputs[1]
    assert boundaries[0] == boundaries[1]
    result = dict(status='passed', observed_utc=queue.now(), arms=results,
        paired_input_mask_and_probe_prefix_exact=True, partitions_and_exclusions_verified=True,
        all_recorded_sources_unchanged=True, optimizer_updates_inferred_from_records=True,
        scope='Existing logged prefix arithmetic/identity/hash audit, not raw-image/gradient replay, '
              'checkpoint convergence, a development evaluation or official tracking metrics.')
    output = queue.ROOT/'reports'/('server28_partitioned_a_prefix%04d_20261102.json' % args.microsteps)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
