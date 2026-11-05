"""Read completed paired R3 evidence; never inspect a growing input manifest."""
import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

from research.localization_r3.objective import ARMS
from research.localization_r3.prepare import CAMPAIGN, check_evidence
from research.protocol_r.common import read, sha


def verify(campaign=CAMPAIGN):
    campaign = Path(campaign)
    paths = [campaign / (arm + '.json') for arm in ARMS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return dict(status='waiting', missing=missing, dispatch_authorized=False)
    configs = [read(path) for path in paths]
    roots = [Path(config['output']) for config in configs]
    missing = [str(root / 'completed.json') for root in roots
               if not (root / 'completed.json').is_file()]
    if missing:
        return dict(status='waiting', missing=missing, dispatch_authorized=False)
    assert {k: v for k, v in configs[0].items() if k not in ('arm', 'output')} == {
        k: v for k, v in configs[1].items() if k not in ('arm', 'output')}
    selected, fixed_epoch20 = {}, {}
    initial = read(roots[0] / 'initial_state.json')
    for arm, config, root in zip(ARMS, configs, roots):
        check_evidence(config)
        assert read(root / 'run_config.json') == config
        assert read(root / 'initial_state.json') == initial
        done, history = read(root / 'completed.json'), read(root / 'development_history.json')
        assert done['status'] == 'complete' and done['epochs'] == 20
        assert done['updates'] == 50000 and not done['early_stopped']
        assert done['all_checkpoints_retained'] and not done['official_test_selection_used']
        assert [row['epoch'] for row in history] == list(range(1, 21))
        for row in history:
            assert row['samples'] == 128 and row['updates'] == 2500 * row['epoch']
            assert set(row['task_loss']) == {'11', '01', '10'}
            assert all(math.isfinite(value) for value in row['task_loss'].values())
            assert math.isclose(row['mean_task_loss'],
                                sum(row['task_loss'].values()) / 3, abs_tol=1e-12)
        chosen = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
        assert done['selected_epoch'] == chosen['epoch']
        assert done['selection_rule'] == config['development_selection']
        assert done['primary_development_epoch20_complete_loss'] == history[-1]['task_loss']['11']
        assert [row['epoch'] for row in done['final_five']] == list(range(16, 21))
        assert (root / 'epoch10.pth.tar').is_file()
        for row in done['final_five']:
            assert Path(row['path']) == root / ('epoch%02d.pth.tar' % row['epoch'])
            assert sha(row['path']) == row['sha256']
        assert sha(root / 'input_manifest.jsonl') == done['input_manifest_sha256']
        selected[arm] = chosen['epoch']
        fixed_epoch20[arm] = history[-1]['task_loss']['11']
    count = 0
    with (roots[0] / 'input_manifest.jsonl').open() as left, \
            (roots[1] / 'input_manifest.jsonl').open() as right:
        for first, second in zip_longest(left, right):
            assert first is not None and second is not None
            first, second = json.loads(first), json.loads(second)
            assert first == second
            assert first['epoch'] == count // 5000 + 1 and first['index'] == count % 5000
            assert len(first['identities']) == 4
            count += 1
    assert count == 100000
    count, losses = 0, [0., 0.]
    with (roots[0] / 'steps.jsonl').open() as left, \
            (roots[1] / 'steps.jsonl').open() as right:
        for first, second in zip_longest(left, right):
            assert first is not None and second is not None
            rows = [json.loads(first), json.loads(second)]
            for index, row in enumerate(rows):
                assert row['arm'] == ARMS[index] and row['microstep'] == count
                assert row['epoch'] == count // 5000 + 1
                assert row['model_forwards'] == index + 1
                assert row['mask_applied_to_model'] == bool(index)
                assert math.isfinite(row['loss']) and math.isfinite(row['task_c'])
                if index == 0:
                    assert row['task_m'] is None and row['kd_forward'] is None
                else:
                    assert math.isfinite(row['task_m']) and math.isfinite(row['kd_forward'])
                assert row['seeds'] == dict(complete=74101 + 3 * count,
                    missing=74102 + 3 * count, masks=74103 + 3 * count)
                losses[index] += row['loss'] / 2
                if (count + 1) % 2 == 0:
                    assert row['update'] == (count + 1) // 2
                    assert math.isfinite(row['gradient_norm'])
                    assert math.isclose(row['update_loss_mean'], losses[index], abs_tol=1e-9)
                    losses[index] = 0.
            for key in ('input_sha256', 'candidate_masks_sha256', 'seeds'):
                assert rows[0][key] == rows[1][key]
            count += 1
    assert count == 100000
    contrast = fixed_epoch20['complete_missing'] - fixed_epoch20['complete_only']
    return dict(status='passed', phase='r3_training_pair', paired_microbatches=count,
        paired_accepted_tuples=4 * count, updates_per_arm=50000,
        model_forwards=dict(complete_only=count, complete_missing=2 * count),
        epoch20_complete_task_loss=fixed_epoch20,
        primary_development_contrast=contrast, selected_epochs=selected,
        dispatch_authorized=False,
        scope='Saved paired training evidence; no official scores or mechanism proof')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default=str(CAMPAIGN))
    parser.add_argument('--output')
    args = parser.parse_args()
    result = verify(args.campaign)
    if args.output and result['status'] == 'passed':
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
