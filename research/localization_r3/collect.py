"""Sidecar full R3 collector; only consumes terminal training/evaluation files."""
import argparse
import json
from pathlib import Path

from research.localization_r3.objective import ARMS
from research.localization_r3.prepare import CAMPAIGN, check_evidence
from research.localization_r3.verify import verify as verify_pair
from research.protocol_r.common import ROOT, read, sha

RESULTS = ROOT / 'results/localization_r3_20261104'
BASELINE = ROOT / 'runtime/task_utility_20261018/reports/e0_complete/metrics.json'
GOALS = {'LasHeR-Miss': {'PR': 69.3, 'SR': 54.4},
         'RGBT234-Miss': {'MPR': 88.1, 'MSR': 64.3}}


def assess_selected(selected, baseline):
    assessments = {}
    for arm, row in selected.items():
        complete, miss = row['metrics']['complete'], row['metrics']['miss']
        deltas = {dataset: {key: complete[dataset][key] - baseline[dataset][key]
                           for key in keys}
                  for dataset, keys in [('LasHeR', ('PR', 'NPR', 'SR')),
                                        ('RGBT234', ('MPR', 'MSR'))]}
        targets = {dataset: {key: dict(score=miss[dataset][key], threshold=limit,
                                      passed=miss[dataset][key] > limit)
                             for key, limit in thresholds.items()}
                   for dataset, thresholds in GOALS.items()}
        assessments[arm] = dict(complete_delta_pp=deltas, miss_targets=targets,
            all_miss_targets_met=all(item['passed'] for values in targets.values()
                                     for item in values.values()),
            complete_at_least2pp_reference={dataset: {key: value >= 2
                for key, value in values.items()} for dataset, values in deltas.items()})
    return assessments


def collect(campaign=CAMPAIGN, results=RESULTS, baseline=BASELINE):
    campaign = Path(campaign)
    required = [campaign / (arm + '_official_completed.json') for arm in ARMS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(status='waiting', stage='WAITING_FOR_TWENTY_FULL_RESULTS',
                    missing=missing, dispatch_authorized=False)
    pair = verify_pair(campaign)
    assert pair['status'] == 'passed' and pair['phase'] == 'r3_training_pair'
    from verify_recovery_campaign28 import verify_official
    selected, all_rows, evidence = {}, {}, {}
    for arm, receipt_path in zip(ARMS, required):
        config_path = campaign / (arm + '.json')
        config = read(config_path)
        check_evidence(config)
        root = Path(config['output'])
        done = read(root / 'completed.json')
        terminal = read(receipt_path)
        assert terminal['status'] == 'complete' and terminal['full_evaluations'] == 10
        assert terminal['sequences_per_evaluation'] == 479
        assert terminal['frames_per_evaluation'] == 337353
        assert terminal['all_checkpoints_retained']
        assert not terminal['official_test_selection_used']
        assert terminal['selected_epoch'] == pair['selected_epochs'][arm] == done['selected_epoch']
        assert terminal['training_completed_sha256'] == sha(root / 'completed.json')
        assert terminal['rows'] == read(campaign / (arm + '_verified_official.json'))
        plan = read(campaign / (arm + '_official_plan.json'))
        assert len(plan) == len(terminal['rows']) == 10
        expected_pairs = [(epoch, condition) for epoch in range(16, 21)
                          for condition in ('complete', 'miss')]
        assert [(job['epoch'], job['condition']) for job in plan] == expected_pairs
        weights = {item['epoch']: item for item in done['final_five']}
        measured_rows, chosen = [], {}
        for job, row in zip(plan, terminal['rows']):
            assert {key: row[key] for key in job} == job
            assert job['arm'] == arm
            checkpoint = weights[job['epoch']]
            assert job['checkpoint'] == checkpoint['path']
            assert sha(job['checkpoint']) == checkpoint['sha256'] == job['checkpoint_sha256']
            assert job['development_selected'] == (job['epoch'] == done['selected_epoch'])
            destination = Path(results) / arm / (
                'epoch%02d_%s' % (job['epoch'], job['condition']))
            assert Path(job['output']) == destination and row['independently_recomputed']
            for path, expected in row['evidence_sha256'].items():
                assert sha(path) == expected
                evidence[path] = expected
            actual = verify_official(destination, job['checkpoint_sha256'],
                                     job['condition'], row['metrics'])
            assert actual == row['metrics']
            measured_rows.append(dict(row, metrics=actual))
            if job['development_selected']:
                chosen[job['condition']] = actual
            evidence[job['checkpoint']] = job['checkpoint_sha256']
        assert set(chosen) == {'complete', 'miss'}
        selected[arm] = dict(epoch=done['selected_epoch'], metrics=chosen,
            checkpoint_sha256=weights[done['selected_epoch']]['sha256'])
        all_rows[arm] = measured_rows
        for path in (receipt_path, config_path, root / 'completed.json',
                     root / 'development_history.json', campaign / (arm + '_official_plan.json'),
                     campaign / (arm + '_verified_official.json')):
            evidence[str(path)] = sha(path)
    assessments = assess_selected(selected, read(baseline))
    evidence[str(baseline)] = sha(baseline)
    return dict(status='passed', phase='r3_full_campaign', training_pair=pair,
        full_evaluations_recomputed=20, sequences_per_evaluation=479,
        frames_per_evaluation=337353, trajectories_verified=9580, frames_verified=6747060,
        selected=selected, selected_goal_assessment=assessments,
        arms=all_rows, evidence_sha256=evidence,
        all_checkpoints_retained=True, official_test_selection_used=False,
        dispatch_authorized=False,
        scope='Full saved-output R3 reporting with development selection; '
              '2pp is a reporting reference, not a selection/routing gate; '
              'not multi-seed significance, three-module proof or a replacement for failed R1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default=str(CAMPAIGN))
    parser.add_argument('--output')
    args = parser.parse_args()
    result = collect(args.campaign)
    if args.output and result['status'] == 'passed':
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
