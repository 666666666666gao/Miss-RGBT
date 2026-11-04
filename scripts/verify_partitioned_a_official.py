"""Read-only final audit of the actual A official-queue terminal artifacts."""
import argparse
import json
from pathlib import Path

from scripts import partitioned_a_official_queue_v2 as protocol


def verify():
    queue, campaign = protocol.QUEUE, protocol.CAMPAIGN
    required = [queue / 'completed.json', campaign / 'verification_v2.json']
    required += [queue / (arm + '_completed.json') for arm in ('utility', 'random')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(status='waiting', stage='WAITING_FOR_A_OFFICIAL_COMPLETION',
                    missing=missing, dispatch_authorized=False)
    from verify_recovery_campaign28 import verify_official
    protocol.check_sources()
    read, sha = protocol.read, protocol.sha
    terminal, trained = read(required[0]), read(required[1])
    assert terminal['status'] == 'complete'
    assert terminal['official_evaluations'] == 20
    assert terminal['trajectories_verified'] == 9580
    assert terminal['frames_verified'] == 6747060
    assert terminal['official_test_selection_used'] is False
    assert terminal['all_checkpoints_retained'] is True
    assert trained['status'] == 'passed' and trained['phase'] == 'formal'
    assert set(terminal['arms']) == set(trained['arms']) == {'utility', 'random'}
    evidence = {str(path): sha(path) for path in required}
    evidence[str(queue / 'source_sha256.json')] = sha(queue / 'source_sha256.json')
    all_rows, selected_rows = {}, {}
    for arm in ('utility', 'random'):
        child = read(queue / (arm + '_completed.json'))
        assert terminal['arms'][arm] == child and child['status'] == 'complete'
        assert child['all_checkpoints_retained'] is True
        assert child['official_test_selection_used'] is False
        root = protocol.ROOT / 'runs/server28_partitioned_a_20261102_formal' / arm
        done, history = read(root / 'completed.json'), read(root / 'development_history.json')
        assert trained['arms'][arm]['completed_sha256'] == sha(root / 'completed.json')
        assert len(history) == done['epochs']
        selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
        assert selected['epoch'] == done['selected_epoch'] == child['selected_epoch']
        assert done['selected_sha256'] == child['selected_sha256']
        for epoch in range(1, done['epochs'] + 1):
            assert (root / ('epoch%02d.pth.tar' % epoch)).is_file()
        jobs = protocol.jobs_for(arm, done)
        assert read(queue / (arm + '_plan.json')) == jobs
        assert read(queue / (arm + '_verified_metrics.json')) == child['metrics']
        assert len(child['metrics']) == len(jobs) == 10
        weights, measured_rows, chosen = {}, [], {}
        for job, row in zip(jobs, child['metrics']):
            assert {key: value for key, value in row.items()
                    if key not in ('metrics', 'independently_recomputed')} == job
            assert row['independently_recomputed'] is True
            checkpoint = job['checkpoint']
            if checkpoint not in weights:
                weights[checkpoint] = sha(checkpoint)
            assert weights[checkpoint] == job['checkpoint_sha256']
            name = '%s_epoch%02d_%s' % (arm, job['epoch'], job['condition'])
            exit_path = queue / (name + '_exit.json')
            assert read(exit_path)['returncode'] == 0
            output = Path(job['output'])
            actual = verify_official(output, job['checkpoint_sha256'],
                                     job['condition'], row['metrics'])
            assert actual == row['metrics']
            measured_rows.append(dict(job, metrics=actual, independently_recomputed=True))
            if job['development_selected']:
                assert job['checkpoint_sha256'] == done['selected_sha256']
                chosen[job['condition']] = actual
            evidence[str(exit_path)] = sha(exit_path)
            for filename in ('run.json', 'completed.json', 'metrics.json', 'all_receipts.json'):
                evidence[str(output / filename)] = sha(output / filename)
        assert set(chosen) == {'complete', 'miss'}
        selected_rows[arm] = dict(epoch=selected['epoch'],
            checkpoint_sha256=done['selected_sha256'], metrics=chosen)
        all_rows[arm] = measured_rows
        evidence.update(weights)
        for path in (root / 'completed.json', root / 'development_history.json',
                     queue / (arm + '_plan.json'), queue / (arm + '_verified_metrics.json')):
            evidence[str(path)] = sha(path)
    evidence[str(Path(__file__).resolve())] = sha(__file__)
    return dict(status='passed', phase='formal_official_queue',
        official_evaluations_recomputed=20, sequences_per_evaluation=479,
        frames_per_evaluation=337353, trajectories_verified=9580,
        frames_verified=6747060, selected=selected_rows, arms=all_rows,
        evidence_sha256=evidence, all_checkpoints_retained=True,
        official_test_selection_used=False, dispatch_authorized=False,
        scope='Actual terminal A queue and twenty full saved-output audits; '
              'not three-module efficacy or authorization to bypass other parent gates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = verify()
    if args.output and result['status'] == 'passed':
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
