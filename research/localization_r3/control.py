"""Sidecar R3 controller; execute with the frozen R3 snapshot on PYTHONPATH."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from research.localization_r3.objective import ARMS
from research.localization_r3.prepare import CAMPAIGN, check_evidence
from research.localization_r3.train import check_smokes
from research.protocol_r.common import ROOT, read, save, sha
from research.protocol_r.launch import gpu_ready

RESULTS = ROOT / 'results/localization_r3_20261104'


def now():
    return datetime.datetime.utcnow().isoformat() + 'Z'


def readiness(config, phase, review):
    assert phase in ('smoke', 'train')
    check_evidence(config)
    review = Path(review)
    if not review.is_file():
        return dict(ready=False, stage='WAITING_FOR_INDEPENDENT_REVIEW',
                    missing=[str(review)], dispatch_authorized=False)
    record = read(review)
    assert record['status'] == 'passed' and record['blocking_issues'] == []
    assert record['review_independence'] == 'same-family'
    assert record['acceptance_status'] == 'provisional'
    assert record['model'] == 'gpt-6-astra' and record['reasoning_effort'] == 'max'
    if phase == 'train':
        missing = [str(Path(read(Path(config['source_manifest']).parent /
                     (arm + '.json'))['output'] + '_smoke') / 'smoke_verified.json')
                   for arm in ARMS]
        missing = [path for path in missing if not Path(path).is_file()]
        if missing:
            return dict(ready=False, stage='WAITING_FOR_PAIRED_IMAGE_SMOKES',
                        missing=missing, dispatch_authorized=False)
        check_smokes(config)
    return dict(ready=True, stage='READY_FOR_' + phase.upper(),
                review=str(review), review_sha256=sha(review),
                scope='Readiness only; GPU preflight and exclusive launch still required')


def verify_training(config):
    """Read only this arm after its completion receipt has been written."""
    root = Path(config['output'])
    assert (root / 'completed.json').is_file()
    check_evidence(config)
    assert read(root / 'run_config.json') == config
    done, history = read(root / 'completed.json'), read(root / 'development_history.json')
    assert done['status'] == 'complete' and done['epochs'] == 20 and done['updates'] == 50000
    assert not done['early_stopped'] and not done['official_test_selection_used']
    assert done['all_checkpoints_retained'] and not done['resume_supported']
    assert [row['epoch'] for row in history] == list(range(1, 21))
    for row in history:
        assert row['samples'] == 128 and row['updates'] == 2500 * row['epoch']
        assert set(row['task_loss']) == {'11', '01', '10'}
        assert all(math.isfinite(value) for value in row['task_loss'].values())
        assert math.isclose(row['mean_task_loss'], sum(row['task_loss'].values()) / 3)
    chosen = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
    assert done['selected_epoch'] == chosen['epoch']
    assert done['selection_rule'] == config['development_selection']
    assert done['primary_development_epoch20_complete_loss'] == history[-1]['task_loss']['11']
    assert [row['epoch'] for row in done['final_five']] == list(range(16, 21))
    assert (root / 'epoch10.pth.tar').is_file()
    for item in done['final_five']:
        assert Path(item['path']) == root / ('epoch%02d.pth.tar' % item['epoch'])
        assert sha(item['path']) == item['sha256']
    assert sha(root / 'input_manifest.jsonl') == done['input_manifest_sha256']
    count, updates = 0, 0
    with (root / 'steps.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row['microstep'] == count and row['arm'] == config['arm']
            assert row['epoch'] == count // 5000 + 1 and math.isfinite(row['loss'])
            count += 1
            if count % 2 == 0:
                updates += 1
                assert row['update'] == updates and math.isfinite(row['gradient_norm'])
    assert count == 100000 and updates == 50000
    return done


def official_plan(config, done, results=RESULTS):
    return [dict(arm=config['arm'], epoch=item['epoch'],
        checkpoint=item['path'], checkpoint_sha256=item['sha256'],
        condition=condition, development_selected=item['epoch'] == done['selected_epoch'],
        output=str(Path(results) / config['arm'] /
                   ('epoch%02d_%s' % (item['epoch'], condition))))
        for item in done['final_five'] for condition in ('complete', 'miss')]


def run(command, env, snapshot, log_path, status_path, stage, **extra):
    with Path(log_path).open('xb') as log:
        child = subprocess.Popen(command, cwd=str(snapshot), env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    save(status_path, dict(stage=stage, pid=child.pid, command=command,
                           started_utc=now(), **extra))
    while child.poll() is None:
        time.sleep(300)
    save(status_path, dict(stage=stage + '_EXITED', pid=child.pid,
        returncode=child.returncode, ended_utc=now(), **extra))
    assert child.returncode == 0, (stage, child.returncode, str(log_path))


def evaluate(config, gpu, campaign, env, results=RESULTS):
    from verify_recovery_campaign28 import verify_official
    done = verify_training(config)
    jobs = official_plan(config, done, results)
    assert len(jobs) == 10
    assert all(not Path(job['output']).exists() for job in jobs)
    arm, rows = config['arm'], []
    save(campaign / (arm + '_official_plan.json'), jobs)
    for job in jobs:
        check_evidence(config)
        preflight = gpu_ready(gpu)
        assert sha(job['checkpoint']) == job['checkpoint_sha256']
        command = [sys.executable, '-u', str(ROOT / 'evaluate_candidate_official28.py'),
            '--checkpoint', job['checkpoint'], '--condition', job['condition'],
            '--gpus', str(gpu), '--workers-per-gpu', '4', '--output', job['output']]
        label = '%s_epoch%02d_%s' % (arm, job['epoch'], job['condition'])
        run(command, env, config['snapshot'], campaign / (label + '.log'),
            campaign / (arm + '_status.json'), 'OFFICIAL', job=job, preflight=preflight)
        destination = Path(job['output'])
        measured = verify_official(destination, job['checkpoint_sha256'],
                                   job['condition'], read(destination / 'metrics.json'))
        rows.append(dict(job, metrics=measured, independently_recomputed=True,
            evidence_sha256={str(destination / name): sha(destination / name) for name in
                             ('run.json', 'completed.json', 'metrics.json', 'all_receipts.json')}))
        save(campaign / (arm + '_verified_official.json'), rows)
    terminal = dict(status='complete', rows=rows, selected_epoch=done['selected_epoch'],
        full_evaluations=10, sequences_per_evaluation=479, frames_per_evaluation=337353,
        training_completed_sha256=sha(Path(config['output']) / 'completed.json'),
        all_checkpoints_retained=True, official_test_selection_used=False, ended_utc=now())
    save(campaign / (arm + '_official_completed.json'), terminal)
    return terminal


def execute(config, phase, gpu, review, results=RESULTS):
    state = readiness(config, phase, review)
    if not state['ready']:
        return state
    campaign = Path(config['source_manifest']).parent
    arm = config['arm']
    preflight = gpu_ready(gpu)
    record = dict(controller_pid=os.getpid(), phase=phase, gpu=gpu, preflight=preflight,
        config=config, review=state['review'], review_sha256=state['review_sha256'],
        controller_sha256=sha(__file__), poll_seconds=300, started_utc=now())
    with (campaign / (arm + '_' + phase + '_launch.json')).open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=config['snapshot'] + ':' + str(ROOT))
    command = [sys.executable, '-u', str(Path(config['snapshot']) /
               'research/localization_r3/train.py'),
               '--config', str(campaign / (arm + '.json'))]
    if phase == 'smoke':
        command += ['--max-steps', '4']
    run(command, env, config['snapshot'], campaign / (arm + '_' + phase + '.log'),
        campaign / (arm + '_status.json'), phase.upper(), preflight=preflight)
    if phase == 'smoke':
        receipt = read(Path(config['output'] + '_smoke') / 'smoke_verified.json')
        assert receipt['status'] == 'passed' and receipt['updates'] == 4
        return dict(status='smoke_complete', formal_training_authorized=False,
                    scope='Both paired smokes must pass before either formal arm')
    terminal = evaluate(config, gpu, campaign, env, results)
    save(campaign / (arm + '_status.json'), dict(stage='COMPLETE', ended_utc=now()))
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default=str(CAMPAIGN))
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument('--review')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    campaign = Path(args.campaign)
    config = read(campaign / (args.arm + '.json'))
    review = args.review or str(campaign / 'independent_review.json')
    result = readiness(config, args.phase, review) if args.check_only else \
        execute(config, args.phase, args.gpu, review)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
