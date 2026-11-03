"""Run one new-protocol smoke or formal arm, then its ten full evaluations."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from research.curriculum_p.engine import ARMS
from research.curriculum_p.protocol_prepare import CAMPAIGN, SNAPSHOT
from research.curriculum_p.protocol_train import check_config, check_evidence, check_smokes
from research.curriculum_p.protocol_verify import verify_arm
from research.protocol_r.common import ROOT, read, save, sha
from research.protocol_r.launch import gpu_ready, now

RUNTIME = ROOT / 'runtime/task_utility_20261018'


def run_child(command, env, log_path, status_path, stage, **extra):
    with log_path.open('xb') as stream:
        child = subprocess.Popen(command, cwd=str(SNAPSHOT), env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    save(status_path, dict(stage=stage, pid=child.pid, command=command,
                           started_utc=now(), **extra))
    while child.poll() is None:
        time.sleep(300)
    exit_record = dict(stage=stage + '_EXITED', pid=child.pid, command=command,
                      returncode=child.returncode, ended_utc=now(), **extra)
    save(log_path.with_suffix('.exit.json'), exit_record)
    save(status_path, exit_record)
    assert child.returncode == 0, (stage, child.returncode, str(log_path))


def jobs_for(arm, done):
    assert arm in ARMS and done['status'] == 'complete' and done['epochs'] == 20
    assert done['official_test_selection_used'] is False and done['all_checkpoints_retained']
    assert done['selected_epoch'] in range(16, 21)
    assert [item['epoch'] for item in done['final_five']] == list(range(16, 21))
    return [dict(arm=arm, epoch=item['epoch'], condition=condition,
        checkpoint=item['path'], checkpoint_sha256=item['sha256'],
        development_selected=item['epoch'] == done['selected_epoch'],
        output=str(ROOT / 'results/rppm_ostrack_20261103' / arm /
                   ('epoch%02d_%s' % (item['epoch'], condition))),
        expected_sequences=479, expected_frames=337353)
        for item in done['final_five'] for condition in ('complete', 'miss')]


def command_for(job, gpu):
    assert gpu in (0, 1, 2, 3) and job['expected_sequences'] == 479
    assert job['expected_frames'] == 337353 and job['condition'] in ('complete', 'miss')
    return [sys.executable, '-u', str(ROOT / 'evaluate_candidate_official28.py'),
        '--checkpoint', job['checkpoint'], '--condition', job['condition'],
        '--gpus', str(gpu), '--workers-per-gpu', '4', '--output', job['output']]


def official_sources(config):
    # Inference uses the existing frozen A implementation, with identical model code.
    for folder in ('lib', 'miss_rgbt', 'experiments/cadtrack'):
        for path in sorted((SNAPSHOT / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.txt'):
                assert sha(path) == sha(RUNTIME / path.relative_to(SNAPSHOT)), str(path)
    paths = [ROOT / name for name in ('evaluate_candidate_official28.py',
        'verify_recovery_campaign28.py', 'evaluate_val97_recovery28.py',
        'analyze_epoch05_health28.py')]
    paths += [RUNTIME / 'reports/e0/inventory.json', Path(__file__),
              Path(__file__).with_name('protocol_verify.py')]
    for folder in ('lib', 'miss_rgbt', 'scripts'):
        paths += sorted((RUNTIME / folder).rglob('*.py'))
    paths += [RUNTIME / 'experiments/cadtrack/cadtrack.yaml', Path(config['source_manifest'])]
    return {str(path): sha(path) for path in paths}


def official(arm, gpu, config, verified, env, queue):
    from verify_recovery_campaign28 import verify_official
    assert verified['status'] == 'passed' and verified['arm'] == arm
    assert verified['verifier_sha256'] == sha(Path(__file__).with_name('protocol_verify.py'))
    done = read(Path(config['output']) / 'completed.json')
    jobs = jobs_for(arm, done)
    frozen = official_sources(config)
    frozen.update(verified['evidence_sha256'])
    save(queue / 'official_sources.json', frozen)
    save(queue / 'official_registration.json', dict(jobs=jobs, gpu=gpu,
        poll_seconds=300, workers_per_gpu=4, training_verified=verified,
        official_test_selection_used=False, all_checkpoints_retained=True))
    results = []
    for job in jobs:
        check_evidence(config)
        assert all(sha(path) == digest for path, digest in frozen.items())
        assert sha(job['checkpoint']) == job['checkpoint_sha256']
        assert not Path(job['output']).exists()
        preflight = gpu_ready(gpu)
        name = 'epoch%02d_%s' % (job['epoch'], job['condition'])
        command = command_for(job, gpu)
        run_child(command, env, queue / (name + '.log'), queue / 'status.json',
                  'OFFICIAL', job=job, preflight=preflight)
        measured = verify_official(Path(job['output']), job['checkpoint_sha256'],
                                  job['condition'], read(Path(job['output']) / 'metrics.json'))
        assert all(sha(path) == digest for path, digest in frozen.items())
        results.append(dict(job, metrics=measured, independently_recomputed=True,
            completed_sha256=sha(Path(job['output']) / 'completed.json')))
        save(queue / 'verified_metrics.json', results)
    assert len(results) == 10
    save(queue / 'official_completed.json', dict(status='complete', rows=results,
        selected_epoch=done['selected_epoch'], official_test_selection_used=False,
        all_checkpoints_retained=True, full_evaluations=10,
        sequences_per_evaluation=479, frames_per_evaluation=337353, ended_utc=now(),
        six_arm_pairing_pending=not (CAMPAIGN / 'paired_training_verified.json').exists()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument('--phase', choices=['smoke', 'formal'], required=True)
    args = parser.parse_args()
    assert Path(__file__).resolve().parents[2] == SNAPSHOT
    config_path = CAMPAIGN / (args.arm + '.json')
    config = read(config_path)
    check_config(config)
    check_evidence(config)
    if args.phase == 'formal':
        check_smokes(config)
    preflight = gpu_ready(args.gpu)
    queue = CAMPAIGN / (args.arm + '_' + args.phase + '_controller')
    queue.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(SNAPSHOT) + ':' + str(ROOT))
    command = [sys.executable, '-u', str(SNAPSHOT / 'research/curriculum_p/protocol_train.py'),
               '--config', str(config_path)]
    if args.phase == 'smoke':
        command += ['--max-steps', '4']
    save(queue / 'launch.json', dict(controller_pid=os.getpid(), config=config,
        config_sha256=sha(config_path), command=command, preflight=preflight,
        phase=args.phase, poll_seconds=300, started_utc=now()))
    run_child(command, env, queue / 'training.log', queue / 'status.json', 'TRAINING')
    if args.phase == 'smoke':
        smoke = read(Path(config['output'] + '_smoke') / 'smoke_verified.json')
        assert smoke['status'] == 'passed' and smoke['updates'] == 4
        assert smoke['strict_reload_bitwise_equal']
    else:
        verified = verify_arm(args.arm)
        receipt = CAMPAIGN / (args.arm + '_training_verified.json')
        assert not receipt.exists()
        save(receipt, verified)
        official(args.arm, args.gpu, config, verified, env, queue)
    save(queue / 'status.json', dict(stage='COMPLETE', phase=args.phase, ended_utc=now()))


if __name__ == '__main__':
    main()
