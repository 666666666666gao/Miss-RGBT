"""Wait for the registered training pair, verify it, then run full official OPE."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/gaob/Miss-RGBT')
RUNTIME = ROOT/'runtime/task_utility_20261018'
CAMPAIGN = ROOT/'reports/server28_partitioned_a_20261102_formal'
QUEUE = ROOT/'reports/server28_partitioned_a_20261102_official_queue'
RESULTS = ROOT/'results/server28_partitioned_a_20261102'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def now():
    return datetime.datetime.utcnow().isoformat()+'Z'


def check_sources():
    for path, expected in read(QUEUE/'source_sha256.json').items():
        assert sha(path) == expected, path


def training_alive(record):
    status = subprocess.run(['ps', '-p', str(record['pid']), '-o', 'stat=', '-o', 'args='],
                            stdout=subprocess.PIPE, universal_newlines=True)
    assert status.returncode in [0, 1]
    if status.returncode == 1:
        return False
    state, command = status.stdout.strip().split(None, 1)
    if state.startswith('Z'):
        return False
    assert record['command'][2] in command and record['command'][-1] in command, command
    return True


def gpu_ready(gpu):
    row = subprocess.check_output(['nvidia-smi', '--id='+str(gpu),
        '--query-gpu=memory.used,temperature.gpu,power.limit', '--format=csv,noheader,nounits'],
        universal_newlines=True)
    memory, temperature, power = [float(value.strip()) for value in row.split(',')]
    assert memory < 500 and temperature < 85 and power == 250, row
    return dict(memory_mib=memory, temperature_c=temperature, power_limit_w=power)


def jobs_for(selector, completed):
    assert completed['status'] == 'complete' and completed['official_test_selection_used'] is False
    assert completed['all_checkpoints_retained'] and len(completed['final_five']) == 5
    epochs = [item['epoch'] for item in completed['final_five']]
    assert epochs == list(range(completed['epochs']-4, completed['epochs']+1))
    assert completed['selected_epoch'] in epochs
    jobs = []
    for item in completed['final_five']:
        for condition in ['complete', 'miss']:
            output = RESULTS/('%s_epoch%02d_%s' % (selector, item['epoch'], condition))
            jobs.append(dict(selector=selector, epoch=item['epoch'], condition=condition,
                checkpoint=item['path'], checkpoint_sha256=item['sha256'],
                output=str(output), development_selected=item['epoch'] == completed['selected_epoch'],
                expected_sequences=479, expected_frames=337353))
    assert len(jobs) == 10
    return jobs


def command_for(job, gpu):
    return [sys.executable, '-u', str(ROOT/'evaluate_candidate_official28.py'),
            '--checkpoint', job['checkpoint'], '--condition', job['condition'],
            '--gpus', str(gpu), '--workers-per-gpu', '4', '--output', job['output']]


def arm_main(selector):
    check_sources()
    verified = read(CAMPAIGN/'verification.json')
    assert verified['status'] == 'passed' and verified['phase'] == 'formal'
    completed = read(ROOT/'runs/server28_partitioned_a_20261102_formal'/selector/'completed.json')
    assert verified['arms'][selector]['completed_sha256'] == sha(
        ROOT/'runs/server28_partitioned_a_20261102_formal'/selector/'completed.json')
    gpu = {'utility': 2, 'random': 3}[selector]
    jobs = jobs_for(selector, completed)
    save(QUEUE/(selector+'_plan.json'), jobs)
    metrics = []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(RUNTIME)+':'+str(ROOT))
    # Only call the isolated official verifier; never call its development main.
    from verify_recovery_campaign28 import verify_official
    for job in jobs:
        check_sources()
        assert sha(job['checkpoint']) == job['checkpoint_sha256']
        assert not Path(job['output']).exists()
        preflight = gpu_ready(gpu)
        name = '%s_epoch%02d_%s' % (selector, job['epoch'], job['condition'])
        command = command_for(job, gpu)
        with (QUEUE/(name+'.log')).open('xb') as stream:
            process = subprocess.Popen(command, cwd=str(RUNTIME), env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
            save(QUEUE/(selector+'_status.json'), dict(stage='OFFICIAL_RUNNING',
                job=job, pid=process.pid, command=command, preflight=preflight, started_utc=now()))
            while process.poll() is None:
                time.sleep(300)
            save(QUEUE/(name+'_exit.json'), dict(pid=process.pid,
                returncode=process.returncode, ended_utc=now()))
            assert process.returncode == 0, (name, process.returncode)
        measured = verify_official(Path(job['output']), job['checkpoint_sha256'],
                                  job['condition'], read(Path(job['output'])/'metrics.json'))
        metrics.append(dict(job, metrics=measured, independently_recomputed=True))
        save(QUEUE/(selector+'_verified_metrics.json'), metrics)
    assert len(metrics) == 10
    save(QUEUE/(selector+'_completed.json'), dict(status='complete', metrics=metrics,
        selected_epoch=completed['selected_epoch'], selected_sha256=completed['selected_sha256'],
        all_checkpoints_retained=True, official_test_selection_used=False, ended_utc=now()))
    save(QUEUE/(selector+'_status.json'), dict(stage='COMPLETE', ended_utc=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['utility', 'random'])
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if args.arm:
        arm_main(args.arm)
        return
    assert args.register
    QUEUE.mkdir(exist_ok=False)
    launch = read(CAMPAIGN/'launch.json')
    assert len(launch) == 2 and all(training_alive(item) for item in launch)
    assert not (CAMPAIGN/'verification.json').exists()
    sources = [Path(__file__), ROOT/'scripts/verify_partitioned_a.py',
        ROOT/'scripts/train_partitioned_a.py', ROOT/'scripts/launch_partitioned_a.py',
        ROOT/'evaluate_candidate_official28.py', ROOT/'verify_recovery_campaign28.py',
        ROOT/'evaluate_val97_recovery28.py', ROOT/'analyze_epoch05_health28.py',
        CAMPAIGN/'launch.json', RUNTIME/'reports/e0/inventory.json',
        ROOT/'docs/PARTITIONED_A_OFFICIAL_QUEUE_20261102.md']
    sources += sorted((RUNTIME/'miss_rgbt').rglob('*.py'))+sorted((RUNTIME/'lib').rglob('*.py'))
    save(QUEUE/'source_sha256.json', {str(path.resolve()): sha(path) for path in sources})
    save(QUEUE/'registration.json', dict(pid=os.getpid(), training=launch,
        poll_seconds=300, official_conditions=['complete', 'miss'],
        evaluations_per_arm=10, sequences_per_evaluation=479, frames_per_evaluation=337353,
        workers_per_gpu=4, physical_gpus=[2, 3], no_subset=True,
        checkpoint_selection='Fixed train882 development loss among final five',
        deletes_checkpoints=False, registered_utc=now()))
    while True:
        states = []
        for item in launch:
            live = training_alive(item)
            complete = (Path(item['config']['output'])/'completed.json').exists()
            assert live or complete, ('Training exited without completion', item['selector'])
            states.append(dict(selector=item['selector'], pid=item['pid'],
                               live=live, completion_present=complete))
        save(QUEUE/'status.json', dict(stage='WAITING_FOR_TRAINING',
                                      training=states, observed_utc=now()))
        if not any(state['live'] for state in states):
            break
        time.sleep(300)
    check_sources()
    command = [sys.executable, '-u', str(ROOT/'scripts/verify_partitioned_a.py'),
               '--phase', 'formal']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(RUNTIME)+':'+str(ROOT))
    with (QUEUE/'training_verification.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment,
            stdout=stream, stderr=subprocess.STDOUT)
        save(QUEUE/'status.json', dict(stage='VERIFYING_TRAINING',
                                      pid=process.pid, command=command, started_utc=now()))
        while process.poll() is None:
            time.sleep(300)
        assert process.returncode == 0, ('Training verification failed', process.returncode)
    assert read(CAMPAIGN/'verification.json')['status'] == 'passed'
    check_sources()
    children = []
    for selector in ['utility', 'random']:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--arm', selector]
        with (QUEUE/(selector+'_controller.log')).open('xb') as stream:
            process = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
        children.append((selector, process))
    save(QUEUE/'status.json', dict(stage='OFFICIAL_RUNNING',
        children=[dict(selector=selector, pid=process.pid) for selector, process in children],
        started_utc=now()))
    while any(process.poll() is None for _, process in children):
        for selector, process in children:
            assert process.poll() in [None, 0], (selector, process.returncode)
        time.sleep(300)
    assert all(process.returncode == 0 for _, process in children)
    results = {selector: read(QUEUE/(selector+'_completed.json')) for selector, _ in children}
    assert all(value['status'] == 'complete' for value in results.values())
    save(QUEUE/'completed.json', dict(status='complete', arms=results,
        official_evaluations=20, trajectories_verified=9580, frames_verified=6747060,
        official_test_selection_used=False, all_checkpoints_retained=True, ended_utc=now(),
        scope='Full official final-five reports; post-hoc comparisons do not validate three modules'))
    save(QUEUE/'status.json', dict(stage='COMPLETE', ended_utc=now()))


if __name__ == '__main__':
    main()
