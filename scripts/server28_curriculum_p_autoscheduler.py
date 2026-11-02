"""Server28 A->P scheduler without requiring a Git checkout."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/gaob/Miss-RGBT')
REPORTS = ROOT / 'reports'
LOG = REPORTS / 'server28_curriculum_p_20261103_autoscheduler.log'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           WANDB_MODE='disabled',
           PYTHONPATH=str(ROOT / 'runtime/task_utility_20261018') + ':' + str(ROOT))


def now():
    return datetime.datetime.utcnow().isoformat() + 'Z'


def write_status(stage, **extra):
    value = dict(stage=stage, observed_utc=now(), **extra)
    (REPORTS / 'server28_curriculum_p_20261103_scheduler_status.json').write_text(
        json.dumps(value, indent=2) + '\n')
    with LOG.open('a') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()


def run(command, label):
    write_status(label, command=command)
    with LOG.open('a') as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), env=ENV,
                                   stdout=stream, stderr=subprocess.STDOUT)
    while process.poll() is None:
        time.sleep(300)
    if process.returncode != 0:
        write_status(label + '_FAILED', returncode=process.returncode)
        raise SystemExit(process.returncode)
    write_status(label + '_DONE')


def launch_pair(phase):
    arms = ['fixed60', 'fixed50', 'progressive', 'shuffled']
    for offset in range(0, len(arms), 4):
        children = []
        for gpu, arm in zip([0, 1, 2, 3], arms[offset:offset + 4]):
            command = [sys.executable, '-u',
                       str(ROOT / 'research/curriculum_p/launch.py'),
                       '--phase', phase, '--arm', arm, '--gpu', str(gpu)]
            with LOG.open('a') as stream:
                children.append((arm, subprocess.Popen(command, cwd=str(ROOT),
                    env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdout=stream, stderr=subprocess.STDOUT)))
        write_status('P_' + phase.upper() + '_RUNNING',
                     arms=[arm for arm, _ in children],
                     pids=[process.pid for _, process in children])
        while any(process.poll() is None for _, process in children):
            time.sleep(300)
        failures = [(arm, process.returncode) for arm, process in children
                    if process.returncode != 0]
        if failures:
            write_status('P_' + phase.upper() + '_FAILED', failures=failures)
            raise SystemExit(1)
        write_status('P_' + phase.upper() + '_PAIR_DONE',
                     arms=[arm for arm, _ in children])


def launch_official_pair():
    arms = ['fixed60', 'fixed50', 'progressive', 'shuffled']
    for offset in range(0, len(arms), 4):
        children = []
        for gpu, arm in zip([0, 1, 2, 3], arms[offset:offset + 4]):
            command = [sys.executable, '-u',
                       str(ROOT / 'research/curriculum_p/official.py'),
                       '--arm', arm, '--gpu', str(gpu)]
            with LOG.open('a') as stream:
                children.append((arm, subprocess.Popen(command, cwd=str(ROOT),
                    env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdout=stream, stderr=subprocess.STDOUT)))
        write_status('P_OFFICIAL_RUNNING',
                     arms=[arm for arm, _ in children],
                     pids=[process.pid for _, process in children])
        while any(process.poll() is None for _, process in children):
            time.sleep(300)
        failures = [(arm, process.returncode) for arm, process in children
                    if process.returncode != 0]
        if failures:
            write_status('P_OFFICIAL_FAILED', failures=failures)
            raise SystemExit(1)
        write_status('P_OFFICIAL_PAIR_DONE',
                     arms=[arm for arm, _ in children])


def main():
    write_status('WAITING_FOR_A')
    formal = REPORTS / 'server28_partitioned_a_20261102_formal'
    completion = [ROOT / 'runs/server28_partitioned_a_20261102_formal' / arm /
                  'completed.json' for arm in ['utility', 'random']]
    while not all(path.exists() for path in completion):
        time.sleep(300)
    write_status('A_COMPLETION_PRESENT')
    run([sys.executable, '-u', str(ROOT / 'scripts/verify_partitioned_a_v2.py'),
         '--phase', 'formal'], 'A_VERIFYING')
    run([sys.executable, '-u', str(ROOT / 'research/curriculum_p/prepare.py'),
         '--phase', 'smoke'], 'P_SMOKE_PREPARING')
    launch_pair('smoke')
    run([sys.executable, '-u', str(ROOT / 'research/curriculum_p/verify.py'),
         '--phase', 'smoke'], 'P_SMOKE_VERIFYING')
    run([sys.executable, '-u', str(ROOT / 'research/curriculum_p/prepare.py'),
         '--phase', 'formal'], 'P_FORMAL_PREPARING')
    launch_pair('formal')
    run([sys.executable, '-u', str(ROOT / 'research/curriculum_p/verify.py'),
         '--phase', 'formal'], 'P_FORMAL_VERIFYING')
    launch_official_pair()
    write_status('COMPLETE')


if __name__ == '__main__':
    main()
