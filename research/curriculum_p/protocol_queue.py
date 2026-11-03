"""Wait for R, then dispatch six-arm smokes and full runs onto free GPUs."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from research.curriculum_p.engine import ARMS
from research.curriculum_p.protocol_prepare import CAMPAIGN, SNAPSHOT
from research.protocol_r.common import ROOT, REPORT, read, save, sha

QUEUE = ROOT / 'reports/server28_rppm_ostrack_20261103_queue'
POLL_SECONDS = 300


def now():
    return datetime.datetime.utcnow().isoformat() + 'Z'


def free_gpus():
    rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,temperature.gpu,power.limit',
        '--format=csv,noheader,nounits'], universal_newlines=True)
    available = []
    for row in rows.splitlines():
        gpu, memory, temperature, power = [float(value.strip()) for value in row.split(',')]
        assert int(gpu) in (0, 1, 2, 3) and power == 250, row
        if memory < 500 and temperature < 85:
            available.append(int(gpu))
    return available


def dispatch(phase, environment, sources):
    pending, active, finished = list(ARMS), {}, []
    while pending or active:
        assert all(sha(path) == digest for path, digest in sources.items())
        for gpu, (arm, child) in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            result = dict(arm=arm, gpu=gpu, phase=phase, pid=child.pid,
                          returncode=code, ended_utc=now())
            save(QUEUE / ('%s_%s_exit.json' % (arm, phase)), result)
            assert code == 0, result
            receipt = (Path(read(CAMPAIGN / (arm + '.json'))['output'] + '_smoke') /
                       'smoke_verified.json' if phase == 'smoke' else
                       CAMPAIGN / (arm + '_formal_controller') / 'official_completed.json')
            done = read(receipt)
            assert done['status'] == ('passed' if phase == 'smoke' else 'complete')
            finished.append(arm)
            del active[gpu]
        if pending:
            for gpu in free_gpus():
                if gpu in active or not pending:
                    continue
                arm = pending.pop(0)
                command = [sys.executable, '-u',
                    str(SNAPSHOT / 'research/curriculum_p/protocol_launch.py'),
                    '--arm', arm, '--gpu', str(gpu), '--phase', phase]
                with (QUEUE / ('%s_%s.log' % (arm, phase))).open('xb') as stream:
                    child = subprocess.Popen(command, cwd=str(SNAPSHOT),
                        env=environment, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (arm, child)
                save(QUEUE / ('%s_%s_launch.json' % (arm, phase)),
                     dict(arm=arm, gpu=gpu, phase=phase, pid=child.pid,
                          command=command, started_utc=now()))
        save(QUEUE / 'status.json', dict(stage=phase.upper(), pending=pending,
            active=[dict(gpu=gpu, arm=arm, pid=child.pid)
                    for gpu, (arm, child) in active.items()],
            finished=finished, observed_utc=now()))
        if pending or active:
            time.sleep(POLL_SECONDS)


def main():
    QUEUE.mkdir(exist_ok=False)
    paths = [Path(__file__)] + sorted((ROOT / 'research/curriculum_p').glob('*.py'))
    paths += sorted((ROOT / 'research/protocol_r').glob('*.py'))
    sources = {str(path.resolve()): sha(path) for path in paths}
    save(QUEUE / 'source_sha256.json', sources)
    save(QUEUE / 'registration.json', dict(pid=os.getpid(), arms=list(ARMS),
        poll_seconds=POLL_SECONDS, gpus=[0, 1, 2, 3], preempts_existing_jobs=False,
        gate='Both R full training verifications and twenty full official evaluations',
        smoke_before_formal='All six four-update real-image smokes',
        formal_epochs=20, full_evaluations_per_arm=10, total_full_evaluations=60,
        full_sequences=479, full_frames=337353, registered_utc=now()))
    required = [REPORT / (arm + suffix) for arm in ('r1', 'r2')
                for suffix in ('_official_completed.json', '_training_verified.json')]
    while True:
        assert all(sha(path) == digest for path, digest in sources.items())
        missing = [str(path) for path in required if not path.is_file()]
        save(QUEUE / 'status.json', dict(stage='WAITING_FOR_R', missing=missing,
                                         observed_utc=now()))
        if not missing:
            break
        time.sleep(POLL_SECONDS)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(ROOT / 'runtime/task_utility_20261018') + ':' + str(ROOT))
    command = [sys.executable, '-u',
               str(ROOT / 'research/curriculum_p/protocol_prepare.py')]
    with (QUEUE / 'preparation.log').open('xb') as stream:
        child = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                                  stdout=stream, stderr=subprocess.STDOUT)
    save(QUEUE / 'status.json', dict(stage='PREPARING', pid=child.pid,
                                     command=command, started_utc=now()))
    while child.poll() is None:
        time.sleep(POLL_SECONDS)
    save(QUEUE / 'preparation_exit.json', dict(returncode=child.returncode, ended_utc=now()))
    assert child.returncode == 0, ('Preparation failed; no arm launched', child.returncode)
    assert all(sha(path) == digest for path, digest in sources.items())
    frozen = read(CAMPAIGN / 'source_sha256.json')
    sources.update(frozen)
    environment.update(PYTHONPATH=str(SNAPSHOT) + ':' + str(ROOT))
    dispatch('smoke', environment, sources)
    dispatch('formal', environment, sources)
    command = [sys.executable, '-u',
               str(SNAPSHOT / 'research/curriculum_p/protocol_verify.py')]
    with (QUEUE / 'paired_verification.log').open('xb') as stream:
        child = subprocess.Popen(command, cwd=str(SNAPSHOT), env=environment,
                                  stdout=stream, stderr=subprocess.STDOUT)
    save(QUEUE / 'status.json', dict(stage='VERIFYING_PAIRING', pid=child.pid,
                                     command=command, started_utc=now()))
    while child.poll() is None:
        time.sleep(POLL_SECONDS)
    save(QUEUE / 'paired_verification_exit.json',
         dict(returncode=child.returncode, ended_utc=now()))
    assert child.returncode == 0, ('Pairing verification failed', child.returncode)
    assert read(CAMPAIGN / 'paired_training_verified.json')['status'] == 'passed'
    save(QUEUE / 'completed.json', dict(status='complete', arms=list(ARMS),
        full_evaluations=60, pairing_verified=True, ended_utc=now()))
    save(QUEUE / 'status.json', dict(stage='COMPLETE', ended_utc=now()))
    print(json.dumps(read(QUEUE / 'completed.json')), flush=True)


if __name__ == '__main__':
    main()
