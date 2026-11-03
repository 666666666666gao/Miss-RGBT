"""Launch each replication on its GPU and queue full official evaluation."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

from research.protocol_r.common import (
    ROOT, REPORT, RUNS, SNAPSHOT, read, save, sha, check_sources,
)


def now():
    return datetime.datetime.utcnow().isoformat()+'Z'


def gpu_ready(gpu):
    row = subprocess.check_output(['nvidia-smi', '--id='+str(gpu),
        '--query-gpu=memory.used,temperature.gpu,power.limit',
        '--format=csv,noheader,nounits'], universal_newlines=True)
    memory, temperature, power = [float(x.strip()) for x in row.split(',')]
    assert memory < 500 and temperature < 85 and power == 250, row
    return dict(memory_mib=memory, temperature_c=temperature, power_limit_w=power)


def verify_training(arm, config):
    check_sources()
    root = RUNS / arm
    assert read(root / 'run_config.json') == config
    done = read(root / 'completed.json')
    history = read(root / 'development_history.json')
    assert done['status'] == 'complete' and done['epochs'] == 20
    expected = 20*config['samples_per_epoch']//(config['microbatch']*config['accumulation'])
    assert done['updates'] == expected
    assert [x['epoch'] for x in history] == list(range(1, 21))
    assert all(x['samples'] == 128 for x in history)
    import math
    assert all(math.isfinite(x['mean_task_loss']) for x in history)
    selected = min(history[-5:], key=lambda x: (x['mean_task_loss'], x['epoch']))
    assert done['selected_epoch'] == selected['epoch']
    assert not done['official_test_selection_used'] and done['all_checkpoints_retained']
    assert [x['epoch'] for x in done['final_five']] == [16, 17, 18, 19, 20]
    for item in done['final_five']:
        assert sha(item['path']) == item['sha256']
    count = 0
    with (root / 'steps.jsonl').open() as stream:
        for line in stream:
            entry = __import__('json').loads(line)
            count += 1
            assert entry['update'] == count and math.isfinite(entry['loss'])
    assert count == expected
    save(REPORT / (arm+'_training_verified.json'), dict(
        status='passed', completed_sha256=sha(root / 'completed.json'),
        updates=expected, selected_epoch=selected['epoch'], verified_utc=now()))
    return done


def run(command, env, log_path, status_path, stage, **extra):
    with log_path.open('xb') as log:
        child = subprocess.Popen(command, cwd=str(SNAPSHOT), env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    save(status_path, dict(stage=stage, pid=child.pid, command=command,
                           started_utc=now(), **extra))
    while child.poll() is None:
        time.sleep(300)
    save(status_path, dict(stage=stage+'_EXITED', pid=child.pid,
                           returncode=child.returncode, ended_utc=now(), **extra))
    assert child.returncode == 0, (stage, child.returncode, str(log_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['r1', 'r2'], required=True)
    args = parser.parse_args()
    arm = args.arm
    config_path = REPORT / (arm+'.json')
    config = read(config_path)
    check_sources()
    gpu = config['physical_gpu']
    preflight = gpu_ready(gpu)
    smoke = read(Path(config['output']+'_smoke') / 'smoke_verified.json')
    assert smoke['status'] == 'passed' and smoke['strict_reload_bitwise_equal']
    assert read(Path(config['output']+'_smoke') / 'run_config.json') == config
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(SNAPSHOT)+':'+str(ROOT))
    status = REPORT / (arm+'_status.json')
    save(REPORT / (arm+'_launch.json'), dict(controller_pid=os.getpid(),
        config=config, config_sha256=sha(config_path), preflight=preflight,
        poll_seconds=300, started_utc=now()))
    command = [sys.executable, '-u', str(SNAPSHOT/'research/protocol_r/train.py'),
               '--config', str(config_path)]
    run(command, env, REPORT/(arm+'_training.log'), status, 'TRAINING')
    done = verify_training(arm, config)
    from verify_recovery_campaign28 import verify_official
    rows = []
    for item in done['final_five']:
        row = dict(epoch=item['epoch'], checkpoint_sha256=item['sha256'],
                   development_selected=item['epoch'] == done['selected_epoch'], conditions={})
        for condition in ['complete', 'miss']:
            check_sources()
            preflight = gpu_ready(gpu)
            assert sha(item['path']) == item['sha256']
            destination = ROOT/'results/r1r2_20261103'/arm/(
                'epoch%02d_%s' % (item['epoch'], condition))
            assert not destination.exists()
            command = [sys.executable, '-u', str(ROOT/'evaluate_candidate_official28.py'),
                '--checkpoint', item['path'], '--condition', condition, '--gpus', str(gpu),
                '--workers-per-gpu', '4', '--output', str(destination)]
            run(command, env, REPORT/('%s_epoch%02d_%s.log' % (arm, item['epoch'], condition)),
                status, 'OFFICIAL', epoch=item['epoch'], condition=condition, preflight=preflight)
            measured = verify_official(destination, item['sha256'], condition,
                                       read(destination/'metrics.json'))
            row['conditions'][condition] = dict(metrics=measured,
                independently_recomputed=True, output=str(destination),
                completed_sha256=sha(destination/'completed.json'))
        rows.append(row)
        save(REPORT/(arm+'_verified_official.json'), rows)
    save(REPORT/(arm+'_official_completed.json'), dict(status='complete', rows=rows,
        selected_epoch=done['selected_epoch'], official_test_selection_used=False,
        full_evaluations=10, sequences_per_evaluation=479, frames_per_evaluation=337353,
        ended_utc=now()))
    save(status, dict(stage='COMPLETE', ended_utc=now()))


if __name__ == '__main__':
    main()
