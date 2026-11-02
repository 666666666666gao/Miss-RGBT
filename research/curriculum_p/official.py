"""Full final-five official reporting for one verified P arm on GPU0/1."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from partitioned_a_official_queue_v2 import command_for, gpu_ready
from prepare import PROJECT, sha
from schedule import ARMS

RUNTIME = PROJECT/'runtime/task_utility_20261018'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def jobs_for(arm, completed):
    assert arm in ARMS and completed['status'] == 'complete'
    assert completed['epochs'] == 20 and completed['official_test_selection_used'] is False
    assert completed['all_checkpoints_retained']
    assert [row['epoch'] for row in completed['final_five']] == list(range(16, 21))
    assert completed['selected_epoch'] in range(16, 21)
    result = []
    for item in completed['final_five']:
        for condition in ['complete', 'miss']:
            result.append(dict(arm=arm, epoch=item['epoch'], condition=condition,
                checkpoint=item['path'], checkpoint_sha256=item['sha256'],
                output=str(PROJECT/'results/server28_curriculum_p_20261102'/(
                    '%s_epoch%02d_%s' % (arm, item['epoch'], condition))),
                development_selected=item['epoch'] == completed['selected_epoch'],
                expected_sequences=479, expected_frames=337353))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    campaign = PROJECT/'reports/server28_curriculum_p_20261102_formal'
    verified = read(campaign/'verification.json')
    assert verified['status'] == 'PASS' and verified['phase'] == 'formal'
    assert verified['prepared_sha256'] == sha(campaign/'prepared.json')
    run = PROJECT/'runs/server28_curriculum_p_20261102_formal'/args.arm
    assert verified['arms'][args.arm]['completed_sha256'] == sha(run/'completed.json')
    assert verified['arms'][args.arm]['input_manifest_sha256'] == sha(run/'input_manifest.jsonl')
    completed = read(run/'completed.json')
    jobs = jobs_for(args.arm, completed)
    assert len(jobs) == 10
    queue = campaign/('official_'+args.arm)
    queue.mkdir(exist_ok=False)
    frozen = read(run/'source_sha256.json')
    for path in [Path(__file__), Path(__file__).with_name('verify.py'),
                 PROJECT/'scripts/partitioned_a_official_queue_v2.py',
                 PROJECT/'evaluate_candidate_official28.py',
                 PROJECT/'verify_recovery_campaign28.py',
                 PROJECT/'evaluate_val97_recovery28.py',
                 PROJECT/'analyze_epoch05_health28.py', campaign/'verification.json',
                 campaign/'prepared.json', run/'completed.json', RUNTIME/'reports/e0/inventory.json']:
        frozen[str(path.resolve())] = sha(path)
    save(queue/'source_sha256.json', frozen)
    save(queue/'registration.json', dict(arm=args.arm, gpu=args.gpu, pid=os.getpid(),
        jobs=jobs, poll_seconds=300, workers_per_gpu=4, deletes_checkpoints=False,
        registered_utc=datetime.datetime.utcnow().isoformat()+'Z'))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(RUNTIME)+':'+str(PROJECT))
    # Use only the isolated official verifier, never its development entry point.
    from verify_recovery_campaign28 import verify_official
    results = []
    for job in jobs:
        assert all(sha(path) == digest for path, digest in frozen.items())
        assert sha(job['checkpoint']) == job['checkpoint_sha256']
        assert not Path(job['output']).exists()
        preflight = gpu_ready(args.gpu)
        command = command_for(job, args.gpu)
        name = 'epoch%02d_%s' % (job['epoch'], job['condition'])
        with (queue/(name+'.log')).open('xb') as stream:
            process = subprocess.Popen(command, cwd=str(RUNTIME), env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
            save(queue/'status.json', dict(stage='OFFICIAL_RUNNING', job=job,
                pid=process.pid, command=command, preflight=preflight))
            while process.poll() is None:
                time.sleep(300)
            save(queue/(name+'_exit.json'), dict(pid=process.pid, returncode=process.returncode))
            assert process.returncode == 0, (name, process.returncode)
        measured = verify_official(Path(job['output']), job['checkpoint_sha256'],
                                  job['condition'], read(Path(job['output'])/'metrics.json'))
        results.append(dict(job, metrics=measured, independently_recomputed=True))
        save(queue/'verified_metrics.json', results)
    assert len(results) == 10
    save(queue/'completed.json', dict(status='complete', metrics=results,
        selected_epoch=completed['selected_epoch'], selected_sha256=completed['selected_sha256'],
        official_test_selection_used=False, all_checkpoints_retained=True,
        scope='Full official final-five comparison; not a validated paper module'))
    save(queue/'status.json', dict(stage='COMPLETE'))


if __name__ == '__main__':
    main()
