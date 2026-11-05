"""Run full final-five F reporting for a verified formal arm on GPU0/1."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from partitioned_a_official_queue_v2 import gpu_ready
from prepare import PROJECT, sha
import train as trainer
from verify_official import verify

RUNTIME = PROJECT/'runtime/task_utility_20261018'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def jobs_for(curriculum, arm, completed):
    assert curriculum in trainer.P_ARMS and arm in trainer.ARMS
    assert completed['status'] == 'complete' and completed['epochs'] == 20
    assert completed['feedback']['mode'] == arm
    assert completed['curriculum_arm'] == curriculum
    assert completed['official_test_selection_used'] is False
    assert completed['all_checkpoints_retained'] is True
    assert [row['epoch'] for row in completed['final_five']] == list(range(16, 21))
    selected = next(row for row in completed['final_five']
                    if row['epoch'] == completed['selected_epoch'])
    assert selected['path'] == completed['selected_checkpoint']
    assert selected['sha256'] == completed['selected_sha256']
    return [dict(arm=arm, curriculum_arm=curriculum, epoch=row['epoch'],
        condition=condition, checkpoint=row['path'], checkpoint_sha256=row['sha256'],
        development_selected=row['epoch'] == completed['selected_epoch'],
        output=str(PROJECT/'results/server28_feedback_f_20261102'/curriculum/(
            '%s_epoch%02d_%s' % (arm, row['epoch'], condition))),
        expected_sequences=479, expected_frames=337353)
        for row in completed['final_five'] for condition in ['complete', 'miss']]


def command_for(job, gpu):
    assert gpu in [0, 1] and job['condition'] in ['complete', 'miss']
    return [sys.executable, '-u', str(Path(__file__).with_name('evaluate.py').resolve()),
        '--checkpoint', job['checkpoint'], '--output', job['output'],
        '--condition', job['condition'], '--gpus', str(gpu), '--workers-per-gpu', '4']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--arm', required=True, choices=trainer.ARMS)
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    campaign = Path(args.campaign)
    checked = read(campaign/'verification.json')
    assert checked['status'] == 'PASS' and checked['phase'] == 'formal'
    assert set(checked['arms']) == set(trainer.ARMS)
    assert checked['prepared_sha256'] == sha(campaign/'prepared.json')
    assert all(sha(path) == digest for path, digest in checked['source_sha256'].items())
    prepared = read(campaign/'prepared.json')
    assert prepared['phase'] == 'formal'
    entry = next(row for row in prepared['arms'] if row['arm'] == args.arm)
    assert sha(entry['config']) == entry['config_sha256']
    config = read(entry['config'])
    trainer.validate_parents(config)
    trainer.validate_smoke(config)
    run = Path(config['output'])
    assert config == read(run/'run_config.json')
    for filename, key in [('completed.json', 'completed_sha256'),
                          ('input_manifest.jsonl', 'input_manifest_sha256')]:
        assert sha(run/filename) == checked['arms'][args.arm][key]
    completed = read(run/'completed.json')
    jobs = jobs_for(prepared['curriculum_arm'], args.arm, completed)
    frozen = read(run/'source_sha256.json')
    assert all(sha(path) == digest for path, digest in frozen.items())
    sources = sorted(Path(__file__).parent.glob('*.py'))
    sources += [PROJECT/'evaluate_candidate_official28.py',
        PROJECT/'scripts/partitioned_a_official_queue_v2.py',
        RUNTIME/'reports/e0/inventory.json', campaign/'verification.json',
        campaign/'prepared.json', run/'completed.json']
    frozen.update({str(path.resolve()): sha(path) for path in sources})
    assert all(sha(path) == digest for path, digest in frozen.items())
    queue = campaign/('official_'+args.arm)
    queue.mkdir(exist_ok=False)
    save(queue/'source_sha256.json', frozen)
    save(queue/'registration.json', dict(arm=args.arm, gpu=args.gpu, pid=os.getpid(),
        jobs=jobs, poll_seconds=300, workers_per_gpu=4, deletes_checkpoints=False,
        registered_utc=datetime.datetime.utcnow().isoformat()+'Z'))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(RUNTIME)+':'+str(PROJECT/'scripts')+':'+str(PROJECT))
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
        measured = verify(Path(job['output']), job['checkpoint_sha256'], job['condition'])
        results.append(dict(job, metrics=measured, independently_recomputed=True))
        save(queue/'verified_metrics.json', results)
    assert len(results) == 10
    save(queue/'completed.json', dict(status='complete', metrics=results,
        selected_epoch=completed['selected_epoch'], selected_sha256=completed['selected_sha256'],
        official_test_selection_used=False, all_checkpoints_retained=True,
        scope='Full final-five reporting, not evidence of a validated paper module'))
    save(queue/'status.json', dict(stage='COMPLETE'))


if __name__ == '__main__':
    main()
