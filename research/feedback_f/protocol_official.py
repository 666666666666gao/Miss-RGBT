"""Full F evaluation controller library; actual parent/smoke gates are required."""
import os
from pathlib import Path
import subprocess
import sys
import time

from research.feedback_f.protocol_official_plan import command_for, jobs_for
from research.feedback_f.protocol_verify import verify_run
from research.protocol_r.common import ROOT, read, save, sha
from research.protocol_r.launch import gpu_ready, now


def official_sources(config):
    snapshot = Path(config['snapshot'])
    runtime = ROOT / 'runtime/task_utility_20261018'
    # The native evaluator inserts the frozen A runtime ahead of PYTHONPATH.
    for folder in ('lib', 'miss_rgbt', 'experiments/cadtrack'):
        for path in sorted((snapshot / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.txt'):
                assert sha(path) == sha(runtime / path.relative_to(snapshot)), str(path)
    paths = [Path(__file__), Path(__file__).with_name('protocol_official_plan.py'),
             Path(__file__).with_name('protocol_verify.py'),
             Path(__file__).parents[1] / 'protocol_r/common.py',
             Path(__file__).parents[1] / 'protocol_r/launch.py',
             ROOT / 'evaluate_candidate_official28.py',
             runtime / 'reports/e0/inventory.json',
             runtime / 'experiments/cadtrack/cadtrack.yaml',
             Path(config['source_manifest'])]
    paths += [snapshot / 'research/feedback_f' / name for name in
              ('evaluate.py', 'verify_official.py', 'tracker.py', 'model.py')]
    for folder in ('lib', 'miss_rgbt', 'scripts'):
        paths += sorted((runtime / folder).rglob('*.py'))
    return {str(path): sha(path) for path in paths}


def run_child(command, environment, snapshot, log_path, status_path, job):
    with log_path.open('xb') as stream:
        child = subprocess.Popen(command, cwd=str(snapshot), env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    save(status_path, dict(stage='OFFICIAL_RUNNING', job=job, pid=child.pid,
                          command=command, started_utc=now()))
    while child.poll() is None:
        time.sleep(300)
    record = dict(stage='OFFICIAL_EXITED', job=job, pid=child.pid,
                  returncode=child.returncode, ended_utc=now())
    save(log_path.with_suffix('.exit.json'), record)
    save(status_path, record)
    assert child.returncode == 0, (job, child.returncode, str(log_path))


def run(config, gpu, results_root, queue, verify_evidence, verify_smokes):
    """Verify a saved F run, then evaluate all five weights under both conditions.

    This has no CLI. The future campaign caller must supply the same actual
    P*/M1/frozen-source and four-smoke gates used by the F training runner.
    """
    from research.feedback_f.verify_official import verify
    assert gpu in (0, 1, 2, 3)
    snapshot = Path(config['snapshot'])
    assert Path(__file__).resolve().parents[2] == snapshot
    verified = verify_run(config, verify_evidence, verify_smokes)
    assert verified['status'] == 'passed'
    root = Path(config['output'])
    done = read(root / 'completed.json')
    assert verified['completed_sha256'] == sha(root / 'completed.json')
    assert verified['selected_epoch'] == done['selected_epoch']
    jobs = jobs_for(config, done, results_root)
    frozen = official_sources(config)
    frozen.update(verified['evidence_sha256'])
    frozen.update(verified['checkpoints_sha256'])
    queue = Path(queue)
    queue.mkdir(exist_ok=False)
    save(queue / 'training_verified.json', verified)
    frozen[str(queue / 'training_verified.json')] = sha(queue / 'training_verified.json')
    save(queue / 'official_sources.json', frozen)
    save(queue / 'registration.json', dict(jobs=jobs, gpu=gpu, pid=os.getpid(),
        poll_seconds=300, workers_per_gpu=4, all_checkpoints_retained=True,
        official_test_selection_used=False, registered_utc=now()))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(snapshot) + ':' + str(ROOT))
    rows = []
    for job in jobs:
        verify_evidence(config)
        verify_smokes(config)
        assert all(sha(path) == digest for path, digest in frozen.items())
        assert sha(job['checkpoint']) == job['checkpoint_sha256']
        assert not Path(job['output']).exists()
        preflight = gpu_ready(gpu)
        name = 'epoch%02d_%s' % (job['epoch'], job['condition'])
        save(queue / (name + '_preflight.json'), preflight)
        run_child(command_for(job, gpu, snapshot, sys.executable),
                  environment, snapshot, queue / (name + '.log'),
                  queue / 'status.json', job)
        measured = verify(Path(job['output']), job['checkpoint_sha256'], job['condition'])
        assert all(sha(path) == digest for path, digest in frozen.items())
        rows.append(dict(job, metrics=measured, independently_recomputed=True,
            completed_sha256=sha(Path(job['output']) / 'completed.json')))
        save(queue / 'verified_metrics.json', rows)
    assert len(rows) == 10
    result = dict(status='complete', rows=rows, selected_epoch=done['selected_epoch'],
        full_evaluations=10, sequences_per_evaluation=479, frames_per_evaluation=337353,
        official_test_selection_used=False, all_checkpoints_retained=True,
        four_arm_pairing_verified=False, ended_utc=now(),
        scope='One verified F arm; caller must still verify four-arm training pairing')
    save(queue / 'official_completed.json', result)
    save(queue / 'status.json', dict(stage='COMPLETE', ended_utc=now()))
    return result
