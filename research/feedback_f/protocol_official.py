"""Full F evaluation controller library; actual parent/smoke gates are required."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from research.feedback_f.protocol_official_plan import command_for, jobs_for
from research.feedback_f.protocol_training import ARMS
from research.feedback_f.protocol_verify import verify_pairs, verify_run
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


def verify_campaign(configs, controllers, results_root, verify_evidence, verify_smokes):
    """Recheck four saved arms and all forty full evaluations before reporting."""
    from research.feedback_f.verify_official import verify
    assert [config['feedback']['mode'] for config in configs] == list(ARMS)
    assert set(controllers) == set(ARMS)
    assert len({str(Path(path).resolve()) for path in controllers.values()}) == 4
    evidence, arms, table = {}, {}, []
    for config in configs:
        mode = config['feedback']['mode']
        controller = Path(controllers[mode])
        verified = verify_run(config, verify_evidence, verify_smokes)
        assert verified['status'] == 'passed'
        assert read(controller / 'training_verified.json') == verified
        frozen = official_sources(config)
        frozen.update(verified['evidence_sha256'])
        frozen.update(verified['checkpoints_sha256'])
        frozen[str(controller / 'training_verified.json')] = sha(
            controller / 'training_verified.json')
        assert read(controller / 'official_sources.json') == frozen
        assert all(sha(path) == digest for path, digest in frozen.items())
        for path, digest in frozen.items():
            assert path not in evidence or evidence[path] == digest
            evidence[path] = digest
        root = Path(config['output'])
        completed = read(root / 'completed.json')
        jobs = jobs_for(config, completed, results_root)
        registration = read(controller / 'registration.json')
        assert registration['jobs'] == jobs and registration['gpu'] in (0, 1, 2, 3)
        assert registration['poll_seconds'] == 300 and registration['workers_per_gpu'] == 4
        assert registration['all_checkpoints_retained'] is True
        assert registration['official_test_selection_used'] is False
        official = read(controller / 'official_completed.json')
        assert official['status'] == 'complete' and official['full_evaluations'] == 10
        assert official['sequences_per_evaluation'] == 479
        assert official['frames_per_evaluation'] == 337353
        assert official['selected_epoch'] == verified['selected_epoch']
        assert official['official_test_selection_used'] is False
        assert official['all_checkpoints_retained'] is True
        assert len(official['rows']) == len(jobs) == 10
        assert read(controller / 'verified_metrics.json') == official['rows']
        epoch_rows = {}
        for job, row in zip(jobs, official['rows']):
            destination = Path(job['output'])
            name = 'epoch%02d_%s' % (job['epoch'], job['condition'])
            exit_path = controller / (name + '.exit.json')
            exited = read(exit_path)
            assert exited['stage'] == 'OFFICIAL_EXITED' and exited['job'] == job
            assert exited['returncode'] == 0
            measured = verify(destination, job['checkpoint_sha256'], job['condition'])
            assert row == dict(job, metrics=measured, independently_recomputed=True,
                completed_sha256=sha(destination / 'completed.json'))
            epoch = job['epoch']
            if epoch not in epoch_rows:
                epoch_rows[epoch] = dict(arm=mode, curriculum_arm=config['curriculum_arm'],
                    epoch=epoch, checkpoint=job['checkpoint'],
                    checkpoint_sha256=job['checkpoint_sha256'],
                    development_selected=job['development_selected'])
            entry = epoch_rows[epoch]
            assert entry['checkpoint_sha256'] == job['checkpoint_sha256']
            for label, keys in (
                    ('LasHeR' + ('-Miss' if job['condition'] == 'miss' else ''),
                     ('PR', 'NPR', 'SR')),
                    ('RGBT234' + ('-Miss' if job['condition'] == 'miss' else ''),
                     ('MPR', 'MSR'))):
                spec = measured[label]
                assert (spec['sequences'], spec['frames']) == (
                    (245, 220703) if label.startswith('LasHeR') else (234, 116650))
                entry.update({label + '_' + key: spec[key] for key in keys})
            for path in [exit_path] + [destination / name for name in (
                    'run.json', 'completed.json', 'metrics.json', 'all_receipts.json')]:
                evidence[str(path)] = sha(path)
        assert sum(row['development_selected'] for row in epoch_rows.values()) == 1
        table.extend(epoch_rows[epoch] for epoch in range(16, 21))
        arms[mode] = verified
        for name in ('registration.json', 'training_verified.json', 'official_sources.json',
                     'official_completed.json', 'verified_metrics.json'):
            evidence[str(controller / name)] = sha(controller / name)
    pairing = verify_pairs(configs)
    for config in configs:
        verify_evidence(config)
        verify_smokes(config)
    assert all(sha(path) == digest for path, digest in evidence.items())
    return dict(status='complete', arms=arms, pairing=pairing, rows=table,
        full_evaluations=40, sequences_per_evaluation=479, frames_per_evaluation=337353,
        four_arm_pairing_verified=True, official_test_selection_used=False,
        all_checkpoints_retained=True, evidence_sha256=evidence,
        scope='Four verified F arms; all five epochs disclosed with paired Complete/Miss '
              'metrics. Development selection only; no module efficacy claim.')


def write_campaign_reports(result, output):
    """Write all twenty paired rows and the four preselected rows after verification."""
    assert result['status'] == 'complete' and result['four_arm_pairing_verified'] is True
    assert result['full_evaluations'] == 40 and len(result['rows']) == 20
    selected = [row for row in result['rows'] if row['development_selected']]
    assert [row['arm'] for row in selected] == list(ARMS)
    assert all(sha(path) == digest for path, digest in result['evidence_sha256'].items())
    output = Path(output)
    output.mkdir(exist_ok=False)
    for name, rows in (('all_epochs.csv', result['rows']), ('selected_epochs.csv', selected)):
        with (output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
            writer.writeheader()
            writer.writerows(rows)
    save(output / 'completed.json', result)
