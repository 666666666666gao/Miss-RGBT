"""Fixed full-budget input-handling, complete-FT and dual-task controls."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

import run_campaign as campaign
from package_e0_availability import sha


ROOT = campaign.ROOT
campaign.RUN = ROOT / 'logs/attribution_20261015_v2'


def state(stage):
    campaign.write('status.json', dict(stage=stage, controller_pid=os.getpid(),
        updated_utc=datetime.datetime.utcnow().isoformat() + 'Z'))


def evaluate(label, checkpoint, condition, gpus):
    state(label.upper() + '_RUNNING')
    folder = ROOT / 'results' / (label + '_20261015')
    process = campaign.launch(label, ['scripts/evaluate_model.py', '--checkpoint', str(checkpoint),
        '--output', str(folder), '--condition', condition, '--gpus', gpus], gpus)
    campaign.finish(label, process)
    campaign.collect(label, folder)
    return folder


def compare_complete(folder):
    original = ROOT / 'results/e0_complete_20261013'
    rows = []
    for path in sorted((folder / 'bbox').rglob('*.txt')):
        previous = np.loadtxt(str(original / 'results' / path.parent.name.replace('-Miss', '') / path.name))
        current = np.loadtxt(str(path))
        assert current.shape == previous.shape and np.isfinite(current).all()
        rows.append(dict(sequence=str(path.relative_to(folder / 'bbox')),
            frames=len(current), identical=bool(np.array_equal(current, previous)),
            max_abs_bbox_difference=float(np.max(np.abs(current - previous)))))
    assert len(rows) == 479 and sum(row['frames'] for row in rows) == 337353
    campaign.write('e0_availability_complete_pair.json', dict(rows=rows,
        identical_sequences=sum(row['identical'] for row in rows),
        scope='Actual whole-trajectory wrapper check. A difference must be disclosed when attributing missing-input gains.'))


def main():
    os.chdir(str(ROOT))
    campaign.RUN.mkdir(parents=True, exist_ok=False)
    configs = {stage: json.loads((ROOT / 'configs' / (stage + '_full20.json')).read_text())
               for stage in ['e0_ft', 'e2']}
    for stage, config in configs.items():
        assert config['epochs'] == 20 and config['samples_per_epoch'] == 20000
        assert config['microbatch'] * config['accumulation'] == 8
        assert config['checkpoint_epochs'] == [10, 20] and config['seed'] == 2027
        assert not config['repair'] and config['forward_weight'] == config['reverse_weight'] == 0
        assert not (ROOT / config['output']).exists()
    assert configs['e0_ft']['stage'] == 'E0-FT' and configs['e2']['stage'] == 'E2'
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['miss_rgbt', 'lib', 'scripts', 'configs', 'experiments']
        for p in sorted((ROOT / folder).rglob('*')) if p.is_file() and p.suffix in ['.py', '.json', '.yaml']}
    campaign.write('source_sha256.json', hashes)
    rows = subprocess.check_output(['nvidia-smi', '--id=2,3', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
    assert len(rows) == 2
    for row in rows:
        gpu, used = [int(value.strip()) for value in row.split(',')]
        assert gpu in [2, 3] and used < 300, row
    assert shutil.disk_usage(str(ROOT)).free > 1024 ** 3
    for config in configs.values():
        assert shutil.disk_usage(str(Path(config['output']).parent)).free > 10 * 1024 ** 3
    campaign.write('campaign.json', dict(source_version=(ROOT / 'reports/ATTRIBUTION_SOURCE_VERSION.txt').read_text().strip(),
        started_utc=datetime.datetime.utcnow().isoformat() + 'Z', configs=configs,
        formal_budget_per_model=dict(epochs=20, sampled_tuples_per_epoch=20000,
            effective_batch=8, optimizer_updates=50000, seed=2027),
        evaluation_epoch=20, selection='fixed final epoch; no test-based selection',
        bn_policy='Original box-head-only running-stat freeze; fusion BN remains training for phase1 controls.',
        concurrency='At most one training per card. After E0-FT finishes, its eval uses only GPU2 while E2 may continue on GPU3.',
        admission_gpus=rows, free_bytes=shutil.disk_usage(str(ROOT)).free))
    started = time.time()
    packaged = ROOT / 'checkpoints/e0_availability/epoch20.pth.tar'
    package_record = json.loads(packaged.with_suffix('.json').read_text())
    assert package_record['original_sha256'] == 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
    assert package_record['tensor_equality'] and package_record['optimizer_updates'] == 0
    assert sha(packaged) == package_record['packaged_sha256']
    complete = evaluate('e0_availability_complete', packaged, 'complete', '2,3')
    compare_complete(complete)
    evaluate('e0_availability_miss', packaged, 'miss', '2,3')
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    state('E0_FT_E2_FULL20_RUNNING')
    ft = campaign.launch('e0_ft_full20_train', ['-m', 'miss_rgbt.train',
        '--config', 'configs/e0_ft_full20.json'], '2')
    dual = campaign.launch('e2_full20_train', ['-m', 'miss_rgbt.train',
        '--config', 'configs/e2_full20.json'], '3')
    campaign.finish('e0_ft_full20_train', ft)
    for stage, process, gpus in [('e0_ft', ft, '2'), ('e2', dual, '2,3')]:
        if stage == 'e2':
            state('WAITING_E2_FULL20')
            campaign.finish('e2_full20_train', process)
        training = ROOT / configs[stage]['output']
        completed = json.loads((training / 'completed.json').read_text())
        assert completed['epochs'] == 20 and completed['updates'] == 50000
        for condition in ['complete', 'miss']:
            evaluate(stage + '_full20_' + condition, training / 'epoch20.pth.tar', condition, gpus)
    state('CAMPAIGN_COMPLETE')
    campaign.write('completed.json', dict(seconds=time.time() - started, research_goal='requires attribution review'))
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
