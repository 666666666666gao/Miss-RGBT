"""Finish the first paired campaign, then train E3/E4 for the native 20 epochs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_campaign as campaign


ROOT = campaign.ROOT
PREVIOUS = ROOT / 'logs/campaign_20261013_v1'
campaign.RUN = ROOT / 'logs/full20_20261013_v1'


def main():
    os.chdir(str(ROOT))
    campaign.RUN.mkdir(parents=True, exist_ok=False)
    configs = {stage: json.loads((ROOT / 'configs' / (stage + '_full20.json')).read_text())
               for stage in ['e3', 'e4']}
    for stage, config in configs.items():
        assert config['epochs'] == 20 and config['samples_per_epoch'] == 20000
        assert config['microbatch'] * config['accumulation'] == 8
        assert config['checkpoint_epochs'] == [10, 20]
        assert config['stage'].lower() == stage and config['reverse_weight'] == 0
        assert not (ROOT / config['output']).exists()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ['miss_rgbt', 'lib', 'scripts', 'configs', 'experiments']
              for p in sorted((ROOT / folder).rglob('*'))
              if p.is_file() and p.suffix in ['.py', '.json', '.yaml']}
    campaign.write('source_sha256.json', hashes)
    campaign.write('campaign.json', dict(
        source_version=(ROOT / 'reports/FULL20_SOURCE_VERSION.txt').read_text().strip(),
        queued_utc=datetime.datetime.utcnow().isoformat() + 'Z',
        prerequisite=str(PREVIOUS), configs=configs,
        formal_budget_per_model=dict(epochs=20, samples_per_epoch=20000,
            effective_batch=8, optimizer_updates=50000, seed=2027),
        evaluation_epoch=20, selection='fixed final epoch; no test-based selection'))
    campaign.write('status.json', dict(stage='WAITING_FIRST_CAMPAIGN', controller_pid=os.getpid()))
    print('WAITING_FIRST_CAMPAIGN', flush=True)
    previous = json.loads((PREVIOUS / 'status.json').read_text())
    while previous['stage'] != 'CAMPAIGN_COMPLETE':
        assert Path('/proc/' + str(previous['controller_pid'])).exists(), previous
        time.sleep(300)
        previous = json.loads((PREVIOUS / 'status.json').read_text())

    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    # Current free space is 34 GB: retain only epoch10 and epoch20 in each full run.
    assert shutil.disk_usage(str(ROOT)).free > 10 * 1024 ** 3
    rows = subprocess.check_output(['nvidia-smi', '--id=2,3', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
    assert len(rows) == 2
    for row in rows:
        index, memory = [int(x.strip()) for x in row.split(',')]
        assert index in [2, 3] and memory < 300, row
    campaign.write('admission.json', dict(gpus=rows,
        free_bytes=shutil.disk_usage(str(ROOT)).free,
        admitted_utc=datetime.datetime.utcnow().isoformat() + 'Z'))
    started = time.time()
    campaign.write('status.json', dict(stage='FULL20_E3_E4_RUNNING', controller_pid=os.getpid(),
        started_utc=datetime.datetime.utcnow().isoformat() + 'Z'))
    p3 = campaign.launch('e3_full20_train',
        ['-m', 'miss_rgbt.train', '--config', 'configs/e3_full20.json'], '2')
    p4 = campaign.launch('e4_full20_train',
        ['-m', 'miss_rgbt.train', '--config', 'configs/e4_full20.json'], '3')
    campaign.finish('e3_full20_train', p3)
    campaign.finish('e4_full20_train', p4)
    for stage in ['e3', 'e4']:
        training = ROOT / configs[stage]['output']
        completed = json.loads((training / 'completed.json').read_text())
        assert completed['epochs'] == 20 and completed['updates'] == 50000, completed
        checkpoint = str(training / 'epoch20.pth.tar')
        for condition in ['complete', 'miss']:
            label = stage + '_full20_' + condition
            campaign.write('status.json', dict(stage=label.upper() + '_RUNNING', controller_pid=os.getpid()))
            folder = ROOT / 'results' / (label + '_20261013')
            process = campaign.launch(label, ['scripts/evaluate_model.py', '--checkpoint', checkpoint,
                '--output', str(folder), '--condition', condition], '2,3')
            campaign.finish(label, process)
            campaign.collect(label, folder)
    campaign.write('status.json', dict(stage='CAMPAIGN_COMPLETE', controller_pid=os.getpid(),
        seconds=time.time() - started, research_goal='requires metric review and reverse-feedback ablation'))
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
