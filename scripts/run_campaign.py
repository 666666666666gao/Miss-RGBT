"""One immutable E0 -> E3/E4 -> complete/Miss campaign; errors remain visible."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'logs/campaign_20261013_v1'
PYTHON = ROOT / 'conda-envs/miss_rgbt/bin/python'


def write(name, data):
    (RUN / name).write_text(json.dumps(data, indent=2), encoding='utf-8')


def launch(label, args, devices):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, PYTHONPATH=str(ROOT),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled')
    command = [str(PYTHON)] + args
    with (RUN / (label + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    write(label + '_launch.json', dict(command=command, pid=process.pid, cuda_visible_devices=devices,
          started_utc=datetime.datetime.utcnow().isoformat() + 'Z'))
    print('LAUNCHED {} pid={} CUDA={}'.format(label, process.pid, devices), flush=True)
    return process


def finish(label, process):
    code = process.wait()
    write(label + '_exit.json', dict(pid=process.pid, exit_code=code,
                                    ended_utc=datetime.datetime.utcnow().isoformat() + 'Z'))
    assert code == 0, '{} failed with exit {}, see original log'.format(label, code)


def collect(label, folder):
    target = ROOT / 'reports' / label
    target.mkdir(parents=True)
    for path in [folder / 'metrics.json', folder / 'completed.json'] + list(folder.glob('*_curves.json')):
        shutil.copy2(str(path), str(target / path.name))


def main():
    os.chdir(str(ROOT))
    RUN.mkdir(parents=True, exist_ok=False)
    # Admission only; no temperature or power query and no process preemption.
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                    '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
    admitted = []
    for row in rows:
        index, uuid, memory = [x.strip() for x in row.split(',')]
        if int(index) in [2, 3]:
            assert int(memory) < 300, row
            admitted.append(dict(index=int(index), uuid=uuid, memory_used_mib=int(memory)))
    assert len(admitted) == 2
    assert shutil.disk_usage(str(ROOT)).free > 20 * 1024 ** 3
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['miss_rgbt', 'lib', 'scripts', 'configs', 'experiments']
        for p in sorted((ROOT / folder).rglob('*')) if p.is_file() and p.suffix in ['.py', '.json', '.yaml']}
    write('source_sha256.json', hashes)
    write('campaign.json', dict(source_version=(ROOT / 'reports/SOURCE_VERSION.txt').read_text().strip(),
          started_utc=datetime.datetime.utcnow().isoformat() + 'Z', admitted_gpus=admitted,
          formal_budget_per_model=dict(epochs=5, samples_per_epoch=20000, effective_batch=8,
                                      optimizer_updates=12500, seed=2027),
          sequence=['E0 complete', 'E3/E4 concurrent train', 'E3 complete', 'E3 miss', 'E4 complete', 'E4 miss']))
    started = time.time()
    write('status.json', dict(stage='E0_COMPLETE_RUNNING', controller_pid=os.getpid()))
    p = launch('e0_complete', ['scripts/evaluate_complete.py', 'run'], '2,3')
    finish('e0_complete', p)
    collect('e0_complete', ROOT / 'results/e0_complete_20261013')
    write('status.json', dict(stage='FORMAL_E3_E4_RUNNING', controller_pid=os.getpid()))
    p3 = launch('e3_train', ['-m', 'miss_rgbt.train', '--config', 'configs/e3.json'], '2')
    p4 = launch('e4_train', ['-m', 'miss_rgbt.train', '--config', 'configs/e4.json'], '3')
    finish('e3_train', p3)
    finish('e4_train', p4)
    for stage in ['e3', 'e4']:
        for condition in ['complete', 'miss']:
            label = stage + '_' + condition
            write('status.json', dict(stage=label.upper() + '_RUNNING', controller_pid=os.getpid()))
            checkpoint = 'checkpoints/{}_seed2027/epoch05.pth.tar'.format(stage)
            folder = ROOT / 'results' / (label + '_20261013')
            p = launch(label, ['scripts/evaluate_model.py', '--checkpoint', checkpoint,
                '--output', str(folder), '--condition', condition], '2,3')
            finish(label, p)
            collect(label, folder)
    write('status.json', dict(stage='CAMPAIGN_COMPLETE', controller_pid=os.getpid(), seconds=time.time() - started,
                             research_goal='still requires metric review and module ablations'))
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
