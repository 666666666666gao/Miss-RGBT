"""Launch the paired train882-only replication; never invoke an evaluator."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['smoke', 'formal'])
    args = parser.parse_args()
    root = Path('/home/gaob/Miss-RGBT')
    runtime = root/'runtime/task_utility_20261018'
    stem = 'server28_partitioned_a_20261102_'+args.phase
    campaign = root/'reports'/stem
    assert not campaign.exists()
    rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,temperature.gpu,power.limit', '--format=csv,noheader'],
        universal_newlines=True).splitlines()
    for gpu in [2, 3]:
        values = rows[gpu].split(',')
        assert int(values[0]) == gpu
        memory, temperature, power = [float(value.strip().split()[0]) for value in values[1:]]
        assert memory < 500 and temperature < 85 and power == 250
    excluded = []
    if args.phase == 'formal':
        for selector in ['utility', 'random']:
            previous = root/'runs/server28_partitioned_a_20261102_smoke'/selector
            completed = json.loads((previous/'completed.json').read_text())
            assert completed['status'] == 'complete' and completed['updates'] == 12
            excluded.append(str(previous/'input_manifest.jsonl'))
        assert (root/'reports/server28_partitioned_a_20261102_smoke/verification.json').is_file()
    campaign.mkdir()
    protocol = root/'docs/PARTITIONED_A_REPLICATION_20261102.md'
    (campaign/'preregistered_protocol.md').write_bytes(protocol.read_bytes())
    base = json.loads((root/'reports/server28-candidates-20261030/batch10_utility_seed2027/config.json').read_text())
    records = []
    for selector, gpu in [('utility', 2), ('random', 3)]:
        config = dict(base)
        config['task_utility'] = dict(base['task_utility'], selection=selector, selection_seed=2028)
        for key in ['validation_seed', 'validation_samples', 'checkpoint_epochs']:
            config.pop(key)
        config.update(seed=2028 if args.phase == 'formal' else 63201,
            development_seed=63102 if args.phase == 'formal' else 63202,
            development_samples=128 if args.phase == 'formal' else 8,
            output=str(root/'runs'/stem/selector), exclude_manifests=excluded,
            purpose='Paired replication; not novel method',
            epochs=20 if args.phase == 'formal' else 6,
            samples_per_epoch=20000 if args.phase == 'formal' else 20,
            lr_drop_epoch=10 if args.phase == 'formal' else 2,
            early_stop_min_epochs=14 if args.phase == 'formal' else 6)
        config_path = campaign/(selector+'.json')
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
            PYTHONPATH=str(runtime)+':'+str(root))
        script = root/'scripts/train_partitioned_a.py'
        command = [sys.executable, '-u', str(script), '--config', str(config_path)]
        with (campaign/(selector+'.log')).open('xb') as stream:
            process = subprocess.Popen(command, cwd=str(runtime), env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(selector=selector, gpu=gpu, pid=process.pid, command=command,
            log=str(campaign/(selector+'.log')), config=config, cwd=str(runtime),
            script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
            launched_utc=datetime.datetime.utcnow().isoformat()+'Z')
        records.append(record)
        (campaign/'launch.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
