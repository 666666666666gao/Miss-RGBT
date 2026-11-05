"""Launch a matched F arm on an idle GPU0/1 after parent validation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from partitioned_a_official_queue_v2 import gpu_ready
from prepare import PROJECT, sha
import train as trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--arm', required=True, choices=trainer.ARMS)
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    campaign = Path(args.campaign)
    prepared = json.loads((campaign/'prepared.json').read_text())
    assert prepared['phase'] in ['smoke', 'formal']
    for path, digest in prepared['source_sha256'].items():
        assert sha(path) == digest
    entry = next(row for row in prepared['arms'] if row['arm'] == args.arm)
    assert sha(entry['config']) == entry['config_sha256']
    config = json.loads(Path(entry['config']).read_text())
    assert config['feedback']['mode'] == args.arm
    assert config['phase'] == prepared['phase']
    assert config['curriculum_arm'] == prepared['curriculum_arm']
    trainer.validate_parents(config)
    if config['phase'] == 'formal':
        trainer.validate_smoke(config)
    assert not Path(config['output']).exists()
    receipt = campaign/('launch_'+args.arm+'.json')
    assert not receipt.exists()
    preflight = gpu_ready(args.gpu)
    runtime = PROJECT/'runtime/task_utility_20261018'
    script = Path(__file__).with_name('train.py').resolve()
    command = [sys.executable, '-u', str(script), '--config', entry['config']]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(runtime)+':'+str(PROJECT/'scripts')+':'+str(PROJECT))
    with (campaign/(args.arm+'.log')).open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(runtime), env=environment,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=process.pid, arm=args.arm, gpu=args.gpu, phase=config['phase'],
        command=command, cwd=str(runtime), config=entry['config'],
        config_sha256=entry['config_sha256'], trainer_sha256=sha(script),
        prepared_sha256=sha(campaign/'prepared.json'), preflight=preflight,
        started_utc=datetime.datetime.utcnow().isoformat()+'Z')
    with receipt.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
