"""Launch one prepared P arm on any verified Server 28 GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from partitioned_a_official_queue_v2 import gpu_ready
from prepare import PROJECT, sha
from schedule import ARMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['smoke', 'formal'])
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    campaign = PROJECT/'reports'/('server28_curriculum_p_20261102_'+args.phase)
    prepared = json.loads((campaign/'prepared.json').read_text())
    entry = next(row for row in prepared['arms'] if row['arm'] == args.arm)
    assert sha(entry['config']) == entry['config_sha256']
    config = json.loads(Path(entry['config']).read_text())
    assert not Path(config['output']).exists()
    for path, digest in config['parent_completion_sha256'].items():
        assert sha(path) == digest and json.loads(Path(path).read_text())['status'] == 'complete'
    for path, digest in config['exclude_manifest_sha256'].items():
        assert sha(path) == digest
    if args.phase == 'formal':
        verification = PROJECT/'reports/server28_curriculum_p_20261102_smoke/verification.json'
        checked = json.loads(verification.read_text())
        assert checked['status'] == 'PASS' and checked['phase'] == 'smoke'
        assert set(checked['arms']) == set(ARMS)
    preflight = gpu_ready(args.gpu)
    runtime = PROJECT/'runtime/task_utility_20261018'
    script = Path(__file__).with_name('train.py').resolve()
    command = [sys.executable, '-u', str(script), '--config', entry['config']]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(runtime)+':'+str(PROJECT/'scripts')+':'+str(PROJECT))
    receipt = campaign/('launch_'+args.arm+'.json')
    assert not receipt.exists()
    with (campaign/(args.arm+'.log')).open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(runtime), env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=process.pid, arm=args.arm, phase=args.phase, gpu=args.gpu, command=command,
        cwd=str(runtime), config=entry['config'], config_sha256=entry['config_sha256'],
        trainer_sha256=sha(script), protocol_sha256=sha(campaign/'protocol.md'),
        preflight=preflight, started_utc=datetime.datetime.utcnow().isoformat()+'Z')
    with receipt.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
