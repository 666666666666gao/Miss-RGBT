"""Start the isolated CPU-only R replay with a persistent log and launch receipt."""
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
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    log = Path(str(output) + '.log')
    receipt = Path(str(output) + '.launch.json')
    assert not output.exists() and not log.exists() and not receipt.exists()
    root = Path('/home/gaob/Miss-RGBT')
    script = root / 'scripts/replay_protocol_r_frame_manifest.py'
    command = ['nice', '-n', '10', 'ionice', '-c', '3', sys.executable,
               '-u', str(script), '--output-dir', str(output)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(root / 'runtime/protocol_r_20261103') + ':' + str(root))
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=str(root / 'runtime/protocol_r_20261103'),
            env=environment, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(status='launched', pid=child.pid, command=command,
        output=str(output), log=str(log),
        launched_utc=datetime.datetime.utcnow().isoformat() + 'Z',
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda_visible_devices='', cpu_workers=2, cpu_nice=10, io_priority_class=3,
        full_epoch_replay_completed=False, freshness_gate_passed=False)
    with receipt.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
