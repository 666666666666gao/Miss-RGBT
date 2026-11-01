"""Launch one paired head-training pilot in the existing server environment."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    project = Path('/home/gaob/Miss-RGBT')
    runtime = project / 'runtime/task_utility_20261018'
    output = runtime / 'reports/server28_head_supervision_20261102'
    used = subprocess.check_output(['nvidia-smi', '--id=2',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        universal_newlines=True).strip()
    assert int(used) < 500 and not output.exists()
    sources = [runtime / 'scripts/probe_head_supervision.py',
               runtime / 'scripts/launch_head_supervision.py',
               project / 'docs/HEAD_SUPERVISION_PILOT_20261102.md']
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2', PYTHONPATH=str(runtime),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-u', 'scripts/probe_head_supervision.py',
               '--output', str(output)]
    with output.with_suffix('.log').open('x') as stream:
        worker = subprocess.Popen(command, cwd=str(runtime), env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    receipt = dict(pid=worker.pid, command=command, physical_gpu=2,
        started_utc=datetime.datetime.utcnow().isoformat() + 'Z',
        source_sha256=hashes, poll_seconds=240)
    with output.with_name(output.name + '_launch.json').open('x', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
