import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path('/home/gaob/Miss-RGBT')
RUNTIME = ROOT / 'runtime/task_utility_20261018'
OUTPUT = RUNTIME / 'reports/server28_fusion_adapter_pilot_20261102'


def main():
    rows = subprocess.check_output(
        ['nvidia-smi', '--id=2', '--query-gpu=memory.used',
         '--format=csv,noheader,nounits'], universal_newlines=True).splitlines()
    assert len(rows) == 1 and int(rows[0]) < 500, rows
    assert not OUTPUT.exists()
    log_path = OUTPUT.with_suffix('.log')
    command = [sys.executable, '-u', 'scripts/probe_fusion_adapter.py',
               '--output', str(OUTPUT)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2', PYTHONPATH=str(RUNTIME),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sources = [RUNTIME / 'scripts/probe_fusion_adapter.py',
               RUNTIME / 'scripts/launch_fusion_adapter_pilot.py',
               ROOT / 'docs/FUSION_ADAPTER_PILOT_20261102.md']
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=str(RUNTIME), env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    receipt = dict(pid=process.pid, command=command, physical_gpu=2,
                   started_utc=datetime.datetime.utcnow().isoformat() + 'Z',
                   output=str(OUTPUT), log=str(log_path),
                   source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sources}, poll_seconds=300)
    with OUTPUT.with_name(OUTPUT.name + '_launch.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
