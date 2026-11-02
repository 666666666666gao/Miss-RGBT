"""Launch fixed candidate box consensus on free GPU2."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/home/gaob/Miss-RGBT')
stem = 'server28_box_consensus_20261102'
script = root/'scripts/diagnose_box_consensus.py'
protocol = root/'docs/BOX_CONSENSUS_DIAGNOSTIC_20261102.md'
output = root/'reports'/stem
receipt = root/'reports'/(stem+'.launch.json')
log = root/'reports'/(stem+'.log')
snapshot = root/'reports'/(stem+'.preregistered_protocol.md')
assert not any(path.exists() for path in [output, receipt, log, snapshot])
rows = subprocess.check_output(['nvidia-smi',
    '--query-gpu=index,memory.used,temperature.gpu,power.limit',
    '--format=csv,noheader'], universal_newlines=True).splitlines()
gpu = [row.split(',') for row in rows if row.split(',')[0].strip() == '2'][0]
memory, temperature, power = [float(value.strip().split()[0]) for value in gpu[1:]]
assert memory < 500 and temperature < 85 and power == 250
runtime = root/'runtime/task_utility_20261018'
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(runtime))
snapshot.write_bytes(protocol.read_bytes())
command = [sys.executable, '-u', str(script), '--output', str(output)]
with log.open('xb') as stream:
    process = subprocess.Popen(command, cwd=str(runtime), env=environment,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=process.pid, gpu=2, memory_mib=memory, temperature_c=temperature,
    power_limit_w=power, command=command, cwd=str(runtime), output=str(output), log=str(log),
    protocol_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    launched_utc=datetime.datetime.utcnow().isoformat()+'Z')
with receipt.open('x', newline='\n') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record))
