"""Launch the frozen positional-token diagnostic on server28 GPU2."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/home/gaob/Miss-RGBT')
runtime = root / 'runtime/task_utility_20261018'
script = root / 'scripts/diagnose_absent_position.py'
stem = 'server28_absent_position_20261102_v4'
output = root / 'reports' / stem
receipt = root / 'reports' / (stem + '.launch.json')
log_path = root / 'reports' / (stem + '.log')
assert not output.exists()
assert not receipt.exists()
assert not log_path.exists()

gpu_rows = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,memory.used,temperature.gpu,power.limit',
    '--format=csv,noheader',
], universal_newlines=True).strip().splitlines()
gpu2 = [row.split(',') for row in gpu_rows if row.split(',')[0].strip() == '2'][0]
memory_mib = float(gpu2[1].strip().split()[0])
temperature_c = float(gpu2[2].strip().split()[0])
power_limit_w = float(gpu2[3].strip().split()[0])
assert memory_mib < 500
assert temperature_c < 85
assert power_limit_w == 250

environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    PYTHONPATH=str(runtime))
command = [sys.executable, '-u', str(script), '--output', str(output)]
with log_path.open('xb') as log:
    process = subprocess.Popen(command, cwd=str(runtime), env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
launch = dict(pid=process.pid, gpu=2, memory_used_mib=memory_mib,
    temperature_c=temperature_c, power_limit_w=power_limit_w,
    command=command, cwd=str(runtime), output=str(output), log=str(log_path),
    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    launched_utc=datetime.datetime.utcnow().isoformat() + 'Z')
with receipt.open('x') as stream:
    json.dump(launch, stream, indent=2)
    stream.write('\n')
print(json.dumps(launch))
