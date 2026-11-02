"""Launch only the amended delayed audit from immutable saved selections."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


project = Path('/home/gaob/Miss-RGBT')
root = project / 'reports/server28_causal_crop_training_20261102'
script = project / 'scripts/resume_causal_crop_audit.py'
protocol = project / 'docs/CAUSAL_CROP_AUDIT_AMENDMENT_20261102.md'
receipt = root / 'audit_resume.launch.json'
snapshot = root / 'audit_amendment.md'
log = root / 'audit_resume.log'
assert not any(path.exists() for path in [receipt, snapshot, log, root / 'audit_clips.json'])
assert subprocess.run(['ps', '-p', '877938'], stdout=subprocess.DEVNULL).returncode == 1
assert 'AssertionError' in (root.parent / (root.name + '.log')).read_text()
rows = subprocess.check_output(['nvidia-smi',
    '--query-gpu=index,memory.used,temperature.gpu,power.limit',
    '--format=csv,noheader'], universal_newlines=True).splitlines()
gpu = [row.split(',') for row in rows if row.split(',')[0].strip() == '2'][0]
memory, temperature, power = [float(value.strip().split()[0]) for value in gpu[1:]]
assert memory < 500 and temperature < 85 and power == 250
runtime = project / 'runtime/task_utility_20261018'
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(runtime))
snapshot.write_bytes(protocol.read_bytes())
# The child waits on this receipt before importing the audit worker.
bootstrap = ('import runpy,time,sys;from pathlib import Path;'
    'sys.path.insert(0,%r);'
    'p=Path(%r)\nwhile not p.exists(): time.sleep(.1)\nrunpy.run_path(%r,run_name="__main__")'
    % (str(script.parent), str(receipt), str(script)))
command = [sys.executable, '-u', '-c', bootstrap]
with log.open('xb') as stream:
    process = subprocess.Popen(command, cwd=str(runtime), env=environment,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=process.pid, gpu=2, command=command, cwd=str(runtime),
    launched_utc=datetime.datetime.utcnow().isoformat() + 'Z',
    script_path=str(script), script_sha256=sha(script), protocol_sha256=sha(snapshot),
    inventory_sha256=sha(root / 'audit_inventory_v2.json'),
    selection_sha256={mode: sha(root / mode / 'selection.json') for mode in ['native', 'causal']},
    count=24, length=64, seed=58103, original_pid=877938,
    memory_mib=memory, temperature_c=temperature, power_limit_w=power)
staged = root / 'audit_resume.launch.pending.json'
with staged.open('x', newline='\n') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
staged.rename(receipt)
print(json.dumps(record))
