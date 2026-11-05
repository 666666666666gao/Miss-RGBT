"""Launch one registered frozen-E0 rollout diagnostic on an idle GPU2."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
SCRIPT = ROOT / "scripts/diagnose_burst_mask_rollout.py"
STEM = "server28_burst_mask_rollout_20261102"
OUTPUT = ROOT / "runtime/task_utility_20261018/reports" / STEM
RECEIPT = ROOT / "runtime/task_utility_20261018/reports" / (STEM + ".launch.json")
LOG = ROOT / "runtime/task_utility_20261018/reports" / (STEM + ".log")

assert not OUTPUT.exists()
assert not RECEIPT.exists()
assert not LOG.exists()

rows = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.used,temperature.gpu,power.limit",
    "--format=csv,noheader",
], universal_newlines=True).strip().splitlines()
gpu = [row.split(",") for row in rows if row.split(",")[0].strip() == "2"][0]
memory_mib = float(gpu[1].strip().split()[0])
temperature_c = float(gpu[2].strip().split()[0])
power_limit_w = float(gpu[3].strip().split()[0])
assert memory_mib < 500
assert temperature_c < 85
assert power_limit_w == 250

environment = dict(os.environ, CUDA_VISIBLE_DEVICES="2",
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    PYTHONPATH=str(RUNTIME))
command = [sys.executable, "-u", str(SCRIPT), "--output", str(OUTPUT)]
with LOG.open("xb") as log:
    process = subprocess.Popen(command, cwd=str(RUNTIME), env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(pid=process.pid, gpu=2, memory_used_mib=memory_mib,
    temperature_c=temperature_c, power_limit_w=power_limit_w,
    command=command, cwd=str(RUNTIME), output=str(OUTPUT), log=str(LOG),
    script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    launched_utc=datetime.datetime.utcnow().isoformat() + "Z")
with RECEIPT.open("x") as stream:
    json.dump(receipt, stream, indent=2)
    stream.write("\n")
print(json.dumps(receipt))
