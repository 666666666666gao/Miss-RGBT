"""Launch the frozen absent-search K/V diagnostic on an idle server28 GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--witness", action="store_true")
parser.add_argument("--run-id", default="v2")
args = parser.parse_args()

root = Path("/home/gaob/Miss-RGBT")
runtime = root / "runtime/task_utility_20261018"
script = root / "scripts/diagnose_absent_search_kv.py"
stem = "server28_absent_search_kv_20261102_" + args.run_id
output = root / "reports" / stem
receipt = root / "reports" / (stem + ".launch.json")
log_path = root / "reports" / (stem + (".witness.log" if args.witness else ".log"))
witness_receipt = root / "reports" / (stem + ".witness.launch.json")
if not args.witness:
    assert not output.exists()
else:
    assert not witness_receipt.exists()
assert not receipt.exists()
assert not log_path.exists()

gpu_rows = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,temperature.gpu,power.limit",
    "--format=csv,noheader",
], universal_newlines=True).strip().splitlines()
gpu2 = [row.split(",") for row in gpu_rows if row.split(",")[0].strip() == "2"][0]
memory_mib = float(gpu2[1].strip().split()[0])
temperature_c = float(gpu2[2].strip().split()[0])
power_limit_w = float(gpu2[3].strip().split()[0])
assert memory_mib < 500
assert temperature_c < 85
assert power_limit_w == 250

environment = dict(os.environ, CUDA_VISIBLE_DEVICES="2",
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    PYTHONPATH=str(runtime))
command = [sys.executable, "-u", str(script), "--output", str(output)]
if args.witness:
    command = [sys.executable, "-u", str(script), "--witness-only"]
with log_path.open("xb") as log:
    process = subprocess.Popen(command, cwd=str(runtime), env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
source_hash = hashlib.sha256(script.read_bytes()).hexdigest()
launcher_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
launch = dict(pid=process.pid, gpu=2, memory_used_mib=memory_mib,
    temperature_c=temperature_c, power_limit_w=power_limit_w,
    command=command, cwd=str(runtime), output=str(output), log=str(log_path),
    script_sha256=source_hash, launcher_sha256=launcher_hash,
    launched_utc=datetime.datetime.utcnow().isoformat() + "Z")
if args.witness:
    launch["output"] = None
    receipt = witness_receipt
with receipt.open("x") as stream:
    json.dump(launch, stream, indent=2)
    stream.write("\n")
print(json.dumps(launch))
