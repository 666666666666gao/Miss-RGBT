"""Launch four independent checkpoint continuations after a verified reboot."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
output = root / "reports/server28-recovery-20261031"
audit = json.loads((root / "reports/server28-reboot-checkpoint-audit-20261031.json").read_text())
assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == audit["boot_id"]
memory = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    universal_newlines=True)
assert len(memory.splitlines()) == 4
assert all(int(row.split(",")[1]) < 500 for row in memory.splitlines())
if output.exists():
    assert not list(output.iterdir())
else:
    output.mkdir()
records = {}
for gpu, label in enumerate(["a_utility_full20_seed2027", "a_random_full20_seed2027",
                             "batch10_utility_seed2027", "batch10_random_seed2027"]):
    command = [sys.executable, "-u", str(root / "recovery_campaign28.py"),
               "--label", label, "--gpu", str(gpu)]
    with (output / (label + "_controller.log")).open("xb") as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                            OPENBLAS_NUM_THREADS="1", WANDB_MODE="disabled"))
    records[label] = dict(controller_pid=process.pid, physical_gpu=gpu, command=command)
receipt = dict(started_utc=datetime.datetime.utcnow().isoformat() + "Z",
               boot_id=audit["boot_id"], memory_before_launch=memory, campaigns=records)
with (output / "launch.json").open("x") as stream:
    json.dump(receipt, stream, indent=2)
    stream.write("\n")
print(json.dumps(receipt, indent=2), flush=True)
