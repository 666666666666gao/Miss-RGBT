"""Register read-only independent verification after each campaign completes."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_val97_recovery28 import sha


root = Path("/home/gaob/Miss-RGBT")
folder = root / "reports/server28-recovery-20261031"
labels = ["a_utility_full20_seed2027", "a_random_full20_seed2027",
          "batch10_utility_seed2027", "batch10_random_seed2027"]
for label in labels:
    launch = json.loads((folder / label / "launch.json").read_text())
    command = (Path("/proc") / str(launch["controller_pid"]) / "cmdline").read_bytes()
    assert b"/home/gaob/Miss-RGBT/recovery_campaign28.py" in command
    assert label.encode() in command
with (folder / "independent_verification_queue.json").open("x") as receipt:
    jobs = {}
    for label in labels:
        with (folder / (label + "_verification.log")).open("xb") as log:
            command = [sys.executable, "-u", str(root / "verify_recovery_campaign28.py"),
                       "--label", label, "--wait"]
            process = subprocess.Popen(
                command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                         MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
            jobs[label] = dict(pid=process.pid, command=command)
    result = dict(
        registered_utc=datetime.datetime.utcnow().isoformat() + "Z", jobs=jobs,
        poll_seconds=300, read_only_verification=True, deletes_weights=False,
        source_sha256={str(path): sha(path) for path in [
            Path(__file__), root / "verify_recovery_campaign28.py",
            root / "analyze_epoch05_health28.py", root / "evaluate_val97_recovery28.py"]})
    json.dump(result, receipt, indent=2)
    receipt.write("\n")
print(json.dumps(result, indent=2), flush=True)
