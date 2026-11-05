"""Launch one registered gap controller after the existing geometry campaign."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path("/home/gaob/Miss-RGBT")
previous_pid = 1256793
previous_command = (Path("/proc") / str(previous_pid) / "cmdline").read_bytes()
assert b"run_spatial_calibration28.py" in previous_command
receipt = root / "reports/server28-calibration-gap-launch-20261031.json"
assert not receipt.exists()
assert not (root / "reports/server28-calibration-gap4-20261031").exists()
command = [sys.executable, "-u", str(root / "run_calibration_gap28.py"),
           "--mode", "controller", "--wait-pid", str(previous_pid)]
with (root / "reports/server28-calibration-gap4-20261031.log").open("xb") as log:
    process = subprocess.Popen(command, cwd=str(root), stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True,
                               env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
result = dict(pid=process.pid, previous_pid=previous_pid, command=command,
              launched_utc=datetime.datetime.utcnow().isoformat() + "Z")
with receipt.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))
