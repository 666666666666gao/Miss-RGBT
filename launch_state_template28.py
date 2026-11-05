"""Launch the isolated state-template control in the existing environment."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
receipt = root / "reports/server28-state-template-launch-20261031.json"
assert not receipt.exists()
assert not (root / "reports/server28-state-template8-20261031").exists()
command = [sys.executable, "-u", str(root / "run_state_template28.py"), "--mode", "controller"]
with (root / "reports/server28-state-template8-20261031.log").open("xb") as log:
    process = subprocess.Popen(command, cwd=str(root), stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True,
                               env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                        OPENBLAS_NUM_THREADS="1"))
with receipt.open("x") as stream:
    json.dump(dict(pid=process.pid, command=command,
                   launched_utc=datetime.datetime.utcnow().isoformat() + "Z"), stream, indent=2)
    stream.write("\n")
print(receipt.read_text())
