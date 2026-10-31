"""Launch the registered scale component control beside existing trainers."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path("/home/gaob/Miss-RGBT")
receipt = root / "reports/server28-scale-only-launch-20261031.json"
assert not receipt.exists()
assert not (root / "reports/server28-scale-only4-20261031").exists()
command = [sys.executable, "-u", str(root / "run_scale_only28.py"), "--mode", "controller"]
with (root / "reports/server28-scale-only4-20261031.log").open("xb") as log:
    process = subprocess.Popen(command, cwd=str(root), stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True,
                               env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                        OPENBLAS_NUM_THREADS="1"))
result = dict(pid=process.pid, command=command,
              launched_utc=datetime.datetime.utcnow().isoformat() + "Z")
with receipt.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))
