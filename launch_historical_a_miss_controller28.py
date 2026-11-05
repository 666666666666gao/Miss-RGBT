"""Detach the verified historical-A Miss evaluation controller."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
log_path = root / "reports/server28-historical-a-miss-control-v2.log"
with log_path.open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "run_historical_a_miss_after_smoke28.py")],
        cwd=str(root), env=dict(os.environ), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=process.pid, log=str(log_path))))
