import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
with (root / "reports/server28-global-proposals-20261030.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "audit_global_proposals28.py")],
        cwd=str(root), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=process.pid, device="cpu", cpu_threads=4)))
