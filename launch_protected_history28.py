"""Launch the registered CPU-only dynamic-template diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
with (root / "reports/server28-protected-history-20261031.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "audit_protected_history28.py")],
        cwd=str(root), env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=process.pid, device="cpu", cpu_threads=4)))
