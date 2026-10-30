"""Launch the bounded CPU pilot without occupying a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
with (root / "reports/server28-protected-sixepoch-20261030.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "probe_protected_sixepoch28.py")],
        cwd=str(root), env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=process.pid, device="cpu", cpu_threads=4)))
