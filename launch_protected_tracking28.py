"""Launch the fixed paired CPU tracking control; wait for the existing pilot."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
with (root / "reports/server28-protected-tracking-development-20261030.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "evaluate_protected_development28.py")],
        cwd=str(root), env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
print(json.dumps(dict(pid=process.pid, device="cpu", cpu_threads=4)))
