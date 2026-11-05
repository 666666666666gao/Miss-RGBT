"""Run one isolated CPU intervention audit without changing live campaigns."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
with (root / "reports/server28-candidate-repair-audit-20261031.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "audit_candidate_repair28.py")],
        cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                 MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"),
        stdin=subprocess.DEVNULL, start_new_session=True)
print(json.dumps(dict(pid=process.pid, device="cpu", cpu_threads=4)))
