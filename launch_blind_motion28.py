"""Launch exactly one GPU0 development evaluator alongside original training."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
with (root / "reports/server28-blind-motion-closed-loop-20261031.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "evaluate_blind_motion28.py")],
        cwd=str(root), env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=process.pid, physical_gpu=0, evaluator_workers=1)))
