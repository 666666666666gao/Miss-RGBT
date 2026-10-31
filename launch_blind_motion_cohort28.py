"""Launch the two-GPU development controller and persist its log."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
with (root / "reports/server28-blind-motion-next16-20261031.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "run_blind_motion_cohort28.py")],
        cwd=str(root), env=dict(os.environ, OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(controller_pid=process.pid, gpus=[0, 1], evaluators_per_gpu=1)))
