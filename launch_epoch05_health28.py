"""Launch two single-worker validation queues in remaining GPU0/1 memory."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
with (root / "reports/server28-epoch05-tracking-health-20261031.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "evaluate_epoch05_health28.py")],
        cwd=str(root), env=dict(os.environ, OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(controller_pid=process.pid, physical_gpus=[0, 1],
                      workers_per_gpu=1)))
