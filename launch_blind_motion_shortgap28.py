"""Launch the registered one-frame-gap control after the long-gap audit."""
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_val97_recovery28 import sha


root = Path("/home/gaob/Miss-RGBT")
previous = root / "reports/server28-blind-motion-next16-20261031"
completion = json.loads((previous / "completed.json").read_text())
assert sha(previous / "report.json") == completion["report_sha256"]
assert completion["trajectories_verified"] == 64 and completion["sequences"] == 16
with (root / "reports/server28-blind-motion-shortgap16-20261031.log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "run_blind_motion_cohort28.py"), "--frame-block", "1"],
        cwd=str(root), env=dict(os.environ, OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(controller_pid=process.pid, gpus=[0, 1],
                      evaluators_per_gpu=1, frame_block=1)))
