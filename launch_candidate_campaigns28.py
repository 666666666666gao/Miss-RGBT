"""Launch the larger-batch parameter pair on physical GPU2/3."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
output = root / "reports/server28-candidates-20261030"
output.mkdir(exist_ok=False)
receipts = {}
for selection, gpu in [("utility", "2"), ("random", "3")]:
    command = [sys.executable, "-u", str(root / "candidate_campaign28.py"),
               "--selection", selection, "--gpu", gpu]
    with (output / (selection + "_controller.log")).open("wb") as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
    receipts[selection] = dict(controller_pid=process.pid, physical_gpu=gpu, command=command)
(output / "launch.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
