"""Paired head decoding on the same fixed native-scale proposals."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/home/gaob/Miss-RGBT")
output = root / "reports/server28-native-grid-decode-20261030"
with output.with_suffix(".log").open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "audit_native_grid28.py"),
         "--output", str(output)],
        cwd=str(root), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=process.pid, device="cpu", cpu_threads=4,
                      output=str(output))))
