"""Wait for the registered CPU audit, then verify its completed report."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_val97_recovery28 import sha


root = Path("/home/gaob/Miss-RGBT")
output = root / "reports/server28-candidate-repair-audit-20261031"
launch = json.loads((output / "launch.json").read_text())
analyzer = root / "analyze_candidate_repair28.py"
expected = sha(analyzer)
with (output / "analysis_queue.json").open("x") as stream:
    json.dump(dict(pid=os.getpid(), producer_pid=launch["pid"],
                   producer_launch_sha256=sha(output / "launch.json"),
                   analyzer_sha256=expected, controller_sha256=sha(__file__),
                   poll_seconds=300), stream, indent=2)
    stream.write("\n")
while (Path("/proc") / str(launch["pid"])).exists():
    command = (Path("/proc") / str(launch["pid"]) / "cmdline").read_bytes()
    assert b"/home/gaob/Miss-RGBT/audit_candidate_repair28.py" in command
    time.sleep(300)
assert sha(analyzer) == expected
assert (output / "completed.json").is_file()
subprocess.check_call([sys.executable, str(analyzer)], cwd=str(root))
with (output / "analysis_completed.json").open("x") as stream:
    json.dump(dict(analysis_sha256=sha(output / "analysis.json"),
                   analyzer_sha256=expected), stream, indent=2)
    stream.write("\n")
print("ANALYSIS_COMPLETE", flush=True)
