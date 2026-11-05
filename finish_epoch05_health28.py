"""Run paired verification once the existing tracking controller has exited."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_val97_recovery28 import sha

ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-epoch05-tracking-health-20261031"
ANALYZER = ROOT / "analyze_epoch05_health28.py"


def main():
    launch = json.loads((OUTPUT / "launch.json").read_text())
    controller = launch["pid"]
    process = Path("/proc") / str(controller)
    assert process.exists(), "Start this queue while the tracking controller is live"
    assert b"evaluate_epoch05_health28.py" in (process / "cmdline").read_bytes()
    analyzer_sha = sha(ANALYZER)
    queue = dict(pid=os.getpid(), tracking_controller_pid=controller,
                 started_utc=datetime.datetime.utcnow().isoformat() + "Z",
                 poll_seconds=300, analyzer_sha256=analyzer_sha,
                 purpose="One-shot postprocessing, no evaluation restart or checkpoint selection")
    with (OUTPUT / "analysis_queue.json").open("x") as stream:
        json.dump(queue, stream, indent=2)
        stream.write("\n")
    print("WAITING " + json.dumps(queue), flush=True)
    while process.exists():
        time.sleep(300)
    assert (OUTPUT / "completed.json").is_file(), "Tracking exited without six-cell completion"
    assert sha(ANALYZER) == analyzer_sha, "Analyzer changed after queue launch"
    subprocess.run([sys.executable, str(ANALYZER), "--report-dir", str(OUTPUT)],
                   cwd=str(ROOT), check=True)
    result = OUTPUT / "paired_analysis.json"
    receipt = dict(queue=queue, paired_analysis_sha256=sha(result),
                   ended_utc=datetime.datetime.utcnow().isoformat() + "Z")
    with (OUTPUT / "analysis_completed.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print("ANALYSIS_COMPLETE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
