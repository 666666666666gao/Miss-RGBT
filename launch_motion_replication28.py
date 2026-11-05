"""Run fixed adjacent and availability histories on the registered fresh cohort."""
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")


def main():
    receipt = ROOT / "reports/server28-motion-replication16-launch-20261031.json"
    assert not receipt.exists()
    runs = {}
    for history in ["adjacent", "availability"]:
        subprocess.check_call([
            sys.executable, "-u", str(ROOT / "run_blind_motion_cohort28.py"),
            "--frame-block", "1", "--velocity-history", history, "--replication"])
        runs[history] = ROOT / ("reports/server28-motion-replication16-" +
                               history + "-20261031")
    subprocess.check_call([
        sys.executable, str(ROOT / "analyze_state_motion_control28.py"),
        "--reference", str(runs["adjacent"]), "--candidate", str(runs["availability"]),
        "--replication"])
    result = dict(pid=os.getpid(), completed=True,
                  comparison_sha256=sha(runs["availability"] / "comparison.json"),
                  registration_sha256=sha(
                      ROOT / "reports/server28-motion-replication16-cohort-20261031.json"))
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
