"""Finish the registered long-gap audit before deploying the short-gap control."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
PREVIOUS = ROOT / "reports/server28-blind-motion-next16-20261031"
STAGED = ROOT / "development/shortgap_staged_20261031"
FILES = ["evaluate_blind_motion28.py", "run_blind_motion_cohort28.py",
         "launch_blind_motion_shortgap28.py"]


def read(path):
    return json.loads(path.read_text())


def main():
    launch = read(PREVIOUS / "launch.json")
    staged_hashes = {name: sha(STAGED / name) for name in FILES}
    print("WAIT " + json.dumps(dict(
        previous_controller_pid=launch["pid"], staged_sha256=staged_hashes,
        poll_seconds=300)), flush=True)
    while subprocess.call(["kill", "-0", str(launch["pid"])],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        time.sleep(300)
    completed = read(PREVIOUS / "completed.json")
    report = read(PREVIOUS / "report.json")
    assert sha(PREVIOUS / "report.json") == completed["report_sha256"]
    assert completed["trajectories_verified"] == 64 and completed["sequences"] == 16
    for gpu in [0, 1]:
        folder = PREVIOUS / ("gpu%d" % gpu)
        receipt = report["worker_receipts"][str(gpu)]
        assert sha(folder / "report.json") == receipt["report_sha256"]
        assert sha(folder / "analysis.json") == receipt["analysis_sha256"]
        assert read(folder / "analysis.json")["trajectories_verified"] == 32
    archive = PREVIOUS / "verified_source"
    archive.mkdir()
    for name in ["evaluate_blind_motion28.py", "run_blind_motion_cohort28.py",
                 "analyze_blind_motion28.py"]:
        source = ROOT / name
        assert sha(source) == launch["source_sha256"][str(source)]
        shutil.copyfile(str(source), str(archive / name))
    for name in FILES:
        assert sha(STAGED / name) == staged_hashes[name]
        shutil.copyfile(str(STAGED / name), str(ROOT / name))
    print("DEPLOYED " + json.dumps(staged_hashes), flush=True)
    subprocess.check_call([sys.executable, str(ROOT / "launch_blind_motion_shortgap28.py")])


if __name__ == "__main__":
    main()
