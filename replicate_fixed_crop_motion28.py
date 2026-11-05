"""Wait for registered fresh crop references and repeat the fixed CPU probe."""
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
SOURCE = ROOT / "reports/server28-motion-replication16-adjacent-20261031"
OUTPUT = ROOT / "reports/server28-fixed-crop-motion-replication16-20261031"


def main():
    launch = json.loads((SOURCE / "launch.json").read_text())
    registration = ROOT / "reports/server28-motion-replication16-cohort-20261031.json"
    assert launch["registration_sha256"] == sha(registration)
    assert launch["development_indices"] == list(range(20, 36))
    commands = [
        ["probe_fixed_crop_motion28.py", "--source", str(SOURCE), "--output", str(OUTPUT)],
        ["analyze_fixed_crop_motion28.py", "--output", str(OUTPUT)],
        ["analyze_motion_state_factorial28.py", "--source", str(OUTPUT)],
    ]
    sources = {command[0]: sha(ROOT / command[0]) for command in commands}
    print("WAIT " + json.dumps(dict(pid=launch["pid"], poll_seconds=300,
                                    source_sha256=sources)), flush=True)
    while subprocess.call(["kill", "-0", str(launch["pid"])],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        time.sleep(300)
    completion = json.loads((SOURCE / "completed.json").read_text())
    assert completion["report_sha256"] == sha(SOURCE / "report.json")
    for command in commands:
        assert sha(ROOT / command[0]) == sources[command[0]]
        subprocess.check_call([sys.executable, "-u", str(ROOT / command[0])] + command[1:])
    print("REPLICATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
