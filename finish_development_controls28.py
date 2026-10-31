"""Analyze the two already-running development controls after process exit."""
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
SHORT = ROOT / "reports/server28-blind-motion-shortgap16-20261031"
CPU = ROOT / "reports/server28-protected-mixed-pair-20261031"
RECEIPT = ROOT / "reports/server28-development-controls-analysis-20261031.json"


def read(path):
    return json.loads(path.read_text())


def main():
    tasks = {
        "shortgap": dict(
            pid=read(SHORT / "launch.json")["pid"], completion=SHORT / "completed.json",
            commands=[
                ["analyze_motion_gap_contrast28.py"],
                ["audit_motion_estimates28.py", "--report-dir", str(SHORT)]],
            artifacts=[ROOT / "reports/server28-motion-gap-contrast-20261031.json",
                       SHORT / "motion_estimate_audit.json"]),
        "template_pair": dict(
            pid=read(CPU / "launch.json")["pid"], completion=CPU / "completed.json",
            commands=[["analyze_protected_mixed_pair28.py"]],
            artifacts=[CPU / "analysis.json"]),
    }
    sources = {command[0]: sha(ROOT / command[0])
               for task in tasks.values() for command in task["commands"]}
    result = dict(source_sha256=sources, poll_seconds=300, completed={})
    print("WAIT " + json.dumps({name: task["pid"] for name, task in tasks.items()}), flush=True)
    while tasks:
        for name in list(tasks):
            task = tasks[name]
            alive = subprocess.call(["kill", "-0", str(task["pid"])],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
            if alive:
                continue
            assert task["completion"].is_file(), (name, "Process exited without completion")
            for command in task["commands"]:
                assert sha(ROOT / command[0]) == sources[command[0]]
                subprocess.check_call([sys.executable, str(ROOT / command[0])] + command[1:])
            result["completed"][name] = {
                str(path): sha(path) for path in task["artifacts"] + [task["completion"]]}
            RECEIPT.write_text(json.dumps(result, indent=2) + "\n")
            print("VERIFIED " + name, flush=True)
            del tasks[name]
        if tasks:
            time.sleep(300)
    print("ALL_ANALYSES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
