"""Register four new videos, wait for active motion study, run offset controls."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-state-offset-control4-20261031"
PREVIOUS = ROOT / "reports/server28-motion-replication16-availability-20261031"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    global OUTPUT, PREVIOUS
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--start", type=int, default=36)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--previous", type=Path, default=PREVIOUS)
    args = parser.parse_args()
    OUTPUT, PREVIOUS = args.output, args.previous
    entry_path = ROOT / "reports/server28-state-offset-entry-20261031.json"
    entry = json.loads(entry_path.read_text())
    assert entry["sham_bitwise_equal"] and entry["complete_bitwise_equal"]
    assert entry["baseline_state_unchanged"]
    for path, expected in entry["source_sha256"].items():
        assert sha(path) == expected, path
    OUTPUT.mkdir(exist_ok=False)
    manifest_path = ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json"
    manifest = json.loads(manifest_path.read_text())
    cohort = []
    indices = list(range(args.start, args.start + args.count))
    for index in indices:
        name = manifest["development_sequences"][index]
        folder = Path("/data/wangwj/dataset/LasHeR/trainingset") / name
        gt_path = folder / "init.txt"
        gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
        assert np.isfinite(gt).all() and (gt[0, 2:] > 0).all()
        assert len(list((folder / "visible").glob("*.jpg"))) == len(gt)
        assert len(list((folder / "infrared").glob("*.jpg"))) == len(gt)
        cohort.append(dict(name=name, folder=str(folder), development_index=index,
                           video_frames=len(gt), gt_sha256=sha(gt_path)))
    assert len(cohort) == args.count
    assert not {row["name"] for row in cohort} & (
        set(manifest["train_sequences"]) | set(manifest["heldout_sequences"]) |
        set(manifest["development_sequences"][:args.start]))
    parts, loads = {0: [], 1: []}, {0: 0, 1: 0}
    for row in sorted(cohort, key=lambda item: -item["video_frames"]):
        gpu = min(loads, key=lambda value: (loads[value], value))
        parts[gpu].append(row)
        loads[gpu] += row["video_frames"]
    for gpu in parts:
        parts[gpu].sort(key=lambda row: row["development_index"])
    sources = {str(ROOT / name): sha(ROOT / name) for name in [
        "run_state_offset_control28.py", "evaluate_state_offset28.py",
        "state_offset_tracker28.py", "analyze_state_offset28.py"]}
    previous_pid = args.wait_pid
    launch = dict(
        pid=os.getpid(), registered_utc=datetime.datetime.utcnow().isoformat() + "Z",
        input_manifest_sha256=sha(manifest_path), cohort=cohort, parts=parts, loads=loads,
        entry_witness_sha256=sha(entry_path),
        development_indices=indices, source_sha256=sources,
        previous_pid=previous_pid, poll_seconds=300, children={},
        scope="Sequential fresh development videos; fixed parameter-free correction "
              "and equal-compute sham, no outcomes used for cohort selection or tuning.")
    save(OUTPUT / "registration.json", launch)
    launch["registration_sha256"] = sha(OUTPUT / "registration.json")
    save(OUTPUT / "launch.json", launch)
    print("WAIT " + json.dumps(launch), flush=True)
    while subprocess.call(["kill", "-0", str(previous_pid)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        time.sleep(300)
    completion = json.loads((PREVIOUS / "completed.json").read_text())
    assert completion["report_sha256"] == sha(PREVIOUS / "report.json")
    if args.start == 36:
        comparison = json.loads((PREVIOUS / "comparison.json").read_text())
        assert comparison["same_cohort_gpu_checkpoint"] and all(comparison["controls_equal"].values())
    else:
        report = json.loads((PREVIOUS / "report.json").read_text())
        assert report["sham_and_complete_bitwise_equal"] and report["past_only_offsets_verified"]
    for path, expected in sources.items():
        assert sha(path) == expected, path
    memory = subprocess.check_output([
        "nvidia-smi", "--id=0,1", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    assert all(int(line.split(",")[1]) >= 3000 for line in memory.splitlines())
    processes = {}
    for gpu, selected in parts.items():
        spec = OUTPUT / ("gpu%d_spec.json" % gpu)
        save(spec, dict(cohort=selected, registration_sha256=sha(OUTPUT / "registration.json")))
        with (OUTPUT / ("gpu%d.log" % gpu)).open("xb") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "evaluate_state_offset28.py"),
                 "--gpu", str(gpu), "--spec", str(spec),
                 "--output", str(OUTPUT / ("gpu%d" % gpu))],
                cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                       OPENBLAS_NUM_THREADS="1"),
                stdout=log, stderr=subprocess.STDOUT)
        processes[gpu] = process
        launch["children"][str(gpu)] = process.pid
    launch["memory_free_at_launch"] = memory
    launch["started_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    save(OUTPUT / "launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    while any(process.poll() is None for process in processes.values()):
        time.sleep(300)
    assert all(process.returncode == 0 for process in processes.values())
    subprocess.check_call([sys.executable, str(ROOT / "analyze_state_offset28.py"),
                           "--output", str(OUTPUT)])
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
