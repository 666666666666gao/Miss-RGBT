"""Run and verify fixed motion controls on the next16 full development videos."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from analyze_epoch05_health28 import compare
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-blind-motion-next16-20261031"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    OUTPUT.mkdir(exist_ok=False)
    memory = subprocess.check_output([
        "nvidia-smi", "--id=0,1", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    rows = [line.split(",") for line in memory.splitlines()]
    assert len(rows) == 2 and all(int(row[1]) >= 3000 for row in rows), memory
    manifest_path = ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json"
    manifest = json.loads(manifest_path.read_text())
    data_root = Path("/data/wangwj/dataset/LasHeR/trainingset")
    cohort = []
    for index in range(4, 20):
        name = manifest["development_sequences"][index]
        folder = data_root / name
        gt_path = folder / "init.txt"
        gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
        assert np.isfinite(gt).all() and (gt[0, 2:] > 0).all()
        cohort.append(dict(name=name, folder=str(folder), video_frames=len(gt),
                           evaluated_frames=len(gt), gt_sha256=sha(gt_path),
                           development_index=index))
    assert len(set(item["name"] for item in cohort)) == 16
    assert sum(item["video_frames"] for item in cohort) == 7776
    parts, loads = {0: [], 1: []}, {0: 0, 1: 0}
    for item in sorted(cohort, key=lambda row: (-row["video_frames"], row["development_index"])):
        eligible = [gpu for gpu in parts if len(parts[gpu]) < 8]
        gpu = min(eligible, key=lambda value: (loads[value], value))
        parts[gpu].append(item)
        loads[gpu] += item["video_frames"]
    for gpu in parts:
        parts[gpu].sort(key=lambda item: item["development_index"])
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        scope="Next16 full train882 development videos; unchanged ordinary CV, no tuning or official test",
        input_manifest_sha256=sha(manifest_path), development_indices=list(range(4, 20)),
        sequences=16, frames_per_arm_condition=7776, total_tracking_frames=31104,
        parts=parts, part_frames=loads, workers_per_gpu=1, memory_free_at_launch=memory,
        assignment="Longest-video-first frame balancing, capacity8 videos per GPU; "
                   "hold and velocity for a video always share one GPU",
        source_sha256={str(ROOT / name): sha(ROOT / name) for name in [
            "run_blind_motion_cohort28.py", "evaluate_blind_motion28.py",
            "analyze_blind_motion28.py"]}, children={}, poll_seconds=300)
    processes = {}
    for gpu, selected in parts.items():
        spec_path = OUTPUT / ("gpu%d_cohort.json" % gpu)
        save(spec_path, dict(
            input_manifest_sha256=sha(manifest_path),
            development_indices=[item["development_index"] for item in selected], cohort=selected))
        destination = OUTPUT / ("gpu%d" % gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (OUTPUT / ("gpu%d.log" % gpu)).open("xb") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "evaluate_blind_motion28.py"),
                 "--gpu", str(gpu), "--cohort-spec", str(spec_path), "--output", str(destination)],
                cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        processes[gpu] = process
        launch["children"][str(gpu)] = process.pid
    save(OUTPUT / "launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    while any(process.poll() is None for process in processes.values()):
        time.sleep(300)
    for gpu, process in processes.items():
        assert process.returncode == 0, (gpu, process.returncode)
        subprocess.check_call(
            [sys.executable, str(ROOT / "analyze_blind_motion28.py"),
             "--report-dir", str(OUTPUT / ("gpu%d" % gpu))])
    combined, groups, receipts = {}, {}, {}
    for gpu in parts:
        folder = OUTPUT / ("gpu%d" % gpu)
        worker = json.loads((folder / "report.json").read_text())
        analysis = json.loads((folder / "analysis.json").read_text())
        assert analysis["report_sha256"] == sha(folder / "report.json")
        receipts[str(gpu)] = dict(report_sha256=sha(folder / "report.json"),
                                  analysis_sha256=sha(folder / "analysis.json"))
        for arm, videos in worker["records"].items():
            combined.setdefault(arm, {})
            assert not set(combined[arm]) & set(videos)
            combined[arm].update(videos)
        for group, values in analysis["per_sequence"].items():
            groups.setdefault(group, []).extend(values)
    names = sorted(item["name"] for item in cohort)
    assert all(set(records) == set(names) for records in combined.values())
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    contrasts, aggregate = {}, {}
    for condition in ["complete", "alternating_outage"]:
        values = {arm: {name: record[condition] for name, record in videos.items()}
                  for arm, videos in combined.items()}
        contrasts[condition] = compare(values["velocity"], values["hold"], names, indices)
        aggregate[condition] = {
            arm: {metric: float(np.mean([values[arm][name][metric] for name in names]))
                  for metric in ["PR20", "SR"]} for arm in values}
    group_summary = {}
    for group, values in groups.items():
        assert len(values) == len(set(row["sequence"] for row in values))
        group_summary[group] = dict(
            sequences=len(values), frames=sum(row["frames"] for row in values),
            delta={metric: float(np.mean([row["velocity_minus_hold"][metric] for row in values]))
                   for metric in ["PR20", "SR"]})
    report = dict(
        launch=launch, aggregate=aggregate, paired_contrasts=contrasts,
        groups=group_summary, per_sequence_groups=groups, worker_receipts=receipts,
        trajectories_verified=64, source_sha256=sha(Path(__file__)),
        bootstrap_draws=10000, bootstrap_seed=2027,
        uncertainty_scope="Paired video resampling on16 registered development videos; "
                          "not training-seed uncertainty, official results or a novel module")
    save(OUTPUT / "report.json", report)
    save(OUTPUT / "completed.json", dict(
        report_sha256=sha(OUTPUT / "report.json"),
        ended_utc=datetime.datetime.utcnow().isoformat() + "Z",
        trajectories_verified=64, sequences=16))
    print("COMPLETE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
