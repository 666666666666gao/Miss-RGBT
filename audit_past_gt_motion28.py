"""Separate measured-velocity error from long-gap model mismatch offline."""
import json
from pathlib import Path

import numpy as np

from analyze_e0_availability28 import frame_metrics
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
FOLDER = ROOT / "reports/server28-blind-motion-next16-20261031"


def read(path):
    return json.loads(path.read_text())


def center(box):
    return box[:2] + box[2:] / 2


def main():
    parent = read(FOLDER / "report.json")
    assert sha(FOLDER / "report.json") == read(FOLDER / "completed.json")["report_sha256"]
    estimates = read(FOLDER / "motion_estimate_audit.json")
    assert estimates["report_sha256"] == sha(FOLDER / "report.json")
    sequences = {}
    for gpu in [0, 1]:
        folder = FOLDER / ("gpu%d" % gpu)
        worker = read(folder / "report.json")
        assert sha(folder / "report.json") == parent["worker_receipts"][str(gpu)]["report_sha256"]
        for item in worker["launch"]["cohort"]:
            gt_path = Path(item["folder"]) / "init.txt"
            path = folder / ("velocity_" + item["name"] + "_alternating_outage.txt")
            assert sha(gt_path) == item["gt_sha256"]
            assert sha(path) == worker["records"]["velocity"][item["name"]]["alternating_outage"]["bbox_sha256"]
            sequences[item["name"]] = (
                np.loadtxt(str(gt_path), delimiter=",", ndmin=2),
                np.loadtxt(str(path), ndmin=2))
    rows, excluded = [], []
    for record in estimates["records"]:
        name = record["sequence"]
        gt, boxes = sequences[name]
        before, last = record["observed_frames"]
        frames = np.arange(record["start"], record["end_exclusive"])
        frames = frames[(gt[frames, 2:] > 0).all(1)]
        if not (gt[[before, last], 2:] > 0).all() or not len(frames):
            excluded.append(dict(sequence=name, start=record["start"],
                                 reason="Invalid past GT geometry or no valid blind GT"))
            continue
        measured_velocity = (center(boxes[last]) - center(boxes[before])) / (last - before)
        privileged_velocity = (center(gt[last]) - center(gt[before])) / (last - before)
        hold = np.tile(boxes[last], (len(frames), 1))
        elapsed = (frames - last)[:, None]
        measured = hold.copy()
        measured[:, :2] += elapsed * measured_velocity
        assert np.allclose(measured, boxes[frames], rtol=0, atol=1e-9)
        privileged = hold.copy()
        privileged[:, :2] += elapsed * privileged_velocity
        metrics = {}
        for arm, predicted in [("same_last_hold", hold), ("measured_velocity", measured),
                               ("past_gt_velocity", privileged)]:
            metrics[arm] = {key: float(value.mean())
                            for key, value in frame_metrics(predicted, gt[frames]).items()}
        rows.append(dict(sequence=name, start=record["start"], frames=len(frames), metrics=metrics))
    names = sorted({row["sequence"] for row in rows})
    per_video, summary = {}, {}
    for name in names:
        selected = [row for row in rows if row["sequence"] == name]
        total = sum(row["frames"] for row in selected)
        per_video[name] = {
            arm: {metric: sum(row["metrics"][arm][metric] * row["frames"] for row in selected) / total
                  for metric in ["PR20", "SR"]}
            for arm in ["same_last_hold", "measured_velocity", "past_gt_velocity"]}
    for arm in ["same_last_hold", "measured_velocity", "past_gt_velocity"]:
        summary[arm] = {metric: float(np.mean([per_video[name][arm][metric] for name in names]))
                        for metric in ["PR20", "SR"]}
    comparison = {}
    for reference in ["same_last_hold", "measured_velocity"]:
        deltas = [per_video[name]["past_gt_velocity"]["SR"] - per_video[name][reference]["SR"]
                  for name in names]
        comparison[reference] = dict(
            past_gt_velocity_SR_delta=float(np.mean(deltas)),
            wins=sum(value > 0 for value in deltas), ties=sum(value == 0 for value in deltas),
            losses=sum(value < 0 for value in deltas))
    result = dict(
        parent_report_sha256=sha(FOLDER / "report.json"),
        estimate_audit_sha256=sha(FOLDER / "motion_estimate_audit.json"),
        source_sha256=sha(Path(__file__)), intervals=len(rows), excluded=excluded,
        videos=len(names), frames=sum(row["frames"] for row in rows),
        blind_video_macro=summary, privileged_contrasts=comparison,
        per_video=per_video, per_interval=rows,
        scope="Offline privileged past-GT velocity anchored to the same saved CV last box. "
              "No future GT enters prediction. Future valid GT only scores the prediction. "
              "This does not recompute closed-loop recovery, provide a performance bound, "
              "select parameters or define a deployable method.")
    with (FOLDER / "past_gt_motion_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in [
        "intervals", "excluded", "videos", "frames", "blind_video_macro", "privileged_contrasts"]},
        indent=2))


if __name__ == "__main__":
    main()
