"""Offline center-velocity controls on saved train882 development outages."""
import datetime
import json
import os
from pathlib import Path

import numpy as np

from analyze_e0_availability28 import frame_metrics
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
TRACKS = ROOT / "reports/server28-protected-tracking-development-20261030"
OUTPUT = ROOT / "reports/server28-blind-motion-20261031"
ARMS = ["native_hold", "tracker_velocity", "gt_velocity_native_anchor",
        "gt_anchor_hold", "gt_anchor_velocity"]


def center(box):
    return box[:2] + box[2:] / 2


def main():
    OUTPUT.mkdir(exist_ok=False)
    source = json.loads((TRACKS / "report.json").read_text())
    launch = source["launch"]
    result = dict(
        observed_utc=datetime.datetime.utcnow().isoformat() + "Z", pid=os.getpid(),
        scope="Offline saved train882 dev gap controls; not closed-loop tracking or official scores",
        source_sha256=sha(Path(__file__)), tracking_report_sha256=sha(TRACKS / "report.json"),
        arms=ARMS, records={}, aggregate={})
    for label in launch["models"]:
        records = {}
        for item in launch["cohort"]:
            name = item["name"]
            prefix = TRACKS / (label + "_" + name + "_alternating_outage")
            receipt = source["records"][label][name]["alternating_outage"]
            assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
            assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
            gt_path = Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)[:item["evaluated_frames"]]
            boxes = np.loadtxt(str(prefix) + ".txt", ndmin=2)
            trace = json.loads(Path(str(prefix) + ".json").read_text())
            assert gt.shape == boxes.shape == (item["evaluated_frames"], 4)
            assert len(trace) == len(boxes) - 1
            assert [row["frame"] for row in trace] == list(range(1, len(boxes)))
            missing = np.zeros(len(boxes), dtype=bool)
            for row in trace:
                missing[row["frame"]] = not any(row["available"])
            assert int(missing.sum()) == receipt["holds"]
            if not missing.any():
                records[name] = dict(blind_frames=0, gaps=[], metrics={})
                continue
            predictions = {arm: boxes.copy() for arm in ARMS}
            starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
            ends = np.flatnonzero(missing & ~np.r_[missing[1:], False]) + 1
            gaps = []
            for start, end in zip(starts, ends):
                assert start >= 2 and not missing[start - 1] and not missing[start - 2]
                assert (gt[start - 2:start, 2:] > 0).all()
                assert np.array_equal(boxes[start:end], np.repeat(boxes[start - 1:start], end - start, 0))
                velocity = center(boxes[start - 1]) - center(boxes[start - 2])
                gt_velocity = center(gt[start - 1]) - center(gt[start - 2])
                lead = np.arange(1, end - start + 1)[:, None]
                for arm, anchor, speed in [
                    ("tracker_velocity", boxes[start - 1], velocity),
                    ("gt_velocity_native_anchor", boxes[start - 1], gt_velocity),
                    ("gt_anchor_hold", gt[start - 1], np.zeros(2)),
                    ("gt_anchor_velocity", gt[start - 1], gt_velocity),
                ]:
                    predicted = predictions[arm][start:end]
                    predicted[:, 2:] = anchor[2:]
                    predicted[:, :2] = center(anchor) + lead * speed - anchor[2:] / 2
                gaps.append(dict(
                    start_frame=int(start), end_frame_exclusive=int(end), frames=int(end - start),
                    tracker_velocity=velocity.tolist(), gt_velocity=gt_velocity.tolist(),
                    tracker_anchor=boxes[start - 1].tolist(), gt_anchor=gt[start - 1].tolist()))
            valid = missing & (gt[:, 2:] > 0).all(1)
            metrics = {}
            for arm, predicted in predictions.items():
                assert np.array_equal(predicted[~missing], boxes[~missing])
                assert np.isfinite(predicted).all()
                metrics[arm] = {metric: float(values.mean())
                                for metric, values in frame_metrics(predicted[valid], gt[valid]).items()}
            records[name] = dict(blind_frames=int(missing.sum()), scored_frames=int(valid.sum()),
                                 gaps=gaps, metrics=metrics)
        active = [row for row in records.values() if row["blind_frames"]]
        result["records"][label] = records
        result["aggregate"][label] = dict(
            sequences_with_gaps=len(active), scored_frames=sum(row["scored_frames"] for row in active),
            sequence_macro={arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in active]))
                for metric in ["PR20", "SR"]} for arm in ARMS},
            frame_weighted={arm: {
                metric: float(np.average([row["metrics"][arm][metric] for row in active],
                                         weights=[row["scored_frames"] for row in active]))
                for metric in ["PR20", "SR"]} for arm in ARMS})
    result["interpretation"] = (
        "Only gap frames are changed/scored. Observed boxes are original saved trajectories, "
        "not re-run after a changed search position. Velocity is fixed from the immediately "
        "preceding two observed frames, with fixed pre-gap box size and no image clipping. "
        "GT arms use privileged past states, not future frames. They are diagnostics, not "
        "implementable predictors or performance upper bounds. Four reused development "
        "videos (three with gaps), overlapping frames across two models, no tuning or bootstrap.")
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUTPUT / "completed.json").write_text(json.dumps(dict(
        report_sha256=sha(OUTPUT / "report.json"), models=2,
        completed_utc=datetime.datetime.utcnow().isoformat() + "Z"), indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
