"""Locate long-gap CV regressions in verified development trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from analyze_e0_availability28 import frame_metrics
from evaluate_val97_recovery28 import geometry, sha


ROOT = Path("/home/gaob/Miss-RGBT")
FOLDER = ROOT / "reports/server28-blind-motion-next16-20261031"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="interval_analysis.json")
    args = parser.parse_args()
    config_path = ROOT / "runtime/task_utility_20261018/experiments/cadtrack/cadtrack.yaml"
    factor = yaml.safe_load(config_path.read_text())["TEST"]["SEARCH_FACTOR"]
    parent = read(FOLDER / "report.json")
    assert sha(FOLDER / "report.json") == read(FOLDER / "completed.json")["report_sha256"]
    intervals, videos = [], []
    for gpu in [0, 1]:
        folder = FOLDER / ("gpu%d" % gpu)
        worker = read(folder / "report.json")
        assert sha(folder / "report.json") == parent["worker_receipts"][str(gpu)]["report_sha256"]
        for item in worker["launch"]["cohort"]:
            name = item["name"]
            gt_path = Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            valid = (gt[:, 2:] > 0).all(1)
            boxes = {}
            for arm in ["hold", "velocity"]:
                path = folder / (arm + "_" + name + "_alternating_outage.txt")
                assert sha(path) == worker["records"][arm][name]["alternating_outage"]["bbox_sha256"]
                boxes[arm] = np.loadtxt(str(path), ndmin=2)
            trace_path = folder / ("velocity_" + name + "_alternating_outage.json")
            assert sha(trace_path) == worker["records"]["velocity"][name]["alternating_outage"]["trace_sha256"]
            trace = read(trace_path)
            blind = np.asarray([False] + [not any(row["available"]) for row in trace])
            padded = np.r_[False, blind, False].astype(int)
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1)
            video = dict(sequence=name, blind_intervals=len(starts),
                         full_SR_delta=worker["records"]["velocity"][name]["alternating_outage"]["SR"]
                                       - worker["records"]["hold"][name]["alternating_outage"]["SR"])
            videos.append(video)
            for index, (start, end) in enumerate(zip(starts, ends)):
                next_start = starts[index + 1] if index + 1 < len(starts) else len(gt)
                record = dict(sequence=name, start=int(start), end_exclusive=int(end),
                              gap_frames=int(end - start), groups={})
                for label, first, stop in [
                    ("blind", start, end), ("observed_until_next_gap", end, next_start)]:
                    selected = np.arange(first, stop)
                    selected = selected[valid[selected]]
                    if not len(selected):
                        continue
                    scores = {}
                    for arm in boxes:
                        measured = frame_metrics(boxes[arm][selected], gt[selected])
                        scores[arm] = {metric: float(value.mean()) for metric, value in measured.items()}
                    record["groups"][label] = dict(
                        frames=len(selected), scores=scores,
                        velocity_minus_hold={metric: scores["velocity"][metric] - scores["hold"][metric]
                                             for metric in ["PR20", "SR"]})
                    if label == "observed_until_next_gap":
                        support = {}
                        for arm in boxes:
                            rows = [geometry(boxes[arm][frame - 1], gt[frame],
                                             boxes[arm][frame], factor) for frame in selected]
                            support[arm] = dict(
                                target_outside_crop=sum(not row["target_overlaps_previous_crop"]
                                                        for row in rows),
                                target_center_outside_crop=sum(not row["target_center_in_previous_crop"]
                                                               for row in rows),
                                first_scored_frame=int(selected[0]),
                                first_frame_geometry=rows[0])
                        record["groups"][label]["crop_support"] = support
                intervals.append(record)
    result = dict(
        parent_report_sha256=sha(FOLDER / "report.json"), analyzer_sha256=sha(Path(__file__)),
        crop_config_sha256=sha(config_path), search_factor=factor,
        videos=sorted(videos, key=lambda row: row["full_SR_delta"]), intervals=intervals,
        scope="Retrospective train882 development localization of fixed-CV regressions. "
              "Recovery windows end at the next blind interval or video end. "
              "No parameter fitting, intervention, official result or causal mechanism claim.")
    with (FOLDER / args.output_name).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(
        intervals=len(intervals), videos=result["videos"],
        worst_blind_intervals=sorted(
            [row for row in intervals if "blind" in row["groups"]],
            key=lambda row: row["groups"]["blind"]["velocity_minus_hold"]["SR"])[:3]), indent=2))


if __name__ == "__main__":
    main()
