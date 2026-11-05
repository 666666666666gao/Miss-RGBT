"""Retrospectively inspect fixed-CV estimates at blind-gap entry."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import yaml

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")


def read(path):
    return json.loads(path.read_text())


def center(box):
    return np.asarray(box[:2]) + np.asarray(box[2:]) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_dir
    parent = read(root / "report.json")
    assert sha(root / "report.json") == read(root / "completed.json")["report_sha256"]
    config = ROOT / "runtime/task_utility_20261018/experiments/cadtrack/cadtrack.yaml"
    factor = yaml.safe_load(config.read_text())["TEST"]["SEARCH_FACTOR"]
    records = []
    for gpu in [0, 1]:
        folder = root / ("gpu%d" % gpu)
        worker = read(folder / "report.json")
        assert sha(folder / "report.json") == parent["worker_receipts"][str(gpu)]["report_sha256"]
        for item in worker["launch"]["cohort"]:
            name = item["name"]
            gt_path = Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            target = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            prefix = folder / ("velocity_" + name + "_alternating_outage")
            receipt = worker["records"]["velocity"][name]["alternating_outage"]
            assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
            assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
            boxes = np.loadtxt(str(prefix) + ".txt", ndmin=2)
            trace = read(Path(str(prefix) + ".json"))
            available = np.asarray([[1, 1]] + [row["available"] for row in trace])
            observed = available.any(1)
            blind = ~observed
            padded = np.r_[False, blind, False].astype(int)
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1)
            for start, end in zip(starts, ends):
                before, last = np.flatnonzero(observed[:start])[-2:]
                velocity = (center(boxes[last]) - center(boxes[before])) / (last - before)
                forecast = boxes[last].copy()
                forecast[:2] += (end - 1 - last) * velocity
                assert np.allclose(forecast, boxes[end - 1], rtol=0, atol=1e-9)
                crop_side = float(np.ceil(np.sqrt(boxes[last, 2:].prod()) * factor))
                states = ["".join(str(int(v)) for v in available[index])
                          for index in [before, last]]
                row = dict(
                    sequence=name, start=int(start), end_exclusive=int(end),
                    observed_frames=[int(before), int(last)], observed_states=states,
                    velocity_px_per_frame=velocity.tolist(),
                    forecast_displacement_crop_sides=float(
                        np.linalg.norm((end - 1 - last) * velocity) / crop_side),
                    last_observed_confidence=trace[last - 1]["confidence"] if last else None,
                    recovery_state=available[end].tolist() if end < len(boxes) else None)
                if (target[[before, last, end - 1], 2:] > 0).all():
                    true_velocity = (center(target[last]) - center(target[before])) / (last - before)
                    row["retrospective_gt"] = dict(
                        recent_velocity_px_per_frame=true_velocity.tolist(),
                        observed_velocity_error_px_per_frame=float(np.linalg.norm(velocity - true_velocity)),
                        last_observed_center_error_px=float(
                            np.linalg.norm(center(boxes[last]) - center(target[last]))),
                        endpoint_center_error_cv_px=float(
                            np.linalg.norm(center(boxes[end - 1]) - center(target[end - 1]))),
                        endpoint_center_error_hold_from_same_last_px=float(
                            np.linalg.norm(center(boxes[last]) - center(target[end - 1]))))
                records.append(row)
    transitions = Counter("->".join(row["observed_states"]) for row in records)
    result = dict(
        report_sha256=sha(root / "report.json"), analyzer_sha256=sha(Path(__file__)),
        config_sha256=sha(config), intervals=len(records), observed_state_pairs=dict(transitions),
        records=records,
        scope="GT is used only after saved tracking to diagnose velocity and localization errors. "
              "Hold-from-same-last is a local counterfactual, not the separate hold trajectory. "
              "No fitted threshold, inference intervention, benchmark result or causal attribution.")
    with (root / "motion_estimate_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(intervals=len(records), observed_state_pairs=dict(transitions),
                          largest_forecast_displacements=sorted(
                              records, key=lambda row: row["forecast_displacement_crop_sides"],
                              reverse=True)[:3]), indent=2))


if __name__ == "__main__":
    main()
