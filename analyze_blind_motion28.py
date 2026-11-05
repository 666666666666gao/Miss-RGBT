"""Verify closed-loop saved trajectories and separate blind/recovery scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_e0_availability28 import frame_metrics
from analyze_epoch05_health28 import metrics
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
REPORT = ROOT / "reports/server28-blind-motion-closed-loop-20261031"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    args = parser.parse_args()
    root = args.report_dir
    completion = json.loads((root / "completed.json").read_text())
    assert sha(root / "report.json") == completion["report_sha256"]
    original = json.loads((root / "report.json").read_text())
    launch = original["launch"]
    assert sha(launch["checkpoint"]) == launch["checkpoint_sha256"]
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected, path
    groups = {name: [] for name in [
        "all_valid", "blind", "observed_before_first_blind",
        "observed_after_first_blind", "both_observed_after_first_blind"]}
    trajectories = 0
    for item in launch["cohort"]:
        name = item["name"]
        gt_path = Path(item["folder"]) / "init.txt"
        assert sha(gt_path) == item["gt_sha256"]
        gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)[:item["evaluated_frames"]]
        valid = (gt[:, 2:] > 0).all(1)
        valid[0] = False
        predictions = {}
        for arm in launch["arms"]:
            for condition in launch["conditions"]:
                prefix = root / (arm + "_" + name + "_" + condition)
                receipt = original["records"][arm][name][condition]
                assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
                assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
                boxes = np.loadtxt(str(prefix) + ".txt", ndmin=2)
                assert boxes.shape == gt.shape and np.isfinite(boxes).all()
                actual = metrics(boxes, gt)
                for key, value in actual.items():
                    assert np.isclose(value, receipt[key], rtol=0, atol=1e-9)
                predictions[(arm, condition)] = boxes
                trajectories += 1
        assert np.array_equal(predictions[("hold", "complete")],
                              predictions[("velocity", "complete")])
        pattern = np.asarray(launch["pattern"])
        available = pattern[(np.arange(len(gt)) // launch["frame_block"]) % len(pattern)]
        blind = ~available.any(1)
        after = np.maximum.accumulate(blind)
        selections = dict(
            all_valid=valid, blind=valid & blind,
            observed_before_first_blind=valid & ~blind & ~after,
            observed_after_first_blind=valid & ~blind & after,
            both_observed_after_first_blind=valid & available.all(1) & after)
        before = ~after
        assert np.array_equal(predictions[("hold", "alternating_outage")][before],
                              predictions[("velocity", "alternating_outage")][before])
        for group, selected in selections.items():
            if not selected.any():
                continue
            record = dict(sequence=name, frames=int(selected.sum()), metrics={})
            for arm in launch["arms"]:
                values = frame_metrics(
                    predictions[(arm, "alternating_outage")][selected], gt[selected])
                record["metrics"][arm] = {metric: float(value.mean())
                                          for metric, value in values.items()}
            record["velocity_minus_hold"] = {
                metric: record["metrics"]["velocity"][metric] - record["metrics"]["hold"][metric]
                for metric in ["PR20", "SR"]}
            groups[group].append(record)
    assert trajectories == len(launch["cohort"]) * 4
    summaries = {}
    for group, rows in groups.items():
        scores = {arm: {metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                        for metric in ["PR20", "SR"]} for arm in launch["arms"]}
        summaries[group] = dict(
            sequences=len(rows), scored_frames=sum(row["frames"] for row in rows),
            sequence_macro=scores,
            velocity_minus_hold={metric: scores["velocity"][metric] - scores["hold"][metric]
                                 for metric in ["PR20", "SR"]})
    for arm in launch["arms"]:
        for metric in ["PR20", "SR"]:
            assert np.isclose(summaries["all_valid"]["sequence_macro"][arm][metric],
                              original["aggregate"]["alternating_outage"][arm][metric],
                              rtol=0, atol=1e-9)
    result = dict(
        scope="Verified same-GPU registered train882 dev closed-loop motion control; not official scores",
        report_sha256=sha(root / "report.json"), analyzer_sha256=sha(Path(__file__)),
        trajectories_verified=trajectories, groups=summaries, per_sequence=groups,
        complete_outputs_bitwise_equal=True, pre_blind_prefix_bitwise_equal=True,
        interpretation="Groups use different video/frame membership and cannot be added as "
                       "sequence macros. Video durations determine blind/recovery membership. "
                       "Ordinary fixed CV, registered development data, no novelty claim.")
    with (root / "analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
