"""Reuse saved counterfactual boxes to vary both sensor states at equal lag1."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
SOURCE = ROOT / "reports/server28-fixed-crop-motion-factorial-20261031"
STATES = ["11", "01", "10"]


def center(box):
    box = np.asarray(box)
    return box[:2] + box[2:] / 2


def main(source):
    report = json.loads((source / "report.json").read_text())
    completion = json.loads((source / "completed.json").read_text())
    verified = json.loads((source / "analysis.json").read_text())
    expected_sha = sha(source / "report.json")
    assert completion["report_sha256"] == verified["report_sha256"] == expected_sha
    assert verified["calls_verified"] == 576 and verified["rows_verified"] == 384
    boxes = {}
    for row in report["records"]:
        key = (row["sequence"], row["before"], row["prior_state"])
        if key in boxes:
            assert boxes[key] == row["before_box"]
        boxes[key] = row["before_box"]
    names = sorted({key[0] for key in boxes})
    assert len(names) == 16
    rows = []
    for name in names:
        gt = np.loadtxt(str(Path("/data/wangwj/dataset/LasHeR/trainingset") / name / "init.txt"),
                        delimiter=",", ndmin=2)
        for original_target in report["launch"]["blind_targets"]:
            before, last, target = original_target - 3, original_target - 2, original_target - 1
            for current in STATES:
                for prior in STATES:
                    previous_box, last_box = boxes[(name, before, prior)], boxes[(name, last, current)]
                    velocity = center(last_box) - center(previous_box)
                    row = dict(sequence=name, before=before, last=last, forecast_frame=target,
                               current_state=current, prior_state=prior, velocity=velocity.tolist())
                    if (gt[[before, last, target], 2:] > 0).all():
                        row["retrospective_gt"] = dict(
                            velocity_error=float(np.linalg.norm(
                                velocity - (center(gt[last]) - center(gt[before])))),
                            forecast_center_error=float(np.linalg.norm(
                                center(last_box) + velocity - center(gt[target]))))
                    rows.append(row)
    assert len(rows) == 576
    metrics = ["velocity_error", "forecast_center_error"]
    per_video = {}
    counts = {}
    for name in names:
        per_video[name], counts[name] = {}, {}
        for current in STATES:
            for prior in STATES:
                selected = [row for row in rows if row["sequence"] == name and
                            row["current_state"] == current and row["prior_state"] == prior]
                assert len(selected) == 4
                valid = [row["retrospective_gt"] for row in selected if "retrospective_gt" in row]
                assert valid
                cell = current + "_" + prior
                counts[name][cell] = len(valid)
                per_video[name][cell] = {
                    metric: float(np.mean([row[metric] for row in valid])) for metric in metrics}
        assert len(set(counts[name].values())) == 1
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    contrasts = {}
    for current in STATES:
        for prior in STATES:
            if current == prior:
                continue
            left, right = current + "_" + current, current + "_" + prior
            label = "current%s_same%s_minus_prior%s" % (current, current, prior)
            contrasts[label] = {}
            for metric in metrics:
                delta = np.asarray([per_video[name][left][metric] -
                                    per_video[name][right][metric] for name in names])
                contrasts[label][metric] = dict(
                    mean_delta=float(delta.mean()),
                    paired_video_bootstrap95=np.percentile(
                        delta[indices].mean(1), [2.5, 97.5]).tolist(),
                    lower_error_videos=int((delta < 0).sum()),
                    equal_videos=int((delta == 0).sum()),
                    higher_error_videos=int((delta > 0).sum()))
    result = dict(
        report_sha256=expected_sha, original_analysis_sha256=sha(source / "analysis.json"),
        analyzer_sha256=sha(Path(__file__)), new_model_calls=0,
        reused_model_calls=576, factorial_rows=576, interval=1,
        rows=rows, valid_triplets=counts, per_video=per_video, contrasts=contrasts,
        video_macro={cell: {metric: float(np.mean([per_video[name][cell][metric] for name in names]))
                            for metric in metrics} for cell in per_video[names[0]]},
        scope="Post-hoc equal-lag3x3 current/prior availability diagnostic reusing saved early "
              "fixed-crop predictions. Forecast frames4/10/16/22 precede original blind targets "
              "by one frame; no added inference or fitted parameter. Negative error contrast "
              "favors same-state history. Reused16 videos, frozen complete first-frame templates "
              "and complete-reference crop geometry; no closed-loop, official or novelty claim.")
    with (source / "state_factorial.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(video_macro=result["video_macro"], contrasts=contrasts), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    main(args.source)
