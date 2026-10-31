"""Check factorial receipts and compare equal-lag sensor-state pairs."""
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-fixed-crop-motion-factorial-20261031"


def center(box):
    box = np.asarray(box)
    return box[:2] + box[2:] / 2


def main():
    report = json.loads((OUTPUT / "report.json").read_text())
    completion = json.loads((OUTPUT / "completed.json").read_text())
    assert sha(OUTPUT / "report.json") == completion["report_sha256"]
    assert report["calls"] == completion["calls"] == 576
    assert report["baseline_state_unchanged"]
    for path, expected in report["launch"]["source_sha256"].items():
        assert sha(path) == expected, path
    rows = report["records"]
    names = sorted(set(row["sequence"] for row in rows))
    assert len(names) == 16 and len(rows) == 384
    targets = {}
    for name in names:
        targets[name] = np.loadtxt(
            str(Path("/data/wangwj/dataset/LasHeR/trainingset") / name / "init.txt"),
            delimiter=",", ndmin=2)
    for row in rows:
        assert row["last"] == row["blind_target"] - 1
        assert row["before"] == row["last"] - row["lag"]
        velocity = (center(row["last_box"]) - center(row["before_box"])) / row["lag"]
        assert np.array_equal(velocity, np.asarray(row["velocity"]))
        gt = targets[row["sequence"]]
        valid = bool((gt[[row["before"], row["last"], row["blind_target"]], 2:] > 0).all())
        assert valid == ("retrospective_gt" in row)
        if valid:
            true_velocity = (center(gt[row["last"]]) - center(gt[row["before"]])) / row["lag"]
            expected = dict(
                velocity_error=float(np.linalg.norm(velocity - true_velocity)),
                forecast_center_error=float(np.linalg.norm(
                    center(row["last_box"]) + velocity - center(gt[row["blind_target"]]))))
            assert expected == row["retrospective_gt"]
    metrics = ["velocity_error", "forecast_center_error"]
    per_video = {}
    for name in names:
        per_video[name] = {}
        for lag in [1, 2]:
            for state in ["11", "01", "10"]:
                chosen = [row for row in rows if row["sequence"] == name and
                          row["lag"] == lag and row["prior_state"] == state]
                assert len(chosen) == 4
                valid = [row["retrospective_gt"] for row in chosen if "retrospective_gt" in row]
                assert valid
                per_video[name][str(lag) + "_" + state] = {
                    metric: float(np.mean([row[metric] for row in valid])) for metric in metrics}
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))

    def contrast(left, right):
        result = {}
        for metric in metrics:
            delta = np.asarray([per_video[name][left][metric] -
                                per_video[name][right][metric] for name in names])
            result[metric] = dict(mean_delta=float(delta.mean()),
                                  paired_video_bootstrap95=np.percentile(
                                      delta[indices].mean(1), [2.5, 97.5]).tolist(),
                                  lower_error_videos=int((delta < 0).sum()),
                                  equal_videos=int((delta == 0).sum()),
                                  higher_error_videos=int((delta > 0).sum()))
        return result

    contrasts = {}
    for lag in [1, 2]:
        for state in ["01", "10"]:
            contrasts["lag%d_same11_minus_cross%s" % (lag, state)] = contrast(
                str(lag) + "_11", str(lag) + "_" + state)
    for state in ["11", "01", "10"]:
        contrasts["state%s_lag2_minus_lag1" % state] = contrast("2_" + state, "1_" + state)
    result = dict(
        report_sha256=sha(OUTPUT / "report.json"), analyzer_sha256=sha(Path(__file__)),
        calls_verified=576, rows_verified=384, per_video=per_video, contrasts=contrasts,
        video_macro={cell: {metric: float(np.mean([per_video[name][cell][metric] for name in names]))
                            for metric in metrics} for cell in per_video[names[0]]},
        scope="Paired16-video fixed-crop CPU diagnostic with four early triplets/video. "
              "Negative error deltas favor left condition. Equal-lag state contrasts isolate "
              "the prior observation availability in this fixed context, not whole-tracker "
              "causal mechanisms or deployment gains. Complete prior GPU crop histories and "
              "fixed initial templates limit transfer. Reused development, no official scores.")
    with (OUTPUT / "analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(video_macro=result["video_macro"], contrasts=contrasts), indent=2))


if __name__ == "__main__":
    main()
