"""Evaluate past-only state-offset correction from saved fixed-crop predictions."""
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
STUDIES = {
    "original": ROOT / "reports/server28-fixed-crop-motion-factorial-20261031",
    "fresh": ROOT / "reports/server28-fixed-crop-motion-replication16-20261031",
}
OUTPUT = ROOT / "reports/server28-causal-state-offset-audit-20261031.json"


def read(path):
    return json.loads(path.read_text())


def center(box):
    box = np.asarray(box)
    return box[:2] + box[2:] / 2


def summarize(records, metric, names):
    return {name: float(np.mean([row[metric] for row in records if row["sequence"] == name]))
            for name in names}


def main():
    studies = {}
    inventories = {}
    for label, folder in STUDIES.items():
        report = read(folder / "report.json")
        assert sha(folder / "report.json") == read(folder / "completed.json")["report_sha256"]
        assert read(folder / "analysis.json")["report_sha256"] == sha(folder / "report.json")
        factorial = read(folder / "state_factorial.json")
        assert factorial["report_sha256"] == sha(folder / "report.json")
        # The prior boxes preserve all states at two consecutive observation times.
        boxes = {(row["sequence"], row["before"], row["prior_state"]): row["before_box"]
                 for row in report["records"]}
        names = sorted({key[0] for key in boxes})
        assert len(names) == 16
        inventories[label] = set(names)
        records = []
        for name in names:
            gt = np.loadtxt(
                str(Path("/data/wangwj/dataset/LasHeR/trainingset") / name / "init.txt"),
                delimiter=",", ndmin=2)
            for original_target in report["launch"]["blind_targets"]:
                before, last, target = original_target - 3, original_target - 2, original_target - 1
                for state in ["01", "10"]:
                    previous_full = center(boxes[(name, before, "11")])
                    previous_missing = center(boxes[(name, before, state)])
                    current_missing = center(boxes[(name, last, state)])
                    previous_offset = previous_missing - previous_full
                    aligned_current = current_missing - previous_offset
                    assert np.allclose(aligned_current - previous_full,
                                       current_missing - previous_missing, rtol=0, atol=1e-10)
                    forecasts = dict(
                        raw_cross=2 * current_missing - previous_full,
                        same_state_motion=2 * current_missing - previous_missing,
                        offset_corrected=2 * aligned_current - previous_full)
                    if not (gt[[before, last, target], 2:] > 0).all():
                        continue
                    row = dict(sequence=name, state=state, before=before, last=last, target=target,
                               previous_offset=previous_offset.tolist(),
                               current_offset_drift=float(np.linalg.norm(
                                   (current_missing - center(boxes[(name, last, "11")])) -
                                   previous_offset)),
                               previous_offset_norm=float(np.linalg.norm(previous_offset)),
                               raw_current_error=float(np.linalg.norm(current_missing - center(gt[last]))),
                               corrected_current_error=float(np.linalg.norm(aligned_current - center(gt[last]))))
                    for arm, predicted in forecasts.items():
                        row[arm + "_forecast_error"] = float(np.linalg.norm(predicted - center(gt[target])))
                    records.append(row)
        indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
        states = {}
        for state in ["01", "10"]:
            selected = [row for row in records if row["state"] == state]
            fields = ["raw_current_error", "corrected_current_error", "raw_cross_forecast_error",
                      "same_state_motion_forecast_error", "offset_corrected_forecast_error",
                      "current_offset_drift", "previous_offset_norm"]
            values = {key: summarize(selected, key, names) for key in fields}
            contrasts = {}
            for left, right in [
                ("corrected_current_error", "raw_current_error"),
                ("offset_corrected_forecast_error", "raw_cross_forecast_error"),
                ("offset_corrected_forecast_error", "same_state_motion_forecast_error")]:
                delta = np.asarray([values[left][name] - values[right][name] for name in names])
                contrasts[left + "_minus_" + right] = dict(
                    mean_delta_px=float(delta.mean()),
                    paired_video_bootstrap95_px=np.percentile(
                        delta[indices].mean(1), [2.5, 97.5]).tolist(),
                    lower_error_videos=int((delta < 0).sum()),
                    equal_videos=int((delta == 0).sum()),
                    higher_error_videos=int((delta > 0).sum()))
            states[state] = dict(valid_pairs=len(selected),
                                 video_macro_px={key: float(np.mean(list(row.values())))
                                                 for key, row in values.items()},
                                 contrasts=contrasts)
        studies[label] = dict(report_sha256=sha(folder / "report.json"), states=states, records=records)
    assert not inventories["original"] & inventories["fresh"]
    result = dict(
        source_sha256=sha(Path(__file__)), studies=studies, new_model_calls=0,
        deployable_inputs="Past fully observed-frame predictions under11/01/10; current missing "
                          "prediction. Current full prediction used only for retrospective offset "
                          "drift, never correction or forecast. No GT in prediction.",
        scope="Post-hoc saved fixed-crop development counterfactual. Correction devised after "
              "viewing both state factorials; fresh set is not a prospective test of this new "
              "correction. No fitted coefficient, closed-loop, official or novel-module claim.")
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({label: value["states"] for label, value in studies.items()}, indent=2))


if __name__ == "__main__":
    main()
