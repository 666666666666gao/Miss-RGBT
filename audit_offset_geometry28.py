"""Post-hoc center/size decomposition of existing fixed-crop predictions."""
import json
from pathlib import Path

import numpy as np

from audit_offset_mechanism28 import scores
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
STUDIES = {
    "original": ("server28-fixed-crop-motion-factorial-20261031",
                 "server28-blind-motion-shortgap16-20261031"),
    "fresh": ("server28-fixed-crop-motion-replication16-20261031",
              "server28-motion-replication16-adjacent-20261031"),
}
OUTPUT = ROOT / "reports/server28-state-offset-geometry-20261031.json"


def read(path):
    return json.loads(path.read_text())


def center(box):
    box = np.asarray(box)
    return box[:2] + box[2:] / 2


def make_box(location, size):
    return np.concatenate([location - size / 2, size])


def main():
    studies = {}
    inventories = []
    for label, (folder_name, parent_name) in STUDIES.items():
        folder, parent_folder = ROOT / "reports" / folder_name, ROOT / "reports" / parent_name
        report, parent = read(folder / "report.json"), read(parent_folder / "report.json")
        assert sha(folder / "report.json") == read(folder / "completed.json")["report_sha256"]
        assert sha(parent_folder / "report.json") == read(parent_folder / "completed.json")["report_sha256"]
        assert report["launch"]["parent_report_sha256"] == sha(parent_folder / "report.json")
        assert read(folder / "analysis.json")["calls_verified"] == 576
        inventory = {item["name"]: item for items in parent["launch"]["parts"].values()
                     for item in items}
        assert len(inventory) == 16
        inventories.append(set(inventory))
        boxes = {}
        for row in report["records"]:
            key = (row["sequence"], row["before"], row["prior_state"])
            if key in boxes:
                assert np.array_equal(boxes[key], row["before_box"])
            boxes[key] = np.asarray(row["before_box"])
        assert set(key[0] for key in boxes) == set(inventory)
        records = {"01": [], "10": []}
        gt_receipts = {}
        for name, item in inventory.items():
            gt_path = Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt_receipts[name] = sha(gt_path)
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            for target in report["launch"]["blind_targets"]:
                previous, current = target - 3, target - 2
                if not (gt[current, 2:] > 0).all():
                    continue
                full = boxes[(name, current, "11")]
                previous_full = boxes[(name, previous, "11")]
                for state in records:
                    raw = boxes[(name, current, state)]
                    prior = boxes[(name, previous, state)]
                    assert np.isfinite([full, previous_full, raw, prior]).all()
                    assert (np.asarray([full, previous_full, raw, prior])[:, 2:] > 0).all()
                    corrected_center = center(raw) - (center(prior) - center(previous_full))
                    candidates = {
                        "raw": raw,
                        "current_full_center": make_box(center(full), raw[2:]),
                        "current_full_size": make_box(center(raw), full[2:]),
                        "current_full_box": full,
                        "past_center": make_box(corrected_center, raw[2:]),
                        "past_center_and_size": make_box(
                            corrected_center, raw[2:] * previous_full[2:] / prior[2:]),
                    }
                    measured = {}
                    for arm, candidate in candidates.items():
                        measured[arm] = scores(candidate, gt[current])
                        measured[arm]["mean_abs_log_size_error"] = float(
                            np.abs(np.log(candidate[2:] / gt[current, 2:])).mean())
                    records[state].append(dict(sequence=name, frame=current, measured=measured))
        names = sorted(inventory)
        indices = np.random.RandomState(2027).randint(0, 16, (10000, 16))
        states = {}
        for state, rows in records.items():
            per_video = {}
            for name in names:
                selected = [row for row in rows if row["sequence"] == name]
                assert selected
                per_video[name] = {
                    arm: {key: float(np.mean([row["measured"][arm][key] for row in selected]))
                          for key in selected[0]["measured"][arm]}
                    for arm in selected[0]["measured"]}
            contrasts = {}
            for first, second in [
                ("current_full_center", "raw"), ("current_full_size", "raw"),
                ("current_full_box", "raw"), ("past_center", "raw"),
                ("past_center_and_size", "past_center"),
            ]:
                contrasts[first + "_minus_" + second] = {}
                for metric in ["center_error_px", "PR20", "SR", "mean_abs_log_size_error"]:
                    delta = np.asarray([per_video[name][first][metric] -
                                        per_video[name][second][metric] for name in names])
                    contrasts[first + "_minus_" + second][metric] = dict(
                        mean_delta=float(delta.mean()),
                        paired_video_bootstrap95=np.percentile(
                            delta[indices].mean(1), [2.5, 97.5]).tolist(),
                        negative_videos=int((delta < 0).sum()),
                        zero_videos=int((delta == 0).sum()),
                        positive_videos=int((delta > 0).sum()))
            states[state] = dict(
                valid_pairs=len(rows), per_video=per_video, contrasts=contrasts,
                macro={arm: {metric: float(np.mean([per_video[name][arm][metric] for name in names]))
                             for metric in per_video[names[0]][arm]}
                       for arm in per_video[names[0]]})
        studies[label] = dict(report_sha256=sha(folder / "report.json"),
                              parent_report_sha256=sha(parent_folder / "report.json"),
                              gt_sha256=gt_receipts, states=states)
    assert not inventories[0] & inventories[1]
    result = dict(
        studies=studies, new_model_calls=0, fitted_parameters=0,
        source_sha256={str(path): sha(path) for path in [
            Path(__file__), ROOT / "audit_offset_mechanism28.py"]},
        scope="Post-hoc geometry decomposition on both previously inspected fixed-crop cohorts. "
              "Current-full-center/size/box arms use unavailable sensor predictions and are "
              "diagnostic only. Past arms use prior complete/missing predictions but synthetic "
              "boxes are not clipped and are not closed-loop or prospective evaluations. "
              "Only GT with positive size is scored; GT never constructs predictions. "
              "No tuning or change to the active registered16-video center-only experiment. "
              "Video bootstrap is descriptive, not training-seed or official benchmark evidence.")
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({label: {state: values["contrasts"] for state, values in study["states"].items()}
                      for label, study in studies.items()}, indent=2))


if __name__ == "__main__":
    main()
