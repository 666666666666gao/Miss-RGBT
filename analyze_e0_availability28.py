"""Describe full-val97 paired E0 errors by the fixed observed sensor state."""
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha

ROOT = Path("/home/gaob/Miss-RGBT")
REPORT = ROOT / "reports/server28-epoch05-tracking-health-20261031"
PROTOCOL = ROOT / "development/recovery_val97_20261030/protocol.json"


def frame_metrics(boxes, gt):
    error = np.linalg.norm(boxes[:, :2] + boxes[:, 2:] / 2 -
                           gt[:, :2] - gt[:, 2:] / 2, axis=1)
    extent = np.maximum(0, np.minimum(boxes[:, :2] + boxes[:, 2:], gt[:, :2] + gt[:, 2:]) -
                        np.maximum(boxes[:, :2], gt[:, :2]))
    intersection = extent.prod(1)
    union = boxes[:, 2:].prod(1) + gt[:, 2:].prod(1) - intersection
    iou = intersection / union
    return dict(PR20=(error <= 20).astype(float) * 100,
                SR=(iou[:, None] > np.linspace(0, 1, 21)).mean(1) * 100)


def main():
    protocol = json.loads(PROTOCOL.read_text())
    spec = protocol["conditions"]["alternating_outage"]
    assert sha(spec["path"]) == spec["sha256"]
    masks = json.loads(Path(spec["path"]).read_text())
    receipts = {}
    for condition in ["complete", "alternating_outage"]:
        folder = REPORT / ("E0_" + condition)
        completed = json.loads((folder / "completed.json").read_text())
        rows = json.loads((folder / "receipts.json").read_text())
        receipts[condition] = {item["sequence"]: item for item in rows}
        assert len(rows) == len(receipts[condition]) == completed["sequences"] == 97
    groups = {label: [] for label in ["00", "01", "10", "11"]}
    sequence_gaps = []
    for item in protocol["sequences"]:
        name = item["name"]
        assert sha(item["gt_path"]) == item["gt_sha256"]
        gt = np.loadtxt(item["gt_path"], delimiter=",", ndmin=2)
        valid = (gt[:, 2:] > 0).all(1)
        valid[0] = False
        mask = np.asarray(masks[name]["data"])
        values = {}
        for condition in receipts:
            path = REPORT / ("E0_" + condition) / (name + ".txt")
            assert sha(path) == receipts[condition][name]["bbox_sha256"]
            boxes = np.loadtxt(str(path), ndmin=2)
            assert boxes.shape == gt.shape == (item["frames"], 4)
            values[condition] = frame_metrics(boxes[valid], gt[valid])
        state = mask[valid, 0] * 2 + mask[valid, 1]
        for label in groups:
            selected = state == int(label, 2)
            if not selected.any():
                continue
            groups[label].append(dict(
                sequence=name, frames=int(selected.sum()),
                **{metric: dict(
                    complete=float(values["complete"][metric][selected].mean()),
                    outage=float(values["alternating_outage"][metric][selected].mean()))
                   for metric in ["PR20", "SR"]}))
        sequence_gaps.append(dict(
            sequence=name, **{metric: float(
                values["alternating_outage"][metric].mean() -
                values["complete"][metric].mean()) for metric in ["PR20", "SR"]}))
    summaries = {}
    for label, rows in groups.items():
        summaries[label] = dict(sequences=len(rows), scored_frames=sum(row["frames"] for row in rows))
        for metric in ["PR20", "SR"]:
            summaries[label][metric] = {
                weighting: dict(
                    complete=float(np.average([row[metric]["complete"] for row in rows], weights=weights)),
                    outage=float(np.average([row[metric]["outage"] for row in rows], weights=weights)),
                    outage_minus_complete_pp=float(np.average([
                        row[metric]["outage"] - row[metric]["complete"] for row in rows], weights=weights)))
                for weighting, weights in [
                    ("sequence_macro", None), ("frame_weighted", [row["frames"] for row in rows])]}
    assert sum(row["scored_frames"] for row in summaries.values()) == 54289
    report = dict(
        scope="Descriptive full-val97 E0 paired same-frame availability groups; not causal or official test",
        protocol_sha256=sha(PROTOCOL), source_sha256=sha(Path(__file__)),
        groups=summaries, sequence_gaps=sequence_gaps, per_sequence_state=groups,
        interpretation="11 frames can retain effects of prior outages and dynamic templates. "
                       "00 holds, state timing and scene difficulty are confounded; "
                       "group gaps do not isolate sensor effects or select thresholds.")
    with (REPORT / "e0_availability_analysis.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
