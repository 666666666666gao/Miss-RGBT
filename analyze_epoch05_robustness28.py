"""Describe paired outage penalties by sensor state for the completed epoch5 tracks."""
import json
from pathlib import Path

import numpy as np

from analyze_e0_availability28 import frame_metrics
from analyze_epoch05_health28 import compare
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
REPORT = ROOT / "reports/server28-epoch05-tracking-health-20261031"
PROTOCOL = ROOT / "development/recovery_val97_20261030/protocol.json"


def main():
    completion = json.loads((REPORT / "analysis_completed.json").read_text())
    assert sha(REPORT / "paired_analysis.json") == completion["paired_analysis_sha256"]
    verified = json.loads((REPORT / "paired_analysis.json").read_text())
    protocol = json.loads(PROTOCOL.read_text())
    assert sha(PROTOCOL) == verified["protocol_sha256"]
    mask_spec = protocol["conditions"]["alternating_outage"]
    assert sha(mask_spec["path"]) == mask_spec["sha256"]
    masks = json.loads(Path(mask_spec["path"]).read_text())
    states = ["all", "00", "01", "10", "11"]
    models = ["E0", "utility", "random"]
    cells = {label: {state: {} for state in states} for label in models}
    receipts = {}
    trajectories = 0
    for label in models:
        for condition in ["complete", "alternating_outage"]:
            rows = json.loads((REPORT / (label + "_" + condition) / "receipts.json").read_text())
            receipts[(label, condition)] = {row["sequence"]: row for row in rows}
            assert len(rows) == len(receipts[(label, condition)]) == 97
    counts = {state: 0 for state in states}
    for item in protocol["sequences"]:
        name = item["name"]
        assert sha(item["gt_path"]) == item["gt_sha256"]
        gt = np.loadtxt(item["gt_path"], delimiter=",", ndmin=2)
        valid = (gt[:, 2:] > 0).all(1)
        valid[0] = False
        mask = np.asarray(masks[name]["data"])
        sensor = (mask[valid, 0] * 2 + mask[valid, 1]).astype(int)
        selections = {"all": np.ones(int(valid.sum()), dtype=bool)}
        selections.update({state: sensor == int(state, 2) for state in states[1:]})
        for state, selected in selections.items():
            counts[state] += int(selected.sum())
        for label in models:
            values = {}
            for condition in ["complete", "alternating_outage"]:
                path = REPORT / (label + "_" + condition) / (name + ".txt")
                receipt = receipts[(label, condition)][name]
                assert sha(path) == receipt["bbox_sha256"]
                boxes = np.loadtxt(str(path), ndmin=2)
                assert boxes.shape == gt.shape == (item["frames"], 4)
                assert np.isfinite(boxes).all()
                values[condition] = frame_metrics(boxes[valid], gt[valid])
                for metric in ["PR20", "SR"]:
                    assert np.isclose(values[condition][metric].mean(), receipt[metric],
                                      rtol=0, atol=1e-9)
                trajectories += 1
            for state, selected in selections.items():
                if selected.any():
                    cells[label][state][name] = {
                        condition: {metric: float(value[selected].mean())
                                    for metric, value in measured.items()}
                        for condition, measured in values.items()}
    assert trajectories == 582 and counts["all"] == 54289
    assert sum(counts[state] for state in states[1:]) == counts["all"]
    results = {}
    for state in states:
        names = sorted(cells["E0"][state])
        indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
        penalties = {}
        means = {}
        for label in models:
            assert set(cells[label][state]) == set(names)
            penalties[label] = {
                name: {metric: row["alternating_outage"][metric] - row["complete"][metric]
                       for metric in ["PR20", "SR"]}
                for name, row in cells[label][state].items()}
            means[label] = {
                metric: dict(
                    complete=float(np.mean([cells[label][state][name]["complete"][metric]
                                            for name in names])),
                    outage=float(np.mean([cells[label][state][name]["alternating_outage"][metric]
                                          for name in names])),
                    outage_minus_complete_pp=float(np.mean(
                        [penalties[label][name][metric] for name in names])))
                for metric in ["PR20", "SR"]}
        results[state] = dict(
            sequences=len(names), scored_frames=counts[state], means=means,
            penalty_contrasts={
                first + "_minus_" + second: compare(penalties[first], penalties[second], names, indices)
                for first, second in [("utility", "E0"), ("random", "E0"), ("utility", "random")]})
    report = dict(
        scope="Post-hoc val97 description of matched outage penalties, not causal or official scores",
        paired_analysis_sha256=sha(REPORT / "paired_analysis.json"),
        protocol_sha256=sha(PROTOCOL), source_sha256=sha(Path(__file__)),
        checkpoint_sha256=verified["checkpoint_sha256"], trajectories_verified=trajectories,
        groups=results, bootstrap_draws=10000, bootstrap_seed=2027,
        interpretation="Positive penalty contrast means less outage degradation relative to the "
                       "same model's complete trajectory. This subtracts complete-input gain "
                       "descriptively, but does not isolate a causal missing-input mechanism. "
                       "State groups have different video membership. One training seed only; "
                       "no threshold fitting or checkpoint selection.")
    with (REPORT / "robustness_analysis.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({state: dict(sequences=row["sequences"], scored_frames=row["scored_frames"],
                                 means=row["means"], penalty_contrasts={
                                     contrast: {metric: {key: value for key, value in result.items()
                                                        if key in ["mean_delta_pp",
                                                                   "paired_sequence_bootstrap95_pp"]}
                                                for metric, result in metrics.items()}
                                     for contrast, metrics in row["penalty_contrasts"].items()})
                      for state, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
