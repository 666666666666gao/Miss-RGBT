"""Verify all trajectories and summarize paired sequence-level validation effects."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha


def metrics(predicted, gt):
    valid = (gt[:, 2:] > 0).all(1)
    valid[0] = False
    p, target = predicted[valid], gt[valid]
    error = np.linalg.norm(p[:, :2] + p[:, 2:] / 2 -
                           target[:, :2] - target[:, 2:] / 2, axis=1)
    extent = np.maximum(0., np.minimum(p[:, :2] + p[:, 2:], target[:, :2] + target[:, 2:]) -
                        np.maximum(p[:, :2], target[:, :2]))
    intersection = extent.prod(1)
    overlap = intersection / (p[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection)
    return dict(PR20=float((error <= 20).mean() * 100),
                SR=float((overlap[:, None] > np.linspace(0, 1, 21)).mean() * 100),
                scored_frames=int(valid.sum()))


def compare(first, second, names, indices):
    result = {}
    for metric in ["PR20", "SR"]:
        delta = np.asarray([first[name][metric] - second[name][metric] for name in names])
        interval = np.percentile(delta[indices].mean(1), [2.5, 97.5]).tolist()
        ordered = sorted(zip(names, delta.tolist()), key=lambda item: item[1])
        result[metric] = dict(
            mean_delta_pp=float(delta.mean()), paired_sequence_bootstrap95_pp=interval,
            wins=int((delta > 0).sum()), ties=int((delta == 0).sum()),
            losses=int((delta < 0).sum()),
            worst5=[dict(sequence=name, delta_pp=value) for name, value in ordered[:5]],
            best5=[dict(sequence=name, delta_pp=value) for name, value in reversed(ordered[-5:])])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path(
        "/home/gaob/Miss-RGBT/reports/server28-epoch05-tracking-health-20261031"))
    parser.add_argument("--protocol", type=Path, default=Path(
        "/home/gaob/Miss-RGBT/development/recovery_val97_20261030/protocol.json"))
    args = parser.parse_args()
    root = args.report_dir
    completed = json.loads((root / "completed.json").read_text())
    assert completed["official_test_used"] is False
    assert completed["checkpoint_selection_used"] is False
    assert len(completed["results"]) == 6
    expected = {(item["label"], item["condition"]): item for item in completed["results"]}
    assert len(expected) == 6
    protocol = json.loads(args.protocol.read_text())
    inventory = {item["name"]: item for item in protocol["sequences"]}
    names = sorted(inventory)
    assert len(names) == 97 and sum(item["frames"] for item in inventory.values()) == 54646
    gt = {}
    for name, item in inventory.items():
        assert sha(item["gt_path"]) == item["gt_sha256"], name
        gt[name] = np.loadtxt(item["gt_path"], delimiter=",", ndmin=2)
        assert gt[name].shape == (item["frames"], 4)
    cells = {}
    source_hashes = []
    checkpoint_hashes = {}
    for label in ["E0", "utility", "random"]:
        for condition in ["complete", "alternating_outage"]:
            destination = root / (label + "_" + condition)
            launch = json.loads((destination / "launch.json").read_text())
            actual = json.loads((destination / "completed.json").read_text())
            receipts = json.loads((destination / "receipts.json").read_text())
            assert actual == expected[(label, condition)]
            assert launch["protocol_sha256"] == sha(args.protocol)
            assert launch["mask_sha256"] == protocol["conditions"][condition]["sha256"]
            assert actual["sequences"] == 97 and actual["frames"] == 54646
            assert actual["checkpoint_sha256"] == launch["checkpoint_sha256"]
            checkpoint_hashes.setdefault(label, launch["checkpoint_sha256"])
            assert checkpoint_hashes[label] == launch["checkpoint_sha256"]
            source_hashes.append(launch["source_sha256"])
            by_name = {item["sequence"]: item for item in receipts}
            assert len(receipts) == len(by_name) == 97 and set(by_name) == set(names)
            verified = {}
            for name in names:
                item = by_name[name]
                path = destination / (name + ".txt")
                assert sha(path) == item["bbox_sha256"], (label, condition, name)
                boxes = np.loadtxt(str(path), ndmin=2)
                assert boxes.shape == gt[name].shape and np.isfinite(boxes).all()
                measured = metrics(boxes, gt[name])
                for key in ["PR20", "SR", "scored_frames"]:
                    assert np.isclose(measured[key], item[key], rtol=0, atol=1e-9), (
                        label, condition, name, key)
                assert item["video_frames"] == inventory[name]["frames"]
                verified[name] = measured
            for metric in ["PR20", "SR"]:
                assert np.isclose(np.mean([item[metric] for item in verified.values()]),
                                  actual[metric], rtol=0, atol=1e-9)
            assert sum(item["scored_frames"] for item in verified.values()) == actual["scored_frames"]
            cells[(label, condition)] = verified
    assert all(source == source_hashes[0] for source in source_hashes)
    # Resample paired videos, not correlated frames, using the same draws for every contrast.
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    contrasts = {}
    for condition in ["complete", "alternating_outage"]:
        contrasts[condition] = {}
        for first, second in [("utility", "E0"), ("random", "E0"), ("utility", "random")]:
            contrasts[condition][first + "_minus_" + second] = compare(
                cells[(first, condition)], cells[(second, condition)], names, indices)
    report = dict(
        scope="Full val97 development health at epoch5; not official scores or final checkpoint selection",
        checkpoint_sha256=checkpoint_hashes, contrasts=contrasts,
        aggregates=completed["results"], sequences=97, frames_per_condition=54646,
        trajectories_verified=582, bootstrap_draws=10000, bootstrap_seed=2027,
        uncertainty_scope="Paired sequence resampling for this one training seed; "
                          "does not estimate training-seed uncertainty or certify unseen E0 data",
        protocol_sha256=sha(args.protocol), analyzer_sha256=sha(Path(__file__)))
    with (root / "paired_analysis.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
