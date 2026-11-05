"""Verify arithmetic and pairing in the saved burst-mask rollout report."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 61101


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(float(row["difference"]))
    sequences = sorted(grouped)
    means = np.asarray([np.mean(grouped[name]) for name in sequences])
    rng = np.random.default_rng(SEED + 1)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    sampled = means[draws].mean(axis=1)
    return dict(mean=float(means.mean()),
        ci95_low=float(np.quantile(sampled, 0.025)),
        ci95_high=float(np.quantile(sampled, 0.975)),
        paired_sequences=len(means))


def box_iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * \
        max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--verify-source-hashes", action="store_true")
    args = parser.parse_args()
    root = Path(args.directory)
    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    completed = json.loads((root / "completed.json").read_text())
    assert report["status"] == completed["status"] == "complete"
    assert sha256(report_path) == completed["report_sha256"]
    assert report["state_unchanged"] and report["gradients_absent"]
    assert len(report["clips"]) == 24
    assert len(report["rows"]) == 24 * 63 * 2

    rows = defaultdict(list)
    for row in report["rows"]:
        rows[(row["sequence"], row["policy"])].append(row)
        assert row["state"] in ("11", "01", "10")
        assert 0.0 <= row["iou"] <= 1.0
    assert len(rows) == 48
    for clip in report["clips"]:
        sequence = clip["sequence"]
        iid = rows[(sequence, "iid")]
        burst = rows[(sequence, "clustered")]
        assert len(iid) == len(burst) == 63
        assert clip["state_counts"] == {
            state: clip["iid_schedule"].count(state) for state in ("11", "01", "10")}
        assert clip["state_counts"] == {
            state: clip["clustered_schedule"].count(state) for state in ("11", "01", "10")}
        assert clip["iid_schedule"][0] == clip["clustered_schedule"][0] == "11"
        assert clip["iid_schedule"].count("01") == clip["clustered_schedule"].count("01")
        assert clip["iid_schedule"].count("10") == clip["clustered_schedule"].count("10")
        for policy, schedule in (("iid", clip["iid_schedule"]),
                                  ("clustered", clip["clustered_schedule"])):
            selected = sorted(rows[(sequence, policy)], key=lambda row: row["frame"])
            previous_missing = False
            recovery_left = 0
            for frame, row in enumerate(selected, 1):
                state = schedule[frame]
                missing = state != "11"
                if previous_missing and not missing:
                    recovery_left = 5
                expected_recovery = state == "11" and recovery_left > 0
                if expected_recovery:
                    recovery_left -= 1
                assert row["frame"] == frame and row["state"] == state
                assert row["recovery"] == expected_recovery
                assert np.isclose(row["iou"], box_iou(row["bbox"], row["gt_bbox"]),
                                  atol=1e-12)
                assert np.isfinite(row["confidence"])
                assert all(np.isfinite(value) for value in row["bbox"])
                previous_missing = missing

    predicates = {
        "all_missing": lambda row: row["state"] in ("01", "10"),
        "rgb_only": lambda row: row["state"] == "10",
        "tir_only": lambda row: row["state"] == "01",
        "recovery": lambda row: row["recovery"],
    }
    expected = {}
    for category, predicate in predicates.items():
        paired = []
        for clip in report["clips"]:
            sequence = clip["sequence"]
            selected_iid = [row["iou"] for row in rows[(sequence, "iid")] if predicate(row)]
            selected_burst = [row["iou"] for row in rows[(sequence, "clustered")] if predicate(row)]
            if selected_iid and selected_burst:
                paired.append(dict(sequence=sequence,
                    difference=float(np.mean(selected_burst) - np.mean(selected_iid))))
        expected[category] = summary(paired)
        stored = report["effects"][category]
        assert np.isclose(expected[category]["mean"], stored["mean"], atol=1e-12)
        assert np.isclose(expected[category]["ci95_low"], stored["ci95_low"], atol=1e-12)
        assert np.isclose(expected[category]["ci95_high"], stored["ci95_high"], atol=1e-12)
        assert expected[category]["paired_sequences"] == stored["sequences"]

    missing = report["effects"]["all_missing"]
    recovery = report["effects"]["recovery"]
    gate = ((missing["mean"] <= -0.01 and missing["ci95_high"] < 0)
            or (expected["recovery"]["paired_sequences"] >= 24
                and recovery["mean"] <= -0.01
                and recovery["ci95_high"] < 0))
    assert bool(gate) == report["advance"] == completed["advance"]

    if args.verify_source_hashes:
        for path, expected_hash in report["source_sha256"].items():
            assert sha256(path) == expected_hash, path
        assert sha256(report["checkpoint"]) == report["checkpoint_sha256"]

    effects = {category: dict(mean=stats["mean"],
        ci95_low=stats["ci95_low"], ci95_high=stats["ci95_high"],
        paired_sequences=stats["paired_sequences"])
        for category, stats in expected.items()}
    result = dict(status="saved_arithmetic_and_bootstrap_verified",
        source_hashes_checked=args.verify_source_hashes,
        sequences=len(report["clips"]), rows=len(report["rows"]),
        effects=effects, advance=report["advance"],
        report_sha256=completed["report_sha256"])
    with (root / "analysis_recomputed.json").open("x", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
