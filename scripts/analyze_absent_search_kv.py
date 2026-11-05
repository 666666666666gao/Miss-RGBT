"""Independently verify the saved absent-search K/V diagnostic evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/server28_absent_search_kv_20261102_v2"
PREVIOUS = [
    ("server28_fusion_adapter_pilot_20261102/report.json", "cache_manifests"),
    ("server28_head_supervision_20261102/report.json", "cache_manifests"),
    ("server28_modality_heads_20261102/report.json", "cache_manifests"),
    ("server28_response_geometry_20261102/report.json", "manifest"),
]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clustered(rows, key, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["identity"]["sequence"]].append(row[key])
    means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    draws = np.random.default_rng(seed).integers(
        0, len(means), size=(10000, len(means)))
    low, high = np.quantile(means[draws].mean(1), [.025, .975])
    return dict(mean=float(means.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(means), rows=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-source-hashes", action="store_true")
    args = parser.parse_args()
    report_path = REPORT / "report.json"
    report = json.loads(report_path.read_text())
    completed = json.loads((REPORT / "completed.json").read_text())
    assert completed["status"] == report["status"] == "complete"
    assert sha256(report_path) == completed["report_sha256"]
    assert report["seed"] == 43101 and report["advance"] is False
    assert report["baseline_state_unchanged"] and report["baseline_gradients_absent"]
    assert report["exact_previous_tuple_overlap"] is False
    assert len(report["sequence_names"]) == len(set(report["sequence_names"])) == 705
    assert len(report["rows"]) == 1152

    previous = set()
    for relative, key in PREVIOUS:
        record = json.loads((ROOT / "reports" / relative).read_text())
        manifests = record[key].values() if key == "cache_manifests" else [record[key]]
        for manifest in manifests:
            for item in manifest["input_manifest"]:
                identity = item["identity"]
                previous.add((identity["sequence"], tuple(identity["template_frames"]),
                              tuple(identity["search_frames"])))

    identities = defaultdict(set)
    grouped = defaultdict(list)
    for row in report["rows"]:
        identity = row["identity"]
        key = (identity["sequence"], tuple(identity["template_frames"]),
               tuple(identity["search_frames"]))
        assert key not in previous
        identities[key].add(row["cell"])
        grouped[row["cell"]].append(row)
        assert row["native_sham_exact"] is True
    assert len(identities) == 128
    assert all(len(cells) == 9 for cells in identities.values())
    assert set(grouped) == {dynamic + "_" + search
                            for dynamic in ("11", "01", "10")
                            for search in ("11", "01", "10")}
    for cell, rows in grouped.items():
        assert len(rows) == 128
        if cell.endswith("_11"):
            assert all(row["complete_equal"] for row in rows)

    expected = {}
    for cell, rows in grouped.items():
        expected[cell] = {
            "loss_delta": clustered(rows, "loss_delta", 43102),
            "iou_delta": clustered(rows, "iou_delta", 43103),
        }
    effects = {}
    for state in ("01", "10"):
        rows = [row for row in report["rows"] if row["cell"].endswith("_" + state)]
        effects[state] = clustered(rows, "iou_delta", 43104)
    assert expected == report["cells"]
    assert effects == report["effects"]
    assert all(value["mean"] < 0 and value["ci95_high"] < 0
               for value in effects.values())
    assert completed["advance"] is False

    if args.verify_source_hashes:
        for relative, expected_hash in report["source_sha256"].items():
            relative = relative.replace("/home/gaob/Miss-RGBT/", "")
            assert sha256(ROOT / relative) == expected_hash, relative

    print(json.dumps(dict(
        status="verified",
        tuples=len(identities),
        sequences=report["unique_sequences"],
        rows=len(report["rows"]),
        native_sham_exact_rows=sum(row["native_sham_exact"] for row in report["rows"]),
        complete_rows=sum(row["complete_equal"] for row in report["rows"]),
        effects=effects,
        advance=False,
        report_sha256=sha256(report_path),
    ), indent=2))


if __name__ == "__main__":
    main()
