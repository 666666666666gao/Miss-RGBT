"""Verify the three completed full-input validation cells without selecting a weight."""
import json
from pathlib import Path

import numpy as np

from analyze_epoch05_health28 import compare, metrics
from evaluate_val97_recovery28 import sha

ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-epoch05-tracking-health-20261031"
PROTOCOL = ROOT / "development/recovery_val97_20261030/protocol.json"


def main():
    protocol = json.loads(PROTOCOL.read_text())
    inventory = {item["name"]: item for item in protocol["sequences"]}
    names = sorted(inventory)
    assert len(names) == 97
    gt = {}
    for name, item in inventory.items():
        assert sha(item["gt_path"]) == item["gt_sha256"]
        gt[name] = np.loadtxt(item["gt_path"], delimiter=",", ndmin=2)
        assert gt[name].shape == (item["frames"], 4)
    cells, aggregates, sources, checkpoint_hashes = {}, {}, [], {}
    for label in ["E0", "utility", "random"]:
        folder = OUTPUT / (label + "_complete")
        launch = json.loads((folder / "launch.json").read_text())
        completed = json.loads((folder / "completed.json").read_text())
        receipts = json.loads((folder / "receipts.json").read_text())
        by_name = {row["sequence"]: row for row in receipts}
        assert len(receipts) == len(by_name) == 97 and set(by_name) == set(names)
        assert launch["condition"] == completed["condition"] == "complete"
        assert launch["protocol_sha256"] == sha(PROTOCOL)
        assert launch["mask_sha256"] == protocol["conditions"]["complete"]["sha256"]
        assert completed["sequences"] == 97 and completed["frames"] == 54646
        assert completed["checkpoint_sha256"] == launch["checkpoint_sha256"]
        checkpoint_hashes[label] = launch["checkpoint_sha256"]
        sources.append(launch["source_sha256"])
        verified = {}
        for name in names:
            path = folder / (name + ".txt")
            receipt = by_name[name]
            assert sha(path) == receipt["bbox_sha256"]
            boxes = np.loadtxt(str(path), ndmin=2)
            assert boxes.shape == gt[name].shape and np.isfinite(boxes).all()
            actual = metrics(boxes, gt[name])
            assert all(np.isclose(actual[key], receipt[key], rtol=0, atol=1e-9)
                       for key in actual)
            assert receipt["video_frames"] == inventory[name]["frames"]
            verified[name] = actual
        assert sum(row["video_frames"] for row in receipts) == 54646
        assert sum(row["scored_frames"] for row in verified.values()) == completed["scored_frames"] == 54289
        for key in ["PR20", "SR"]:
            assert np.isclose(np.mean([row[key] for row in verified.values()]),
                              completed[key], rtol=0, atol=1e-9)
        cells[label], aggregates[label] = verified, completed
    assert all(source == sources[0] for source in sources)
    indices = np.random.RandomState(2027).randint(0, 97, (10000, 97))
    contrasts = {first + "_minus_" + second: compare(cells[first], cells[second], names, indices)
                 for first, second in [("utility", "E0"), ("random", "E0"), ("utility", "random")]}
    result = dict(
        scope="Full-input val97 epoch5 health only; no weight selection or official test",
        trajectories_verified=291, aggregates=aggregates, contrasts=contrasts,
        checkpoint_sha256=checkpoint_hashes, source_sha256=sha(Path(__file__)),
        helper_sha256=sha(ROOT / "analyze_epoch05_health28.py"),
        uncertainty="10000 paired-video bootstrap draws, seed2027; one training seed only",
        missing_condition="Alternating-outage candidate tracking still requires separate completion")
    with (OUTPUT / "complete_paired_analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(aggregates=aggregates, contrasts=contrasts), indent=2))


if __name__ == "__main__":
    main()
