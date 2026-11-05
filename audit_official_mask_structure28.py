"""Describe published availability schedules without reading tracking outcomes."""
from collections import Counter
from itertools import groupby
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
INVENTORY = ROOT / "runtime/task_utility_20261018/reports/e0/inventory.json"
OUTPUT = ROOT / "reports/server28-official-mask-structure-20261031.json"


def describe(values):
    values = np.asarray(values)
    return dict(count=len(values), minimum=int(values.min()), median=float(np.median(values)),
                mean=float(values.mean()), p95=float(np.percentile(values, 95)),
                maximum=int(values.max()))


def main():
    inventory = json.loads(INVENTORY.read_text())
    results = {}
    for dataset, spec in inventory["datasets"].items():
        mask_path = Path(spec["mask"])
        assert sha(mask_path) == spec["mask_sha256"]
        masks = json.loads(mask_path.read_text())
        assert len(masks) == spec["sequences"]
        raw_counts, effective_counts, transitions = Counter(), Counter(), Counter()
        recovery, intervals, per_sequence = Counter(), [], []
        first_overrides = 0
        expected = {item["name"]: item["frames"] for item in spec["sequence_details"]}
        assert set(expected) == set(masks)
        for name in sorted(masks):
            values = np.asarray(masks[name]["data"])
            assert values.shape == (expected[name], 2) and np.isin(values, [0, 1]).all()
            labels = ["".join(str(int(value)) for value in row) for row in values]
            raw_counts.update(labels)
            first_overrides += int(labels[0] != "11")
            labels[0] = "11"
            effective_counts.update(labels)
            transitions.update(before + "->" + after for before, after in zip(labels, labels[1:]))
            sequence_intervals = []
            for state, grouped in groupby(enumerate(labels), key=lambda item: item[1]):
                positions = [index for index, _ in grouped]
                if state == "00":
                    start, end = positions[0], positions[-1] + 1
                    record = dict(sequence=name, start=start, end_exclusive=end,
                                  frames=end - start, previous_state=labels[start - 1],
                                  recovery_state=labels[end] if end < len(labels) else "video_end")
                    sequence_intervals.append(record)
                    recovery[record["recovery_state"]] += 1
            intervals.extend(sequence_intervals)
            per_sequence.append(dict(
                sequence=name, frames=len(labels), blind_frames=labels.count("00"),
                blind_intervals=len(sequence_intervals)))
        total = sum(effective_counts.values())
        assert total == sum(expected.values())
        results[dataset] = dict(
            mask_path=str(mask_path), mask_sha256=sha(mask_path), sequences=len(masks),
            total_frames=total, raw_state_counts=dict(raw_counts),
            evaluated_state_counts=dict(effective_counts), first_frame_overrides=first_overrides,
            blind_fraction=effective_counts["00"] / total,
            sequences_with_blind_frames=sum(row["blind_frames"] > 0 for row in per_sequence),
            blind_duration_frames=describe([row["frames"] for row in intervals]),
            recovery_state_counts=dict(recovery), transitions=dict(transitions),
            per_sequence=per_sequence, blind_intervals=intervals)
    result = dict(
        inventory_sha256=sha(INVENTORY), source_sha256=sha(Path(__file__)), datasets=results,
        scope="Availability schedules only; no images, boxes, ground truth or outcome scores read. "
              "First frame is forced complete to match evaluator initialization. "
              "No parameters selected and no official evaluation performance claim.")
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({name: {key: value for key, value in report.items()
                            if key not in ["per_sequence", "blind_intervals", "transitions"]}
                      for name, report in results.items()}, indent=2))


if __name__ == "__main__":
    main()
