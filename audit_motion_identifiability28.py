"""Verify whether state matching and fixed lag2 are separable in saved studies."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
STUDIES = {
    "adjacent": ROOT / "reports/server28-blind-motion-shortgap16-20261031",
    "availability": ROOT / "reports/server28-state-motion-shortgap16-20261031",
}
OUTPUT = ROOT / "reports/server28-motion-identifiability-20261031.json"


def read(path):
    return json.loads(path.read_text())


def center(box):
    return box[:2] + box[2:] / 2


def main():
    studies = {}
    for label, root in STUDIES.items():
        parent = read(root / "report.json")
        assert sha(root / "report.json") == read(root / "completed.json")["report_sha256"]
        intervals = Counter()
        trajectories = []
        max_reconstruction_error = 0.0
        for gpu in [0, 1]:
            folder = root / ("gpu%d" % gpu)
            worker = read(folder / "report.json")
            assert sha(folder / "report.json") == parent["worker_receipts"][str(gpu)]["report_sha256"]
            assert worker["launch"]["frame_block"] == 1
            for item in worker["launch"]["cohort"]:
                name = item["name"]
                prefix = folder / ("velocity_" + name + "_alternating_outage")
                receipt = worker["records"]["velocity"][name]["alternating_outage"]
                assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
                assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
                boxes = np.loadtxt(str(prefix) + ".txt", ndmin=2)
                trace = read(Path(str(prefix) + ".json"))
                available = np.asarray([[1, 1]] + [row["available"] for row in trace])
                assert len(available) == len(boxes) == item["video_frames"]
                count = 0
                for frame in np.flatnonzero(~available.any(1)):
                    observed = np.flatnonzero(available[:frame].any(1))
                    adjacent_before, last = observed[-2:]
                    same = observed[(available[observed] == available[last]).all(1)]
                    matched_before = same[-2]
                    assert last == frame - 1
                    assert adjacent_before == last - 1
                    assert matched_before == last - 2
                    assert available[adjacent_before].tolist() == [1, 0]
                    assert available[last].tolist() == available[matched_before].tolist() == [1, 1]
                    intervals["adjacent_dt1_cross_state"] += 1
                    intervals["matched_dt2_same_state"] += 1
                    lag2 = (center(boxes[last]) - center(boxes[last - 2])) / 2
                    matched = (center(boxes[last]) - center(boxes[matched_before])) / (last - matched_before)
                    assert np.array_equal(lag2, matched)
                    velocity = ((center(boxes[last]) - center(boxes[adjacent_before]))
                                if label == "adjacent" else lag2)
                    predicted = boxes[last].copy()
                    predicted[:2] += velocity
                    error = float(np.max(np.abs(predicted - boxes[frame])))
                    assert error <= 1e-9
                    max_reconstruction_error = max(max_reconstruction_error, error)
                    count += 1
                trajectories.append(dict(sequence=name, physical_gpu=gpu, blind_frames=count))
        studies[label] = dict(
            report_sha256=sha(root / "report.json"), sequences=len(trajectories),
            blind_frames=sum(row["blind_frames"] for row in trajectories),
            history_counts=dict(intervals), trajectories=trajectories,
            max_saved_forecast_reconstruction_error=max_reconstruction_error)
    result = dict(
        source_sha256=sha(Path(__file__)), studies=studies,
        matched_velocity_equals_fixed_lag2_on_every_blind_frame=True,
        conclusion="This schedule perfectly couples sensor-state matching to a two-frame "
                   "difference. Availability-history results cannot distinguish it from ordinary "
                   "lag2 velocity. Fresh videos under the same schedule do not remove this "
                   "identifiability problem.",
        next_evidence="Use equal temporal intervals with same/different state pairs, or a "
                      "counterbalanced schedule, before claiming a sensor-state mechanism.",
        scope="Read-only reconstruction of completed development tracks, including invalid-GT "
              "frames. No GT consulted, model calls, tuning, inference change or new metric gain.")
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
