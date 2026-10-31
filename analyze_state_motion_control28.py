"""Wait for and verify closed-loop availability-history versus adjacent CV."""
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from analyze_epoch05_health28 import compare
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
REFERENCE = ROOT / "reports/server28-blind-motion-shortgap16-20261031"
CANDIDATE = ROOT / "reports/server28-state-motion-shortgap16-20261031"


def read(path):
    return json.loads(path.read_text())


def main():
    launch = read(CANDIDATE / "launch.json")
    print("WAIT controller=" + str(launch["pid"]) + " poll_seconds=300", flush=True)
    while subprocess.call(["kill", "-0", str(launch["pid"])],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        time.sleep(300)
    records, groups, inventories, receipts = {}, {}, {}, {}
    for label, root in [("adjacent", REFERENCE), ("availability", CANDIDATE)]:
        parent = read(root / "report.json")
        completion = read(root / "completed.json")
        assert sha(root / "report.json") == completion["report_sha256"]
        assert completion["trajectories_verified"] == 64 and completion["sequences"] == 16
        receipts[label] = sha(root / "report.json")
        records[label], groups[label], inventories[label] = {}, {}, {}
        for gpu in [0, 1]:
            folder = root / ("gpu%d" % gpu)
            report = read(folder / "report.json")
            analysis = read(folder / "analysis.json")
            receipt = parent["worker_receipts"][str(gpu)]
            assert sha(folder / "report.json") == receipt["report_sha256"]
            assert sha(folder / "analysis.json") == receipt["analysis_sha256"]
            assert analysis["report_sha256"] == receipt["report_sha256"]
            assert analysis["trajectories_verified"] == 32
            assert report["launch"]["frame_block"] == 1
            if label == "availability":
                assert report["launch"]["velocity_history"] == "availability"
            for item in report["launch"]["cohort"]:
                name = item["name"]
                inventories[label][name] = dict(
                    item=item, gpu=gpu, checkpoint=report["launch"]["checkpoint_sha256"])
                records[label][name] = report["records"]["velocity"][name]["alternating_outage"]
                for arm, condition in [("hold", "complete"), ("velocity", "complete"),
                                       ("hold", "alternating_outage")]:
                    path = folder / (arm + "_" + name + "_" + condition + ".txt")
                    assert sha(path) == report["records"][arm][name][condition]["bbox_sha256"]
                    if label == "availability":
                        original = REFERENCE / ("gpu%d" % gpu) / path.name
                        assert np.array_equal(np.loadtxt(str(path)), np.loadtxt(str(original)))
            for group, values in analysis["per_sequence"].items():
                groups[label].setdefault(group, {}).update({
                    row["sequence"]: row["metrics"]["velocity"] for row in values})
    assert inventories["adjacent"] == inventories["availability"]
    names = sorted(records["adjacent"])
    assert len(names) == 16
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    contrasts = compare(records["availability"], records["adjacent"], names, indices)
    group_contrasts = {}
    for group in groups["adjacent"]:
        assert set(groups["adjacent"][group]) == set(groups["availability"][group])
        selected = sorted(groups["adjacent"][group])
        draws = np.random.RandomState(2027).randint(0, len(selected), (10000, len(selected)))
        group_contrasts[group] = compare(
            groups["availability"][group], groups["adjacent"][group], selected, draws)
    result = dict(
        report_sha256=receipts, analyzer_sha256=sha(Path(__file__)),
        full_outage_availability_minus_adjacent=contrasts, group_contrasts=group_contrasts,
        controls_equal=dict(complete_hold=True, complete_velocity=True, outage_hold=True),
        same_cohort_gpu_checkpoint=True, sequences=16,
        scope="Same16 development videos used in the preceding offline diagnostic; not fresh "
              "generalization evidence. Matching state also changes velocity time interval. "
              "No official score, fitted threshold or novel-module claim.")
    with (CANDIDATE / "comparison.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
