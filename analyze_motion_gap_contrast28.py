"""Compare verified fixed-CV effects under one-frame and32-frame dev gaps."""
import json
from pathlib import Path

import numpy as np

from analyze_epoch05_health28 import compare
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
FOLDERS = {
    32: ROOT / "reports/server28-blind-motion-next16-20261031",
    1: ROOT / "reports/server28-blind-motion-shortgap16-20261031",
}


def read(path):
    return json.loads(path.read_text())


def main():
    reports, effects, complete, receipts, cohorts = {}, {}, {}, {}, {}
    for duration, folder in FOLDERS.items():
        report = read(folder / "report.json")
        completion = read(folder / "completed.json")
        assert completion["report_sha256"] == sha(folder / "report.json")
        assert completion["trajectories_verified"] == 64 and completion["sequences"] == 16
        reports[duration] = report
        receipts[str(duration)] = sha(folder / "report.json")
        effects[duration], complete[duration], cohorts[duration] = {}, {}, {}
        for gpu in [0, 1]:
            worker_folder = folder / ("gpu%d" % gpu)
            worker = read(worker_folder / "report.json")
            analysis = read(worker_folder / "analysis.json")
            receipt = report["worker_receipts"][str(gpu)]
            assert receipt["report_sha256"] == sha(worker_folder / "report.json")
            assert receipt["analysis_sha256"] == sha(worker_folder / "analysis.json")
            assert analysis["report_sha256"] == receipt["report_sha256"]
            assert analysis["trajectories_verified"] == 32
            assert worker["launch"]["frame_block"] == duration
            assert worker["launch"]["physical_gpu"] == gpu
            for item in worker["launch"]["cohort"]:
                name = item["name"]
                assert name not in effects[duration]
                cohorts[duration][name] = dict(item=item, gpu=gpu,
                                               checkpoint=worker["launch"]["checkpoint_sha256"])
                records = worker["records"]
                effects[duration][name] = {
                    metric: records["velocity"][name]["alternating_outage"][metric]
                            - records["hold"][name]["alternating_outage"][metric]
                    for metric in ["PR20", "SR"]}
                complete[duration][name] = {}
                for arm in ["hold", "velocity"]:
                    path = worker_folder / (arm + "_" + name + "_complete.txt")
                    assert sha(path) == records[arm][name]["complete"]["bbox_sha256"]
                    complete[duration][name][arm] = np.loadtxt(str(path), ndmin=2)
        assert len(effects[duration]) == 16
    assert cohorts[1] == cohorts[32]
    names = sorted(effects[1])
    for name in names:
        for arm in ["hold", "velocity"]:
            assert np.array_equal(complete[1][name][arm], complete[32][name][arm])
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    result = dict(
        report_sha256=receipts, analyzer_sha256=sha(Path(__file__)),
        sequences=16, complete_outputs_equal_across_durations=True,
        per_sequence_effects=effects,
        one_frame_effect_minus_32_frame_effect=compare(effects[1], effects[32], names, indices),
        raw_aggregates={str(duration): report["aggregate"] for duration, report in reports.items()},
        scope="Paired development-video contrast of fixed CV effects; distinct outage schedules "
              "change both gap duration and switching frequency. No official performance claim, "
              "parameter selection, training-seed uncertainty or novel module.")
    path = ROOT / "reports/server28-motion-gap-contrast-20261031.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
