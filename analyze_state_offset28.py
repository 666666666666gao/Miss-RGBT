"""Independently verify offset-control trajectories and past-only corrections."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from analyze_epoch05_health28 import compare, metrics
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-state-offset-control4-20261031"
sys.path.insert(0, str(ROOT / "runtime/task_utility_20261018"))
from lib.utils.box_ops import clip_box


def read(path):
    return json.loads(path.read_text())


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output
    launch = read(OUTPUT / "launch.json")
    assert launch["registration_sha256"] == sha(OUTPUT / "registration.json")
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected, path
    all_records, receipts = {}, {}
    total = 0
    for gpu in [0, 1]:
        folder = OUTPUT / ("gpu%d" % gpu)
        report = read(folder / "report.json")
        assert read(folder / "completed.json")["report_sha256"] == sha(folder / "report.json")
        assert report["baseline_state_unchanged"]
        assert report["sham_outputs_bitwise_equal"] and report["complete_outputs_bitwise_equal"]
        assert report["launch"]["cohort"] == launch["parts"][str(gpu)]
        assert report["launch"]["spec_sha256"] == sha(OUTPUT / ("gpu%d_spec.json" % gpu))
        spec = read(OUTPUT / ("gpu%d_spec.json" % gpu))
        assert spec["registration_sha256"] == launch["registration_sha256"]
        receipts[str(gpu)] = dict(report_sha256=sha(folder / "report.json"))
        for item in report["launch"]["cohort"]:
            name, video = item["name"], Path(item["folder"])
            assert sha(video / "init.txt") == item["gt_sha256"]
            gt = np.loadtxt(str(video / "init.txt"), delimiter=",", ndmin=2)
            all_records[name] = {}
            references = {}
            for arm in ["native", "calibration_sham", "offset_corrected"]:
                all_records[name][arm] = {}
                for condition in ["complete", "alternating_outage"]:
                    prefix = folder / (arm + "_" + name + "_" + condition)
                    receipt = report["records"][arm][name][condition]
                    assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
                    assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
                    boxes = np.loadtxt(str(prefix) + ".txt", ndmin=2)
                    trace = read(Path(str(prefix) + ".json"))
                    assert boxes.shape == gt.shape and len(trace) == len(gt) - 1
                    assert np.isfinite(boxes).all()
                    measured = metrics(boxes, gt)
                    for key, value in measured.items():
                        assert np.isclose(value, receipt[key], rtol=0, atol=1e-10)
                    if arm == "native":
                        references[condition] = boxes
                    elif condition == "complete" or arm == "calibration_sham":
                        assert np.array_equal(boxes, references[condition])
                    offsets = {"01": np.zeros(2), "10": np.zeros(2)}
                    model_calls = 0
                    tracker_calls = 0
                    last_calibration = None
                    for frame, row in enumerate(trace, start=1):
                        expected = ([1, 1] if condition == "complete" else
                                    report["launch"]["pattern"][frame % 6])
                        assert row["frame"] == frame and row["available"] == expected
                        assert np.array_equal(boxes[frame], np.asarray(row["bbox"]))
                        if not any(expected):
                            assert np.array_equal(boxes[frame], boxes[frame - 1])
                            assert not row["template_written"]
                            continue
                        tracker_calls += 1
                        model_calls += (1 if arm == "native" else 3 if expected == [1, 1] else 1)
                        if arm == "native":
                            continue
                        state = "".join(map(str, expected))
                        if state == "11":
                            assert set(row["calibration_offsets"]) == {"01", "10"}
                            offsets = {key: np.asarray(value)
                                       for key, value in row["calibration_offsets"].items()}
                            last_calibration = tracker_calls
                        applied = (offsets[state] if arm == "offset_corrected" and state != "11"
                                   else np.zeros(2))
                        assert np.array_equal(np.asarray(row["applied_offset"]), applied)
                        assert row["last_calibration_call"] == last_calibration
                        raw = list(row["unclipped_bbox"])
                        raw[0] -= float(applied[0])
                        raw[1] -= float(applied[1])
                        h, w = row["image_shape"]
                        assert np.array_equal(np.asarray(clip_box(raw, h, w, margin=10)), boxes[frame])
                    assert model_calls == receipt["model_calls"]
                    all_records[name][arm][condition] = measured
                    total += 1
    names = sorted(all_records)
    assert set(names) == {item["name"] for item in launch["cohort"]}
    assert total == len(names) * 6
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    aggregate, contrasts = {}, {}
    for condition in ["complete", "alternating_outage"]:
        values = {arm: {name: all_records[name][arm][condition] for name in names}
                  for arm in ["native", "calibration_sham", "offset_corrected"]}
        aggregate[condition] = {
            arm: {metric: float(np.mean([values[arm][name][metric] for name in names]))
                  for metric in ["PR20", "SR"]} for arm in values}
        contrasts[condition] = compare(values["offset_corrected"], values["native"], names, indices)
    result = dict(
        launch=launch, aggregate=aggregate, paired_contrasts=contrasts, per_video=all_records,
        worker_receipts=receipts, trajectories_verified=total,
        sham_and_complete_bitwise_equal=True, past_only_offsets_verified=True,
        analyzer_sha256=sha(Path(__file__)),
        scope="Fresh full development videos, parameter-free offset calibration control; "
              "small paired-video uncertainty, no official result or novelty claim.")
    with (OUTPUT / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    with (OUTPUT / "completed.json").open("x") as stream:
        json.dump(dict(report_sha256=sha(OUTPUT / "report.json"), trajectories_verified=total), stream)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
