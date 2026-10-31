"""Compare calibration ages under published masks and the development schedule."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/home/gaob/Miss-RGBT")
INVENTORY = ROOT / "runtime/task_utility_20261018/reports/e0/inventory.json"
OUTPUT = ROOT / "reports/server28-calibration-age-20261031.json"
PATTERN = np.asarray([[1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    result = {"count": len(values)}
    if values:
        values = np.asarray(values)
        result.update(
            mean=float(values.mean()),
            percentiles=dict(zip(["p50", "p90", "p95", "p99"],
                                 np.percentile(values, [50, 90, 95, 99]).tolist())),
            maximum=int(values.max()),
            greater_than_one=int((values > 1).sum()),
            greater_than_one_fraction=float((values > 1).mean()))
    return result


def scan(mask):
    counts = {state: 0 for state in ["11", "01", "10", "00"]}
    ages = {state: [] for state in ["01", "10"]}
    call_ages = {state: [] for state in ["01", "10"]}
    uncalibrated = {state: 0 for state in ["01", "10"]}
    call, last_frame, last_call = 0, None, None
    for frame, flags in enumerate(mask[1:], start=1):
        state = "".join(str(int(value)) for value in flags)
        counts[state] += 1
        if state == "00":
            continue
        call += 1
        if state == "11":
            last_frame, last_call = frame, call
        elif last_frame is None:
            uncalibrated[state] += 1
        else:
            ages[state].append(frame - last_frame)
            call_ages[state].append(call - last_call)
    assert sum(counts.values()) == len(mask) - 1
    for state in ages:
        assert len(ages[state]) + uncalibrated[state] == counts[state]
    return counts, ages, call_ages, uncalibrated


def main():
    inventory = json.loads(INVENTORY.read_text())
    datasets = {}
    inputs = {}
    for name, spec in inventory["datasets"].items():
        assert sha(spec["mask"]) == spec["mask_sha256"]
        masks = json.loads(Path(spec["mask"]).read_text())
        assert len(masks) == spec["sequences"]
        inputs[name] = dict(
            mask=spec["mask"], sha256=spec["mask_sha256"],
            stored_first_frame_states=dict(Counter(
                "".join(str(int(value)) for value in item["data"][0])
                for item in masks.values())))
        schedules = {}
        for schedule in ["published", "development_block1"]:
            counts = {state: 0 for state in ["11", "01", "10", "00"]}
            ages = {state: [] for state in ["01", "10"]}
            call_ages = {state: [] for state in ["01", "10"]}
            uncalibrated = {state: 0 for state in ["01", "10"]}
            per_video = {}
            for item in spec["sequence_details"]:
                sequence, frames = item["name"], item["frames"]
                mask = np.asarray(masks[sequence]["data"], dtype=np.uint8)
                assert mask.shape == (frames, 2) and np.isin(mask, [0, 1]).all()
                if schedule == "development_block1":
                    mask = PATTERN[np.arange(frames) % len(PATTERN)]
                vc, va, vca, vu = scan(mask)
                for state in counts:
                    counts[state] += vc[state]
                for state in ages:
                    ages[state].extend(va[state])
                    call_ages[state].extend(vca[state])
                    uncalibrated[state] += vu[state]
                per_video[sequence] = dict(
                    state_counts=vc, uncalibrated_single_frames=vu,
                    calibrated_age_gt1={state: sum(age > 1 for age in va[state])
                                        for state in va})
            schedules[schedule] = dict(
                state_counts=counts, video_frame_age={s: summarize(v) for s, v in ages.items()},
                observed_call_age={s: summarize(v) for s, v in call_ages.items()},
                uncalibrated_single_frames=uncalibrated, per_video=per_video)
        datasets[name] = schedules
    result = dict(
        inventory_sha256=sha(INVENTORY), masks=inputs, source_sha256=sha(__file__),
        datasets=datasets, new_model_calls=0, fitted_parameters=0,
        protocol="Initialization frame0 is complete but does not calibrate the actual tracker. "
                 "A later11 call refreshes calibration;00 frames advance video time but not "
                 "tracker calls. Ages are measured only on single-modality frames after refresh. "
                 "Synthetic block1 uses identical official sequence lengths to isolate schedule. "
                 "Distribution-only diagnosis; no GT, inference, threshold tuning or efficacy claim.")
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({name: {schedule: {key: value for key, value in data.items()
                                       if key != "per_video"}
                            for schedule, data in schedules.items()}
                      for name, schedules in datasets.items()}, indent=2))
    print("REPORT_SHA256 " + sha(OUTPUT))


if __name__ == "__main__":
    main()
