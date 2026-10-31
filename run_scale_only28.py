"""Prospective scale-only component control on fresh development videos."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from analyze_epoch05_health28 import compare, metrics
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
OUTPUT = ROOT / "reports/server28-scale-only4-20261031"
PREVIOUS = ROOT / "reports/server28-calibration-gap4-20261031-recovered"
WITNESS = ROOT / "reports/server28-scale-only-entry-20261031.json"
PATTERN = [[1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]]
ARMS = ["native", "sham", "scale", "geometry"]
BLOCKS = {"complete": 0, "alternating_outage": 1, "block32_outage": 32,
          "block128_outage": 128}
CONDITIONS = ["complete", "block32_outage", "block128_outage"]


def flags_for(frame, condition):
    block = BLOCKS[condition]
    return [1, 1] if block == 0 else PATTERN[(frame // block) % len(PATTERN)]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def runtime(cpu=False):
    os.environ.update(OMP_NUM_THREADS="4" if cpu else "1",
                      MKL_NUM_THREADS="4" if cpu else "1", OPENBLAS_NUM_THREADS="1")
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    import torch
    torch.set_num_threads(4 if cpu else 1)
    cv2.setNumThreads(1)
    sys.path.insert(0, str(RUNTIME))
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_x_frame
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    return cfg, get_x_frame


def images(folder):
    return sorted((folder / "visible").glob("*.jpg")), sorted((folder / "infrared").glob("*.jpg"))


def observed_images(rgb, tir, frame, flags, get_frame):
    r, t = (rgb[frame] if flags[0] else tir[frame], tir[frame] if flags[1] else rgb[frame])
    image = get_frame(str(r), str(t), dtype="rgbrgb")
    calibration = ({
        (0, 1): get_frame(str(tir[frame]), str(tir[frame]), dtype="rgbrgb"),
        (1, 0): get_frame(str(rgb[frame]), str(rgb[frame]), dtype="rgbrgb"),
    } if flags == [1, 1] else {})
    return image, calibration


def witness():
    cfg, get_frame = runtime(cpu=True)
    from cpu_diagnostic_tracker import CpuDiagnosticTracker
    from evaluate_blind_motion28 import network_digest
    from spatial_calibration_tracker28 import SpatialCalibrationTracker
    tracker = CpuDiagnosticTracker(cfg, str(ROOT / "checkpoints/e0_availability/epoch20.pth.tar"))
    tracker.network.requires_grad_(False)
    digest = network_digest(tracker.network)
    manifest = read(ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json")
    folder = Path("/data/wangwj/dataset/LasHeR/trainingset") / manifest["development_sequences"][36]
    gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
    rgb, tir = images(folder)
    records = {}
    nonidentity_scale_frames = 0
    for condition in ["complete", "alternating_outage"]:
        tracker.initialize(get_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                           dict(init_bbox=gt[0].tolist()))
        candidates = {}
        for arm in ["sham", "scale", "geometry"]:
            cls = SpatialCalibrationTracker
            candidate = cls.__new__(cls)
            candidate.__dict__.update(tracker.__dict__)
            candidate.templates = list(tracker.templates)
            candidate.availability = list(tracker.availability)
            candidate.state = list(tracker.state)
            candidate.apply_offset = arm == "geometry"
            candidate.apply_scale = arm in ["scale", "geometry"]
            candidate.offsets = {(0, 1): np.zeros(2), (1, 0): np.zeros(2)}
            candidate.scales = {(0, 1): np.ones(2), (1, 0): np.ones(2)}
            candidate.calibration_sizes = {}
            candidate.calibrating = False
            candidate.last_calibration_frame = None
            candidates[arm] = candidate
        rows = []
        for frame in range(1, 8):
            flags = flags_for(frame, condition)
            if not any(flags):
                continue
            image, calibration = observed_images(rgb, tir, frame, flags, get_frame)
            baseline = tracker.track(image, flags)["target_bbox"]
            actual = {arm: candidate.track(image, flags, calibration)
                      for arm, candidate in candidates.items()}
            assert np.array_equal(baseline, actual["sham"]["target_bbox"])
            assert np.array_equal(actual["scale"]["applied_offset"], np.zeros(2))
            if flags != [1, 1]:
                value = actual["scale"]
                raw = np.asarray(value["before_scale_bbox"])
                unclipped = np.asarray(value["unclipped_target_bbox"])
                assert np.allclose(raw[:2] + raw[2:] / 2,
                                   unclipped[:2] + unclipped[2:] / 2, rtol=0, atol=1e-9)
                nonidentity_scale_frames += int(not np.array_equal(value["applied_scale"], np.ones(2)))
            if condition == "complete":
                assert all(np.array_equal(baseline, value["target_bbox"]) for value in actual.values())
            assert all(np.isfinite(value["target_bbox"]).all() for value in actual.values())
            rows.append(dict(frame=frame, available=flags, native_bbox=baseline, predictions=actual))
        records[condition] = rows
    assert network_digest(tracker.network) == digest
    assert nonidentity_scale_frames > 0
    save(WITNESS, dict(
        records=records, sham_equal=True, complete_equal=True, scale_offset_zero=True,
        nonidentity_scale_frames=nonidentity_scale_frames,
        baseline_state_unchanged=True,
        source_sha256={str(path): sha(path) for path in [
            Path(__file__), ROOT / "spatial_calibration_tracker28.py",
            ROOT / "state_offset_tracker28.py", ROOT / "cpu_diagnostic_tracker.py"]},
        scope="Seven-frame CPU track-body witness with shared frozen E0 and initialized CPU "
              "attributes; not CUDA constructor/initialization or full-trajectory verification."))


def worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg, get_frame = runtime()
    import torch
    from evaluate_blind_motion28 import network_digest
    from miss_rgbt.tracker import MissingTracker
    from spatial_calibration_tracker28 import SpatialCalibrationTracker
    launch = read(OUTPUT / "launch.json")
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    spec = read(OUTPUT / ("gpu%s_spec.json" % args.gpu))
    assert spec["registration_sha256"] == launch["registration_sha256"]
    destination = OUTPUT / ("gpu" + args.gpu)
    destination.mkdir(exist_ok=False)
    checkpoint = ROOT / "checkpoints/e0_availability/epoch20.pth.tar"
    package = read(checkpoint.with_suffix(".json"))
    assert sha(checkpoint) == package["packaged_sha256"]
    assert package["optimizer_updates"] == 0 and package["tensor_equality"]
    save(destination / "launch.json", dict(
        pid=os.getpid(), physical_gpu=int(args.gpu), checkpoint_sha256=sha(checkpoint),
        registration_sha256=launch["registration_sha256"], cohort=spec["cohort"]))
    records, references = {}, {}
    started = time.time()
    for arm in ARMS:
        tracker = (MissingTracker(cfg, str(checkpoint)) if arm == "native" else
                   SpatialCalibrationTracker(cfg, str(checkpoint),
                                             apply_offset=arm == "geometry",
                                             apply_scale=arm in ["scale", "geometry"]))
        tracker.network.requires_grad_(False)
        digest = network_digest(tracker.network)
        records[arm] = {}
        for item in spec["cohort"]:
            name, folder = item["name"], Path(item["folder"])
            assert sha(folder / "init.txt") == item["gt_sha256"]
            gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
            rgb, tir = images(folder)
            assert len(rgb) == len(tir) == len(gt) == item["video_frames"]
            records[arm][name] = {}
            for condition in CONDITIONS:
                tracker.initialize(get_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                                   dict(init_bbox=gt[0].tolist()))
                boxes, trace, calls = [list(tracker.state)], [], 0
                for frame in range(1, len(gt)):
                    flags = flags_for(frame, condition)
                    templates = list(tracker.templates)
                    previous_call = tracker.frame_id
                    result = {}
                    shape = None
                    if any(flags):
                        image, calibration = observed_images(rgb, tir, frame, flags, get_frame)
                        shape = list(image.shape[:2])
                        result = (tracker.track(image, flags) if arm == "native" else
                                  tracker.track(image, flags, calibration))
                        calls += 1 if arm == "native" else result["model_calls"]
                        assert tracker.frame_id == previous_call + 1
                    else:
                        assert tracker.frame_id == previous_call
                        assert all(a is b for a, b in zip(templates, tracker.templates))
                    box = list(tracker.state)
                    assert np.isfinite(box).all() and (np.asarray(box)[2:] > 0).all()
                    boxes.append(box)
                    trace.append(dict(frame=frame, available=flags, bbox=box, prediction=result,
                                      image_shape=shape,
                                      template_written=any(a is not b for a, b in zip(templates, tracker.templates))))
                boxes = np.asarray(boxes)
                if arm == "native":
                    references[(name, condition)] = boxes.copy()
                elif arm == "sham" or condition == "complete":
                    assert np.array_equal(boxes, references[(name, condition)])
                prefix = destination / (arm + "_" + name + "_" + condition)
                np.savetxt(str(prefix) + ".txt", boxes)
                save(Path(str(prefix) + ".json"), trace)
                measured = metrics(boxes, gt)
                measured.update(bbox_sha256=sha(str(prefix) + ".txt"),
                                trace_sha256=sha(str(prefix) + ".json"), model_calls=calls)
                records[arm][name][condition] = measured
                print("TRACK " + json.dumps(dict(arm=arm, sequence=name, condition=condition,
                                                metrics=measured)), flush=True)
        assert network_digest(tracker.network) == digest
        del tracker
        torch.cuda.empty_cache()
    save(destination / "report.json", dict(
        records=records, cohort=spec["cohort"], seconds=time.time() - started,
        baseline_state_unchanged=True, controls_bitwise_equal=True))
    save(destination / "completed.json", dict(report_sha256=sha(destination / "report.json")))


def analyze():
    sys.path.insert(0, str(RUNTIME))
    from lib.utils.box_ops import clip_box
    launch = read(OUTPUT / "launch.json")
    assert sha(OUTPUT / "registration.json") == launch["registration_sha256"]
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    all_records, worker_receipts, calibration_ages = {}, {}, {}
    trajectories = 0
    for gpu, cohort in launch["parts"].items():
        folder = OUTPUT / ("gpu" + gpu)
        report = read(folder / "report.json")
        worker_launch = read(folder / "launch.json")
        assert worker_launch["registration_sha256"] == launch["registration_sha256"]
        assert report["cohort"] == worker_launch["cohort"] == cohort
        assert worker_launch["physical_gpu"] == int(gpu)
        assert report["baseline_state_unchanged"] and report["controls_bitwise_equal"]
        assert sha(folder / "report.json") == read(folder / "completed.json")["report_sha256"]
        worker_receipts[gpu] = dict(report_sha256=sha(folder / "report.json"),
                                    checkpoint_sha256=worker_launch["checkpoint_sha256"])
        for item in cohort:
            name, gt_path = item["name"], Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            all_records[name], references = {}, {}
            for arm in ARMS:
                all_records[name][arm] = {}
                for condition in CONDITIONS:
                    prefix = folder / (arm + "_" + name + "_" + condition)
                    receipt = report["records"][arm][name][condition]
                    assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
                    assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
                    boxes, trace = np.loadtxt(str(prefix) + ".txt", ndmin=2), read(Path(str(prefix) + ".json"))
                    assert boxes.shape == gt.shape and len(trace) == len(gt) - 1
                    measured = metrics(boxes, gt)
                    assert all(np.isclose(value, receipt[key], rtol=0, atol=1e-10)
                               for key, value in measured.items())
                    if arm == "native":
                        references[condition] = boxes
                    elif arm == "sham" or condition == "complete":
                        assert np.array_equal(boxes, references[condition])
                    offsets, scales = {"01": np.zeros(2), "10": np.zeros(2)}, {
                        "01": np.ones(2), "10": np.ones(2)}
                    calls = tracker_calls = 0
                    last_calibration = None
                    last_video_calibration, age_samples, uncalibrated = None, [], 0
                    for frame, row in enumerate(trace, start=1):
                        flags = flags_for(frame, condition)
                        assert row["frame"] == frame and row["available"] == flags
                        assert np.array_equal(row["bbox"], boxes[frame])
                        if not any(flags):
                            assert np.array_equal(boxes[frame], boxes[frame - 1])
                            assert not row["template_written"] and not row["prediction"]
                            continue
                        tracker_calls += 1
                        calls += 1 if arm == "native" else 3 if flags == [1, 1] else 1
                        if flags == [1, 1]:
                            last_video_calibration = frame
                        elif last_video_calibration is None:
                            uncalibrated += 1
                        else:
                            age_samples.append(frame - last_video_calibration)
                        if arm == "native":
                            continue
                        result, state = row["prediction"], "".join(map(str, flags))
                        if state == "11":
                            offsets = {key: np.asarray(value)
                                       for key, value in result["calibration_offsets"].items()}
                            assert set(result["calibration_sizes"]) == {"11", "01", "10"}
                            full = np.asarray(result["calibration_sizes"]["11"])
                            for key in ["01", "10"]:
                                scales[key] = full / np.asarray(result["calibration_sizes"][key])
                                assert np.array_equal(scales[key], result["calibration_scales"][key])
                            last_calibration = tracker_calls
                        expected_offset = offsets[state] if arm == "geometry" and state != "11" else np.zeros(2)
                        expected_scale = scales[state] if arm in ["scale", "geometry"] and state != "11" else np.ones(2)
                        assert np.array_equal(expected_offset, result["applied_offset"])
                        assert np.array_equal(expected_scale, result["applied_scale"])
                        assert result["last_calibration_call"] == last_calibration
                        raw = np.asarray(result["before_scale_bbox"])
                        size = raw[2:] * expected_scale
                        expected_box = np.concatenate([
                            raw[:2] + raw[2:] / 2 - size / 2 - expected_offset, size])
                        assert np.allclose(clip_box(expected_box.tolist(), *row["image_shape"], margin=10),
                                           boxes[frame], rtol=0, atol=1e-9)
                    assert calls == receipt["model_calls"]
                    if arm == "native":
                        calibration_ages.setdefault(name, {})[condition] = dict(
                            calibrated_single_frames=len(age_samples),
                            uncalibrated_single_frames=uncalibrated,
                            video_age_percentiles=(np.percentile(
                                age_samples, [50, 90, 95, 99, 100]).tolist()
                                if age_samples else []))
                    all_records[name][arm][condition] = measured
                    trajectories += 1
    names = sorted(all_records)
    assert len(names) == 4 and trajectories == 48
    assert len({row["checkpoint_sha256"] for row in worker_receipts.values()}) == 1
    draws = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    aggregates, contrasts = {}, {}
    for condition in CONDITIONS:
        values = {arm: {name: all_records[name][arm][condition] for name in names} for arm in ARMS}
        aggregates[condition] = {
            arm: {key: float(np.mean([values[arm][name][key] for name in names])) for key in ["PR20", "SR"]}
            for arm in ARMS}
        contrasts[condition] = {
            first + "_minus_" + second: compare(values[first], values[second], names, draws)
            for first, second in [("scale", "native"), ("geometry", "native"), ("scale", "geometry")]}
    gap_interactions = {}
    for condition in ["block128_outage"]:
        gap_interactions[condition] = {}
        for first, second in [("scale", "native"), ("geometry", "native"), ("scale", "geometry")]:
            gap_interactions[condition][first + "_minus_" + second] = {}
            for metric in ["PR20", "SR"]:
                delta = np.asarray([
                    (all_records[name][first][condition][metric] -
                     all_records[name][second][condition][metric]) -
                    (all_records[name][first]["block32_outage"][metric] -
                     all_records[name][second]["block32_outage"][metric])
                    for name in names])
                gap_interactions[condition][first + "_minus_" + second][metric] = dict(
                    mean_delta=float(delta.mean()),
                    paired_video_bootstrap95=np.percentile(
                        delta[draws].mean(1), [2.5, 97.5]).tolist())
    save(OUTPUT / "report.json", dict(
        launch=launch, aggregate=aggregates, paired_contrasts=contrasts, per_video=all_records,
        effect_change_from_block32=gap_interactions,
        calibration_age_by_video=calibration_ages,
        worker_receipts=worker_receipts, trajectories_verified=trajectories,
        past_only_center_and_scale_verified=True, controls_bitwise_equal=True,
        scope="Four prospectively registered fresh development videos, complete/block32/block128 "
              "scale-only versus native and center-plus-scale. Block interactions change "
              "all modality histories, not isolated cache-age causality. Ordinary component control, "
              "not official benchmark or novel-module evidence."))
    save(OUTPUT / "completed.json", dict(report_sha256=sha(OUTPUT / "report.json"),
                                         trajectories_verified=trajectories))


def controller(args):
    witness_report = read(WITNESS)
    assert all(witness_report[key] for key in [
        "sham_equal", "complete_equal", "scale_offset_zero", "baseline_state_unchanged"])
    for path, expected in witness_report["source_sha256"].items():
        assert sha(path) == expected
    manifest_path = ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json"
    manifest = read(manifest_path)
    cohort = []
    for index in range(64, 68):
        name = manifest["development_sequences"][index]
        folder = Path("/data/wangwj/dataset/LasHeR/trainingset") / name
        gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
        rgb, tir = images(folder)
        assert len(rgb) == len(tir) == len(gt) and np.isfinite(gt).all()
        assert (gt[0, 2:] > 0).all()
        cohort.append(dict(name=name, folder=str(folder), development_index=index,
                           video_frames=len(gt), gt_sha256=sha(folder / "init.txt")))
    assert not {item["name"] for item in cohort} & (
        set(manifest["train_sequences"]) | set(manifest["heldout_sequences"]) |
        set(manifest["development_sequences"][:64]))
    parts, loads = {"0": [], "1": []}, {"0": 0, "1": 0}
    for item in sorted(cohort, key=lambda row: -row["video_frames"]):
        gpu = min(loads, key=lambda key: (loads[key], key))
        parts[gpu].append(item)
        loads[gpu] += item["video_frames"]
    sources = {str(path): sha(path) for path in [
        Path(__file__), ROOT / "spatial_calibration_tracker28.py", ROOT / "state_offset_tracker28.py",
        ROOT / "analyze_epoch05_health28.py", ROOT / "evaluate_val97_recovery28.py",
        RUNTIME / "miss_rgbt/model.py", RUNTIME / "miss_rgbt/tracker.py"]}
    OUTPUT.mkdir(exist_ok=False)
    registration = dict(
        pid=os.getpid(), registered_utc=datetime.datetime.utcnow().isoformat() + "Z",
        input_manifest_sha256=sha(manifest_path), development_indices=list(range(64, 68)),
        cohort=cohort, parts=parts, loads=loads, arms=ARMS, conditions=CONDITIONS,
        frame_blocks=BLOCKS, pattern=PATTERN,
        source_sha256=sources, witness_sha256=sha(WITNESS), poll_seconds=300,
        scope="Sequential fresh development indices64..67, no outcome or duration filtering. "
              "Native/sham/scale-only/center-plus-scale, complete/block32/block128. "
              "Same E0 and hardware per video; no fitted gate, expiry, decay or weight updates.")
    save(OUTPUT / "registration.json", registration)
    print("REGISTERED " + json.dumps(registration), flush=True)
    previous = read(PREVIOUS / "report.json")
    assert sha(PREVIOUS / "report.json") == read(PREVIOUS / "completed.json")["report_sha256"]
    assert previous["past_only_center_and_scale_verified"] and previous["controls_bitwise_equal"]
    for path, expected in sources.items():
        assert sha(path) == expected
    memory = subprocess.check_output([
        "nvidia-smi", "--id=0,1", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    assert all(int(line.split(",")[1]) >= 3000 for line in memory.splitlines())
    launch = dict(registration, registration_sha256=sha(OUTPUT / "registration.json"),
                  started_utc=datetime.datetime.utcnow().isoformat() + "Z",
                  previous_report_sha256=sha(PREVIOUS / "report.json"))
    save(OUTPUT / "launch.json", launch)
    processes = {}
    for gpu, selected in parts.items():
        save(OUTPUT / ("gpu%s_spec.json" % gpu), dict(
            cohort=selected, registration_sha256=launch["registration_sha256"]))
        with (OUTPUT / ("gpu%s.log" % gpu)).open("xb") as log:
            processes[gpu] = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__)), "--mode", "worker", "--gpu", gpu],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    save(OUTPUT / "children.json", {gpu: process.pid for gpu, process in processes.items()})
    while any(process.poll() is None for process in processes.values()):
        time.sleep(300)
    assert all(process.returncode == 0 for process in processes.values())
    subprocess.check_call([sys.executable, str(Path(__file__)), "--mode", "analyze"], cwd=str(ROOT))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["witness", "controller", "worker", "analyze"], required=True)
    parser.add_argument("--gpu", choices=["0", "1"])
    args = parser.parse_args()
    if args.mode == "witness":
        witness()
    elif args.mode == "controller":
        controller(args)
    elif args.mode == "worker":
        assert args.gpu is not None
        worker(args)
    else:
        analyze()
