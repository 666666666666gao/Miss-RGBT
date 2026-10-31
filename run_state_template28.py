"""Prospective state-template control with native and recording-only controls."""
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
OUTPUT = ROOT / "reports/server28-state-template8-20261031"
WITNESS = ROOT / "reports/server28-state-template-entry-20261031.json"
ARMS = ["native", "sham", "matched"]
CONDITIONS = ["complete", "block32_outage"]
PATTERN = [[1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def flags_for(frame, condition):
    return [1, 1] if condition == "complete" else PATTERN[(frame // 32) % 6]


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


def frame_image(rgb, tir, frame, flags, reader):
    return reader(str(rgb[frame] if flags[0] else tir[frame]),
                  str(tir[frame] if flags[1] else rgb[frame]), dtype="rgbrgb")


def witness():
    cfg, reader = runtime(cpu=True)
    from cpu_diagnostic_tracker import CpuDiagnosticTracker
    from evaluate_blind_motion28 import network_digest
    from state_template_tracker28 import StateTemplateTracker
    checkpoint = ROOT / "checkpoints/e0_availability/epoch20.pth.tar"
    native = CpuDiagnosticTracker(cfg, str(checkpoint))
    native.network.requires_grad_(False)
    digest = network_digest(native.network)
    manifest = read(ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json")
    folder = Path("/data/wangwj/dataset/LasHeR/trainingset") / manifest["development_sequences"][36]
    rgb, tir = images(folder)
    gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
    records, differing_sources, actual_writes = {}, 0, 0
    # Twenty-one actual track calls expose native modulo5 writes across three states.
    for condition in ["complete", "switch_every_five"]:
        native.initialize(reader(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                          dict(init_bbox=gt[0].tolist()))
        trackers = {}
        for arm in ["sham", "matched"]:
            tracker = StateTemplateTracker.__new__(StateTemplateTracker)
            tracker.__dict__.update(native.__dict__)
            tracker.templates, tracker.availability = list(native.templates), list(native.availability)
            tracker.state, tracker.match_state = list(native.state), arm == "matched"
            tracker.reset_bank()
            trackers[arm] = tracker
        rows = []
        for frame in range(1, 22):
            flags = [1, 1] if condition == "complete" else PATTERN[((frame - 1) // 5) % 5]
            image = frame_image(rgb, tir, frame, flags, reader)
            reference = native.track(image, flags)
            predictions = {arm: tracker.track(image, flags) for arm, tracker in trackers.items()}
            assert np.array_equal(reference["target_bbox"], predictions["sham"]["target_bbox"])
            assert reference["best_score"] == predictions["sham"]["best_score"]
            if condition == "complete":
                assert np.array_equal(reference["target_bbox"], predictions["matched"]["target_bbox"])
            else:
                differing_sources += int(predictions["matched"]["selected_source_call"] !=
                                         predictions["sham"]["selected_source_call"])
            actual_writes += sum(value["template_written"] for value in predictions.values())
            rows.append(dict(frame=frame, available=flags, predictions=predictions))
        records[condition] = rows
    assert actual_writes > 0 and differing_sources > 0
    assert network_digest(native.network) == digest
    save(WITNESS, dict(records=records, sham_equal=True, complete_equal=True,
                       actual_writes=actual_writes, differing_sources=differing_sources,
                       frozen_network=True,
                       source_sha256={str(path): sha(path) for path in [
                           Path(__file__), ROOT / "state_template_tracker28.py",
                           ROOT / "cpu_diagnostic_tracker.py"]},
                       scope="CPU actual track-body witness,21calls/condition; "
                             "not CUDA initialization or full-trajectory verification."))


def worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg, reader = runtime()
    import torch
    from evaluate_blind_motion28 import network_digest
    from miss_rgbt.tracker import MissingTracker
    from state_template_tracker28 import StateTemplateTracker
    launch = read(OUTPUT / "launch.json")
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    cohort = launch["parts"][args.gpu]
    destination = OUTPUT / ("gpu" + args.gpu)
    destination.mkdir(exist_ok=False)
    checkpoint = ROOT / "checkpoints/e0_availability/epoch20.pth.tar"
    package = read(checkpoint.with_suffix(".json"))
    assert sha(checkpoint) == package["packaged_sha256"]
    assert package["optimizer_updates"] == 0 and package["tensor_equality"]
    save(destination / "launch.json", dict(
        pid=os.getpid(), gpu=args.gpu, checkpoint_sha256=sha(checkpoint),
        registration_sha256=launch["registration_sha256"], cohort=cohort))
    records, references = {}, {}
    started = time.time()
    for arm in ARMS:
        tracker = (MissingTracker(cfg, str(checkpoint)) if arm == "native" else
                   StateTemplateTracker(cfg, str(checkpoint)))
        if arm != "native":
            tracker.match_state = arm == "matched"
        tracker.network.requires_grad_(False)
        digest = network_digest(tracker.network)
        records[arm] = {}
        for item in cohort:
            folder, name = Path(item["folder"]), item["name"]
            assert sha(folder / "init.txt") == item["gt_sha256"]
            gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
            rgb, tir = images(folder)
            assert len(rgb) == len(tir) == len(gt) == item["video_frames"]
            records[arm][name] = {}
            for condition in CONDITIONS:
                tracker.initialize(reader(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                                   dict(init_bbox=gt[0].tolist()))
                boxes, trace, calls = [list(tracker.state)], [], 0
                for frame in range(1, len(gt)):
                    flags = flags_for(frame, condition)
                    prediction = {}
                    written = False
                    if any(flags):
                        previous_template = tracker.templates[-1]
                        prediction = tracker.track(frame_image(rgb, tir, frame, flags, reader), flags)
                        written = (tracker.templates[-1] is not previous_template if arm == "native"
                                   else prediction["template_written"])
                        calls += 1
                    box = list(tracker.state)
                    assert tracker.frame_id == calls
                    assert np.isfinite(box).all() and (np.asarray(box)[2:] > 0).all()
                    boxes.append(box)
                    trace.append(dict(frame=frame, available=flags, track_call=calls,
                                      bbox=box, prediction=prediction, template_written=written))
                boxes = np.asarray(boxes)
                if arm == "native":
                    references[(name, condition)] = boxes.copy()
                elif arm == "sham" or condition == "complete":
                    assert np.array_equal(boxes, references[(name, condition)])
                prefix = destination / (arm + "_" + name + "_" + condition)
                np.savetxt(str(prefix) + ".txt", boxes)
                save(str(prefix) + ".json", trace)
                measured = dict(metrics(boxes, gt), bbox_sha256=sha(str(prefix) + ".txt"),
                                trace_sha256=sha(str(prefix) + ".json"), model_calls=calls)
                records[arm][name][condition] = measured
                print("TRACK " + json.dumps(dict(
                    arm=arm, sequence=name, condition=condition, metrics=measured)), flush=True)
        assert network_digest(tracker.network) == digest
        del tracker
        torch.cuda.empty_cache()
    save(destination / "report.json", dict(
        records=records, cohort=cohort, frozen_network=True, controls_equal=True,
        seconds=time.time() - started))
    save(destination / "completed.json", dict(report_sha256=sha(destination / "report.json")))


def analyze():
    launch = read(OUTPUT / "launch.json")
    assert sha(OUTPUT / "registration.json") == launch["registration_sha256"]
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    records, receipts, exposures, trajectories = {}, {}, {}, 0
    for gpu, cohort in launch["parts"].items():
        folder = OUTPUT / ("gpu" + gpu)
        report, worker_launch = read(folder / "report.json"), read(folder / "launch.json")
        assert report["cohort"] == worker_launch["cohort"] == cohort
        assert worker_launch["registration_sha256"] == launch["registration_sha256"]
        assert worker_launch["gpu"] == gpu and report["frozen_network"] and report["controls_equal"]
        assert sha(folder / "report.json") == read(folder / "completed.json")["report_sha256"]
        receipts[gpu] = dict(report_sha256=sha(folder / "report.json"),
                             checkpoint_sha256=worker_launch["checkpoint_sha256"])
        for item in cohort:
            name = item["name"]
            gt_path = Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            records[name], references, exposures[name] = {}, {}, {}
            for arm in ARMS:
                records[name][arm], exposures[name][arm] = {}, {}
                for condition in CONDITIONS:
                    prefix = folder / (arm + "_" + name + "_" + condition)
                    receipt = report["records"][arm][name][condition]
                    assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
                    assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
                    boxes = np.loadtxt(str(prefix) + ".txt", ndmin=2)
                    trace = read(str(prefix) + ".json")
                    assert boxes.shape == gt.shape and len(trace) == len(gt) - 1
                    measured = metrics(boxes, gt)
                    assert all(np.isclose(value, receipt[key], rtol=0, atol=1e-10)
                               for key, value in measured.items())
                    if arm == "native":
                        references[condition] = boxes
                    elif arm == "sham" or condition == "complete":
                        assert np.array_equal(boxes, references[condition])
                    bank = {key: 0 for key in ["11", "01", "10"]}
                    availability, latest, calls, changed, writes = {0: [1, 1]}, 0, 0, 0, 0
                    for frame, row in enumerate(trace, 1):
                        flags = flags_for(frame, condition)
                        assert row["frame"] == frame and row["available"] == flags
                        assert np.array_equal(row["bbox"], boxes[frame])
                        if any(flags):
                            calls += 1
                            availability[calls] = flags
                            written = (calls % launch["update_interval"] == 0 and
                                       row["prediction"]["best_score"] > launch["update_threshold"])
                            assert row["template_written"] == written
                            writes += int(written)
                            if arm != "native":
                                result, key = row["prediction"], "".join(map(str, flags))
                                selected = bank[key] if arm == "matched" else latest
                                assert result["selected_source_call"] == selected < calls
                                assert result["selected_availability"] == availability[selected]
                                changed += int(selected != latest)
                                # Use native YAML predicate recorded before launch.
                                assert result["template_written"] == written
                                if written:
                                    latest, bank[key] = calls, calls
                                assert result["bank_source_calls"] == bank
                        else:
                            assert not row["prediction"]
                            assert not row["template_written"]
                            assert np.array_equal(boxes[frame], boxes[frame - 1])
                        assert row["track_call"] == calls
                    assert calls == receipt["model_calls"]
                    records[name][arm][condition] = measured
                    if condition == "block32_outage":
                        current11 = np.asarray([
                            flags_for(frame, condition) == [1, 1] for frame in range(len(gt))])
                        group_gt = gt.copy()
                        group_gt[~current11, 2:] = 0
                        records[name][arm][condition]["current11"] = metrics(boxes, group_gt)
                    exposures[name][arm][condition] = dict(
                        calls=calls, writes=writes,
                        selected_different_from_latest=changed if arm != "native" else None)
                    trajectories += 1
    names = sorted(records)
    assert len(names) == 8 and trajectories == 48
    assert len({value["checkpoint_sha256"] for value in receipts.values()}) == 1
    draws = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    aggregates, contrasts = {}, {}
    for condition in CONDITIONS:
        values = {arm: {name: records[name][arm][condition] for name in names} for arm in ARMS}
        aggregates[condition] = {
            arm: {key: float(np.mean([values[arm][name][key] for name in names]))
                  for key in ["PR20", "SR"]} for arm in ARMS}
        contrasts[condition] = compare(values["matched"], values["native"], names, draws)
    current11 = {arm: {name: records[name][arm]["block32_outage"]["current11"] for name in names}
                 for arm in ARMS}
    assert sum(exposures[name]["matched"]["block32_outage"]["selected_different_from_latest"]
               for name in names) > 0
    save(OUTPUT / "report.json", dict(
        launch=launch, aggregate=aggregates, matched_minus_native=contrasts,
        per_video=records, exposures=exposures, worker_receipts=receipts,
        current11_matched_minus_native=compare(
            current11["matched"], current11["native"], names, draws),
        current11_aggregate={
            arm: {key: float(np.mean([current11[arm][name][key] for name in names]))
                  for key in ["PR20", "SR"]} for arm in ARMS},
        trajectories_verified=trajectories, past_only_bank_verified=True, controls_equal=True,
        scope="Eight prospective development videos; paired-video resampling for one fixed E0. "
              "Ordinary state-history control, not official scores or novel module evidence. "
              "Template/crop histories evolve jointly; effects do not isolate template age."))
    save(OUTPUT / "completed.json", dict(
        report_sha256=sha(OUTPUT / "report.json"), trajectories_verified=trajectories))


def controller():
    cfg, _ = runtime()
    witness_report = read(WITNESS)
    assert all(witness_report[key] for key in ["sham_equal", "complete_equal", "frozen_network"])
    for path, expected in witness_report["source_sha256"].items():
        assert sha(path) == expected
    manifest_path = ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json"
    manifest = read(manifest_path)
    cohort = []
    for index in range(68, 76):
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
        set(manifest["development_sequences"][:68]))
    parts, loads = {"0": [], "1": []}, {"0": 0, "1": 0}
    for item in sorted(cohort, key=lambda row: -row["video_frames"]):
        gpu = min(loads, key=lambda key: (loads[key], key))
        parts[gpu].append(item)
        loads[gpu] += item["video_frames"]
    sources = {str(path): sha(path) for path in [
        Path(__file__), ROOT / "state_template_tracker28.py", ROOT / "analyze_epoch05_health28.py",
        ROOT / "evaluate_val97_recovery28.py", ROOT / "evaluate_blind_motion28.py",
        RUNTIME / "miss_rgbt/model.py", RUNTIME / "miss_rgbt/tracker.py",
        RUNTIME / "experiments/cadtrack/cadtrack.yaml"]}
    OUTPUT.mkdir(exist_ok=False)
    registration = dict(
        pid=os.getpid(), registered_utc=datetime.datetime.utcnow().isoformat() + "Z",
        input_manifest_sha256=sha(manifest_path), development_indices=list(range(68, 76)),
        cohort=cohort, parts=parts, loads=loads, arms=ARMS, conditions=CONDITIONS,
        source_sha256=sources, witness_sha256=sha(WITNESS), pattern=PATTERN, frame_block=32,
        update_interval=cfg.TEST.UPDATE_INTERVALS, update_threshold=cfg.TEST.UPDATE_THRESHOLD,
        poll_seconds=300, scope="Sequential fresh indices68..75, no outcome/duration filtering. "
        "Each state bank initialized with the real complete initial template and availability11. "
        "Native write predicate unchanged; selected cache is read before any current write. "
        "No fitted gate, extra forward, optimizer or baseline runtime edit.")
    save(OUTPUT / "registration.json", registration)
    memory = subprocess.check_output([
        "nvidia-smi", "--id=0,1", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    assert all(int(line.split(",")[1]) >= 3000 for line in memory.splitlines())
    save(OUTPUT / "launch.json", dict(
        registration, registration_sha256=sha(OUTPUT / "registration.json")))
    processes = {}
    for gpu in parts:
        with (OUTPUT / ("gpu" + gpu + ".log")).open("xb") as log:
            processes[gpu] = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__)), "--mode", "worker", "--gpu", gpu],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    save(OUTPUT / "children.json", {gpu: process.pid for gpu, process in processes.items()})
    print("LAUNCHED " + json.dumps(read(OUTPUT / "children.json")), flush=True)
    while any(process.poll() is None for process in processes.values()):
        time.sleep(300)
    assert all(process.returncode == 0 for process in processes.values())
    subprocess.check_call([sys.executable, str(Path(__file__)), "--mode", "analyze"], cwd=str(ROOT))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["witness", "worker", "analyze", "controller"], required=True)
    parser.add_argument("--gpu", choices=["0", "1"])
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args)
    else:
        {"witness": witness, "analyze": analyze, "controller": controller}[args.mode]()
