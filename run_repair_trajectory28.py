"""Closed-loop native/sham/repair-off control for the retained epoch6 pair."""
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
from run_state_template28 import runtime, images, frame_image, flags_for, read, save

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
OUTPUT = ROOT / "reports/server28-repair-trajectory8-20261031"
PREVIOUS = ROOT / "reports/server28-state-template8-20261031"
WITNESS = ROOT / "reports/server28-repair-trajectory-entry-20261031.json"
ARMS = ["native", "sham", "off"]
CONDITIONS = ["complete", "block32_outage"]
MODELS = ["utility", "random"]


def sources():
    return {str(path): sha(path) for path in [
        Path(__file__), ROOT / "run_state_template28.py", ROOT / "cpu_diagnostic_tracker.py",
        ROOT / "evaluate_blind_motion28.py", ROOT / "analyze_epoch05_health28.py",
        ROOT / "evaluate_val97_recovery28.py",
        RUNTIME / "miss_rgbt/model.py", RUNTIME / "miss_rgbt/task_utility.py",
        RUNTIME / "miss_rgbt/tracker.py", RUNTIME / "experiments/cadtrack/cadtrack.yaml"]}


def checkpoints():
    report = read(ROOT / "reports/server28-candidate-repair-audit-20261031/report.json")
    result = {model: {key: report["models"][model][key]
                      for key in ["checkpoint", "checkpoint_sha256"]} for model in MODELS}
    for item in result.values():
        assert sha(item["checkpoint"]) == item["checkpoint_sha256"]
    return result


def install_hook(utility, arm, calls):
    def hook(module, inputs, output):
        calls[0] += 1
        if arm == "off":
            # The native tail consumes these same pre-compensation tensor objects.
            return inputs[0], inputs[1], output[2]
        return output
    return utility.register_forward_hook(hook)


def witness():
    cfg, reader = runtime(cpu=True)
    from cpu_diagnostic_tracker import CpuDiagnosticTracker
    from evaluate_blind_motion28 import network_digest
    manifest = read(ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json")
    folder = Path("/data/wangwj/dataset/LasHeR/trainingset") / manifest["development_sequences"][36]
    rgb, tir = images(folder)
    gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
    weights, records = checkpoints(), {}
    for model, item in weights.items():
        tracker = CpuDiagnosticTracker(cfg, item["checkpoint"])
        tracker.network.requires_grad_(False)
        utility = tracker.network.utility_repair
        original_rng = utility.selection_rng_state.clone()
        digest = network_digest(tracker.network)
        records[model] = {}
        for arm in ["native", "sham", "off", "disabled"]:
            tracker.initialize(reader(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                               dict(init_bbox=gt[0].tolist()))
            calls, handle = [0], None
            if arm in ["sham", "off"]:
                handle = install_hook(utility, arm, calls)
            if arm == "disabled":
                tracker.network.utility_repair = None
            rows = []
            for frame in range(1, 8):
                flags = [[1, 1], [0, 1], [1, 0]][(frame - 1) % 3]
                result = tracker.track(frame_image(rgb, tir, frame, flags, reader), flags)
                rows.append(dict(frame=frame, available=flags, bbox=list(result["target_bbox"]),
                                 confidence=result["best_score"]))
            if handle is not None:
                handle.remove()
                assert calls[0] == 7
            tracker.network.utility_repair = utility
            records[model][arm] = rows
        assert records[model]["native"] == records[model]["sham"]
        assert records[model]["off"] == records[model]["disabled"]
        utility.selection_rng_state.copy_(original_rng)
        assert network_digest(tracker.network) == digest
    save(WITNESS, dict(
        records=records, checkpoints=weights, source_sha256=sources(),
        sham_equal=True, off_equals_module_disabled=True, restored_full_state_equal=True,
        scope="Seven observed CPU track calls/model/arm, including11/01/10. "
              "No claim of CUDA equivalence, training ablation or trajectory benefit."))


def worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg, reader = runtime()
    import torch
    from miss_rgbt.tracker import MissingTracker
    from evaluate_blind_motion28 import network_digest
    launch = read(OUTPUT / "launch.json")
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    folder = OUTPUT / ("gpu" + args.gpu)
    folder.mkdir(exist_ok=False)
    cohort = launch["parts"][args.gpu]
    save(folder / "launch.json", dict(
        pid=os.getpid(), physical_gpu=args.gpu, cohort=cohort,
        registration_sha256=launch["registration_sha256"], checkpoints=launch["checkpoints"]))
    records, references = {}, {}
    started = time.time()
    for model, weight in launch["checkpoints"].items():
        assert sha(weight["checkpoint"]) == weight["checkpoint_sha256"]
        tracker = MissingTracker(cfg, weight["checkpoint"])
        tracker.network.requires_grad_(False)
        utility = tracker.network.utility_repair
        assert utility is not None
        original_rng = utility.selection_rng_state.clone()
        digest = network_digest(tracker.network)
        records[model] = {}
        for arm in ARMS:
            count = [0]
            handle = install_hook(utility, arm, count) if arm != "native" else None
            records[model][arm] = {}
            for item in cohort:
                name, data_folder = item["name"], Path(item["folder"])
                gt_path = data_folder / "init.txt"
                assert sha(gt_path) == item["gt_sha256"]
                gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
                rgb, tir = images(data_folder)
                assert len(rgb) == len(tir) == len(gt) == item["video_frames"]
                records[model][arm][name] = {}
                for condition in CONDITIONS:
                    tracker.initialize(reader(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                                       dict(init_bbox=gt[0].tolist()))
                    boxes, trace, calls = [list(tracker.state)], [], 0
                    for frame in range(1, len(gt)):
                        flags = flags_for(frame, condition)
                        before_hooks = count[0]
                        if any(flags):
                            tracker.track(frame_image(rgb, tir, frame, flags, reader), flags)
                            calls += 1
                        fired = count[0] - before_hooks
                        assert fired == int(arm != "native" and any(flags))
                        box = list(tracker.state)
                        assert np.isfinite(box).all() and (np.asarray(box)[2:] > 0).all()
                        assert tracker.frame_id == calls
                        boxes.append(box)
                        trace.append(dict(frame=frame, available=flags, bbox=box,
                                          track_call=calls, hook_calls=fired))
                    boxes = np.asarray(boxes)
                    if arm == "native":
                        references[(model, name, condition)] = boxes.copy()
                    elif arm == "sham":
                        assert np.array_equal(boxes, references[(model, name, condition)])
                    prefix = folder / (model + "_" + arm + "_" + name + "_" + condition)
                    np.savetxt(str(prefix) + ".txt", boxes)
                    save(str(prefix) + ".json", trace)
                    measured = dict(metrics(boxes, gt), bbox_sha256=sha(str(prefix) + ".txt"),
                                    trace_sha256=sha(str(prefix) + ".json"), model_calls=calls)
                    records[model][arm][name][condition] = measured
                    print("TRACK " + json.dumps(dict(model=model, arm=arm, sequence=name,
                                                     condition=condition, metrics=measured)), flush=True)
            if handle is not None:
                handle.remove()
        utility.selection_rng_state.copy_(original_rng)
        assert network_digest(tracker.network) == digest
        del tracker
        torch.cuda.empty_cache()
    save(folder / "report.json", dict(records=records, cohort=cohort,
                                      frozen_state_restored=True, sham_equal=True,
                                      seconds=time.time() - started))
    save(folder / "completed.json", dict(report_sha256=sha(folder / "report.json")))


def analyze():
    launch = read(OUTPUT / "launch.json")
    assert sha(OUTPUT / "registration.json") == launch["registration_sha256"]
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    records, workers, trajectories = {}, {}, 0
    for gpu, cohort in launch["parts"].items():
        folder = OUTPUT / ("gpu" + gpu)
        report, worker_launch = read(folder / "report.json"), read(folder / "launch.json")
        assert report["cohort"] == worker_launch["cohort"] == cohort
        assert worker_launch["physical_gpu"] == gpu
        assert worker_launch["registration_sha256"] == launch["registration_sha256"]
        assert worker_launch["checkpoints"] == launch["checkpoints"]
        assert report["sham_equal"] and report["frozen_state_restored"]
        assert sha(folder / "report.json") == read(folder / "completed.json")["report_sha256"]
        workers[gpu] = sha(folder / "report.json")
        for item in cohort:
            name = item["name"]
            gt_path = Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            records[name], references = {}, {}
            for model in MODELS:
                records[name][model] = {}
                for arm in ARMS:
                    records[name][model][arm] = {}
                    for condition in CONDITIONS:
                        prefix = folder / (model + "_" + arm + "_" + name + "_" + condition)
                        receipt = report["records"][model][arm][name][condition]
                        assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
                        assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
                        boxes = np.loadtxt(str(prefix) + ".txt", ndmin=2)
                        trace = read(str(prefix) + ".json")
                        assert boxes.shape == gt.shape and len(trace) == len(gt) - 1
                        measured = metrics(boxes, gt)
                        assert all(np.isclose(value, receipt[key], rtol=0, atol=1e-10)
                                   for key, value in measured.items())
                        if arm == "native":
                            references[(model, condition)] = boxes
                        elif arm == "sham":
                            assert np.array_equal(boxes, references[(model, condition)])
                        calls = 0
                        for frame, row in enumerate(trace, 1):
                            flags = flags_for(frame, condition)
                            assert row["frame"] == frame and row["available"] == flags
                            calls += int(any(flags))
                            assert row["track_call"] == calls
                            assert row["hook_calls"] == int(arm != "native" and any(flags))
                            assert np.array_equal(row["bbox"], boxes[frame])
                            if not any(flags):
                                assert np.array_equal(boxes[frame], boxes[frame - 1])
                        assert calls == receipt["model_calls"]
                        records[name][model][arm][condition] = measured
                        trajectories += 1
    names = sorted(records)
    assert len(names) == 8 and trajectories == 96
    draws = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    aggregates, contrasts = {}, {}
    for condition in CONDITIONS:
        values = {model + "_" + arm: {
            name: records[name][model][arm][condition] for name in names}
            for model in MODELS for arm in ARMS}
        aggregates[condition] = {
            label: {key: float(np.mean([by_name[name][key] for name in names]))
                    for key in ["PR20", "SR"]} for label, by_name in values.items()}
        contrasts[condition] = {
            first + "_minus_" + second: compare(values[first], values[second], names, draws)
            for first, second in [
                ("utility_native", "utility_off"), ("random_native", "random_off"),
                ("utility_native", "random_native"), ("utility_off", "random_off")]}
    save(OUTPUT / "report.json", dict(
        launch=launch, aggregate=aggregates, contrasts=contrasts, per_video=records,
        worker_report_sha256=workers, trajectories_verified=trajectories, sham_equal=True,
        scope="Fresh eight-video closed-loop inference intervention on retained epoch6 models. "
              "Off discards utility compensation but executes selection and preserves its overhead. "
              "Not matched retraining, selector-only causality, official attainment, convergence "
              "or module novelty. Frozen-state restoration is a worker witness, not network replay."))
    save(OUTPUT / "completed.json", dict(
        report_sha256=sha(OUTPUT / "report.json"), trajectories_verified=trajectories))


def controller():
    witness_report = read(WITNESS)
    assert all(witness_report[key] for key in [
        "sham_equal", "off_equals_module_disabled", "restored_full_state_equal"])
    assert sources() == witness_report["source_sha256"]
    weights = checkpoints()
    assert weights == witness_report["checkpoints"]
    manifest_path = ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json"
    manifest, cohort = read(manifest_path), []
    for index in range(76, 84):
        name = manifest["development_sequences"][index]
        folder = Path("/data/wangwj/dataset/LasHeR/trainingset") / name
        gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
        rgb, tir = images(folder)
        assert len(rgb) == len(tir) == len(gt) and np.isfinite(gt).all()
        assert (gt[0, 2:] > 0).all()
        cohort.append(dict(name=name, folder=str(folder), development_index=index,
                           video_frames=len(gt), gt_sha256=sha(folder / "init.txt")))
    assert not {row["name"] for row in cohort} & (
        set(manifest["train_sequences"]) | set(manifest["heldout_sequences"]) |
        set(manifest["development_sequences"][:76]))
    parts, loads = {"0": [], "1": []}, {"0": 0, "1": 0}
    for row in sorted(cohort, key=lambda row: -row["video_frames"]):
        gpu = min(loads, key=lambda key: (loads[key], key))
        parts[gpu].append(row)
        loads[gpu] += row["video_frames"]
    previous = read(PREVIOUS / "registration.json")
    prior_pid = previous["pid"]
    OUTPUT.mkdir(exist_ok=False)
    registration = dict(
        pid=os.getpid(), registered_utc=datetime.datetime.utcnow().isoformat() + "Z",
        cohort=cohort, parts=parts, loads=loads, arms=ARMS, models=MODELS,
        conditions=CONDITIONS, source_sha256=sources(), checkpoints=weights,
        previous_pid=prior_pid, previous_registration_sha256=sha(PREVIOUS / "registration.json"),
        input_manifest_sha256=sha(manifest_path), witness_sha256=sha(WITNESS),
        poll_seconds=300, scope="Sequential fresh development76..83, no filtering or training.")
    save(OUTPUT / "registration.json", registration)
    print("WAITING " + json.dumps(dict(previous_pid=prior_pid, loads=loads)), flush=True)
    while (Path("/proc") / str(prior_pid)).exists():
        command = (Path("/proc") / str(prior_pid) / "cmdline").read_bytes()
        assert b"/home/gaob/Miss-RGBT/run_state_template28.py" in command
        time.sleep(300)
    prior_report = read(PREVIOUS / "report.json")
    assert sha(PREVIOUS / "report.json") == read(PREVIOUS / "completed.json")["report_sha256"]
    assert prior_report["controls_equal"] and prior_report["past_only_bank_verified"]
    assert sources() == registration["source_sha256"] and checkpoints() == weights
    memory = subprocess.check_output([
        "nvidia-smi", "--id=0,1", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    assert all(int(line.split(",")[1]) >= 3000 for line in memory.splitlines())
    save(OUTPUT / "launch.json", dict(
        registration, registration_sha256=sha(OUTPUT / "registration.json"),
        previous_report_sha256=sha(PREVIOUS / "report.json"),
        started_utc=datetime.datetime.utcnow().isoformat() + "Z"))
    processes = {}
    for gpu in parts:
        with (OUTPUT / ("gpu" + gpu + ".log")).open("xb") as log:
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
    parser.add_argument("--mode", choices=["witness", "worker", "analyze", "controller"], required=True)
    parser.add_argument("--gpu", choices=["0", "1"])
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args)
    else:
        {"witness": witness, "analyze": analyze, "controller": controller}[args.mode]()
