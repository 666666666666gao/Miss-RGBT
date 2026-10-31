"""Single-GPU matched closed-loop constant-velocity development control."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from analyze_epoch05_health28 import metrics
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
SOURCE = ROOT / "reports/server28-protected-tracking-development-20261030"
OUTPUT = ROOT / "reports/server28-blind-motion-closed-loop-20261031"
PATTERN = [[1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]]


def center(box):
    return np.asarray(box[:2]) + np.asarray(box[2:]) / 2


def network_digest(network):
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=["0", "1"], default="0")
    parser.add_argument("--cohort-spec", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    free = subprocess.check_output([
        "nvidia-smi", "--id=" + args.gpu, "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    assert int(free.strip()) >= 3000, free
    import torch
    sys.path.insert(0, str(RUNTIME))
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_x_frame
    from miss_rgbt.tracker import MissingTracker
    torch.set_num_threads(1)
    output.mkdir(exist_ok=False)
    checkpoint = ROOT / "checkpoints/e0_availability/epoch20.pth.tar"
    packaged = json.loads(checkpoint.with_suffix(".json").read_text())
    assert sha(checkpoint) == packaged["packaged_sha256"]
    assert packaged["tensor_equality"] and packaged["optimizer_updates"] == 0
    original = json.loads((SOURCE / "report.json").read_text())
    manifest_path = ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json"
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == original["launch"]["input_manifest_sha256"]
    if args.cohort_spec is None:
        cohort = original["launch"]["cohort"]
        assert [item["name"] for item in cohort] == manifest["development_sequences"][:4]
    else:
        spec = json.loads(args.cohort_spec.read_text())
        assert spec["input_manifest_sha256"] == sha(manifest_path)
        assert len(spec["development_indices"]) == len(set(spec["development_indices"])) == 8
        assert set(spec["development_indices"]) <= set(range(4, 20))
        cohort = spec["cohort"]
        assert [item["name"] for item in cohort] == [
            manifest["development_sequences"][index] for index in spec["development_indices"]]
        assert all(item["evaluated_frames"] == item["video_frames"] for item in cohort)
    assert not set(item["name"] for item in cohort) & (
        set(manifest["train_sequences"]) | set(manifest["heldout_sequences"]))
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    tracker = MissingTracker(cfg, str(checkpoint))
    tracker.network.requires_grad_(False)
    baseline_sha = network_digest(tracker.network)
    started = time.time()
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        physical_gpu=int(args.gpu), evaluator_workers=1, memory_free_at_launch_mib=int(free),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        source_tracking_report_sha256=sha(SOURCE / "report.json"),
        cohort=cohort, cohort_spec_sha256=sha(args.cohort_spec) if args.cohort_spec else None,
        arms=["hold", "velocity"], conditions=["complete", "alternating_outage"],
        frame_block=32, pattern=PATTERN,
        source_sha256={str(path): sha(path) for path in [
            Path(__file__), ROOT / "analyze_epoch05_health28.py",
            ROOT / "evaluate_val97_recovery28.py", RUNTIME / "miss_rgbt/tracker.py",
            RUNTIME / "miss_rgbt/model.py", RUNTIME / "experiments/cadtrack/cadtrack.yaml"]},
        rule="Fixed center velocity from last two observed video-frame states; no GT, "
             "size change, clipping or template write on blind frames",
        scope="Registered train882 development videos, same GPU paired closed-loop control; "
              "ordinary constant velocity, no novel module or official result claim")
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print("LAUNCH " + json.dumps(launch), flush=True)
    records, references = {}, {}
    for arm in launch["arms"]:
        records[arm] = {}
        for item in cohort:
            name = item["name"]
            folder = Path(item["folder"])
            gt_path = folder / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            rgb = sorted((folder / "visible").glob("*.jpg"))
            tir = sorted((folder / "infrared").glob("*.jpg"))
            assert len(rgb) == len(tir) == len(gt) == item["video_frames"]
            frames = item["evaluated_frames"]
            records[arm][name] = {}
            for condition in launch["conditions"]:
                tracker.initialize(get_x_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                                   dict(init_bbox=gt[0].tolist()))
                history = [(0, list(tracker.state))]
                boxes, trace, blind = [list(tracker.state)], [], [False]
                for index in range(1, frames):
                    available = [1, 1] if condition == "complete" else PATTERN[(index // 32) % 6]
                    previous = list(tracker.state)
                    calls = tracker.frame_id
                    template = tracker.templates[-1]
                    confidence = None
                    if any(available):
                        r = rgb[index] if available[0] else tir[index]
                        t = tir[index] if available[1] else rgb[index]
                        actual = tracker.track(get_x_frame(str(r), str(t), dtype="rgbrgb"), available)
                        confidence = actual["best_score"]
                        assert np.isfinite(confidence) and tracker.frame_id == calls + 1
                        history.append((index, list(tracker.state)))
                        history = history[-2:]
                    else:
                        if arm == "velocity":
                            assert len(history) == 2
                            (before_time, before), (last_time, last) = history
                            velocity = (center(last) - center(before)) / (last_time - before_time)
                            forecast = np.asarray(last).copy()
                            forecast[:2] += (index - last_time) * velocity
                            tracker.state = forecast.tolist()
                        else:
                            assert tracker.state == previous
                        assert tracker.frame_id == calls and tracker.templates[-1] is template
                    assert np.isfinite(tracker.state).all() and (np.asarray(tracker.state)[2:] > 0).all()
                    boxes.append(list(tracker.state))
                    blind.append(not any(available))
                    trace.append(dict(
                        frame=index, available=available, bbox=list(tracker.state),
                        track_call=tracker.frame_id, template_written=tracker.templates[-1] is not template,
                        confidence=confidence))
                predicted = np.asarray(boxes)
                target = gt[:frames]
                key = (name, condition)
                if arm == "hold":
                    references[key] = predicted.copy()
                elif condition == "complete" or not any(blind):
                    assert np.array_equal(predicted, references[key])
                else:
                    first_blind = blind.index(True)
                    assert np.array_equal(predicted[:first_blind], references[key][:first_blind])
                basename = arm + "_" + name + "_" + condition
                bbox_path = output / (basename + ".txt")
                trace_path = output / (basename + ".json")
                np.savetxt(str(bbox_path), predicted)
                trace_path.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
                measured = metrics(predicted, target)
                measured.update(
                    frames=frames, blind_frames=sum(blind), bbox_sha256=sha(bbox_path),
                    trace_sha256=sha(trace_path), seconds=time.time() - started)
                records[arm][name][condition] = measured
                (output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
                print("TRACK " + json.dumps(dict(arm=arm, sequence=name, condition=condition,
                                                 metrics=measured)), flush=True)
    assert network_digest(tracker.network) == baseline_sha
    aggregate = {
        condition: {arm: {metric: float(np.mean([
            records[arm][item["name"]][condition][metric] for item in cohort]))
            for metric in ["PR20", "SR"]} for arm in launch["arms"]}
        for condition in launch["conditions"]}
    report = dict(
        launch=launch, records=records, aggregate=aggregate,
        velocity_minus_hold={condition: {
            metric: aggregate[condition]["velocity"][metric] - aggregate[condition]["hold"][metric]
            for metric in ["PR20", "SR"]} for condition in launch["conditions"]},
        baseline_state_unchanged=True, complete_outputs_bitwise_equal=True,
        pre_blind_prefix_bitwise_equal=True, no_blind_video_outputs_bitwise_equal=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.time() - started)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "completed.json").write_text(json.dumps(dict(
        report_sha256=sha(output / "report.json"), model_conditions=4, sequences=len(cohort),
        seconds=time.time() - started), indent=2) + "\n")
    print("COMPLETE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
