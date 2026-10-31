"""Paired native, calibration-sham and offset-corrected E0 development tracks."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from analyze_epoch05_health28 import metrics
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
PATTERN = [[1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, choices=["0", "1"])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import torch
    sys.path.insert(0, str(RUNTIME))
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_x_frame
    from miss_rgbt.tracker import MissingTracker
    from state_offset_tracker28 import StateOffsetTracker
    from evaluate_blind_motion28 import network_digest
    torch.set_num_threads(1)
    args.output.mkdir(exist_ok=False)
    spec = json.loads(args.spec.read_text())
    checkpoint = ROOT / "checkpoints/e0_availability/epoch20.pth.tar"
    packaged = json.loads(checkpoint.with_suffix(".json").read_text())
    assert sha(checkpoint) == packaged["packaged_sha256"]
    assert packaged["optimizer_updates"] == 0 and packaged["tensor_equality"]
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        physical_gpu=int(args.gpu), checkpoint_sha256=sha(checkpoint),
        spec_sha256=sha(args.spec), cohort=spec["cohort"],
        arms=["native", "calibration_sham", "offset_corrected"],
        conditions=["complete", "alternating_outage"], pattern=PATTERN, frame_block=1,
        source_sha256={str(p): sha(p) for p in [
            Path(__file__), ROOT / "state_offset_tracker28.py", RUNTIME / "miss_rgbt/tracker.py",
            RUNTIME / "miss_rgbt/model.py", ROOT / "analyze_epoch05_health28.py"]},
        scope="New sequential development36..39, parameter-free E0 calibration control. "
              "No GT beyond initialization in inference, official score or novel-module claim.")
    save(args.output / "launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    started = time.time()
    records, reference = {}, {}
    for arm in launch["arms"]:
        tracker = (MissingTracker(cfg, str(checkpoint)) if arm == "native" else
                   StateOffsetTracker(cfg, str(checkpoint), apply_offset=arm == "offset_corrected"))
        tracker.network.requires_grad_(False)
        digest = network_digest(tracker.network)
        records[arm] = {}
        for item in spec["cohort"]:
            name, folder = item["name"], Path(item["folder"])
            gt_path = folder / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            rgb = sorted((folder / "visible").glob("*.jpg"))
            tir = sorted((folder / "infrared").glob("*.jpg"))
            assert len(rgb) == len(tir) == len(gt) == item["video_frames"]
            records[arm][name] = {}
            for condition in launch["conditions"]:
                tracker.initialize(get_x_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                                   dict(init_bbox=gt[0].tolist()))
                boxes, trace, calls = [list(tracker.state)], [], 0
                for frame in range(1, len(gt)):
                    flags = [1, 1] if condition == "complete" else PATTERN[frame % len(PATTERN)]
                    templates = list(tracker.templates)
                    previous_call = tracker.frame_id
                    if any(flags):
                        r, t = (rgb[frame] if flags[0] else tir[frame],
                                tir[frame] if flags[1] else rgb[frame])
                        image = get_x_frame(str(r), str(t), dtype="rgbrgb")
                        if arm == "native":
                            result = tracker.track(image, flags)
                            calls += 1
                        else:
                            calibration_images = {}
                            if flags == [1, 1]:
                                calibration_images = {
                                    (0, 1): get_x_frame(str(tir[frame]), str(tir[frame]), dtype="rgbrgb"),
                                    (1, 0): get_x_frame(str(rgb[frame]), str(rgb[frame]), dtype="rgbrgb"),
                                }
                            result = tracker.track(image, flags, calibration_images)
                            calls += result["model_calls"]
                        assert tracker.frame_id == previous_call + 1
                    else:
                        result = dict(best_score=None)
                        assert tracker.frame_id == previous_call
                        assert all(a is b for a, b in zip(templates, tracker.templates))
                    box = list(tracker.state)
                    assert np.isfinite(box).all() and (np.asarray(box)[2:] > 0).all()
                    boxes.append(box)
                    trace.append(dict(
                        frame=frame, available=flags, bbox=box, confidence=result["best_score"],
                        applied_offset=result.get("applied_offset"), raw_bbox=result.get("raw_target_bbox"),
                        unclipped_bbox=result.get("unclipped_target_bbox"),
                        calibration_offsets=result.get("calibration_offsets"),
                        image_shape=list(image.shape[:2]) if any(flags) else None,
                        last_calibration_call=result.get("last_calibration_call"),
                        template_written=any(a is not b for a, b in zip(templates, tracker.templates))))
                boxes = np.asarray(boxes)
                if arm == "native":
                    reference[(name, condition)] = boxes.copy()
                elif condition == "complete" or arm == "calibration_sham":
                    assert np.array_equal(boxes, reference[(name, condition)]), (arm, name, condition)
                prefix = args.output / (arm + "_" + name + "_" + condition)
                np.savetxt(str(prefix) + ".txt", boxes)
                save(Path(str(prefix) + ".json"), trace)
                measured = metrics(boxes, gt)
                measured.update(bbox_sha256=sha(str(prefix) + ".txt"), model_calls=calls,
                                trace_sha256=sha(str(prefix) + ".json"), frames=len(gt))
                records[arm][name][condition] = measured
                save(args.output / "progress.json", records)
                print("TRACK " + json.dumps(dict(arm=arm, name=name, condition=condition,
                                                metrics=measured)), flush=True)
        assert network_digest(tracker.network) == digest
        del tracker
        torch.cuda.empty_cache()
    report = dict(launch=launch, records=records, seconds=time.time() - started,
                  sham_outputs_bitwise_equal=True, complete_outputs_bitwise_equal=True,
                  baseline_state_unchanged=True)
    save(args.output / "report.json", report)
    save(args.output / "completed.json", dict(report_sha256=sha(args.output / "report.json")))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
