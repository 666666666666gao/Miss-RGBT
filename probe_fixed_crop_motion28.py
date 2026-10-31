"""CPU factorial diagnostic separating temporal lag from sensor-state pairs."""
import datetime
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                  MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")

import cv2
import numpy as np
import torch

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
sys.path.insert(0, str(RUNTIME))

from cpu_diagnostic_tracker import CpuDiagnosticTracker
from evaluate_blind_motion28 import network_digest
from evaluate_val97_recovery28 import sha
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.dataset.depth_utils import get_x_frame


SOURCE = ROOT / "reports/server28-blind-motion-shortgap16-20261031"
OUTPUT = ROOT / "reports/server28-fixed-crop-motion-factorial-20261031"
STATES = {"11": [1, 1], "01": [0, 1], "10": [1, 0]}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def center(box):
    value = np.asarray(box)
    return value[:2] + value[2:] / 2


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    started = time.time()
    parent = json.loads((SOURCE / "report.json").read_text())
    assert sha(SOURCE / "report.json") == json.loads(
        (SOURCE / "completed.json").read_text())["report_sha256"]
    checkpoint = ROOT / "checkpoints/e0_availability/epoch20.pth.tar"
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    tracker = CpuDiagnosticTracker(cfg, str(checkpoint))
    tracker.network.requires_grad_(False)
    digest = network_digest(tracker.network)
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        checkpoint_sha256=sha(checkpoint), parent_report_sha256=sha(SOURCE / "report.json"),
        device="cpu", threads=4, source_sha256={
            str(path): sha(path) for path in [
                Path(__file__), ROOT / "cpu_diagnostic_tracker.py",
                RUNTIME / "miss_rgbt/model.py", RUNTIME / "miss_rgbt/tracker.py"]},
        blind_targets=[5, 11, 17, 23], lags=[1, 2], prior_states=list(STATES),
        current_state="11", expected_model_calls=576,
        crop_rule="Saved complete-track previous-frame box at each observation time",
        template_rule="First-frame complete templates fixed; reset call index before each probe",
        scope="Reused16 development videos; fixed-crop CPU counterfactual, not closed-loop "
              "tracking or an official/module gain. GT after initialization used only for metrics.")
    save(OUTPUT / "launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    records = []
    calls = 0
    for gpu in [0, 1]:
        folder = SOURCE / ("gpu%d" % gpu)
        worker = json.loads((folder / "report.json").read_text())
        assert sha(folder / "report.json") == parent["worker_receipts"][str(gpu)]["report_sha256"]
        for item in worker["launch"]["cohort"]:
            name = item["name"]
            video = Path(item["folder"])
            assert sha(video / "init.txt") == item["gt_sha256"]
            gt = np.loadtxt(str(video / "init.txt"), delimiter=",", ndmin=2)
            path = folder / ("hold_" + name + "_complete.txt")
            assert sha(path) == worker["records"]["hold"][name]["complete"]["bbox_sha256"]
            reference = np.loadtxt(str(path), ndmin=2)
            rgb = sorted((video / "visible").glob("*.jpg"))
            tir = sorted((video / "infrared").glob("*.jpg"))
            assert len(rgb) == len(tir) == len(gt) == len(reference)
            tracker.initialize(get_x_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                               dict(init_bbox=gt[0].tolist()))
            initial_templates = list(tracker.templates)
            initial_availability = list(tracker.availability)
            observed = {}
            for target in launch["blind_targets"]:
                last = target - 1
                for frame in [last - 2, last - 1, last]:
                    for state, flags in STATES.items():
                        tracker.state = reference[frame - 1].tolist()
                        tracker.frame_id = 0
                        tracker.templates = list(initial_templates)
                        tracker.availability = list(initial_availability)
                        r = rgb[frame] if flags[0] else tir[frame]
                        t = tir[frame] if flags[1] else rgb[frame]
                        actual = tracker.track(get_x_frame(str(r), str(t), dtype="rgbrgb"), flags)
                        assert all(a is b for a, b in zip(tracker.templates, initial_templates))
                        box = actual["target_bbox"]
                        assert np.isfinite(box).all()
                        observed[(frame, state)] = box
                        calls += 1
                for lag in launch["lags"]:
                    before = last - lag
                    for state in STATES:
                        last_box = observed[(last, "11")]
                        prior_box = observed[(before, state)]
                        velocity = (center(last_box) - center(prior_box)) / lag
                        row = dict(
                            sequence=name, blind_target=target, lag=lag, prior_state=state,
                            before=before, last=last, before_box=prior_box, last_box=last_box,
                            velocity=velocity.tolist(),
                            same_state=state == "11")
                        if (gt[[before, last, target], 2:] > 0).all():
                            true_velocity = (center(gt[last]) - center(gt[before])) / lag
                            forecast = center(last_box) + velocity
                            row["retrospective_gt"] = dict(
                                velocity_error=float(np.linalg.norm(velocity - true_velocity)),
                                forecast_center_error=float(np.linalg.norm(forecast - center(gt[target]))))
                        records.append(row)
            save(OUTPUT / "progress.json", dict(calls=calls, records=len(records),
                                                last_sequence=name, seconds=time.time() - started))
            print(json.dumps(dict(sequence=name, calls=calls,
                                  seconds=time.time() - started)), flush=True)
    assert calls == launch["expected_model_calls"]
    assert network_digest(tracker.network) == digest
    save(OUTPUT / "report.json", dict(launch=launch, records=records, calls=calls,
                                     baseline_state_unchanged=True, seconds=time.time() - started))
    save(OUTPUT / "completed.json", dict(report_sha256=sha(OUTPUT / "report.json"),
                                        calls=calls, seconds=time.time() - started))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
