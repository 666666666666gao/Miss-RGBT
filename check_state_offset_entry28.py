"""CPU witness of the new track body using the existing explicit CPU E0 port."""
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                  MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")

import cv2
import numpy as np
import torch

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
sys.path.insert(0, str(RUNTIME))
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.dataset.depth_utils import get_x_frame
from cpu_diagnostic_tracker import CpuDiagnosticTracker
from evaluate_blind_motion28 import network_digest
from evaluate_val97_recovery28 import sha
from state_offset_tracker28 import StateOffsetTracker


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    tracker = CpuDiagnosticTracker(cfg, str(ROOT / "checkpoints/e0_availability/epoch20.pth.tar"))
    tracker.network.requires_grad_(False)
    digest = network_digest(tracker.network)
    manifest = json.loads((ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json").read_text())
    video = Path("/data/wangwj/dataset/LasHeR/trainingset") / manifest["development_sequences"][36]
    gt = np.loadtxt(str(video / "init.txt"), delimiter=",", ndmin=2)
    rgb, tir = sorted((video / "visible").glob("*.jpg")), sorted((video / "infrared").glob("*.jpg"))
    pattern = [[1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]]
    records = {}
    for condition in ["complete", "alternating_outage"]:
        tracker.initialize(get_x_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                           dict(init_bbox=gt[0].tolist()))
        candidates = {}
        for arm in ["sham", "corrected"]:
            candidate = StateOffsetTracker.__new__(StateOffsetTracker)
            candidate.__dict__.update(tracker.__dict__)
            candidate.templates = list(tracker.templates)
            candidate.availability = list(tracker.availability)
            candidate.state = list(tracker.state)
            candidate.apply_offset = arm == "corrected"
            candidate.offsets = {(0, 1): np.zeros(2), (1, 0): np.zeros(2)}
            candidate.last_calibration_frame = None
            candidates[arm] = candidate
        records[condition] = []
        for frame in range(1, 8):
            flags = [1, 1] if condition == "complete" else pattern[frame % 6]
            if not any(flags):
                continue
            r = rgb[frame] if flags[0] else tir[frame]
            t = tir[frame] if flags[1] else rgb[frame]
            image = get_x_frame(str(r), str(t), dtype="rgbrgb")
            calibration = ({
                (0, 1): get_x_frame(str(tir[frame]), str(tir[frame]), dtype="rgbrgb"),
                (1, 0): get_x_frame(str(rgb[frame]), str(rgb[frame]), dtype="rgbrgb")
            } if flags == [1, 1] else {})
            baseline = tracker.track(image, flags)["target_bbox"]
            outputs = {arm: candidate.track(image, flags, calibration)
                       for arm, candidate in candidates.items()}
            assert np.array_equal(baseline, outputs["sham"]["target_bbox"])
            if condition == "complete":
                assert np.array_equal(baseline, outputs["corrected"]["target_bbox"])
            records[condition].append(dict(
                frame=frame, available=flags, native_bbox=baseline,
                corrected_bbox=outputs["corrected"]["target_bbox"],
                correction=outputs["corrected"]["applied_offset"]))
    assert network_digest(tracker.network) == digest
    result = dict(
        source_sha256={str(path): sha(path) for path in [
            Path(__file__), ROOT / "state_offset_tracker28.py", ROOT / "cpu_diagnostic_tracker.py"]},
        records=records, sham_bitwise_equal=True, complete_bitwise_equal=True,
        baseline_state_unchanged=True,
        scope="Seven-frame CPU track-body witness with CPU-initialized attributes, not GPU "
              "constructor/initialization, full tracking, numerical equivalence or metrics.")
    with (ROOT / "reports/server28-state-offset-entry-20261031.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
