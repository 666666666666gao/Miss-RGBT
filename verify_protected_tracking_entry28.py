"""Check the new CPU tracking entry against a zero-initialized repair."""
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import torch

from evaluate_protected_development28 import (
    ROOT, RUNTIME, PILOT, ProtectedCpuTracker, cfg, digest,
    load_pretrained, sha, track, update_config_from_file)


def main():
    started = time.time()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    launch = json.loads((PILOT / "launch.json").read_text())
    checkpoint = Path(launch["checkpoint"])
    assert sha(checkpoint) == launch["checkpoint_sha256"]
    inputs = json.loads((PILOT / "inputs.json").read_text())
    folder = Path("/data/wangwj/dataset/LasHeR/trainingset") / inputs["development_sequences"][0]
    gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
    item = dict(name=folder.name, folder=str(folder), video_frames=len(gt),
                evaluated_frames=33, gt_sha256=sha(folder / "init.txt"))
    boxes, states, results = {}, {}, {}
    for repair in [False, True]:
        torch.manual_seed(2027)
        model = load_pretrained(cfg, str(checkpoint), repair=repair, width=128, repair_layer=6)
        states[repair] = digest(model.baseline.state_dict())
        if repair:
            assert torch.count_nonzero(model.repair.delta.weight) == 0
            assert torch.count_nonzero(model.repair.delta.bias) == 0
        tracker = ProtectedCpuTracker(model)
        results[str(repair)] = {}
        for condition in ["complete", "alternating_outage"]:
            predictions, rows, metrics = track(tracker, item, condition)
            if repair:
                assert np.array_equal(predictions, boxes[condition])
            else:
                boxes[condition] = predictions
            assert metrics["frames"] == 33 and metrics["track_calls"] == 32
            results[str(repair)][condition] = metrics
        assert states[repair] == digest(model.baseline.state_dict())
        del tracker, model
    assert states[False] == states[True]
    result = dict(status="ZERO_REPAIR_TRACKING_ENTRY_PARITY_PASSED", frames_per_condition=33,
                  sequence=item["name"], model_conditions=4, results=results,
                  seconds=time.time() - started,
                  runner_sha256=sha(ROOT / "evaluate_protected_development28.py"),
                  scope="Numerical entry smoke on first33 frames, including first missing "
                        "search observation. Not selected-repair or benchmark evidence.")
    (ROOT / "reports/server28-protected-tracking-entry-smoke-20261030.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
