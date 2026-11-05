"""Native validation trajectories and explicit macro PR20/SR metrics."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
sys.path.insert(0, str(RUNTIME))
from evaluate_val97_recovery28 import sha


def initialize(checkpoint, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_x_frame
    from miss_rgbt.tracker import MissingTracker
    global TRACKER, READ
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    TRACKER = MissingTracker(cfg, checkpoint)
    READ = get_x_frame


def sequence(job):
    item, data_root, mask, destination = job
    name, frames = item["name"], item["frames"]
    assert sha(item["gt_path"]) == item["gt_sha256"]
    gt = np.loadtxt(item["gt_path"], delimiter=",", ndmin=2)
    folder = Path(data_root) / name
    rgb = sorted((folder / "visible").glob("*.jpg"))
    tir = sorted((folder / "infrared").glob("*.jpg"))
    assert len(rgb) == len(tir) == len(gt) == len(mask) == frames
    boxes = [gt[0].tolist()]
    TRACKER.initialize(READ(str(rgb[0]), str(tir[0]), dtype="rgbrgb"), dict(init_bbox=boxes[0]))
    for index in range(1, frames):
        r, t = mask[index]
        if r or t:
            boxes.append(TRACKER.track(READ(str(rgb[index] if r else tir[index]),
                                           str(tir[index] if t else rgb[index]),
                                           dtype="rgbrgb"), [r, t])["target_bbox"])
        else:
            boxes.append(list(boxes[-1]))
    predicted = np.asarray(boxes)
    assert predicted.shape == gt.shape and np.isfinite(predicted).all()
    path = Path(destination) / (name + ".txt")
    np.savetxt(str(path), predicted)
    valid = (gt[:, 2:] > 0).all(1)
    valid[0] = False
    assert valid.any()
    p, target = predicted[valid], gt[valid]
    error = np.linalg.norm(p[:, :2] + p[:, 2:] / 2 - target[:, :2] - target[:, 2:] / 2, axis=1)
    intersection = np.maximum(0., np.minimum(p[:, :2] + p[:, 2:], target[:, :2] + target[:, 2:])
                              - np.maximum(p[:, :2], target[:, :2])).prod(1)
    iou = intersection / (p[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection)
    return dict(sequence=name, video_frames=frames, scored_frames=int(valid.sum()),
                PR20=float((error <= 20).mean() * 100),
                SR=float((iou[:, None] > np.linspace(0, 1, 21)).mean() * 100),
                bbox_sha256=sha(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", required=True, choices=["0", "1", "2", "3"])
    parser.add_argument("--condition", required=True, choices=["complete", "alternating_outage"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "development/recovery_val97_20261030/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    spec = protocol["conditions"][args.condition]
    assert sha(spec["path"]) == spec["sha256"]
    masks = json.loads(Path(spec["path"]).read_text())
    cohort = protocol["sequences"][:args.limit]
    launch = dict(checkpoint=args.checkpoint, checkpoint_sha256=sha(args.checkpoint),
                  physical_gpu=args.gpu, condition=args.condition, workers=args.workers,
                  protocol_sha256=sha(protocol_path), source_sha256=sha(__file__),
                  expected_sequences=len(cohort), expected_frames=sum(item["frames"] for item in cohort),
                  metric_protocol="Macro sequence PR at20px and SR at21 thresholds0..1 with IoU>threshold; "
                                  "initialization/nonpositive-size GT excluded, both-missing held frames included. "
                                  "Development-only metrics, not official benchmark scores.")
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    jobs = [(item, protocol["data_root"], masks[item["name"]]["data"], str(output)) for item in cohort]
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=initialize,
                                      initargs=(args.checkpoint, args.gpu)) as pool:
        results = []
        for receipt in pool.imap_unordered(sequence, jobs):
            results.append(receipt)
            (output / "receipts.json").write_text(json.dumps(results, indent=2) + "\n")
            print("SEQUENCE " + json.dumps(receipt), flush=True)
    assert len(results) == len(cohort)
    metrics = dict(sequences=len(results), frames=sum(item["video_frames"] for item in results),
                   PR20=float(np.mean([item["PR20"] for item in results])),
                   SR=float(np.mean([item["SR"] for item in results])),
                   seconds=time.time() - started, condition=args.condition,
                   checkpoint_sha256=launch["checkpoint_sha256"])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "completed.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("COMPLETE " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
