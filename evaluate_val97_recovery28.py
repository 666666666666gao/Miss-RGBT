"""Development-only closed-loop trajectory and crop-support diagnostics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry(previous, target, prediction, factor):
    if target[2] <= 0 or target[3] <= 0:
        return None
    side = math.ceil(math.sqrt(previous[2] * previous[3]) * factor)
    lower = [round(previous[i] + previous[i + 2] / 2 - side / 2) for i in (0, 1)]
    center = [target[i] + target[i + 2] / 2 for i in (0, 1)]
    center_inside = all(lower[i] <= center[i] < lower[i] + side for i in (0, 1))
    overlap = all(min(lower[i] + side, target[i] + target[i + 2]) >
                  max(lower[i], target[i]) for i in (0, 1))
    overlap_widths = [max(0., min(prediction[i] + prediction[i + 2], target[i] + target[i + 2]) -
                          max(prediction[i], target[i])) for i in (0, 1)]
    intersection = overlap_widths[0] * overlap_widths[1]
    union = prediction[2] * prediction[3] + target[2] * target[3] - intersection
    return dict(target_center_in_previous_crop=center_inside,
                target_overlaps_previous_crop=overlap, iou=intersection / union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", required=True,
                        choices=["complete", "rgb_outage", "tir_outage", "alternating_outage"])
    parser.add_argument("--gpu", required=True, choices=["0", "1"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch

    root = Path("/home/gaob/Miss-RGBT")
    runtime = root / "runtime/task_utility_20261018"
    sys.path.insert(0, str(runtime))
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_x_frame
    from miss_rgbt.tracker import MissingTracker

    torch.set_num_threads(1)
    protocol_path = root / "development/recovery_val97_20261030/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    mask_spec = protocol["conditions"][args.condition]
    mask_path = Path(mask_spec["path"])
    assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == mask_spec["sha256"]
    masks = json.loads(mask_path.read_text())
    update_config_from_file(str(runtime / "experiments/cadtrack/cadtrack.yaml"))
    tracker = MissingTracker(cfg, args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps(dict(
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha(args.checkpoint),
        condition=args.condition, physical_gpu=args.gpu, protocol_sha256=sha(protocol_path),
        diagnostic_source_sha256=sha(__file__),
        tracker_source_sha256=sha(runtime / "miss_rgbt/tracker.py"),
        yaml_sha256=sha(runtime / "experiments/cadtrack/cadtrack.yaml"),
        scope="val97 development diagnostics only; not official benchmark scores",
    ), indent=2) + "\n")
    for item in protocol["sequences"]:
        name, frames = item["name"], item["frames"]
        gt_path = Path(item["gt_path"])
        assert hashlib.sha256(gt_path.read_bytes()).hexdigest() == item["gt_sha256"]
        gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
        folder = Path(protocol["data_root"]) / name
        rgb = sorted((folder / "visible").glob("*.jpg"))
        tir = sorted((folder / "infrared").glob("*.jpg"))
        mask = masks[name]["data"]
        assert len(rgb) == len(tir) == len(gt) == len(mask) == frames
        tracker.initialize(get_x_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                           dict(init_bbox=gt[0].tolist()))
        rows = []
        template_source_frame = 0
        for index in range(1, frames):
            available = mask[index]
            previous = list(tracker.state)
            previous_template = tracker.templates[-1]
            confidence = None
            observed = any(available)
            if observed:
                rgb_path = rgb[index] if available[0] else tir[index]
                tir_path = tir[index] if available[1] else rgb[index]
                result = tracker.track(get_x_frame(str(rgb_path), str(tir_path), dtype="rgbrgb"), available)
                confidence = result["best_score"]
            prediction = list(tracker.state)
            assert np.isfinite(prediction).all()
            template_written = tracker.templates[-1] is not previous_template
            if template_written:
                template_source_frame = index
            diagnostic = geometry(previous, gt[index].tolist(), prediction, cfg.TEST.SEARCH_FACTOR)
            if diagnostic is not None and not observed:
                diagnostic["target_center_in_previous_crop"] = None
                diagnostic["target_overlaps_previous_crop"] = None
            rows.append(dict(frame=index, available=available, observed=observed,
                             restored=observed and not all(mask[index - 1]) and all(available),
                             previous_box=previous, bbox=prediction,
                             track_call=tracker.frame_id, template_written=template_written,
                             dynamic_template_source_frame=template_source_frame,
                             hann_confidence=confidence, diagnostic=diagnostic))
        path = output / (name + ".json")
        path.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
        print(json.dumps(dict(sequence=name, frames=frames, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())), flush=True)
    (output / "completed.json").write_text(json.dumps(dict(
        sequences=97, frames=54646, condition=args.condition,
        scope="development trajectories only; no official metric computation",
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
