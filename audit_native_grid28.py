"""Fixed-budget spatial proposals at observed previous and initialization scales."""
import datetime
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import torch

ROOT = Path("/home/gaob/Miss-RGBT")
sys.path.insert(0, str(ROOT / "runtime/task_utility_20261018"))
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.data.processing_utils import sample_target
from lib.train.dataset.depth_utils import get_x_frame
from lib.utils.box_ops import clip_box
from cpu_diagnostic_tracker import CpuDiagnosticTracker
from evaluate_val97_recovery28 import geometry, sha

OUTPUT = ROOT / "reports/server28-native-grid-20261030"


def summarize(rows):
    valid = [row for row in rows if row["candidates"][0]["diagnostic"] is not None]
    result = dict(observations=len(rows), valid=len(valid))
    if not valid:
        return result
    for rule in ["native", "oracle", "hann"]:
        values = []
        for row in valid:
            candidates = row["candidates"]
            if rule == "native":
                candidate = candidates[0]
            elif rule == "oracle":
                candidate = max(candidates, key=lambda item: item["diagnostic"]["iou"])
            else:
                candidate = max(candidates, key=lambda item: item["confidence"])
            values.append(candidate["diagnostic"]["iou"])
        result[rule] = dict(
            mean_iou=sum(values) / len(values), iou_at_least_half=sum(value >= .5 for value in values),
            harms_over_point_one=sum(value < row["candidates"][0]["diagnostic"]["iou"] - .1
                                     for value, row in zip(values, valid)))
    failures = [row for row in valid
                if not row["candidates"][0]["diagnostic"]["target_overlaps_previous_crop"]]
    result["native_unsupported"] = len(failures)
    for label in ["previous", "initial"]:
        result[label] = dict(
            unsupported_support_restored=sum(any(
                candidate["diagnostic"]["target_overlaps_previous_crop"]
                for candidate in row["candidates"] if candidate["scale"] == label) for row in failures),
            unsupported_center_restored=sum(any(
                candidate["diagnostic"]["target_center_in_previous_crop"]
                for candidate in row["candidates"] if candidate["scale"] == label) for row in failures),
            unsupported_oracle_iou_half=sum(max(
                candidate["diagnostic"]["iou"]
                for candidate in row["candidates"] if candidate["scale"] == label) >= .5 for row in failures))
    return result


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    OUTPUT.mkdir(exist_ok=False)
    source_path = ROOT / "reports/server28-global-proposals-20261030/report.json"
    source = json.loads(source_path.read_text())
    protocol_path = ROOT / "development/recovery_val97_20261030/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    specs = {item["name"]: item for item in protocol["sequences"]}
    checkpoint = ROOT / "artifacts/server25-data2/checkpoints/a_utility_full20_seed2027/epoch20.pth.tar"
    update_config_from_file(str(ROOT / "runtime/task_utility_20261018/experiments/cadtrack/cadtrack.yaml"))
    tracker = CpuDiagnosticTracker(cfg, str(checkpoint))
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        source_sha256=sha(__file__), checkpoint_sha256=sha(checkpoint),
        input_report_sha256=sha(source_path), protocol_sha256=sha(protocol_path),
        protocol="Same46 points as global audit. Native crop plus5x5 image cell centers "
                 "at previous predicted box size and initialization box size,51 crops. "
                 "All native-resolution crops. Initial templates, unchanged Hann/head. "
                 "CPU minibatch8; chunked native prediction checked against prior audit.",
        scope="No current GT generates proposals, chooses frames or ranks responses. "
              "GT only labels predictions after inference. Fixed budget does not guarantee "
              "full-image coverage. One-step audit, not closed-loop recovery, trigger "
              "calibration, final PR/SR, validated module or novelty.",
    )
    (OUTPUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    started, rows, inputs = time.time(), [], {}
    for observation in source["observations"]:
        name, index = observation["sequence"], observation["frame"]
        if name not in inputs:
            spec = specs[name]
            assert sha(spec["gt_path"]) == spec["gt_sha256"]
            folder = Path(protocol["data_root"]) / name
            rgb = sorted((folder / "visible").glob("*.jpg"))
            tir = sorted((folder / "infrared").glob("*.jpg"))
            gt = np.loadtxt(spec["gt_path"], delimiter=",", ndmin=2)
            tracker.initialize(get_x_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                               dict(init_bbox=gt[0].tolist()))
            inputs[name] = (rgb, tir, gt, list(tracker.templates))
        rgb, tir, gt, initial = inputs[name]
        available = observation["available"]
        image = get_x_frame(str(rgb[index] if available[0] else tir[index]),
                            str(tir[index] if available[1] else rgb[index]), dtype="rgbrgb")
        height, width = image.shape[:2]
        previous = observation["candidates"][0]["crop_box"]
        proposals = [dict(scale="native", box=list(previous))]
        for label, size in [("previous", previous[2:]), ("initial", gt[0, 2:].tolist())]:
            for y in range(5):
                for x in range(5):
                    cx, cy = width * (x + .5) / 5., height * (y + .5) / 5.
                    proposals.append(dict(scale=label, box=[
                        cx - size[0] / 2., cy - size[1] / 2., size[0], size[1]]))
        candidates = []
        for start in range(0, len(proposals), 8):
            chunk = proposals[start:start + 8]
            crops, factors = [], []
            for proposal in chunk:
                crop, factor, _ = sample_target(image, proposal["box"], cfg.TEST.SEARCH_FACTOR,
                                                output_sz=cfg.TEST.SEARCH_SIZE)
                crops.append(tracker.preprocessor.process(crop))
                factors.append(factor)
            count = len(chunk)
            with torch.no_grad():
                templates = [template.expand(count, -1, -1, -1) for template in initial]
                availability = [torch.ones(count, 2) for _ in initial]
                availability.append(torch.tensor([available], dtype=torch.float32).expand(count, -1))
                out = tracker.network(templates, [torch.cat(crops)], availability)[0]
                response = tracker.window * out["score_map"]
                predicted = tracker.network.baseline.box_head.cal_bbox(
                    response, out["size_map"], out["offset_map"]).view(count, 4)
            for offset, proposal in enumerate(chunk):
                crop_box = proposal["box"]
                cx, cy, box_width, box_height = (
                    predicted[offset] * cfg.TEST.SEARCH_SIZE / factors[offset]).tolist()
                half = .5 * cfg.TEST.SEARCH_SIZE / factors[offset]
                box = clip_box([
                    cx + crop_box[0] + crop_box[2] / 2. - half - box_width / 2.,
                    cy + crop_box[1] + crop_box[3] / 2. - half - box_height / 2.,
                    box_width, box_height], height, width, margin=10)
                assert np.isfinite(box).all()
                candidates.append(dict(scale=proposal["scale"], crop_box=crop_box, bbox=box,
                                       confidence=float(response[offset].max()),
                                       diagnostic=geometry(crop_box, gt[index].tolist(),
                                                           box, cfg.TEST.SEARCH_FACTOR)))
        old_native = observation["candidates"][0]["hann_bbox"]
        difference = max(abs(first - second)
                         for first, second in zip(old_native, candidates[0]["bbox"]))
        assert np.allclose(old_native, candidates[0]["bbox"], rtol=1e-4, atol=.05)
        rows.append(dict(sequence=name, frame=index, available=available, candidates=candidates,
                         native_vs_prior_max_bbox_difference=difference))
        (OUTPUT / "report.json").write_text(json.dumps(dict(launch=launch, observations=rows),
                                                      indent=2) + "\n")
        print("PROGRESS " + json.dumps(dict(sequence=name, frame=index, observations=len(rows),
                                            seconds=time.time() - started)), flush=True)
    assert len(rows) == 46
    result = dict(summary=summarize(rows),
                  per_sequence={name: summarize([row for row in rows if row["sequence"] == name])
                                for name in inputs},
                  seconds=time.time() - started)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUTPUT / "completed.json").write_text(json.dumps(
        dict(observations=len(rows), candidates=len(rows) * 51, seconds=result["seconds"]),
        indent=2) + "\n")
    print("COMPLETE " + json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
