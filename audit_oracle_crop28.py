"""GT crop interventions for diagnosis only; never deployable recovery."""
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

OUTPUT = ROOT / "reports/server28-oracle-crop-20261030"


def summarize(rows):
    result = dict(observations=len(rows))
    if not rows:
        return result
    for rule in ["native", "global_hann_oracle", "global_raw_oracle",
                 "center_previous_scale", "center_true_scale", "center_tile_scale"]:
        values = [row[rule + "_iou"] for row in rows]
        result[rule] = dict(mean_iou=sum(values) / len(values),
                            iou_at_least_half=sum(value >= .5 for value in values))
    return result


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    OUTPUT.mkdir(exist_ok=False)
    report_path = ROOT / "reports/server28-global-proposals-20261030/report.json"
    source = json.loads(report_path.read_text())
    protocol_path = ROOT / "development/recovery_val97_20261030/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    specs = {item["name"]: item for item in protocol["sequences"]}
    checkpoint = ROOT / "artifacts/server25-data2/checkpoints/a_utility_full20_seed2027/epoch20.pth.tar"
    update_config_from_file(str(ROOT / "runtime/task_utility_20261018/experiments/cadtrack/cadtrack.yaml"))
    tracker = CpuDiagnosticTracker(cfg, str(checkpoint))
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        source_sha256=sha(__file__), checkpoint_sha256=sha(checkpoint),
        proposal_report_sha256=sha(report_path), protocol_sha256=sha(protocol_path),
        protocol="All45 valid-GT points from preregistered global audit. GT centers three "
                 "diagnostic crops: previous predicted box size, true current GT box size, "
                 "half-image crop side. Initial templates, unchanged Hann inference. "
                 "No trainable parameters, thresholds or additional frame sampling.",
        scope="Uses current GT to generate crops. Diagnostic intervention only, not deployable "
              "recovery, official score, guaranteed upper bound or module efficacy. "
              "Centering and scale interact with native Hann prior and context.",
    )
    (OUTPUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    started, records, inputs = time.time(), [], {}
    for item in source["observations"]:
        native = item["candidates"][0]
        if native["hann_diagnostic"] is None:
            continue
        name, index = item["sequence"], item["frame"]
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
        available = item["available"]
        image = get_x_frame(str(rgb[index] if available[0] else tir[index]),
                            str(tir[index] if available[1] else rgb[index]), dtype="rgbrgb")
        target = gt[index].tolist()
        assert target[2] > 0 and target[3] > 0
        center = [target[0] + target[2] / 2., target[1] + target[3] / 2.]
        tile_size = max(image.shape[:2]) / (2. * cfg.TEST.SEARCH_FACTOR)
        sizes = [
            ("center_previous_scale", native["crop_box"][2:]),
            ("center_true_scale", target[2:]),
            ("center_tile_scale", [tile_size, tile_size]),
        ]
        crops, factors, boxes = [], [], []
        for _, size in sizes:
            box = [center[0] - size[0] / 2., center[1] - size[1] / 2.] + list(size)
            crop, factor, _ = sample_target(image, box, cfg.TEST.SEARCH_FACTOR,
                                            output_sz=cfg.TEST.SEARCH_SIZE)
            crops.append(tracker.preprocessor.process(crop))
            factors.append(factor)
            boxes.append(box)
        with torch.no_grad():
            templates = [template.expand(3, -1, -1, -1) for template in initial]
            availability = [torch.ones(3, 2) for _ in templates]
            availability.append(torch.tensor([available], dtype=torch.float32).expand(3, -1))
            out = tracker.network(templates, [torch.cat(crops)], availability)[0]
            predicted = tracker.network.baseline.box_head.cal_bbox(
                tracker.window * out["score_map"], out["size_map"], out["offset_map"]).view(3, 4)
        row = dict(sequence=name, frame=index, available=available, target=target,
                   native_no_support=not native["hann_diagnostic"]["target_overlaps_previous_crop"],
                   native_iou=native["hann_diagnostic"]["iou"],
                   global_hann_oracle_iou=max(candidate["hann_diagnostic"]["iou"]
                                              for candidate in item["candidates"]),
                   global_raw_oracle_iou=max(candidate["raw_diagnostic"]["iou"]
                                             for candidate in item["candidates"]),
                   interventions={})
        for offset, (label, _) in enumerate(sizes):
            cx, cy, width, height = (
                predicted[offset] * cfg.TEST.SEARCH_SIZE / factors[offset]).tolist()
            half = .5 * cfg.TEST.SEARCH_SIZE / factors[offset]
            box = clip_box([cx + center[0] - half - width / 2.,
                            cy + center[1] - half - height / 2., width, height],
                           image.shape[0], image.shape[1], margin=10)
            assert np.isfinite(box).all()
            diagnostic = geometry(boxes[offset], target, box, cfg.TEST.SEARCH_FACTOR)
            row[label + "_iou"] = diagnostic["iou"]
            row["interventions"][label] = dict(crop_box=boxes[offset], bbox=box,
                                               diagnostic=diagnostic,
                                               target_search_pixels=[value * factors[offset]
                                                                     for value in target[2:]])
        records.append(row)
        (OUTPUT / "report.json").write_text(json.dumps(dict(launch=launch, observations=records),
                                                      indent=2) + "\n")
        if len(records) % 6 == 0:
            print("PROGRESS " + json.dumps(dict(observations=len(records),
                                               seconds=time.time() - started)), flush=True)
    assert len(records) == 45
    result = dict(all=summarize(records),
                  unsupported=summarize([row for row in records if row["native_no_support"]]),
                  per_sequence={name: summarize([row for row in records if row["sequence"] == name])
                                for name in inputs},
                  seconds=time.time() - started)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUTPUT / "completed.json").write_text(json.dumps(
        dict(observations=len(records), seconds=result["seconds"]), indent=2) + "\n")
    print("COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
