"""Fixed whole-image/coarse-tile proposal audit with the frozen native tracker."""
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import torch

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
OUTPUT = ROOT / "reports/server28-global-proposals-20261030"
sys.path.insert(0, str(RUNTIME))
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.data.processing_utils import sample_target
from lib.train.dataset.depth_utils import get_x_frame
from lib.utils.box_ops import clip_box
from cpu_diagnostic_tracker import CpuDiagnosticTracker
from evaluate_val97_recovery28 import geometry, sha


def summarize(rows):
    valid = [row for row in rows if row["candidates"][0]["hann_diagnostic"] is not None]
    result = dict(observations=len(rows), valid_observations=len(valid))
    if not valid:
        return result
    native_ious, oracle_ious, hann_ious, raw_ious = [], [], [], []
    missing, covered, oracle_recovered = 0, 0, 0
    for row in valid:
        candidates = row["candidates"]
        native = candidates[0]["hann_diagnostic"]["iou"]
        oracle = max(candidate["hann_diagnostic"]["iou"] for candidate in candidates)
        hann = max(candidates, key=lambda item: item["hann_confidence"])["hann_diagnostic"]["iou"]
        raw = max(candidates, key=lambda item: item["raw_confidence"])["raw_diagnostic"]["iou"]
        native_ious.append(native)
        oracle_ious.append(oracle)
        hann_ious.append(hann)
        raw_ious.append(raw)
        if not candidates[0]["hann_diagnostic"]["target_overlaps_previous_crop"]:
            missing += 1
            covered += int(any(candidate["hann_diagnostic"]["target_overlaps_previous_crop"]
                               for candidate in candidates))
            oracle_recovered += int(oracle >= .5)
    for label, values in [("native", native_ious), ("hann_oracle", oracle_ious),
                          ("hann_selected", hann_ious), ("raw_selected", raw_ious)]:
        result[label] = dict(mean_iou=sum(values) / len(values),
                             iou_at_least_half=sum(value >= .5 for value in values),
                             harms_over_point_one=sum(value < native - .1
                                                      for value, native in zip(values, native_ious)),
                             improvements_over_point_one=sum(value > native + .1
                                                            for value, native in zip(values, native_ious)))
    result.update(native_no_support=missing, global_restores_support=covered,
                  hann_oracle_recovers_iou_half=oracle_recovered)
    return result


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    OUTPUT.mkdir(exist_ok=False)
    protocol_path = ROOT / "development/recovery_val97_20261030/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    checkpoint = ROOT / "artifacts/server25-data2/checkpoints/a_utility_full20_seed2027/epoch20.pth.tar"
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    tracker = CpuDiagnosticTracker(cfg, str(checkpoint))
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        checkpoint_sha256=sha(checkpoint), source_sha256=sha(__file__),
        protocol_sha256=sha(protocol_path), cpu_threads=4,
        selection="Fixed first8 val sequences. For each observed11/01/10 state, choose "
                  "25% and75% positions in that state's full observed-frame list, deduplicated. "
                  "No GT, scores or failure filtering for sampling/proposals.",
        candidates="Native previous-box crop, whole-image square centered on image with "
                   "side=max(height,width), nine half-side squares on image quarter/half/"
                   "three-quarter centers. All resize to native search resolution. Initial "
                   "templates only. Compare native Hann and no-Hann regression/selection.",
        scope="One-step proposal audit on prerecorded GPU trajectories. No closed-loop "
              "recovery, dynamic-template equivalence, trigger calibration, official metric, "
              "novelty or module gain. Single-image tiling is an ordinary baseline. GT only "
              "evaluates support and IoU after all predictions.",
    )
    (OUTPUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    started = time.time()
    records = []
    for item in protocol["sequences"][:8]:
        name = item["name"]
        trace_path = ROOT / "reports/server28-val97-gpu-consistency-20261030/alternating_outage" / (name + ".json")
        trace = json.loads(trace_path.read_text())
        selected = {}
        for state in [[1, 1], [0, 1], [1, 0]]:
            observed = [row for row in trace if row["observed"] and row["available"] == state]
            if observed:
                for fraction in [.25, .75]:
                    row = observed[int(fraction * (len(observed) - 1))]
                    selected[row["frame"]] = row
        folder = Path(protocol["data_root"]) / name
        rgb = sorted((folder / "visible").glob("*.jpg"))
        tir = sorted((folder / "infrared").glob("*.jpg"))
        gt = np.loadtxt(item["gt_path"], delimiter=",", ndmin=2)
        assert sha(item["gt_path"]) == item["gt_sha256"]
        tracker.initialize(get_x_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                           dict(init_bbox=gt[0].tolist()))
        initial_templates = list(tracker.templates)
        for index, row in sorted(selected.items()):
            available = row["available"]
            image = get_x_frame(str(rgb[index] if available[0] else tir[index]),
                                str(tir[index] if available[1] else rgb[index]), dtype="rgbrgb")
            height, width = image.shape[:2]
            side = float(max(height, width))
            specs = [("native", list(row["previous_box"]))]
            for label, center_x, center_y, crop_side in [
                    ("whole", width / 2., height / 2., side)] + [
                    ("tile%d%d" % (x, y), width * x / 4., height * y / 4., side / 2.)
                    for y in [1, 2, 3] for x in [1, 2, 3]]:
                virtual_size = crop_side / cfg.TEST.SEARCH_FACTOR
                specs.append((label, [center_x - virtual_size / 2., center_y - virtual_size / 2.,
                                      virtual_size, virtual_size]))
            crops, factors = [], []
            for _, box in specs:
                crop, factor, _ = sample_target(image, box, cfg.TEST.SEARCH_FACTOR,
                                                output_sz=cfg.TEST.SEARCH_SIZE)
                crops.append(tracker.preprocessor.process(crop))
                factors.append(factor)
            batch = len(crops)
            with torch.no_grad():
                templates = [template.expand(batch, -1, -1, -1) for template in initial_templates]
                availability = [torch.ones(batch, 2) for _ in templates]
                availability.append(torch.tensor([available], dtype=torch.float32).expand(batch, -1))
                out = tracker.network(templates, [torch.cat(crops)], availability)[0]
                hann = tracker.window * out["score_map"]
                predictions = {
                    "hann": tracker.network.baseline.box_head.cal_bbox(
                        hann, out["size_map"], out["offset_map"]).view(batch, 4),
                    "raw": tracker.network.baseline.box_head.cal_bbox(
                        out["score_map"], out["size_map"], out["offset_map"]).view(batch, 4),
                }
            candidates = []
            for candidate_index, (label, crop_box) in enumerate(specs):
                record = dict(label=label, crop_box=crop_box,
                              actual_crop_side=math.ceil(math.sqrt(crop_box[2] * crop_box[3]) *
                                                         cfg.TEST.SEARCH_FACTOR),
                              hann_confidence=float(hann[candidate_index].max()),
                              raw_confidence=float(out["score_map"][candidate_index].max()))
                for rule, values in predictions.items():
                    cx, cy, box_width, box_height = (
                        values[candidate_index] * cfg.TEST.SEARCH_SIZE / factors[candidate_index]).tolist()
                    half = .5 * cfg.TEST.SEARCH_SIZE / factors[candidate_index]
                    box = clip_box([
                        cx + crop_box[0] + crop_box[2] / 2. - half - box_width / 2.,
                        cy + crop_box[1] + crop_box[3] / 2. - half - box_height / 2.,
                        box_width, box_height], height, width, margin=10)
                    assert np.isfinite(box).all()
                    record[rule + "_bbox"] = box
                    record[rule + "_diagnostic"] = geometry(
                        crop_box, gt[index].tolist(), box, cfg.TEST.SEARCH_FACTOR)
                candidates.append(record)
            tracker.state = list(row["previous_box"])
            tracker.frame_id = 0
            tracker.templates = list(initial_templates)
            tracker.availability = [torch.ones(1, 2) for _ in initial_templates]
            reference = tracker.track(image, available)["target_bbox"]
            difference = max(abs(first - second)
                             for first, second in zip(reference, candidates[0]["hann_bbox"]))
            assert np.allclose(reference, candidates[0]["hann_bbox"], rtol=1e-4, atol=.05)
            records.append(dict(sequence=name, frame=index, available=available,
                                source_trace_sha256=sha(trace_path),
                                native_trajectory_iou=row["diagnostic"]["iou"] if row["diagnostic"] else None,
                                native_batch_vs_single_max_bbox_difference=difference,
                                candidates=candidates))
        (OUTPUT / "report.json").write_text(json.dumps(
            dict(launch=launch, observations=records), indent=2) + "\n")
        print("SEQUENCE " + json.dumps(dict(sequence=name, observations=len(selected),
                                          total_observations=len(records),
                                          seconds=time.time() - started)), flush=True)
    result = dict(summary=summarize(records),
                  per_state={state: summarize([
                      row for row in records
                      if "".join(map(str, row["available"])) == state])
                             for state in ["11", "01", "10"]},
                  per_sequence={item["name"]: summarize([
                      row for row in records if row["sequence"] == item["name"]])
                                for item in protocol["sequences"][:8]},
                  seconds=time.time() - started)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUTPUT / "completed.json").write_text(json.dumps(dict(
        sequences=8, observations=len(records), candidates=len(records) * 11,
        seconds=result["seconds"]), indent=2) + "\n")
    print("COMPLETE " + json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
