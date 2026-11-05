"""Paired CPU tracking control after the frozen-baseline branch pilot."""
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
RUNTIME = ROOT / "runtime/task_utility_20261018"
PILOT = ROOT / "reports/server28-protected-sixepoch-20261030"
OUTPUT = ROOT / "reports/server28-protected-tracking-development-20261030"
PILOT_PID = 61899
PATTERN = [[1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]]
sys.path.insert(0, str(RUNTIME))
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.utils.hann import hann2d
from lib.train.dataset.depth_utils import get_x_frame
from miss_rgbt.model import load_pretrained
from cpu_diagnostic_tracker import CpuDiagnosticTracker, CpuPreprocessorMM
from evaluate_val97_recovery28 import geometry, sha
from probe_protected_sixepoch28 import digest


class ProtectedCpuTracker(CpuDiagnosticTracker):
    """Reuse native CPU tracking with an explicitly assembled E0/repair model."""

    def __init__(self, network):
        self.cfg = cfg
        self.network = network.cpu().eval().requires_grad_(False)
        self.preprocessor = CpuPreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        self.num_template = cfg.DATA.TEMPLATE.NUMBER
        size = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.window = hann2d(torch.tensor([size, size]).long(), centered=True)


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def track(tracker, item, condition):
    folder = Path(item["folder"])
    gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
    assert sha(folder / "init.txt") == item["gt_sha256"]
    rgb = sorted((folder / "visible").glob("*.jpg"))
    tir = sorted((folder / "infrared").glob("*.jpg"))
    assert len(rgb) == len(tir) == len(gt) == item["video_frames"]
    assert np.isfinite(gt).all() and (gt[0, 2:] > 0).all()
    frames = item["evaluated_frames"]
    tracker.initialize(get_x_frame(str(rgb[0]), str(tir[0]), dtype="rgbrgb"),
                       dict(init_bbox=gt[0].tolist()))
    boxes, rows = [gt[0].tolist()], []
    for index in range(1, frames):
        available = [1, 1] if condition == "complete" else PATTERN[(index // 32) % 6]
        previous, calls = list(tracker.state), tracker.frame_id
        template = tracker.templates[-1]
        confidence = None
        if any(available):
            r = rgb[index] if available[0] else tir[index]
            t = tir[index] if available[1] else rgb[index]
            result = tracker.track(get_x_frame(str(r), str(t), dtype="rgbrgb"), available)
            confidence = result["best_score"]
            assert np.isfinite(confidence) and tracker.frame_id == calls + 1
        else:
            assert tracker.state == previous and tracker.frame_id == calls
            assert tracker.templates[-1] is template
        prediction = list(tracker.state)
        assert np.isfinite(prediction).all()
        written = tracker.templates[-1] is not template
        if written:
            assert any(available) and tracker.frame_id % cfg.TEST.UPDATE_INTERVALS == 0
            assert confidence > cfg.TEST.UPDATE_THRESHOLD
        diagnostic = geometry(previous, gt[index].tolist(), prediction, cfg.TEST.SEARCH_FACTOR)
        if diagnostic is not None and not any(available):
            diagnostic["target_center_in_previous_crop"] = None
            diagnostic["target_overlaps_previous_crop"] = None
        boxes.append(prediction)
        rows.append(dict(frame=index, available=available, bbox=prediction,
                         track_call=tracker.frame_id, template_written=written,
                         hann_confidence=confidence, diagnostic=diagnostic))
    predicted = np.asarray(boxes)
    target = gt[:frames]
    valid = (target[:, 2:] > 0).all(1)
    valid[0] = False
    assert valid.any()
    p, target = predicted[valid], target[valid]
    error = np.linalg.norm(p[:, :2] + p[:, 2:] / 2 - target[:, :2] - target[:, 2:] / 2, axis=1)
    overlap = np.maximum(0., np.minimum(p[:, :2] + p[:, 2:], target[:, :2] + target[:, 2:])
                         - np.maximum(p[:, :2], target[:, :2])).prod(1)
    iou = overlap / (p[:, 2:].prod(1) + target[:, 2:].prod(1) - overlap)
    observed = [row for row in rows if any(row["available"]) and row["diagnostic"] is not None]
    metrics = dict(frames=frames, scored_frames=int(valid.sum()),
                   PR20=float((error <= 20).mean() * 100),
                   SR=float((iou[:, None] > np.linspace(0, 1, 21)).mean() * 100),
                   mean_iou=float(iou.mean()), track_calls=tracker.frame_id,
                   holds=sum(not any(row["available"]) for row in rows),
                   observed_valid_frames=len(observed),
                   unsupported_observed_frames=sum(
                       not row["diagnostic"]["target_overlaps_previous_crop"] for row in observed))
    return predicted, rows, metrics


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    inputs = json.loads((PILOT / "inputs.json").read_text())
    train = set(inputs["train_sequences"])
    development = inputs["development_sequences"]
    heldout = set(inputs["heldout_sequences"])
    assert len(train) == 705 and len(development) == len(set(development)) == 177
    assert not (train & set(development) or train & heldout or set(development) & heldout)
    data_root = Path("/data/wangwj/dataset/LasHeR/trainingset")
    cohort = []
    for name in development[:4]:
        folder = data_root / name
        gt = np.loadtxt(str(folder / "init.txt"), delimiter=",", ndmin=2)
        cohort.append(dict(name=name, folder=str(folder), video_frames=len(gt),
                           evaluated_frames=min(257, len(gt)), gt_sha256=sha(folder / "init.txt")))
    launch = dict(pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
                  device="cpu", cpu_threads=4, pilot_pid=PILOT_PID, poll_seconds=300,
                  cohort=cohort, input_manifest_sha256=sha(PILOT / "inputs.json"),
                  conditions=["complete", "alternating_outage"], pattern=PATTERN,
                  frame_block=32, models=["E0", "selected_repair"],
                  source_sha256={str(path): sha(path) for path in [
                      Path(__file__), ROOT / "cpu_diagnostic_tracker.py",
                      ROOT / "probe_protected_sixepoch28.py", ROOT / "evaluate_val97_recovery28.py",
                      RUNTIME / "miss_rgbt/model.py",
                      RUNTIME / "experiments/cadtrack/cadtrack.yaml"]},
                  scope="First4 train882 development sequences, at most257 frames each. "
                        "Descriptive CPU tracking control; not official Miss metrics, "
                        "full sequences, full development set or an unseen test. "
                        "No checkpoint or parameter selection from tracking outcomes.")
    save("launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    while (Path("/proc") / str(PILOT_PID)).exists():
        command = ((Path("/proc") / str(PILOT_PID) / "cmdline")
                   .read_bytes().replace(b"\0", b" ").decode())
        assert "probe_protected_sixepoch28.py" in command
        time.sleep(300)
    completed = json.loads((PILOT / "completed.json").read_text())
    report = json.loads((PILOT / "report.json").read_text())
    history = json.loads((PILOT / "development_history.json").read_text())
    assert completed["updates"] == report["updates"] == 192 and completed["cleanup_completed"]
    assert [(row["epoch"], row["updates"]) for row in history] == [
        (epoch, epoch * 32) for epoch in range(1, 7)]
    best = min(history[-5:], key=lambda row: (row["development"]["selection_score"], row["epoch"]))
    assert best == report["selected"] and completed["selected_epoch"] == best["epoch"]
    assert report["baseline_state_unchanged"] and report["complete_outputs_bitwise_equal"]
    retained = Path(report["retained"])
    assert retained.parent == PILOT and sha(retained) == report["retained_sha256"]
    assert sorted(PILOT.glob("repair_epoch*.pth")) == [retained]
    assert len(report["cleanup"]) == 4 and all(
        not Path(item["path"]).exists() for item in report["cleanup"])
    checkpoint = Path(report["launch"]["checkpoint"])
    assert sha(checkpoint) == report["launch"]["checkpoint_sha256"]
    for path, expected in report["launch"]["source_sha256"].items():
        assert sha(path) == expected
    save("pilot_verified.json", dict(
        selected_epoch=best["epoch"], repair_sha256=sha(retained),
        pilot_report_sha256=sha(PILOT / "report.json"), final_five_development_selection=True,
        tracking_selection_used=False))
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    started = time.time()
    records, reference = {}, {}
    baseline_sha = None
    for label in launch["models"]:
        torch.manual_seed(2027)
        model = load_pretrained(cfg, str(checkpoint), repair=label == "selected_repair",
                                width=128, repair_layer=6)
        current_sha = digest(model.baseline.state_dict())
        if label == "E0":
            baseline_sha = current_sha
        else:
            assert current_sha == baseline_sha
            saved = torch.load(str(retained), map_location="cpu")
            assert saved["epoch"] == best["epoch"]
            model.repair.load_state_dict(saved["repair"], strict=True)
        tracker = ProtectedCpuTracker(model)
        records[label] = {}
        for item in cohort:
            name = item["name"]
            records[label][name] = {}
            for condition in launch["conditions"]:
                boxes, rows, metrics = track(tracker, item, condition)
                key = (name, condition)
                if label == "E0":
                    reference[key] = boxes
                elif condition == "complete":
                    assert np.array_equal(boxes, reference[key])
                else:
                    assert np.array_equal(boxes[:32], reference[key][:32])
                basename = label + "_" + name + "_" + condition
                path = OUTPUT / (basename + ".txt")
                np.savetxt(str(path), boxes)
                save(basename + ".json", rows)
                metrics.update(bbox_sha256=sha(path), trace_sha256=sha(
                    OUTPUT / (basename + ".json")))
                records[label][name][condition] = metrics
                save("tracking_progress.json", records)
                print("TRACK " + json.dumps(dict(model=label, sequence=name,
                      condition=condition, seconds=time.time() - started, **metrics)), flush=True)
        assert digest(model.baseline.state_dict()) == baseline_sha
        del tracker, model
    aggregate = {}
    for condition in launch["conditions"]:
        aggregate[condition] = {}
        for label in launch["models"]:
            aggregate[condition][label] = {
                metric: float(np.mean([records[label][item["name"]][condition][metric]
                                       for item in cohort]))
                for metric in ["PR20", "SR", "mean_iou"]}
        aggregate[condition]["selected_minus_E0"] = {
            metric: aggregate[condition]["selected_repair"][metric] - aggregate[condition]["E0"][metric]
            for metric in ["PR20", "SR", "mean_iou"]}
    result = dict(launch=launch, pilot_selected_epoch=best["epoch"], records=records,
                  aggregate=aggregate, complete_predictions_bitwise_equal=True,
                  complete_prefix_before_outage_bitwise_equal=True,
                  baseline_state_unchanged=True, seconds=time.time() - started)
    save("report.json", result)
    save("completed.json", dict(sequences=len(cohort), model_conditions=4,
                               frames_per_model_condition=sum(
                                   item["evaluated_frames"] for item in cohort),
                               seconds=result["seconds"], aggregate=aggregate))
    print("COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
