"""Paired train882 closed-loop rollout under IID and clustered modality masks."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.dataset import LasHeR
from miss_rgbt.tracker import MissingTracker


ROOT = Path("/home/gaob/Miss-RGBT")
DATASET = "/data/wangwj/dataset/LasHeR/trainingset"
CHECKPOINT = ROOT / "checkpoints/e0_availability/epoch20.pth.tar"
SAMPLES = 24
FRAMES = 64
SEED = 61101
STATES = {
    "11": (1.0, 1.0),
    "01": (0.0, 1.0),
    "10": (1.0, 0.0),
}
PROBABILITIES = np.asarray([0.4, 0.3, 0.3], dtype=np.float64)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prior_frames(report_root):
    paths = [
        report_root / "server28_fusion_adapter_pilot_20261102/report.json",
        report_root / "server28_head_supervision_20261102/report.json",
        report_root / "server28_modality_heads_20261102/report.json",
        report_root / "server28_response_geometry_20261102/report.json",
        report_root / "server28_absent_search_kv_20261102_v2/report.json",
        report_root / "server28_train882_block_state_diag_20261102/diagnostic.json",
    ]
    used = defaultdict(set)

    def visit(value):
        if isinstance(value, dict):
            identity = value.get("identity")
            if isinstance(identity, dict) and "sequence" in identity:
                for key in ("template_frames", "search_frames"):
                    used[identity["sequence"]].update(int(frame)
                        for frame in identity.get(key, []))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        if path.exists():
            visit(json.loads(path.read_text()))
    return used


def choose_clips(dataset, names, used, rng):
    selected = []
    order = rng.permutation(len(names)).tolist()
    for sequence_index in order:
        name = names[sequence_index]
        dataset_index = dataset.sequence_list.index(name)
        info = dataset.get_sequence_info(dataset_index)
        boxes = info["bbox"].numpy()
        valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        blocked = used[name]
        starts = [
            start for start in range(max(0, len(boxes) - FRAMES + 1))
            if valid[start:start + FRAMES].all()
            and not blocked.intersection(range(start, start + FRAMES))
        ]
        if not starts:
            continue
        start = int(rng.choice(starts))
        selected.append(dict(name=name, dataset_index=dataset_index,
            start=start, stop=start + FRAMES, boxes=boxes[start:start + FRAMES].tolist()))
        if len(selected) == SAMPLES:
            break
    assert len(selected) == SAMPLES
    assert len({item["name"] for item in selected}) == SAMPLES
    return selected


def make_schedules(rng, count):
    labels = rng.choice(list(STATES), size=count - 1, p=PROBABILITIES)
    iid = labels.copy()
    rng.shuffle(iid)
    block_order = rng.permutation(list(STATES))
    clustered = np.concatenate([labels[labels == state] for state in block_order])
    return ["11"] + iid.tolist(), ["11"] + clustered.tolist()


def box_iou(first, second):
    ax1, ay1, aw, ah = [float(value) for value in first]
    bx1, by1, bw, bh = [float(value) for value in second]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def rollout(tracker, frames, boxes, schedule, sequence, policy):
    tracker.initialize(frames[0], {"init_bbox": boxes[0]})
    rows = []
    previous_missing = False
    recovery_left = 0
    for frame_index in range(1, len(frames)):
        state = schedule[frame_index]
        missing = state != "11"
        if previous_missing and not missing:
            recovery_left = 5
        result = tracker.track(frames[frame_index], STATES[state])
        gt = boxes[frame_index]
        overlap = box_iou(result["target_bbox"], gt)
        recovery = state == "11" and recovery_left > 0
        if recovery:
            recovery_left -= 1
        rows.append(dict(sequence=sequence, policy=policy, frame=frame_index,
            state=state, bbox=[float(value) for value in result["target_bbox"]],
            gt_bbox=[float(value) for value in gt], iou=float(overlap),
            confidence=float(result["best_score"]), recovery=recovery))
        previous_missing = missing
    return rows


def sha_image_sequence(frames):
    digest = hashlib.sha256()
    for image in frames:
        digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def summarize(rows, field, seed):
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence"]].append(float(row[field]))
    means = np.asarray([np.mean(by_sequence[name]) for name in sorted(by_sequence)])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    sampled = means[draws].mean(axis=1)
    return dict(mean=float(means.mean()),
        ci95_low=float(np.quantile(sampled, 0.025)),
        ci95_high=float(np.quantile(sampled, 0.975)),
        sequences=len(means), frames=sum(len(value) for value in by_sequence.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root", default=DATASET)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()

    update_config_from_file("experiments/cadtrack/cadtrack.yaml")
    tracker = MissingTracker(cfg, str(CHECKPOINT))
    tracker.network.eval().requires_grad_(False)
    state_before = tensor_digest(tracker.network.state_dict())

    train_file = Path("lib/train/data_specs/lasher_train.txt")
    names = train_file.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    dataset = LasHeR(root=args.dataset_root, split="train")
    used = prior_frames(ROOT / "runtime/task_utility_20261018/reports")
    rng = np.random.default_rng(SEED)
    clips = choose_clips(dataset, fitting, used, rng)

    all_rows = []
    clip_manifest = []
    for clip_index, clip in enumerate(clips):
        frame_ids = list(range(clip["start"], clip["stop"]))
        frames, annotations, _ = dataset.get_frames(
            clip["dataset_index"], frame_ids, dataset.get_sequence_info(clip["dataset_index"]))
        boxes = [box.tolist() for box in annotations["bbox"]]
        assert len(frames) == len(boxes) == FRAMES
        iid_schedule, burst_schedule = make_schedules(rng, FRAMES)
        state_counts_iid = {state: iid_schedule.count(state) for state in STATES}
        state_counts_burst = {state: burst_schedule.count(state) for state in STATES}
        assert state_counts_iid == state_counts_burst
        image_sha = sha_image_sequence(frames)
        for policy, schedule in (("iid", iid_schedule), ("clustered", burst_schedule)):
            all_rows.extend(rollout(tracker, frames, boxes, schedule, clip["name"], policy))
        clip_manifest.append(dict(sequence=clip["name"], frame_ids=frame_ids,
            image_sha256=image_sha, state_counts=state_counts_iid,
            iid_schedule=iid_schedule, clustered_schedule=burst_schedule))
        print(json.dumps(dict(stage="clip", completed=clip_index + 1,
            total=SAMPLES, peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved())), flush=True)

    assert len(all_rows) == SAMPLES * (FRAMES - 1) * 2
    assert state_before == tensor_digest(tracker.network.state_dict())
    assert all(parameter.grad is None for parameter in tracker.network.parameters())
    assert all(np.isfinite(row["iou"]) and np.isfinite(row["confidence"])
               and all(np.isfinite(value) for value in row["bbox"]) for row in all_rows)

    effects = {}
    categories = {
        "all_missing": lambda row: row["state"] in ("01", "10"),
        "rgb_only": lambda row: row["state"] == "10",
        "tir_only": lambda row: row["state"] == "01",
        "recovery": lambda row: row["recovery"],
    }
    for category, predicate in categories.items():
        values = []
        for clip in clips:
            selected = [row for row in all_rows
                if row["sequence"] == clip["name"] and predicate(row)]
            iid = [row["iou"] for row in selected if row["policy"] == "iid"]
            burst = [row["iou"] for row in selected if row["policy"] == "clustered"]
            if iid and burst:
                values.append(dict(sequence=clip["name"],
                    difference=float(np.mean(burst) - np.mean(iid))))
        effects[category] = summarize(values, "difference", SEED + 1)

    missing = effects["all_missing"]
    recovery = effects["recovery"]
    advance = ((missing["mean"] <= -0.01 and missing["ci95_high"] < 0)
               or (recovery["frames"] >= SAMPLES and recovery["mean"] <= -0.01
                   and recovery["ci95_high"] < 0))
    source_paths = [
        Path(__file__), Path("miss_rgbt/tracker.py"), Path("miss_rgbt/model.py"),
        Path("miss_rgbt/train.py"), Path("miss_rgbt/data.py"),
        Path("lib/train/dataset/lasher.py"),
        Path("lib/train/data_specs/lasher_train.txt"),
        Path("experiments/cadtrack/cadtrack.yaml"),
    ]
    report = dict(status="complete", seed=SEED, checkpoint=str(CHECKPOINT),
        checkpoint_sha256=sha256(CHECKPOINT), baseline_state_sha256=state_before,
        state_unchanged=True, gradients_absent=True, clips=clip_manifest,
        rows=all_rows, effects=effects, advance=bool(advance),
        source_sha256={str(path): sha256(path) for path in source_paths},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.time() - started,
        scope="Frozen-E0, train882 closed-loop mask-autocorrelation diagnostic; "
              "not official metrics, model training, or module validation.")
    report_path = output / "report.json"
    with report_path.open("x", newline="\n") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    completed = dict(status="complete", report_sha256=sha256(report_path),
        rows=len(all_rows), advance=bool(advance), seconds=report["seconds"])
    with (output / "completed.json").open("x", newline="\n") as stream:
        json.dump(completed, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(stage="complete", effects=effects,
        advance=bool(advance), seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
