import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss


SAMPLES = 128
SEED = 1701


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class AvailabilityPolicyBlock(nn.Module):
    def __init__(self, block, index):
        super().__init__()
        self.block = block
        self.index = index
        self.policy = "native"
        self.availability = None
        self.capture_drift = False
        self.drift = {}

    def configure(self, policy, availability, capture_drift):
        self.policy = policy
        self.availability = availability
        self.capture_drift = capture_drift
        self.drift = {}

    def forward(self, rgb, tir):
        next_rgb, next_tir = self.block(rgb, tir)
        if self.policy == "skip_unavailable":
            skip_rgb = self.availability[:, :, 0:1] < 0.5
            skip_tir = self.availability[:, :, 1:2] < 0.5
            next_rgb = torch.where(skip_rgb, rgb, next_rgb)
            next_tir = torch.where(skip_tir, tir, next_tir)
        elif self.policy == "skip_available":
            skip_rgb = self.availability[:, :, 0:1] >= 0.5
            skip_tir = self.availability[:, :, 1:2] >= 0.5
            next_rgb = torch.where(skip_rgb, rgb, next_rgb)
            next_tir = torch.where(skip_tir, tir, next_tir)
        else:
            assert self.policy == "native"

        if self.capture_drift:
            self.drift = {
                "rgb": self._drift(rgb, next_rgb, 0),
                "tir": self._drift(tir, next_tir, 1),
            }
        return next_rgb, next_tir

    def _drift(self, before, after, stream):
        missing = self.availability[:, :, stream] < 0.5
        delta = (after - before).float().square().mean(-1).sqrt()
        scale = before.float().square().mean(-1).sqrt()
        count = missing.sum(1).clamp_min(1)
        absolute = (delta * missing).sum(1) / count
        relative = ((delta / scale.clamp_min(1e-12)) * missing).sum(1) / count
        return {
            "absolute_rms": absolute.detach().cpu().tolist(),
            "relative_rms": relative.detach().cpu().tolist(),
        }


def wrap_blocks(model):
    original = list(model.baseline.backbone.blocks)
    wrapped = [AvailabilityPolicyBlock(block, index) for index, block in enumerate(original)]
    model.baseline.backbone.blocks = nn.Sequential(*wrapped)
    model.eval()
    return wrapped


def make_availability(batch, templates, searches, condition, device):
    if condition == "complete":
        state = [1.0, 1.0]
    elif condition == "rgb_only":
        state = [1.0, 0.0]
    elif condition == "tir_only":
        state = [0.0, 1.0]
    else:
        raise ValueError(condition)

    first = torch.ones(batch, 2, device=device)
    rest = torch.tensor(state, dtype=torch.float32, device=device).view(1, 2)
    return [first] + [rest.expand(batch, -1) for _ in range(len(templates) - 1 + len(searches))]


def make_token_availability(model, templates, searches, availability):
    backbone = model.baseline.backbone
    batch = searches[0].shape[0]
    frames = []
    for index, available in enumerate(availability):
        pos = backbone.pos_embed_z if index < len(templates) else backbone.pos_embed_x
        frames.append(available.unsqueeze(1).expand(-1, pos.shape[1], -1))
    cls = searches[0].new_ones(batch, 1, 2)
    return torch.cat([cls, torch.cat(frames, dim=1)], dim=1)


def set_policy(blocks, policy, availability, capture_drift=False):
    for block in blocks:
        block.configure(policy, availability, capture_drift)


def capture_identities(loader):
    sampler = loader.dataset
    dataset = sampler.datasets[0]
    original_getitem = sampler.getitem
    original_get_frames = dataset.get_frames
    identities = []

    def tracked_getitem():
        calls = []

        def tracked_get_frames(seq_id, frame_ids, anno=None):
            calls.append((seq_id, tuple(int(frame_id) for frame_id in frame_ids)))
            return original_get_frames(seq_id, frame_ids, anno)

        dataset.get_frames = tracked_get_frames
        sample = original_getitem()
        dataset.get_frames = original_get_frames
        template_call, search_call = calls[-2:]
        assert template_call[0] == search_call[0]
        identities.append({
            "sequence": dataset.sequence_list[template_call[0]],
            "template_frames": list(template_call[1]),
            "search_frames": list(search_call[1]),
        })
        return sample

    sampler.getitem = tracked_getitem
    return identities


def hash_sample_inputs(templates, searches, boxes, sample_index):
    template_tensor = torch.stack([frame[sample_index] for frame in templates], dim=0)
    search_tensor = torch.stack([frame[sample_index] for frame in searches], dim=0)
    box_tensor = torch.stack([frame[sample_index] for frame in boxes], dim=0)
    return {
        "template_input_sha256": tensor_sha256(template_tensor),
        "search_input_sha256": tensor_sha256(search_tensor),
        "boxes_sha256": tensor_sha256(box_tensor),
    }


def measure(outputs, boxes, cfg):
    losses = task_loss(outputs, boxes, cfg).detach()
    last_output = outputs[-1]
    prediction = last_output["pred_boxes"][:, 0, :]
    target = boxes[-1]
    prediction_xyxy = box_cxcywh_to_xyxy(prediction)
    target_xyxy = box_xywh_to_xyxy(target).clamp(0, 1)
    _, iou_matrix = generalized_box_iou(prediction_xyxy, target_xyxy)
    iou = torch.diagonal(iou_matrix)
    center_error = torch.norm(prediction[:, :2] - target[:, :2], dim=-1)
    return [
        {
            "task_loss": float(losses[index].cpu()),
            "iou": float(iou[index].cpu()),
            "center_error": float(center_error[index].cpu()),
        }
        for index in range(losses.shape[0])
    ]


def compare_outputs(first, second):
    assert len(first) == len(second)
    exact = True
    max_abs = 0.0
    for out_a, out_b in zip(first, second):
        assert set(out_a) == set(out_b)
        for key in out_a:
            value_a, value_b = out_a[key], out_b[key]
            if isinstance(value_a, torch.Tensor):
                difference = (value_a - value_b).abs().max().item()
                max_abs = max(max_abs, difference)
                exact = exact and torch.equal(value_a, value_b)
            else:
                assert value_a == value_b
    return exact, max_abs


def paired_effect(rows, field, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["identity"]["sequence"]].append(row[field])
    sequences = sorted(grouped)
    means = np.asarray([np.mean(grouped[name]) for name in sequences], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(5000, len(means)))
    bootstrap = means[draws].mean(axis=1)
    return {
        "sequence_macro_mean": float(means.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "sequences": len(sequences),
        "tuples": len(rows),
    }


def native_noise(rows, metric):
    return max(abs(row["native"][metric] - row["native_repeat"][metric]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/gaob/Miss-RGBT/checkpoints/pretrained/CADTrack_ep0020.pth.tar")
    parser.add_argument("--dataset-root", default="/data/wangwj/dataset/LasHeR/trainingset")
    parser.add_argument("--output", default="reports/server28_train882_block_state_diag_20261102")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    assert args.batch_size > 0 and SAMPLES % args.batch_size == 0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    checkpoint = Path(args.checkpoint)
    assert checkpoint.is_file()
    update_config_from_file("experiments/cadtrack/cadtrack.yaml")
    model = load_pretrained(cfg, str(checkpoint), repair=False, task_utility=None).to(device).eval()
    blocks = wrap_blocks(model)

    train_sequences_path = Path("lib/train/data_specs/lasher_train.txt")
    train_sequences = train_sequences_path.read_text().splitlines()
    assert len(train_sequences) == 882 and len(set(train_sequences)) == 882
    loader = make_loader(
        cfg, args.dataset_root, "train", args.batch_size, SAMPLES, workers=0)
    identities = capture_identities(loader)

    results = []
    identity_offset = 0
    complete_parity = True
    complete_max_abs = 0.0
    noise_by_condition = {}
    modes = {
        "complete": ["native", "native_repeat", "intervention"],
        "rgb_only": ["native", "native_repeat", "intervention", "sham"],
        "tir_only": ["native", "native_repeat", "intervention", "sham"],
    }

    for batch_index, data in enumerate(loader):
        templates = [image.cuda(non_blocking=True) for image in data["template_images"]]
        searches = [image.cuda(non_blocking=True) for image in data["search_images"]]
        boxes = data["search_anno"].cuda(non_blocking=True)
        batch = searches[0].shape[0]
        batch_identities = identities[identity_offset:identity_offset + batch]
        assert len(batch_identities) == batch
        identity_offset += batch
        batch_rows = []
        for index, identity in enumerate(batch_identities):
            row = {"identity": identity}
            row.update(hash_sample_inputs(
                data["template_images"], data["search_images"], data["search_anno"], index))
            row["conditions"] = {}
            batch_rows.append(row)

        for condition, condition_modes in modes.items():
            available_frames = make_availability(batch, templates, searches, condition, device)
            token_available = make_token_availability(
                model, templates, searches, available_frames)
            condition_rows = [{} for _ in range(batch)]
            native_outputs = None
            for mode in condition_modes:
                policy = "skip_unavailable" if mode == "intervention" else (
                    "skip_available" if mode == "sham" else "native")
                set_policy(blocks, policy, token_available,
                           capture_drift=condition != "complete")
                with torch.no_grad():
                    outputs = model(templates, searches, availability=available_frames)
                    measured = measure(outputs, boxes, cfg)
                if mode == "native":
                    native_outputs = outputs
                elif condition == "complete" and mode == "intervention":
                    parity, delta = compare_outputs(native_outputs, outputs)
                    complete_parity = complete_parity and parity
                    complete_max_abs = max(complete_max_abs, delta)
                for index in range(batch):
                    condition_rows[index][mode] = measured[index]
                    if condition != "complete":
                        condition_rows[index][mode + "_drift"] = [
                            {
                                "absolute_rms": block.drift[
                                    "tir" if condition == "rgb_only" else "rgb"
                                ]["absolute_rms"][index],
                                "relative_rms": block.drift[
                                    "tir" if condition == "rgb_only" else "rgb"
                                ]["relative_rms"][index],
                            }
                            for block in blocks
                        ]

            for index, row in enumerate(batch_rows):
                condition_rows[index]["intervention_task_loss_improvement"] = (
                    condition_rows[index]["native"]["task_loss"]
                    - condition_rows[index]["intervention"]["task_loss"])
                condition_rows[index]["sham_task_loss_improvement"] = (
                    condition_rows[index]["native"]["task_loss"]
                    - condition_rows[index].get("sham", {}).get("task_loss",
                        condition_rows[index]["native"]["task_loss"]))
                row["conditions"][condition] = condition_rows[index]
        results.extend(batch_rows)
        print(json.dumps({
            "batch": batch_index + 1,
            "batches": SAMPLES // args.batch_size,
            "samples_processed": len(results),
            "complete_parity_so_far": complete_parity,
        }), flush=True)

    assert len(results) == SAMPLES and identity_offset == SAMPLES
    assert all(row["identity"]["sequence"] in train_sequences for row in results)

    summaries = {}
    for condition in ["rgb_only", "tir_only"]:
        rows = []
        for row in results:
            condition_row = dict(row["conditions"][condition])
            condition_row["identity"] = row["identity"]
            rows.append(condition_row)
        noise = native_noise(rows, "task_loss")
        noise_by_condition[condition] = noise
        intervention = paired_effect(rows, "intervention_task_loss_improvement", SEED)
        sham = paired_effect(rows, "sham_task_loss_improvement", SEED + 1)
        intervention_gate = intervention["ci95_low"] > noise
        sham_gate = sham["ci95_low"] <= noise
        summaries[condition] = {
            "native_repeat_max_abs_task_loss_noise": noise,
            "intervention_task_loss_improvement": intervention,
            "sham_task_loss_improvement": sham,
            "intervention_exceeds_repeat_noise": intervention_gate,
            "sham_does_not_reproduce_improvement": sham_gate,
            "intervention_iou_improvement": paired_effect(
                [dict(identity=row["identity"], value=(
                    row["conditions"][condition]["intervention"]["iou"]
                    - row["conditions"][condition]["native"]["iou"]))
                 for row in results], "value", SEED + 2),
            "intervention_center_error_improvement": paired_effect(
                [dict(identity=row["identity"], value=(
                    row["conditions"][condition]["native"]["center_error"]
                    - row["conditions"][condition]["intervention"]["center_error"]))
                 for row in results], "value", SEED + 3),
        }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "train882-only; no parameter updates; no val97 or official-test data",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": SEED,
        "samples": SAMPLES,
        "batch_size": args.batch_size,
        "train_sequences": len(train_sequences),
        "train_sequence_list_sha256": sha256(train_sequences_path),
        "complete_input_bitwise_parity": complete_parity,
        "complete_input_max_abs_output_delta": complete_max_abs,
        "native_repeat_noise_by_condition": noise_by_condition,
        "condition_summaries": summaries,
        "advance_to_matched_training": complete_parity and all(
            item["intervention_exceeds_repeat_noise"]
            and item["sham_does_not_reproduce_improvement"]
            for item in summaries.values()),
        "tuples": results,
    }
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(output / "diagnostic.json"),
        "complete_input_bitwise_parity": complete_parity,
        "condition_summaries": summaries,
        "advance_to_matched_training": report["advance_to_matched_training"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
