import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss


CHECKPOINT_SHA256 = "42ed78bb821a34f2695423a3a5d8db3ea0af0ab7820022b78588378fea803c8d"
CHECKPOINT = "/home/gaob/Miss-RGBT/runs/server28-recovery-20261031/a_utility_full20_seed2027/epoch16.pth.tar"
BASELINE = "/home/gaob/Miss-RGBT/checkpoints/pretrained/CADTrack_ep0020.pth.tar"
CONFIG = "configs/a_utility_full20.json"
TRAIN_LIST = "lib/train/data_specs/lasher_train.txt"
SEED = 28001
SAMPLES = 128


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def capture_identities(loader):
    sampler = loader.dataset
    dataset = sampler.datasets[0]
    original_getitem = sampler.getitem
    original_get_frames = dataset.get_frames
    identities = []

    def tracked_getitem():
        calls = []

        def tracked_get_frames(seq_id, frame_ids, anno=None):
            calls.append((seq_id, tuple(int(value) for value in frame_ids)))
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


def token_availability(model, availability, template_count):
    backbone = model.baseline.backbone
    frames = []
    for index, available in enumerate(availability):
        position = backbone.pos_embed_z if index < template_count else backbone.pos_embed_x
        frames.append(available.unsqueeze(1).expand(-1, position.shape[1], -1))
    cls = availability[0].new_ones(availability[0].shape[0], 1, 2)
    return torch.cat([cls, torch.cat(frames, dim=1)], dim=1)


def sample_hashes(templates, search, box, sample_index):
    template_tensor = torch.stack([frame[sample_index] for frame in templates], dim=0)
    return {
        "templates_sha256": tensor_sha256(template_tensor),
        "search_sha256": tensor_sha256(search[sample_index]),
        "box_sha256": tensor_sha256(box[sample_index]),
    }


def one_loss(output, boxes):
    return task_loss([output], boxes.unsqueeze(0), cfg).detach()


def bootstrap_effect(rows, left, right, seed):
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row["identity"]["sequence"]].append(
            row[right] - row[left])
    names = sorted(by_sequence)
    values = np.asarray(
        [np.mean(by_sequence[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean_sequence_macro": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "sequences": len(names),
        "tuples": len(rows),
        "positive_sequences": int((values > 0).sum()),
        "zero_sequences": int((values == 0).sum()),
        "negative_sequences": int((values < 0).sum()),
    }


def summarize(rows, condition):
    selected = [row for row in rows if row["condition"] == condition]
    keys = ["off", "utility_joint", "random_joint", "similarity_joint"]
    means = {key: float(np.mean([row[key] for row in selected])) for key in keys}
    return {
        "means": means,
        "utility_minus_random_loss_improvement": bootstrap_effect(
            selected, "utility_joint", "random_joint", SEED + 1),
        "utility_minus_similarity_loss_improvement": bootstrap_effect(
            selected, "utility_joint", "similarity_joint", SEED + 2),
        "utility_joint_gain": bootstrap_effect(
            selected, "utility_joint", "off", SEED + 3),
        "utility_additivity_residual": bootstrap_effect(
            selected, "utility_single_gain_sum", "utility_joint_gain", SEED + 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data/wangwj/dataset/LasHeR/trainingset")
    parser.add_argument("--output", default="reports/server28_joint_utility_sets_20261102")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    assert args.batch_size > 0 and SAMPLES % args.batch_size == 0

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    checkpoint = Path(CHECKPOINT)
    assert checkpoint.is_file()
    assert sha256(checkpoint) == CHECKPOINT_SHA256
    config = json.loads(Path(CONFIG).read_text())
    assert config["task_utility"]["selection"] == "utility"
    assert config["task_utility"]["selected_regions"] == 4
    update_config_from_file("experiments/cadtrack/cadtrack.yaml")
    model = load_pretrained(
        cfg, BASELINE, task_utility=config["task_utility"]).cuda().eval()
    saved = torch.load(str(checkpoint), map_location="cpu")
    assert saved["epoch"] == 16
    model.load_state_dict(saved["model"], strict=True)
    utility = model.utility_repair
    utility.eval()

    train_path = Path(TRAIN_LIST)
    train_sequences = train_path.read_text().splitlines()
    assert len(train_sequences) == 882 and len(set(train_sequences)) == 882
    loader = make_loader(cfg, args.dataset_root, "train", args.batch_size, SAMPLES, 0)
    identities = capture_identities(loader)
    random_generator = torch.Generator().manual_seed(SEED + 10)
    captured = {}

    def capture_before_utility(module, inputs, result):
        captured["rgb"], captured["tir"] = result

    hook = model.baseline.backbone.blocks[5].register_forward_hook(capture_before_utility)
    rows = []
    identity_offset = 0
    processed = 0
    progress_path = output / "progress.json"

    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            templates_cpu = list(data["template_images"])
            search_cpu = data["search_images"][0]
            boxes_cpu = data["search_anno"][0]
            batch = search_cpu.shape[0]
            batch_identities = identities[identity_offset:identity_offset + batch]
            assert len(batch_identities) == batch
            identity_offset += batch
            templates = [image.cuda(non_blocking=True) for image in templates_cpu]
            search = search_cpu.cuda(non_blocking=True)
            boxes = boxes_cpu.cuda(non_blocking=True)
            batch_rows = []

            for condition, state in [
                ("complete", [1.0, 1.0]),
                ("rgb_only", [1.0, 0.0]),
                ("tir_only", [0.0, 1.0]),
            ]:
                available = [search.new_ones(batch, 2) for _ in templates]
                available.append(search.new_tensor(state).expand(batch, -1))
                token_mask = token_availability(model, available, len(templates))
                captured.clear()
                native = model(templates, [search], availability=available)[0]
                rgb, tir = captured["rgb"], captured["tir"]
                context, similarity = utility.context(rgb, tir, token_mask)
                region_context = context[:, utility.region_tokens].mean(2)
                logits = utility.selector(region_context.detach()).squeeze(-1)
                utility_regions = logits.topk(utility.selected_regions, dim=1).indices
                similarity_scores = similarity[:, utility.region_tokens].mean(2)
                similarity_regions = similarity_scores.topk(
                    utility.selected_regions, dim=1).indices
                random_scores = torch.rand(
                    logits.shape, generator=random_generator).to(search.device)
                random_regions = random_scores.topk(
                    utility.selected_regions, dim=1).indices

                def predict(regions):
                    if regions is None:
                        return model.predict_tail(
                            rgb, tir,
                            len(templates) * model.baseline.backbone.pos_embed_z.shape[1],
                            model.baseline.backbone.pos_embed_x.shape[1])
                    corrected_rgb, corrected_tir, _ = utility.apply_regions(
                        rgb, tir, token_mask, context, regions)
                    return model.predict_tail(
                        corrected_rgb, corrected_tir,
                        len(templates) * model.baseline.backbone.pos_embed_z.shape[1],
                        model.baseline.backbone.pos_embed_x.shape[1])

                off_output = predict(None)
                utility_output = predict(utility_regions)
                random_output = predict(random_regions)
                similarity_output = predict(similarity_regions)
                native_error = (native["pred_boxes"] - utility_output["pred_boxes"]).abs().max()
                assert float(native_error) < 1e-5, float(native_error)
                off_loss = one_loss(off_output, boxes)
                utility_loss = one_loss(utility_output, boxes)
                random_loss = one_loss(random_output, boxes)
                similarity_loss = one_loss(similarity_output, boxes)
                single_gains = torch.zeros(batch, device=search.device)
                for region_index in range(utility.selected_regions):
                    single_output = predict(utility_regions[:, region_index:region_index + 1])
                    single_gains += off_loss - one_loss(single_output, boxes)

                for index, identity in enumerate(batch_identities):
                    row = {
                        "identity": identity,
                        "condition": condition,
                        "utility_regions": utility_regions[index].cpu().tolist(),
                        "random_regions": random_regions[index].cpu().tolist(),
                        "similarity_regions": similarity_regions[index].cpu().tolist(),
                        "off": float(off_loss[index].cpu()),
                        "utility_joint": float(utility_loss[index].cpu()),
                        "random_joint": float(random_loss[index].cpu()),
                        "similarity_joint": float(similarity_loss[index].cpu()),
                        "utility_single_gain_sum": float(single_gains[index].cpu()),
                        "utility_joint_gain": float((off_loss - utility_loss)[index].cpu()),
                    }
                    row.update(sample_hashes(
                        templates_cpu, search_cpu, boxes_cpu, index))
                    batch_rows.append(row)

            rows.extend(batch_rows)
            processed += batch * 3
            progress = {
                "batches": batch_index + 1,
                "expected_batches": SAMPLES // args.batch_size,
                "tuples_processed": identity_offset,
                "condition_rows_processed": processed,
            }
            progress_path.write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)

    hook.remove()
    assert identity_offset == SAMPLES
    assert len(rows) == SAMPLES * 3
    assert all(row["identity"]["sequence"] in train_sequences for row in rows)
    summaries = {
        condition: summarize(rows, condition)
        for condition in ["complete", "rgb_only", "tir_only"]
    }
    report = {
        "protocol": "train882-only frozen-checkpoint diagnostic; no optimizer updates; no val97 or official-test data",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "baseline_checkpoint": BASELINE,
        "sample_seed": SEED,
        "random_selector_seed": SEED + 10,
        "samples": SAMPLES,
        "batch_size": args.batch_size,
        "conditions": ["complete", "rgb_only", "tir_only"],
        "regions_per_subset": utility.selected_regions,
        "train_sequences": len(train_sequences),
        "train_sequence_list_sha256": sha256(train_path),
        "source_sha256": {
            str(path): sha256(path)
            for folder in ["miss_rgbt", "lib", "scripts"]
            for path in sorted(Path(folder).rglob("*.py"))
        },
        "summaries": summaries,
        "advance_to_joint_utility_training": all(
            summaries[condition]["utility_minus_random_loss_improvement"]["ci95_low"] > 0
            and summaries[condition]["utility_minus_similarity_loss_improvement"]["ci95_low"] > 0
            for condition in ["rgb_only", "tir_only"]),
        "rows": rows,
    }
    report_path = output / "diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    completed = {
        "status": "complete",
        "diagnostic": str(report_path.resolve()),
        "diagnostic_sha256": sha256(report_path),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tuples": SAMPLES,
        "condition_rows": len(rows),
    }
    (output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    progress_path.write_text(json.dumps({
        "stage": "complete",
        "tuples_processed": SAMPLES,
        "condition_rows_processed": len(rows),
        "diagnostic": str(report_path.resolve()),
        "completed": str((output / "completed.json").resolve()),
    }, indent=2) + "\n")
    print(json.dumps({
        "diagnostic": str(report_path.resolve()),
        "summaries": summaries,
        "advance_to_joint_utility_training": report["advance_to_joint_utility_training"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
