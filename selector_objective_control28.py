"""Frozen counterfactual labels: matched utility-regression versus pairwise control."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
OUTPUT = ROOT / "reports/server28-selector-objective-control-20261030"
sys.path.insert(0, str(RUNTIME))
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.data import make_loader
from miss_rgbt.model import MissingTrackerModel
from miss_rgbt.objectives import task_loss
from miss_rgbt.task_utility import utility_targets
from evaluate_val97_recovery28 import sha


def objective(prediction, target, kind):
    if kind == "regression":
        return F.smooth_l1_loss(prediction, target)
    differences = prediction.unsqueeze(-1) - prediction.unsqueeze(-2)
    positive = target.unsqueeze(-1) > target.unsqueeze(-2)
    assert positive.any()
    return F.softplus(-differences[positive]).mean()


@torch.no_grad()
def evaluate(selector, rows, kind):
    scores = selector(torch.stack([row["features"] for row in rows])).squeeze(-1)
    targets = torch.stack([row["targets"] for row in rows])
    correct, total, overlaps, selected_benefits = 0, 0, [], []
    for prediction, target in zip(scores, targets):
        positive = target.unsqueeze(-1) > target.unsqueeze(-2)
        predicted = prediction.unsqueeze(-1) - prediction.unsqueeze(-2)
        correct += float(((predicted > 0) & positive).sum())
        correct += .5 * float(((predicted == 0) & positive).sum())
        total += int(positive.sum())
        selected = prediction.topk(4).indices
        actual = target.topk(4).indices
        overlaps.append(len(set(selected.tolist()) & set(actual.tolist())))
        selected_benefits.append(float(target[selected].mean()))
    return dict(
        slates=len(rows), objective=float(objective(scores, targets, kind)),
        strict_order_pairs=total, pairwise_accuracy=correct / total,
        mean_top4_overlap=sum(overlaps) / len(overlaps),
        mean_selected_single_region_benefit=sum(selected_benefits) / len(selected_benefits))


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    OUTPUT.mkdir(exist_ok=False)
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    checkpoint = ROOT / "artifacts/server25-data2/checkpoints/a_utility_full20_seed2027/epoch20.pth.tar"
    saved = torch.load(str(checkpoint), map_location="cpu")
    config = saved["config"]
    model = MissingTrackerModel(build_cadtrack(cfg, training=False), repair=False,
                                width=128, repair_layer=6, task_utility=config["task_utility"]).cpu().eval()
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False)
    utility = model.utility_repair
    train_spec = RUNTIME / "lib/train/data_specs/lasher_train.txt"
    val_spec = RUNTIME / "lib/train/data_specs/lasher_val.txt"
    train_names = train_spec.read_text().splitlines()
    val_names = val_spec.read_text().splitlines()
    splits = dict(train=train_names[:16], development=train_names[16:20], heldout=val_names[:4])
    assert not set(splits["train"]) & set(splits["development"])
    assert not set(splits["train"] + splits["development"]) & set(splits["heldout"])
    started = time.time()
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        device="cpu", cpu_threads=4, checkpoint_sha256=sha(checkpoint),
        source_sha256=sha(__file__), splits=splits, seed=6027,
        protocol="One causal tuple per sequence, first search, deterministic validation preprocessing "
                 "also for training tuples; full templates, search11/01/10, no patch mask. "
                 "All16 region interventions, same frozen features/correction/head. "
                 "Historical selector initialization, AdamW1e-4 weight_decay1e-4 for both controls.",
        selection="Each objective uses its own development loss, min30/max100epochs, "
                  "patience10 for0.2%relative improvement; choose best final5. "
                  "Heldout joint task loss evaluated only after selection.",
        scope="Small offline objective-control pilot, not full network training, benchmark, "
              "recovery, convergence of tracker or novel ranking algorithm. "
              "Strict floating-point gain ordering may contain numerically weak preferences.",
    )
    (OUTPUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    capture = {}

    def hook(module, inputs, output):
        capture["rgb"], capture["tir"], capture["available"] = [value.detach().clone() for value in inputs]
        capture["context"] = output[2]["context"].detach().clone()

    handle = utility.register_forward_hook(hook)
    dataset = {key: [] for key in splits}
    caches = {}
    manifest = []
    for split, names in splits.items():
        for sequence_index, name in enumerate(names):
            seed = 6027 + sequence_index + (100 if split == "development" else 200 if split == "heldout" else 0)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            loader = make_loader(cfg, "/data/wangwj/dataset/LasHeR/trainingset", "val", 1, 1, 0)
            loader.dataset.datasets[0].sequence_list = [name]
            info = loader.dataset.datasets[0].get_sequence_info(0)
            assert len(info["visible"]) >= 20 and int(info["visible"].sum()) > 6
            data = next(iter(loader))
            templates = list(data["template_images"])
            searches = [data["search_images"][0]]
            boxes = data["search_anno"][:1]
            input_hash = hashlib.sha256(data["search_images"].numpy().tobytes()).hexdigest()
            for flags in [[1., 1.], [0., 1.], [1., 0.]]:
                state = "".join(str(int(value)) for value in flags)
                available = [torch.ones(1, 2) for _ in templates] + [torch.tensor([flags])]
                with torch.no_grad():
                    native = model(templates, searches, available)[0]
                    r, t, mask, context = [capture[key] for key in ["rgb", "tir", "available", "context"]]
                    features = context[:, utility.region_tokens].mean(2)[0].clone()
                    assert torch.equal(utility.selector(features).squeeze(-1), native["utility_logits"][0])
                    lens_z = len(templates) * utility.template_tokens
                    lens_x = utility.search_tokens
                    off_loss = task_loss([model.predict_tail(r, t, lens_z, lens_x)], boxes, cfg)
                    targets = []
                    for region in range(16):
                        on_r, on_t, _ = utility.apply_regions(r, t, mask, context, torch.tensor([[region]]))
                        on_loss = task_loss([model.predict_tail(on_r, on_t, lens_z, lens_x)], boxes, cfg)
                        targets.append(float(utility_targets(off_loss, on_loss)))
                    target = torch.tensor(targets)
                    assert torch.isfinite(target).all()
                    row = dict(sequence=name, state=state, features=features, targets=target,
                               off_loss=float(off_loss),
                               native_loss=float(task_loss([native], boxes, cfg)))
                    dataset[split].append(row)
                    if split == "heldout":
                        caches[(name, state)] = dict(r=r, t=t, mask=mask, context=context,
                                                    boxes=boxes, lens_z=lens_z, lens_x=lens_x)
                    manifest.append(dict(split=split, sequence=name, state=state,
                                         input_sha256=input_hash, targets=targets,
                                         off_loss=row["off_loss"], native_loss=row["native_loss"]))
            (OUTPUT / "labels.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print("LABELS " + json.dumps(dict(split=split, sequence=name,
                                            slates=len(dataset[split]), seconds=time.time() - started)), flush=True)
    handle.remove()
    torch.save(dataset, str(OUTPUT / "features_labels.pth"))
    result = dict(launch=launch, controls={})
    for kind in ["regression", "pairwise"]:
        selector = copy.deepcopy(utility.selector).requires_grad_(True)
        optimizer = torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
        features = torch.stack([row["features"] for row in dataset["train"]])
        targets = torch.stack([row["targets"] for row in dataset["train"]])
        history = []
        best, stale = float("inf"), 0
        for epoch in range(1, 101):
            optimizer.zero_grad(set_to_none=True)
            prediction = selector(features).squeeze(-1)
            loss = objective(prediction, targets, kind)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in selector.parameters())
            optimizer.step()
            development = evaluate(selector, dataset["development"], kind)
            history.append(dict(epoch=epoch, development=development, training_loss=float(loss.detach())))
            torch.save(dict(selector=selector.state_dict(), epoch=epoch, objective=kind),
                       str(OUTPUT / ("%s_epoch%03d.pth" % (kind, epoch))))
            if epoch > 5:
                (OUTPUT / ("%s_epoch%03d.pth" % (kind, epoch - 5))).unlink()
            if development["objective"] < best * .998:
                best, stale = development["objective"], 0
            else:
                stale += 1
            if epoch >= 30 and stale >= 10:
                break
        selected = min(history[-5:], key=lambda value: (value["development"]["objective"], value["epoch"]))
        path = OUTPUT / ("%s_epoch%03d.pth" % (kind, selected["epoch"]))
        selector.load_state_dict(torch.load(str(path), map_location="cpu")["selector"], strict=True)
        joint = []
        generator = torch.Generator().manual_seed(2027)
        with torch.no_grad():
            for row in dataset["heldout"]:
                cached = caches[(row["sequence"], row["state"])]
                logits = selector(row["features"]).squeeze(-1)
                regions = dict(selected=logits.topk(4).indices.reshape(1, 4),
                               random=torch.randperm(16, generator=generator)[:4].reshape(1, 4),
                               oracle=row["targets"].topk(4).indices.reshape(1, 4))
                record = dict(sequence=row["sequence"], state=row["state"], off_loss=row["off_loss"],
                              native_loss=row["native_loss"])
                for rule, indices in regions.items():
                    r, t, _ = utility.apply_regions(cached["r"], cached["t"], cached["mask"],
                                                    cached["context"], indices)
                    value = task_loss([model.predict_tail(r, t, cached["lens_z"], cached["lens_x"])],
                                      cached["boxes"], cfg)
                    record[rule + "_loss"] = float(value)
                joint.append(record)
        cleanup = []
        for item in history[-5:]:
            other = OUTPUT / ("%s_epoch%03d.pth" % (kind, item["epoch"]))
            if other != path:
                cleanup.append(dict(path=str(other), sha256=sha(other)))
        result["controls"][kind] = dict(
            epochs=epoch, selected_epoch=selected["epoch"], selected_sha256=sha(path),
            final_five_epochs=[item["epoch"] for item in history[-5:]],
            development=selected["development"], heldout=evaluate(selector, dataset["heldout"], kind),
            heldout_per_state={state: evaluate(selector, [row for row in dataset["heldout"] if row["state"] == state], kind)
                               for state in ["11", "01", "10"]},
            joint_interventions=joint, history=history, cleanup=cleanup)
        (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        for item in cleanup:
            Path(item["path"]).unlink()
        print("CONTROL " + json.dumps(dict(objective=kind, epoch=selected["epoch"],
                                          heldout=result["controls"][kind]["heldout"])), flush=True)
    result["seconds"] = time.time() - started
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUTPUT / "completed.json").write_text(json.dumps(dict(
        controls=2, train_slates=len(dataset["train"]), development_slates=len(dataset["development"]),
        heldout_slates=len(dataset["heldout"]), seconds=result["seconds"]), indent=2) + "\n")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
