"""Bounded CPU missing-branch pilot with sequence-disjoint model selection."""
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

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
OUTPUT = ROOT / "reports/server28-protected-sixepoch-20261030"
sys.path.insert(0, str(RUNTIME))
from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from evaluate_val97_recovery28 import sha


def digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    seed(2027)
    started = time.time()
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    checkpoint = ROOT / "checkpoints/pretrained/CADTrack_ep0020.pth.tar"
    model = load_pretrained(cfg, str(checkpoint), repair=True, width=128,
                            repair_layer=6).cpu().eval()
    model.baseline.requires_grad_(False)
    baseline_sha = digest(model.baseline.state_dict())
    launch = dict(pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
                  device="cpu", cpu_threads=4, epochs=6, updates_per_epoch=32,
                  effective_batch=4, learning_rate=.0001,
                  checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  source_sha256={str(p): sha(p) for p in [
                      Path(__file__), RUNTIME / "miss_rgbt/model.py",
                      RUNTIME / "miss_rgbt/data.py", RUNTIME / "miss_rgbt/objectives.py"]},
                  selection="Lowest development mean 01/10 task loss among epochs2..6; ties earlier",
                  scope="Existing LocalRepair CPU pilot, not a new contribution or tracking metric")
    save("launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    data_root = "/data/wangwj/dataset/LasHeR/trainingset"
    train_loader = make_loader(cfg, data_root, "train", 2, 64, 0)
    all_train = train_loader.dataset.datasets[0].sequence_list
    development_names = all_train[::5]
    train_names = [name for index, name in enumerate(all_train) if index % 5]
    train_loader.dataset.datasets[0].sequence_list = train_names
    development_loader = make_loader(cfg, data_root, "val", 2, 16, 0)
    heldout_loader = make_loader(cfg, data_root, "val", 2, 32, 0)
    heldout_names = heldout_loader.dataset.datasets[0].sequence_list
    development_loader.dataset.datasets[0].sequence_list = development_names
    assert not (set(train_names) & set(development_names))
    assert not (set(all_train) & set(heldout_names))
    seed(27182)
    development = list(development_loader)
    seed(31415)
    heldout = list(heldout_loader)
    save("inputs.json", dict(
        train_sequences=train_names, development_sequences=development_names,
        heldout_sequences=heldout_names,
        development=[digest({key: batch[key] for key in [
            "template_images", "search_images", "search_anno"]}) for batch in development],
        heldout=[digest({key: batch[key] for key in [
            "template_images", "search_images", "search_anno"]}) for batch in heldout]))

    def forward(batch, flags):
        templates, searches = list(batch["template_images"]), list(batch["search_images"])
        size = searches[0].shape[0]
        available = [torch.ones(size, 2) for _ in templates]
        available += [torch.tensor([flags]).float().expand(size, -1) for _ in searches]
        return model(templates, searches, available)

    def measure(batches):
        totals = {"11": 0., "01": 0., "10": 0.}
        complete = hashlib.sha256()
        with torch.no_grad():
            for batch in batches:
                for label, flags in [("11", [1, 1]), ("01", [0, 1]), ("10", [1, 0])]:
                    output = forward(batch, flags)
                    loss = task_loss(output, batch["search_anno"], cfg).mean()
                    assert torch.isfinite(loss)
                    totals[label] += float(loss) / len(batches)
                    if label == "11":
                        for out in output:
                            for key in ["score_map", "size_map", "offset_map", "pred_boxes"]:
                                complete.update(out[key].contiguous().numpy().tobytes())
        return dict(task_loss=totals, complete_outputs_sha256=complete.hexdigest(),
                    selection_score=(totals["01"] + totals["10"]) / 2)

    initial_development = measure(development)
    initial_heldout = measure(heldout)
    # Heldout losses are recorded only after selecting an epoch from development.
    optimizer = torch.optim.AdamW(model.repair.parameters(), lr=.0001, weight_decay=.0001)
    history = []
    updates = 0
    for epoch in range(1, 7):
        seed(20270 + epoch)
        batches = list(train_loader)
        save("epoch%02d_inputs.json" % epoch, [
            digest({key: batch[key] for key in [
                "template_images", "search_images", "search_anno"]}) for batch in batches])
        for pair in range(len(batches) // 2):
            # Both outage directions see the same effective batch each epoch.
            for flags in [[0, 1], [1, 0]]:
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                for batch in batches[2 * pair:2 * pair + 2]:
                    loss = task_loss(forward(batch, flags), batch["search_anno"], cfg).mean()
                    assert torch.isfinite(loss)
                    (loss / 2).backward()
                    loss_sum += float(loss.detach()) / 2
                norm = torch.nn.utils.clip_grad_norm_(model.repair.parameters(), .1)
                assert torch.isfinite(norm)
                optimizer.step()
                updates += 1
                print("UPDATE " + json.dumps(dict(
                    epoch=epoch, update=updates, available=flags, loss=loss_sum,
                    gradient_norm=float(norm), seconds=time.time() - started)), flush=True)
        validation = measure(development)
        assert validation["complete_outputs_sha256"] == initial_development["complete_outputs_sha256"]
        history.append(dict(epoch=epoch, updates=updates, development=validation))
        if epoch >= 2:
            torch.save(dict(repair=model.repair.state_dict(), epoch=epoch, launch=launch),
                       str(OUTPUT / ("repair_epoch%02d.pth" % epoch)))
        save("development_history.json", history)
        print("EPOCH " + json.dumps(history[-1]), flush=True)
    best = min(history[-5:], key=lambda x: (x["development"]["selection_score"], x["epoch"]))
    retained = OUTPUT / ("repair_epoch%02d.pth" % best["epoch"])
    model.repair.load_state_dict(torch.load(str(retained), map_location="cpu")["repair"], strict=True)
    actual_heldout = measure(heldout)
    assert actual_heldout["complete_outputs_sha256"] == initial_heldout["complete_outputs_sha256"]
    assert baseline_sha == digest(model.baseline.state_dict())
    assert all(parameter.grad is None for parameter in model.baseline.parameters())
    cleanup = []
    for path in sorted(OUTPUT.glob("repair_epoch*.pth")):
        assert path.parent == OUTPUT
        if path != retained:
            cleanup.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    report = dict(
        launch=launch, selected=best, retained=str(retained), retained_sha256=sha(retained),
        initial_development=initial_development, initial_heldout=initial_heldout,
        selected_heldout=actual_heldout,
        heldout_loss_delta={s: actual_heldout["task_loss"][s] - initial_heldout["task_loss"][s]
                            for s in ["11", "01", "10"]},
        baseline_state_unchanged=True, complete_outputs_bitwise_equal=True,
        updates=updates, cleanup=cleanup, seconds=time.time() - started)
    save("report.json", report)
    for item in cleanup:
        Path(item["path"]).unlink()
    save("completed.json", dict(selected_epoch=best["epoch"], updates=updates,
                               cleanup_completed=True, seconds=time.time() - started))
    print("COMPLETE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
