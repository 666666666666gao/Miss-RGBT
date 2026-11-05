"""Matched CPU LocalRepair pilots differing only in dynamic-template masks."""
import datetime
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from probe_protected_sixepoch28 import (
    ROOT, RUNTIME, cfg, cv2, digest, load_pretrained, make_loader, seed,
    sha, task_loss, torch, update_config_from_file,
)


OUTPUT = ROOT / "reports/server28-protected-mixed-pair-20261031"
PREVIOUS = ROOT / "reports/server28-protected-sixepoch-20261030"
STATES = [("11", [1, 1]), ("01", [0, 1]), ("10", [1, 0])]


def save(folder, name, value):
    (folder / name).write_text(json.dumps(value, indent=2) + "\n")


def inputs(batch):
    return digest({key: batch[key] for key in [
        "template_images", "search_images", "search_anno"]})


def forward(model, batch, dynamic, search_flags):
    templates, searches = list(batch["template_images"]), list(batch["search_images"])
    size = searches[0].shape[0]
    assert len(templates) == 2
    available = [torch.ones(size, 2), dynamic]
    available += [torch.tensor([search_flags], dtype=torch.float32).expand(size, -1)
                  for _ in searches]
    return model(templates, searches, available)


def measure(model, batches):
    results = {}
    with torch.no_grad():
        for dynamic_label, dynamic_flags in STATES:
            for search_label, search_flags in STATES:
                key = dynamic_label + "_" + search_label
                losses, hashes = [], []
                for batch in batches:
                    size = batch["search_images"][0].shape[0]
                    dynamic = torch.tensor([dynamic_flags], dtype=torch.float32).expand(size, -1)
                    outputs = forward(model, batch, dynamic, search_flags)
                    loss = task_loss(outputs, batch["search_anno"], cfg)
                    assert torch.isfinite(loss).all()
                    losses.extend(loss.reshape(-1).tolist())
                    if key == "11_11":
                        hashes.append(digest({
                            str(index) + "_" + field: output[field]
                            for index, output in enumerate(outputs)
                            for field in ["score_map", "size_map", "offset_map", "pred_boxes"]}))
                results[key] = dict(mean_task_loss=sum(losses) / len(losses), losses=losses)
                if hashes:
                    results[key]["outputs_sha256"] = hashes
    missing = [row["mean_task_loss"] for key, row in results.items()
               if key.split("_")[1] != "11"]
    return dict(cells=results, selection_score=sum(missing) / len(missing))


def run_arm(label, manifest, started):
    folder = OUTPUT / label
    folder.mkdir()
    seed(2027)
    checkpoint = ROOT / "checkpoints/pretrained/CADTrack_ep0020.pth.tar"
    model = load_pretrained(cfg, str(checkpoint), repair=True, width=128,
                            repair_layer=6).cpu().eval()
    model.baseline.requires_grad_(False)
    initial_repair = digest(model.repair.state_dict())
    baseline = digest(model.baseline.state_dict())
    data_root = "/data/wangwj/dataset/LasHeR/trainingset"
    train_loader = make_loader(cfg, data_root, "train", 2, 64, 0)
    all_names = train_loader.dataset.datasets[0].sequence_list
    development_names = all_names[::5]
    train_names = [name for index, name in enumerate(all_names) if index % 5]
    train_loader.dataset.datasets[0].sequence_list = train_names
    development_loader = make_loader(cfg, data_root, "val", 2, 16, 0)
    assert development_loader.dataset.datasets[0].sequence_list == manifest["heldout_sequences"]
    development_loader.dataset.datasets[0].sequence_list = development_names
    assert train_names == manifest["train_sequences"]
    assert development_names == manifest["development_sequences"]
    assert not set(train_names) & set(development_names)
    assert not set(all_names) & set(manifest["heldout_sequences"])
    seed(27182)
    development = list(development_loader)
    assert [inputs(batch) for batch in development] == manifest["development"]
    initial = measure(model, development)
    save(folder, "initial.json", dict(
        initial_repair_sha256=initial_repair, baseline_sha256=baseline, development=initial))
    print("INITIAL " + json.dumps(dict(arm=label, seconds=time.time() - started,
                                       selection_score=initial["selection_score"])), flush=True)
    optimizer = torch.optim.AdamW(model.repair.parameters(), lr=.0001, weight_decay=.0001)
    states = torch.tensor([[1., 1.], [0., 1.], [1., 0.]])
    probability = torch.tensor([.4, .3, .3])
    history, updates = [], 0
    for epoch in range(1, 7):
        seed(20270 + epoch)
        batches = list(train_loader)
        hashes = [inputs(batch) for batch in batches]
        assert hashes == json.loads((PREVIOUS / ("epoch%02d_inputs.json" % epoch)).read_text())
        save(folder, "epoch%02d_inputs.json" % epoch, hashes)
        generator = torch.Generator().manual_seed(50000 + epoch)
        mask_counts = {state: 0 for state, _ in STATES}
        for pair in range(len(batches) // 2):
            selected = batches[2 * pair:2 * pair + 2]
            dynamic = []
            for batch in selected:
                size = batch["search_images"][0].shape[0]
                mask = states[torch.multinomial(probability, size, replacement=True,
                                                generator=generator)]
                if label == "complete_templates":
                    mask = torch.ones_like(mask)
                dynamic.append(mask)
                for state, flags in STATES:
                    mask_counts[state] += int((mask == torch.tensor(flags)).all(1).sum())
            for search_flags in [[0, 1], [1, 0]]:
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                for batch, mask in zip(selected, dynamic):
                    loss = task_loss(forward(model, batch, mask, search_flags),
                                     batch["search_anno"], cfg).mean()
                    assert torch.isfinite(loss)
                    (loss / 2).backward()
                    loss_sum += float(loss.detach()) / 2
                norm = torch.nn.utils.clip_grad_norm_(model.repair.parameters(), .1)
                assert torch.isfinite(norm)
                optimizer.step()
                updates += 1
                print("UPDATE " + json.dumps(dict(
                    arm=label, epoch=epoch, update=updates, loss=loss_sum,
                    gradient_norm=float(norm), seconds=time.time() - started)), flush=True)
        development_result = measure(model, development)
        assert (development_result["cells"]["11_11"]["outputs_sha256"] ==
                initial["cells"]["11_11"]["outputs_sha256"])
        row = dict(epoch=epoch, updates=updates, dynamic_template_counts=mask_counts,
                   development=development_result, seconds=time.time() - started)
        history.append(row)
        if epoch >= 2:
            torch.save(dict(repair=model.repair.state_dict(), epoch=epoch, arm=label),
                       str(folder / ("repair_epoch%02d.pth" % epoch)))
        save(folder, "development_history.json", history)
        print("EPOCH " + json.dumps(dict(arm=label, epoch=epoch, updates=updates,
                                        selection_score=development_result["selection_score"],
                                        seconds=time.time() - started)), flush=True)
    best = min(history[-5:], key=lambda row: (row["development"]["selection_score"], row["epoch"]))
    retained = folder / ("repair_epoch%02d.pth" % best["epoch"])
    model.repair.load_state_dict(torch.load(str(retained), map_location="cpu")["repair"], strict=True)
    selected_result = measure(model, development)
    assert selected_result == best["development"]
    assert digest(model.baseline.state_dict()) == baseline
    assert all(parameter.grad is None for parameter in model.baseline.parameters())
    assert updates == 192
    cleanup = []
    for path in sorted(folder.glob("repair_epoch*.pth")):
        assert path.parent == folder
        if path != retained:
            cleanup.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    assert len(cleanup) == 4
    report = dict(
        arm=label, selected=best, initial=initial, initial_repair_sha256=initial_repair,
        retained=str(retained), retained_sha256=sha(retained), updates=updates,
        baseline_state_unchanged=True, complete_history_outputs_bitwise_equal=True,
        cleanup=cleanup, seconds=time.time() - started)
    save(folder, "report.json", report)
    for item in cleanup:
        Path(item["path"]).unlink()
    save(folder, "completed.json", dict(updates=updates, selected_epoch=best["epoch"],
                                       cleanup_completed=True, retained=str(retained),
                                       retained_sha256=sha(retained)))
    return report


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    started = time.time()
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    previous = json.loads((PREVIOUS / "report.json").read_text())
    for path, expected in previous["launch"]["source_sha256"].items():
        assert sha(Path(path)) == expected
    assert sha(previous["launch"]["checkpoint"]) == previous["launch"]["checkpoint_sha256"]
    manifest = json.loads((PREVIOUS / "inputs.json").read_text())
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        device="cpu", cpu_threads=4, arms=["complete_templates", "mixed_templates"],
        epochs_per_arm=6, updates_per_arm=192, effective_batch=4, learning_rate=.0001,
        dynamic_template_probability=[.4, .3, .3], mask_seed_by_epoch="50000+epoch",
        input_manifest_sha256=sha(PREVIOUS / "inputs.json"),
        source_sha256={str(Path(__file__)): sha(Path(__file__)),
                       **previous["launch"]["source_sha256"]},
        selection="Lowest mean of six dynamic11/01/10 x search01/10 development losses "
                  "among final five epochs2..6 for BOTH arms; earlier on ties",
        scope="Matched ordinary LocalRepair control, train882 development only; "
              "no val97 scores, tracking, new contribution or official target claim")
    save(OUTPUT, "launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    results = {}
    for arm in launch["arms"]:
        results[arm] = run_arm(arm, manifest, started)
    assert results["complete_templates"]["initial_repair_sha256"] == results[
        "mixed_templates"]["initial_repair_sha256"]
    assert results["complete_templates"]["initial"] == results["mixed_templates"]["initial"]
    comparison = dict(
        launch=launch, arms=results,
        selected_loss_delta_mixed_minus_complete={
            cell: results["mixed_templates"]["selected"]["development"]["cells"][cell]["mean_task_loss"]
                  - results["complete_templates"]["selected"]["development"]["cells"][cell]["mean_task_loss"]
            for cell in results["complete_templates"]["initial"]["cells"]},
        same_initial_state_and_inputs=True, seconds=time.time() - started)
    save(OUTPUT, "comparison.json", comparison)
    save(OUTPUT, "completed.json", dict(
        updates=384, comparison_sha256=sha(OUTPUT / "comparison.json"),
        ended_utc=datetime.datetime.utcnow().isoformat() + "Z", seconds=time.time() - started))
    print("COMPLETE " + json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
