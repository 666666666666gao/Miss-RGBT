"""Frozen E0 control: copy observed features only into unavailable positions."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
OUTPUT = ROOT / "reports/server28-observed-copy-20261031"
STATES = [("11", [1, 1]), ("01", [0, 1]), ("10", [1, 0])]
ARMS = ["native", "observed_copy"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run():
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(RUNTIME))
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.models.cadtrack import build_cadtrack
    from miss_rgbt.data import make_loader
    from miss_rgbt.model import MissingTrackerModel
    from miss_rgbt.objectives import task_loss
    from evaluate_blind_motion28 import network_digest

    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    assert cfg.DATA.TEMPLATE.NUMBER == 2 and cfg.DATA.SEARCH.NUMBER == 3
    checkpoint = ROOT / "checkpoints/e0_availability/epoch20.pth.tar"
    package = read(checkpoint.with_suffix(".json"))
    assert package["optimizer_updates"] == 0 and package["tensor_equality"]
    assert sha(checkpoint) == package["packaged_sha256"]
    saved = torch.load(str(checkpoint), map_location="cpu")
    settings = saved["config"]
    assert settings["repair"] is False and settings.get("task_utility") is None
    model = MissingTrackerModel(
        build_cadtrack(cfg, training=False), repair=False,
        width=settings["repair_width"], repair_layer=settings["repair_layer"])
    model.load_state_dict(saved["model"], strict=True)
    del saved
    model.cpu().eval().requires_grad_(False)
    assert model.baseline.backbone.pos_embed_z.shape[1] == 64
    assert model.baseline.backbone.pos_embed_x.shape[1] == 256
    model_digest = network_digest(model)
    manifest_path = ROOT / "reports/server28-protected-sixepoch-20261030/inputs.json"
    manifest = read(manifest_path)
    names = manifest["train_sequences"]
    assert len(names) == len(set(names)) == 705
    assert not set(names) & (set(manifest["development_sequences"]) |
                             set(manifest["heldout_sequences"]))
    random.seed(1702)
    np.random.seed(1702)
    torch.manual_seed(1702)
    loader = make_loader(cfg, "/data/wangwj/dataset/LasHeR/trainingset", "val", 4, 16, 0)
    loader.dataset.datasets[0].sequence_list = list(names)
    batches = list(loader)
    assert len(batches) == 4
    inputs = hashlib.sha256()
    for batch in batches:
        for key in ["template_images", "search_images", "search_anno"]:
            inputs.update(batch[key].numpy().tobytes())
    sources = [Path(__file__).resolve(), ROOT / "evaluate_blind_motion28.py",
               RUNTIME / "miss_rgbt/model.py", RUNTIME / "miss_rgbt/data.py",
               RUNTIME / "miss_rgbt/objectives.py",
               RUNTIME / "experiments/cadtrack/cadtrack.yaml"]
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        source_sha256={str(path): sha(path) for path in sources},
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        manifest_sha256=sha(manifest_path), samples=16, batch_size=4, seed=1702,
        input_sha256=inputs.hexdigest(), candidate_sequences=names,
        template_frames=2, search_frames=3, intervention_layer=6,
        device="cpu", threads=4, arms=ARMS,
        scope="Fixed16 tuples sampled from existing705 training sequences with val-mode "
              "transforms. No outcome filtering, optimization or new-seed/video independence. "
              "Both arms use native input-level availability masking. GT constructs native "
              "offline crops and scores loss, but does not choose copied features. "
              "This is an ordinary zero-parameter control, not a proposed novel module.")
    save(OUTPUT / "launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    context = dict(arm="native", index=0, masks=[], copied=[])

    def intervene(module, arguments, output):
        rgb, tir = output
        mask = context["masks"][context["index"]]
        context["index"] += 1
        assert rgb.shape == tir.shape and mask.shape == rgb.shape[:2] + (2,)
        if context["arm"] == "native":
            return output
        replace_rgb = (mask[..., 0] == 0) & (mask[..., 1] == 1)
        replace_tir = (mask[..., 1] == 0) & (mask[..., 0] == 1)
        repaired_rgb = torch.where(replace_rgb.unsqueeze(-1), tir, rgb)
        repaired_tir = torch.where(replace_tir.unsqueeze(-1), rgb, tir)
        assert torch.equal(repaired_rgb[~replace_rgb], rgb[~replace_rgb])
        assert torch.equal(repaired_tir[~replace_tir], tir[~replace_tir])
        context["copied"].append(dict(
            rgb=replace_rgb.sum(1).tolist(), tir=replace_tir.sum(1).tolist()))
        return repaired_rgb, repaired_tir

    hook = model.baseline.backbone.blocks[5].register_forward_hook(intervene)
    records = []
    started = time.time()
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            templates, searches = list(batch["template_images"]), list(batch["search_images"])
            assert len(templates) == 2 and len(searches) == 3
            size = searches[0].shape[0]
            for dynamic_label, dynamic_flags in STATES:
                for search_label, search_flags in STATES:
                    flags = [torch.ones(size, 2),
                             torch.tensor([dynamic_flags], dtype=torch.float32).expand(size, -1)]
                    flags += [torch.tensor([search_flags], dtype=torch.float32).expand(size, -1)
                              for _ in searches]
                    template_tokens = model.baseline.backbone.pos_embed_z.shape[1]
                    search_tokens = model.baseline.backbone.pos_embed_x.shape[1]
                    masks = [torch.cat([
                        torch.ones(size, 1, 2),
                        flags[0].unsqueeze(1).expand(-1, template_tokens, -1),
                        flags[1].unsqueeze(1).expand(-1, template_tokens, -1),
                        flags[2 + index].unsqueeze(1).expand(-1, search_tokens, -1)], 1)
                        for index in range(3)]
                    row = dict(batch=batch_index, state=dynamic_label + "_" + search_label,
                               samples=size, arms={})
                    for arm in ARMS:
                        context.update(arm=arm, index=0, masks=masks, copied=[])
                        outputs = model(templates, searches, flags)
                        assert context["index"] == 3
                        loss = task_loss(outputs, batch["search_anno"], cfg)
                        assert loss.shape == (4,) and torch.isfinite(loss).all()
                        output_hash = hashlib.sha256()
                        for out in outputs:
                            for key in ["pred_boxes", "score_map", "size_map", "offset_map"]:
                                assert torch.isfinite(out[key]).all()
                                output_hash.update(out[key].contiguous().numpy().tobytes())
                        row["arms"][arm] = dict(
                            losses=loss.tolist(), output_sha256=output_hash.hexdigest(),
                            copied_tokens_per_search=context["copied"])
                    if row["state"] == "11_11":
                        assert row["arms"]["native"] == dict(
                            row["arms"]["observed_copy"], copied_tokens_per_search=[])
                    records.append(row)
            print("BATCH " + json.dumps(dict(batch=batch_index, samples=(batch_index + 1) * 4,
                                             seconds=time.time() - started)), flush=True)
    hook.remove()
    assert network_digest(model) == model_digest
    save(OUTPUT / "report.json", dict(
        launch=launch, records=records, model_state_digest=model_digest,
        model_state_unchanged=True, complete_history_outputs_bitwise_equal=True,
        observed_positions_unchanged_at_intervention=True, seconds=time.time() - started))
    save(OUTPUT / "completed.json", dict(report_sha256=sha(OUTPUT / "report.json"),
                                         cells=18, samples_per_cell=16))
    analyze()
    print("COMPLETE", flush=True)


def analyze():
    report = read(OUTPUT / "report.json")
    assert sha(OUTPUT / "report.json") == read(OUTPUT / "completed.json")["report_sha256"]
    launch = report["launch"]
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    assert sha(launch["checkpoint"]) == launch["checkpoint_sha256"]
    assert len(report["records"]) == 36
    results = {}
    for dynamic, dynamic_flags in STATES:
        for search, search_flags in STATES:
            key = dynamic + "_" + search
            rows = [row for row in report["records"] if row["state"] == key]
            assert [row["batch"] for row in rows] == list(range(4))
            values = {arm: [value for row in rows for value in row["arms"][arm]["losses"]]
                      for arm in ARMS}
            assert all(len(items) == 16 for items in values.values())
            assert all(math.isfinite(value) for items in values.values() for value in items)
            for row in rows:
                assert row["samples"] == 4 and set(row["arms"]) == set(ARMS)
                copied = row["arms"]["observed_copy"]["copied_tokens_per_search"]
                assert len(copied) == 3
                for item in copied:
                    for view, label in enumerate(["rgb", "tir"]):
                        expected = (64 * int(dynamic_flags[view] == 0) +
                                    256 * int(search_flags[view] == 0))
                        assert item[label] == [expected] * 4
                if key == "11_11":
                    assert row["arms"]["native"]["output_sha256"] == row["arms"]["observed_copy"]["output_sha256"]
            delta = [a - b for a, b in zip(values["observed_copy"], values["native"])]
            results[key] = dict(
                native_mean=sum(values["native"]) / 16,
                observed_copy_mean=sum(values["observed_copy"]) / 16,
                mean_loss_delta=sum(delta) / 16,
                wins=sum(value < 0 for value in delta),
                ties=sum(value == 0 for value in delta),
                losses=sum(value > 0 for value in delta))
    save(OUTPUT / "analysis.json", dict(
        results=results, report_sha256=sha(OUTPUT / "report.json"),
        source_sha256=sha(__file__), new_model_calls=0,
        scope="Separate receipt/source/hash/shape/token-count/aggregate checks on producer "
              "outputs. Execution witnesses are not re-executed. Correlated training tuples "
              "and batch-coupled loss, not PR/SR, closed-loop benefit or a validated module."))
    save(OUTPUT / "analysis_completed.json", dict(analysis_sha256=sha(OUTPUT / "analysis.json")))
    print("ANALYSIS " + json.dumps(results), flush=True)


def launch():
    path = ROOT / "reports/server28-observed-copy-launch-20261031.json"
    assert not OUTPUT.exists() and not path.exists()
    with Path(str(OUTPUT) + ".log").open("xb") as log:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "run"]
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                     MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"))
    result = dict(pid=process.pid, command=command, source_sha256=sha(__file__))
    save(path, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "analyze", "launch"], required=True)
    args = parser.parse_args()
    {"run": run, "analyze": analyze, "launch": launch}[args.mode]()
