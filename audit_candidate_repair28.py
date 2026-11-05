"""Fixed-input inference interventions on a matched retained epoch6 pair."""
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
OUTPUT = ROOT / "reports/server28-candidate-repair-audit-20261031"
sys.path.insert(0, str(RUNTIME))
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.data import make_loader
from miss_rgbt.model import MissingTrackerModel
from miss_rgbt.objectives import task_loss
from evaluate_val97_recovery28 import sha


def digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    OUTPUT.mkdir(exist_ok=False)
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    campaign = ROOT / "reports/server28-candidates-20261030/batch10_utility_seed2027"
    config = json.loads((campaign / "config.json").read_text())
    expected = json.loads((Path(config["output"]) / "validation_inputs.json").read_text())
    seed = config["validation_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    batches = list(make_loader(cfg, config["dataset"], "val", 4, 128, 0))
    inputs = hashlib.sha256()
    for batch in batches:
        for key in ["template_images", "search_images", "search_anno"]:
            inputs.update(batch[key].numpy().tobytes())
    assert inputs.hexdigest() == expected["inputs_sha256"]
    paired = ROOT / "reports/epoch05-full-paired-analysis-20261031.json"
    checkpoints = {label: ROOT / "runs/server28" / (
        "batch10_%s_seed2027" % label) / "epoch06.pth.tar"
        for label in ["utility", "random"]}
    paths = [
        Path(__file__), ROOT / "audit_validation_reference28.py",
        RUNTIME / "miss_rgbt/model.py", RUNTIME / "miss_rgbt/task_utility.py",
        RUNTIME / "miss_rgbt/objectives.py", RUNTIME / "miss_rgbt/data.py",
        RUNTIME / "experiments/cadtrack/cadtrack.yaml",
    ]
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        device="cpu", threads=4, inputs_sha256=inputs.hexdigest(),
        validation_seed=seed, samples=128, batch_size=4,
        motivating_epoch5_tracking_report_sha256=sha(paired),
        checkpoint_epoch=6, checkpoint_updates=12000,
        checkpoint_sha256={key: sha(path) for key, path in checkpoints.items()},
        source_sha256={str(path): sha(path) for path in paths},
        arms=["native4", "off", "random4", "all16"],
        random_control_seed=2027,
        protocol="Exact fixed128 cached tuples, both search frames, full templates, "
                 "search11/01/10, eval mode, no patch masking or optimization. "
                 "Within each model/search, reuse identical pre-repair features and "
                 "context for off, seeded random4 and all16 tail interventions. "
                 "Native4 is re-executed through the same tail as an exact witness.",
        scope="Inference ablations of the retained matched epoch6 checkpoints, "
              "motivated by epoch5 full97 tracking but distinct weights. Random "
              "epoch5 was already removed by rolling retention. Not retrained "
              "ablations, tracking PR/SR, official "
              "scores, independent validation or a novel module. All16 costs more "
              "and is only a diagnostic. CPU/GPU equality is not assumed.")
    save("launch.json", launch)
    print("LAUNCH " + json.dumps(launch), flush=True)
    started, results = time.time(), {}
    for label in ["utility", "random"]:
        checkpoint = checkpoints[label]
        assert sha(checkpoint) == launch["checkpoint_sha256"][label]
        saved = torch.load(str(checkpoint), map_location="cpu")
        settings = saved["config"]
        assert saved["epoch"] == 6 and saved["updates"] == 12000
        model = MissingTrackerModel(
            build_cadtrack(cfg, training=False), repair=settings["repair"],
            width=settings["repair_width"], repair_layer=settings["repair_layer"],
            task_utility=settings["task_utility"]).cpu().eval().requires_grad_(False)
        model.load_state_dict(saved["model"], strict=True)
        del saved
        utility = model.utility_repair
        original_state = utility.selection_rng_state.clone()
        before = digest(model)
        utility.reset_selection()
        generator = torch.Generator().manual_seed(2027)
        captures = []

        def capture(module, args, output):
            captures.append(dict(
                rgb=args[0].detach().clone(), tir=args[1].detach().clone(),
                available=args[2].detach().clone(),
                context=output[2]["context"].detach().clone(),
                regions=output[2]["utility_regions"].detach().clone()))

        handle = utility.register_forward_hook(capture)
        records = []
        with torch.no_grad():
            for index, batch in enumerate(batches):
                templates = list(batch["template_images"])
                searches = list(batch["search_images"])
                size = searches[0].shape[0]
                lens_z = len(templates) * utility.template_tokens
                record = dict(batch=index, samples=size, states={})
                for state, flags in [("11", [1., 1.]), ("01", [0., 1.]),
                                     ("10", [1., 0.])]:
                    captures.clear()
                    available = [torch.ones(size, 2) for _ in templates]
                    available += [torch.tensor([flags]).expand(size, -1) for _ in searches]
                    native = model(templates, searches, available)
                    assert len(captures) == len(searches)
                    outputs = {key: [] for key in ["off", "random4", "all16"]}
                    random_regions, native_regions = [], []
                    for search_index, cached in enumerate(captures):
                        r, t, mask, context = [cached[key] for key in [
                            "rgb", "tir", "available", "context"]]
                        on_r, on_t, _ = utility.apply_regions(
                            r, t, mask, context, cached["regions"])
                        replay = model.predict_tail(on_r, on_t, lens_z, utility.search_tokens)
                        for key in ["pred_boxes", "score_map", "size_map", "offset_map"]:
                            assert torch.equal(replay[key], native[search_index][key]), key
                        outputs["off"].append(model.predict_tail(
                            r, t, lens_z, utility.search_tokens))
                        selected = torch.stack([
                            torch.randperm(16, generator=generator)[:4] for _ in range(size)])
                        random_regions.append(selected.tolist())
                        native_regions.append(cached["regions"].tolist())
                        for arm, regions in [
                            ("random4", selected),
                            ("all16", torch.arange(16).expand(size, -1)),
                        ]:
                            on_r, on_t, _ = utility.apply_regions(r, t, mask, context, regions)
                            outputs[arm].append(model.predict_tail(
                                on_r, on_t, lens_z, utility.search_tokens))
                    outputs["native4"] = native
                    losses = {key: task_loss(value, batch["search_anno"], cfg)
                              for key, value in outputs.items()}
                    assert all(torch.isfinite(value).all() for value in losses.values())
                    record["states"][state] = dict(
                        task_loss={key: value.tolist() for key, value in losses.items()},
                        native_regions=native_regions, random_regions=random_regions,
                        native_tail_bitwise_equal=True)
                records.append(record)
                save(label + "_progress.json", dict(records=records,
                                                    seconds=time.time() - started))
                if (index + 1) % 4 == 0:
                    print("PROGRESS " + json.dumps(dict(
                        model=label, samples=sum(row["samples"] for row in records),
                        seconds=time.time() - started)), flush=True)
        handle.remove()
        utility.selection_rng_state.copy_(original_state)
        assert digest(model) == before
        summary = {}
        for state in ["11", "01", "10"]:
            values = {arm: np.asarray([
                value for row in records for value in row["states"][state]["task_loss"][arm]])
                for arm in launch["arms"]}
            assert all(value.shape == (128,) for value in values.values())
            summary[state] = dict(
                mean_task_loss={key: float(value.mean()) for key, value in values.items()},
                contrasts={first + "_minus_" + second: dict(
                    mean_loss_delta=float((values[first] - values[second]).mean()),
                    wins=int((values[first] < values[second]).sum()),
                    ties=int((values[first] == values[second]).sum()),
                    losses=int((values[first] > values[second]).sum()))
                    for first, second in [("native4", "off"), ("native4", "random4"),
                                          ("all16", "native4")]})
        results[label] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                              state_digest=before, network_state_unchanged=True,
                              native_tail_bitwise_equal=True, summary=summary, records=records)
        save("report.json", dict(launch=launch, models=results, seconds=time.time() - started))
        print("MODEL_COMPLETE " + json.dumps(dict(model=label, summary=summary)), flush=True)
        del model
    save("completed.json", dict(report_sha256=sha(OUTPUT / "report.json"),
                                models=2, samples_per_model=128, states=3,
                                native_tail_bitwise_equal=True, network_state_unchanged=True,
                                seconds=time.time() - started))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
