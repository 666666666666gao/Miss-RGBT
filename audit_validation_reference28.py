"""CPU reference task loss on the candidate campaign's exact cached input set."""
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
OUTPUT = ROOT / "reports/server28-validation-reference-20261030"
sys.path.insert(0, str(RUNTIME))
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.data import make_loader
from miss_rgbt.model import MissingTrackerModel, load_pretrained
from miss_rgbt.objectives import task_loss
from evaluate_val97_recovery28 import sha


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
    batches = list(make_loader(cfg, config["dataset"], "val", 4, config["validation_samples"], 0))
    digest = hashlib.sha256()
    for batch in batches:
        for key in ["template_images", "search_images", "search_anno"]:
            digest.update(batch[key].numpy().tobytes())
    assert digest.hexdigest() == expected["inputs_sha256"], (digest.hexdigest(), expected)
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        source_sha256=sha(__file__), device="cpu", threads=4,
        inputs_sha256=digest.hexdigest(), validation_seed=seed,
        protocol="Same128 cached candidate validation tuples, batch4, full templates, "
                 "all search frames11/01/10, no patch mask, mean native task loss. "
                 "E0 availability-aware wrapper matches candidate initialization path. "
                 "Historical A utility20 loaded strictly; both eval/no parameter updates.",
        scope="Development reference only, not PR/SR, tracker trajectories or test selection. "
              "CPU versus existing GPU loss is not assumed bitwise equal. Historical E0 "
              "val97 exposure remains uncertified. No automatic training/stopping changes.",
    )
    (OUTPUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    references = [
        ("E0", Path(config["checkpoint"])),
        ("historical_A_utility20", ROOT / "artifacts/server25-data2/checkpoints/a_utility_full20_seed2027/epoch20.pth.tar"),
    ]
    started, results = time.time(), {}
    for label, checkpoint in references:
        if label == "E0":
            model = load_pretrained(cfg, str(checkpoint), repair=False, width=128, repair_layer=6)
        else:
            saved = torch.load(str(checkpoint), map_location="cpu")
            model_config = saved["config"]
            model = MissingTrackerModel(build_cadtrack(cfg, training=False),
                                        repair=model_config["repair"], width=model_config["repair_width"],
                                        repair_layer=model_config["repair_layer"],
                                        task_utility=model_config["task_utility"])
            model.load_state_dict(saved["model"], strict=True)
            del saved
        model.cpu().eval().requires_grad_(False)
        if model.utility_repair is not None:
            model.utility_repair.reset_selection()
        totals, count, receipts = {state: 0. for state in ["11", "01", "10"]}, 0, []
        with torch.no_grad():
            for index, batch in enumerate(batches):
                templates = list(batch["template_images"])
                searches = list(batch["search_images"])
                size = searches[0].shape[0]
                record = dict(batch=index, samples=size, loss={})
                for state, flags in [("11", [1., 1.]), ("01", [0., 1.]), ("10", [1., 0.])]:
                    availability = [torch.ones(size, 2) for _ in templates]
                    availability += [torch.tensor([flags]).expand(size, -1) for _ in searches]
                    loss = task_loss(model(templates, searches, availability), batch["search_anno"], cfg)
                    assert torch.isfinite(loss).all()
                    record["loss"][state] = float(loss.mean())
                    totals[state] += float(loss.mean()) * size
                count += size
                receipts.append(record)
                if (index + 1) % 4 == 0:
                    print("PROGRESS " + json.dumps(dict(model=label, samples=count,
                                                       seconds=time.time() - started)), flush=True)
        assert count == 128
        results[label] = dict(samples=count, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                              task_loss={state: value / count for state, value in totals.items()},
                              mean_task_loss=sum(totals.values()) / (3 * count), batches=receipts,
                              seconds=time.time() - started)
        (OUTPUT / "report.json").write_text(json.dumps(dict(launch=launch, references=results),
                                                      indent=2) + "\n")
        print("REFERENCE " + json.dumps(dict(model=label, mean_task_loss=results[label]["mean_task_loss"],
                                            task_loss=results[label]["task_loss"])), flush=True)
        del model
    (OUTPUT / "completed.json").write_text(json.dumps(
        dict(references=2, samples_per_reference=128, seconds=time.time() - started), indent=2) + "\n")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
