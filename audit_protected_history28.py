"""Paired frozen-baseline diagnostic on pre-existing development tuples."""
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
OUTPUT = ROOT / "reports/server28-protected-history-20261031"
PILOT = ROOT / "reports/server28-protected-sixepoch-20261030"
sys.path.insert(0, str(RUNTIME))
from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from probe_protected_sixepoch28 import digest
from evaluate_val97_recovery28 import sha


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    started = time.time()
    pilot = json.loads((PILOT / "report.json").read_text())
    checkpoint = Path(pilot["retained"])
    assert sha(checkpoint) == pilot["retained_sha256"]
    assert sha(Path(pilot["launch"]["checkpoint"])) == pilot["launch"]["checkpoint_sha256"]
    for path, expected in pilot["launch"]["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    model = load_pretrained(cfg, pilot["launch"]["checkpoint"], repair=True,
                            width=128, repair_layer=6).cpu().eval()
    model.requires_grad_(False)
    baseline_sha = digest(model.baseline.state_dict())
    data_root = "/data/wangwj/dataset/LasHeR/trainingset"
    train = make_loader(cfg, data_root, "train", 2, 64, 0)
    names = train.dataset.datasets[0].sequence_list[::5]
    loader = make_loader(cfg, data_root, "val", 2, 16, 0)
    loader.dataset.datasets[0].sequence_list = names
    random.seed(27182)
    np.random.seed(27182)
    torch.manual_seed(27182)
    batches = list(loader)[:2]
    hashes = [digest({key: batch[key] for key in [
        "template_images", "search_images", "search_anno"]}) for batch in batches]
    original = json.loads((PILOT / "inputs.json").read_text())
    assert hashes == original["development"][:2]
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        source_sha256={str(Path(__file__)): sha(Path(__file__)),
                       **pilot["launch"]["source_sha256"]}, checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), input_sha256=hashes,
        samples=4, search_index=0, cpu_threads=4,
        scope="Reused pilot development tuples; no training, tuning or benchmark claim",
        intervention="Initial template complete; dynamic template/search independently 11/01/10")
    (OUTPUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print("LAUNCH " + json.dumps(launch), flush=True)
    states = [("11", [1, 1]), ("01", [0, 1]), ("10", [1, 0])]
    records = {}
    complete_hashes = {}
    with torch.no_grad():
        for arm in ["zero", "selected"]:
            if arm == "selected":
                model.repair.load_state_dict(
                    torch.load(str(checkpoint), map_location="cpu")["repair"], strict=True)
            records[arm] = {}
            for dynamic_label, dynamic_flags in states:
                for search_label, search_flags in states:
                    key = dynamic_label + "_" + search_label
                    losses = []
                    outputs_hash = hashlib.sha256()
                    for batch in batches:
                        templates = list(batch["template_images"])
                        assert len(templates) == 2
                        searches = [batch["search_images"][0]]
                        size = searches[0].shape[0]
                        availability = [torch.ones(size, 2)]
                        availability += [
                            torch.tensor([flags], dtype=torch.float32).expand(size, -1)
                            for flags in [dynamic_flags, search_flags]]
                        outputs = model(templates, searches, availability)
                        loss = task_loss(outputs, batch["search_anno"][:1], cfg)
                        assert torch.isfinite(loss).all()
                        losses.extend(loss.reshape(-1).tolist())
                        for out in outputs:
                            for field in ["score_map", "size_map", "offset_map", "pred_boxes"]:
                                value = out[field]
                                assert torch.isfinite(value).all()
                                outputs_hash.update(value.contiguous().numpy().tobytes())
                    records[arm][key] = dict(
                        mean_task_loss=sum(losses) / len(losses), losses=losses,
                        outputs_sha256=outputs_hash.hexdigest())
                    if key == "11_11":
                        complete_hashes[arm] = outputs_hash.hexdigest()
                    print("CELL " + json.dumps(dict(
                        arm=arm, state=key, result=records[arm][key],
                        seconds=time.time() - started)), flush=True)
                    (OUTPUT / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
    assert complete_hashes["zero"] == complete_hashes["selected"]
    assert digest(model.baseline.state_dict()) == baseline_sha
    report = dict(
        launch=launch, results=records,
        loss_delta={key: records["selected"][key]["mean_task_loss"] -
                    records["zero"][key]["mean_task_loss"] for key in records["zero"]},
        baseline_state_unchanged=True, complete_history_outputs_bitwise_equal=True,
        seconds=time.time() - started)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "completed.json").write_text(json.dumps(dict(
        cells=18, samples_per_cell=4, seconds=time.time() - started), indent=2) + "\n")
    print("COMPLETE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
