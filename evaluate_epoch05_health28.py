"""Matched full-val97 tracking health check alongside existing training."""
import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from evaluate_candidate_val97_28 import initialize, sequence
from evaluate_val97_recovery28 import sha

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
OUTPUT = ROOT / "reports/server28-epoch05-tracking-health-20261031"
PROTOCOL = ROOT / "development/recovery_val97_20261030/protocol.json"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def evaluate(gpu, cases, sources):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    protocol = json.loads(PROTOCOL.read_text())
    assert len(protocol["sequences"]) == 97
    assert sum(item["frames"] for item in protocol["sequences"]) == 54646
    # Longest sequences first expose realistic running time before the short tail.
    cohort = sorted(protocol["sequences"], key=lambda item: item["frames"], reverse=True)
    import torch
    for label, checkpoint, conditions in cases:
        checkpoint = Path(checkpoint)
        checkpoint_sha = sha(checkpoint)
        if label == "E0":
            metadata = json.loads(checkpoint.with_suffix(".json").read_text())
            assert checkpoint_sha == metadata["packaged_sha256"]
            assert metadata["tensor_equality"] and metadata["optimizer_updates"] == 0
        for path, expected in sources.items():
            assert sha(path) == expected, path
        saved = torch.load(str(checkpoint), map_location="cpu")
        assert saved["epoch"] == (20 if label == "E0" else 5)
        if label != "E0":
            assert saved["updates"] == 10000
            assert saved["config"]["task_utility"]["selection"] == label
        del saved
        initialize(str(checkpoint), str(gpu))
        for condition in conditions:
            started = time.time()
            destination = OUTPUT / (label + "_" + condition)
            destination.mkdir(exist_ok=False)
            spec = protocol["conditions"][condition]
            assert sha(spec["path"]) == spec["sha256"]
            masks = json.loads(Path(spec["path"]).read_text())
            save(destination / "launch.json", dict(
                pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
                physical_gpu=gpu, checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_sha, condition=condition,
                source_sha256=sources, protocol_sha256=sha(PROTOCOL),
                mask_sha256=spec["sha256"], sequences=97, frames=54646,
                scope="Epoch5 training health; full val97 development, not final selection or official test",
                metric_protocol="Macro sequence PR20 and 21-threshold SR; initialization and "
                                "nonpositive-size GT excluded; both-missing holds scored"))
            receipts = []
            for item in cohort:
                result = sequence((item, protocol["data_root"],
                                   masks[item["name"]]["data"], str(destination)))
                receipts.append(result)
                save(destination / "receipts.json", receipts)
                print("SEQUENCE " + json.dumps(dict(
                    label=label, condition=condition, physical_gpu=gpu,
                    completed=len(receipts), seconds=time.time() - started,
                    receipt=result)), flush=True)
            assert len(receipts) == 97
            assert sum(item["video_frames"] for item in receipts) == 54646
            assert sha(checkpoint) == checkpoint_sha
            metrics = dict(
                label=label, condition=condition, sequences=97, frames=54646,
                scored_frames=sum(item["scored_frames"] for item in receipts),
                PR20=float(np.mean([item["PR20"] for item in receipts])),
                SR=float(np.mean([item["SR"] for item in receipts])),
                seconds=time.time() - started, checkpoint_sha256=checkpoint_sha)
            save(destination / "completed.json", metrics)
            print("COMPLETE " + json.dumps(metrics), flush=True)
    save(OUTPUT / ("gpu%d_completed.json" % gpu), dict(pid=os.getpid(), physical_gpu=gpu))


def main():
    OUTPUT.mkdir(exist_ok=False)
    memory = subprocess.check_output([
        "nvidia-smi", "--id=0,1", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    rows = [row.split(",") for row in memory.splitlines()]
    assert len(rows) == 2 and all(int(row[1]) >= 3000 for row in rows), memory
    sources = {str(path): sha(path) for folder in ["miss_rgbt", "lib"]
               for path in sorted((RUNTIME / folder).rglob("*.py"))}
    sources.update({str(ROOT / name): sha(ROOT / name) for name in [
        "evaluate_epoch05_health28.py", "evaluate_candidate_val97_28.py",
        "evaluate_val97_recovery28.py"]})
    baseline = str(ROOT / "checkpoints/e0_availability/epoch20.pth.tar")
    conditions = ["complete", "alternating_outage"]
    cases = {
        0: [("E0", baseline, ["complete"]),
            ("utility", str(ROOT / "runs/server28/batch10_utility_seed2027/epoch05.pth.tar"),
             conditions)],
        1: [("E0", baseline, ["alternating_outage"]),
            ("random", str(ROOT / "runs/server28/batch10_random_seed2027/epoch05.pth.tar"),
             conditions)]}
    context = mp.get_context("spawn")
    processes = {}
    for gpu, jobs in cases.items():
        process = context.Process(target=evaluate, args=(gpu, jobs, sources))
        process.start()
        processes[gpu] = process
    save(OUTPUT / "launch.json", dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        children={str(gpu): process.pid for gpu, process in processes.items()},
        cases=cases, memory_free_at_launch=memory, source_sha256=sources,
        purpose="Matched E0/epoch5 full-val97 health check; no checkpoint selection",
        poll_seconds=300))
    for gpu, process in processes.items():
        process.join()
        assert process.exitcode == 0, (gpu, process.exitcode)
    results = [json.loads((OUTPUT / (label + "_" + condition) / "completed.json").read_text())
               for gpu, jobs in cases.items()
               for label, _, selected_conditions in jobs for condition in selected_conditions]
    assert len(results) == 6
    save(OUTPUT / "completed.json", dict(
        results=results, official_test_used=False, checkpoint_selection_used=False,
        ended_utc=datetime.datetime.utcnow().isoformat() + "Z"))


if __name__ == "__main__":
    main()
