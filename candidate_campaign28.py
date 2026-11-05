"""Train one candidate, validate its last five epochs, retain the selected weight."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_val97_recovery28 import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", choices=["utility", "random"], required=True)
    parser.add_argument("--gpu", choices=["2", "3"], required=True)
    args = parser.parse_args()
    root = Path("/home/gaob/Miss-RGBT")
    runtime = root / "runtime/task_utility_20261018"
    label = "batch10_" + args.selection + "_seed2027"
    campaign = root / "reports/server28-candidates-20261030" / label
    campaign.mkdir(parents=True, exist_ok=False)
    config = json.loads((runtime / ("configs/a_%s_full20.json" % args.selection)).read_text())
    config.update(microbatch=5, accumulation=2, learning_rate=.000125, workers=2,
                  validation_seed=1701, validation_samples=128,
                  early_stop_min_epochs=12, early_stop_patience=3, early_stop_relative_delta=.002,
                  output=str(root / "runs/server28" / label))
    config_path = campaign / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-u", str(root / "train_candidates28.py"),
               "--config", str(config_path)]
    with (campaign / "training.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=str(runtime), env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        (campaign / "launch.json").write_text(json.dumps(dict(
            controller_pid=os.getpid(), training_pid=process.pid, physical_gpu=args.gpu,
            command=command, effective_batch=10, learning_rate=.000125,
            maximum_epochs=20, minimum_early_stop_epoch=12,
            selection_rule="Highest mean complete/alternating-outage macro validation SR "
                           "among final5 epochs; ties choose earlier epoch",
            source_sha256={str(root / name): sha(root / name) for name in [
                "train_candidates28.py", "candidate_validation28.py",
                "evaluate_candidate_val97_28.py", "candidate_campaign28.py",
                "evaluate_candidate_official28.py"]},
        ), indent=2) + "\n")
        while process.poll() is None:
            time.sleep(300)
        assert process.returncode == 0, process.returncode
    run = Path(config["output"])
    completed = json.loads((run / "completed.json").read_text())
    epochs = completed["final_five_epochs"]
    assert len(epochs) == 5
    records = []
    for epoch in epochs:
        checkpoint = run / ("epoch%02d.pth.tar" % epoch)
        record = dict(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                      conditions={})
        for condition in ["complete", "alternating_outage"]:
            destination = campaign / ("epoch%02d_%s" % (epoch, condition))
            command = [sys.executable, "-u", str(root / "evaluate_candidate_val97_28.py"),
                       "--checkpoint", str(checkpoint), "--gpu", args.gpu,
                       "--condition", condition, "--output", str(destination)]
            with (campaign / ("epoch%02d_%s.log" % (epoch, condition))).open("wb") as log:
                evaluation = subprocess.Popen(command, cwd=str(runtime), env=env,
                                              stdout=log, stderr=subprocess.STDOUT)
                while evaluation.poll() is None:
                    time.sleep(300)
                assert evaluation.returncode == 0, (epoch, condition, evaluation.returncode)
            metrics = json.loads((destination / "completed.json").read_text())
            assert metrics["sequences"] == 97 and metrics["frames"] == 54646
            assert metrics["checkpoint_sha256"] == record["checkpoint_sha256"]
            record["conditions"][condition] = metrics
        record["selection_score"] = sum(
            metrics["SR"] for metrics in record["conditions"].values()) / 2
        records.append(record)
        (campaign / "last_five_metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    best = max(records, key=lambda record: (record["selection_score"], -record["epoch"]))
    retained = Path(best["checkpoint"])
    assert sha(retained) == best["checkpoint_sha256"]
    cleanup = []
    for record in records:
        path = Path(record["checkpoint"])
        assert path.parent == run
        if path != retained:
            cleanup.append(dict(path=str(path), bytes=path.stat().st_size,
                                sha256=record["checkpoint_sha256"]))
    receipt = dict(best=best, cleanup=cleanup, official_test_selection_used=False)
    (campaign / "selected.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for item in cleanup:
        Path(item["path"]).unlink()
    receipt["cleanup_completed"] = True
    (campaign / "completed.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for condition in ["complete", "miss"]:
        destination = root / "results/server28_candidates_20261030" / (label + "_" + condition)
        command = [sys.executable, "-u", str(root / "evaluate_candidate_official28.py"),
                   "--checkpoint", str(retained), "--gpus", args.gpu,
                   "--condition", condition, "--output", str(destination)]
        with (campaign / ("official_" + condition + ".log")).open("wb") as log:
            evaluation = subprocess.Popen(command, cwd=str(runtime), env=env, stdout=log,
                                          stderr=subprocess.STDOUT)
            while evaluation.poll() is None:
                time.sleep(300)
            assert evaluation.returncode == 0, (condition, evaluation.returncode)
        receipt["official_" + condition] = json.loads((destination / "metrics.json").read_text())
    (campaign / "official_completed.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
