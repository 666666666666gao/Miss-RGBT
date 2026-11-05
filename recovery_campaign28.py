"""Continue a verified reboot checkpoint and evaluate the last five weights."""
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
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", choices=["0", "1", "2", "3"], required=True)
    args = parser.parse_args()
    root = Path("/home/gaob/Miss-RGBT")
    runtime = root / "runtime/task_utility_20261018"
    label = args.label
    audited = json.loads((root / "reports/server28-reboot-checkpoint-audit-20261031.json").read_text())
    record = audited["records"][label]
    assert sha(record["checkpoint"]) == record["sha256"]
    campaign = root / "reports/server28-recovery-20261031" / label
    campaign.mkdir(parents=True, exist_ok=False)
    config = dict(record["config"])
    config.update(validation_seed=1701, validation_samples=128,
                  early_stop_min_epochs=12 if label.startswith("batch10_") else 21,
                  early_stop_patience=3, early_stop_relative_delta=.002,
                  output=str(root / "runs/server28-recovery-20261031" / label))
    config_path = campaign / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-u", str(root / "resume_training28.py"),
               "--config", str(config_path), "--resume", record["checkpoint"]]
    with (campaign / "training.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=str(runtime), env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        (campaign / "launch.json").write_text(json.dumps(dict(
            controller_pid=os.getpid(), training_pid=process.pid, physical_gpu=args.gpu,
            command=command, effective_batch=config["microbatch"] * config["accumulation"],
            learning_rate=config["learning_rate"], resumed_checkpoint=record,
            boot_id=audited["boot_id"],
            maximum_epochs=20, minimum_early_stop_epoch=config["early_stop_min_epochs"],
            selection_rule="Highest mean complete/alternating-outage macro validation SR "
                           "among final5 epochs; ties choose earlier epoch",
            source_sha256={str(root / name): sha(root / name) for name in [
                "resume_training28.py", "candidate_validation28.py",
                "evaluate_recovery_val97_28.py", "recovery_campaign28.py",
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
            command = [sys.executable, "-u", str(root / "evaluate_recovery_val97_28.py"),
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
        destination = root / "results/server28_recovery_20261031" / (label + "_" + condition)
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
