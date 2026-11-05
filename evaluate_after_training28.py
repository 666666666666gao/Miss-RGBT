import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path("/home/gaob/Miss-RGBT")
runtime = root / "runtime/task_utility_20261018"
training = root / "logs/server28_a_full20_20261030"
queue = root / "logs/server28_a_evaluation_20261030"
python = root / "conda-envs/miss_rgbt/bin/python"


def save(name, value):
    (queue / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    queue.mkdir(parents=True, exist_ok=False)
    launch = json.loads((training / "launch.json").read_text())
    controller = launch["controller_pid"]
    save("launch.json", {
        "pid": os.getpid(),
        "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "training_controller_pid": controller,
        "poll_seconds": 300,
        "evaluation_epoch": 20,
        "conditions": ["complete", "miss"],
        "models": ["utility", "random"],
        "physical_gpus": [0, 1],
    })
    save("status.json", {"stage": "WAITING_FOR_TRAINING", "pid": os.getpid()})
    for label in ("utility", "random"):
        receipt = training / (label + "_exit.json")
        while not receipt.exists():
            assert Path("/proc/" + str(controller)).exists(), "Training controller missing"
            time.sleep(300)
        result = json.loads(receipt.read_text())
        assert result["exit_code"] == 0, result
    configs = launch["configs"]
    for label in ("utility", "random"):
        folder = Path(configs[label]["output"])
        completed = json.loads((folder / "completed.json").read_text())
        assert completed["epochs"] == 20 and completed["updates"] == 50000, completed
    expected_source = json.loads((training / "source_sha256.json").read_text())
    for name, digest in expected_source.items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == digest, name
    rows = subprocess.check_output(
        ["nvidia-smi", "--id=0,1", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        universal_newlines=True,
    ).splitlines()
    assert len(rows) == 2 and all(int(row.split(",")[1]) < 500 for row in rows), rows
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", PYTHONPATH=str(runtime),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               WANDB_MODE="disabled")
    metrics = {}
    for label in ("utility", "random"):
        checkpoint = Path(configs[label]["output"]) / "epoch20.pth.tar"
        metrics[label] = {}
        for condition in ("complete", "miss"):
            name = label + "_" + condition
            output = root / "results/server28_20261030" / name
            command = [
                str(python), "-u", "scripts/evaluate_model.py",
                "--checkpoint", str(checkpoint), "--output", str(output),
                "--condition", condition, "--gpus", "0,1",
            ]
            with (queue / (name + ".log")).open("w") as log:
                process = subprocess.Popen(command, cwd=str(runtime), env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
            save(name + "_launch.json", {
                "pid": process.pid, "command": command,
                "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
            })
            save("status.json", {"stage": name.upper() + "_RUNNING",
                                 "pid": os.getpid(), "evaluation_pid": process.pid})
            code = process.wait()
            save(name + "_exit.json", {
                "pid": process.pid, "exit_code": code,
                "ended_utc": datetime.datetime.utcnow().isoformat() + "Z",
            })
            assert code == 0, "Evaluation failed: " + name
            completed = json.loads((output / "completed.json").read_text())
            assert completed["sequences"] == 479 and completed["frames"] == 337353
            metrics[label][condition] = json.loads((output / "metrics.json").read_text())
            save("metrics.json", metrics)
    save("status.json", {"stage": "EVALUATION_COMPLETE", "pid": os.getpid(),
                         "ended_utc": datetime.datetime.utcnow().isoformat() + "Z"})


if __name__ == "__main__":
    main()
