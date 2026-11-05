"""Run historical A Miss scoring only after the concurrent GPU smoke passes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/home/gaob/Miss-RGBT")
PYTHON = ROOT / "conda-envs/miss_rgbt/bin/python"
CHECKPOINT = ROOT / "artifacts/server25-data2/checkpoints/a_utility_full20_seed2027/epoch20.pth.tar"
SMOKE = ROOT / "reports/server28-historical-a-smoke-20261030"
SMOKE_PID = 4085844
CONTROL = ROOT / "reports/server28-historical-a-miss-control-v2-20261030"
OUTPUT = ROOT / "results/server28_historical_a_20261030/miss_single_worker_per_gpu"
LOG = ROOT / "reports/server28-historical-a-miss-20261030.log"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    CONTROL.mkdir(exist_ok=False)
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        smoke_pid=SMOKE_PID, smoke_output=str(SMOKE),
        physical_gpus=[0, 1], workers_per_gpu=1, condition="miss",
        checkpoint=str(CHECKPOINT),
        protocol="Wait for the existing two-sequence official evaluator smoke. "
                 "Proceed only after both sequence receipts, exact smoke completion, "
                 "and both GPU assignments are verified. Never auto-retry.",
        poll_seconds=300)
    (CONTROL / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    while (Path("/proc") / str(SMOKE_PID)).exists():
        command = ((Path("/proc") / str(SMOKE_PID) / "cmdline")
                   .read_bytes().replace(b"\0", b" ").decode())
        assert "evaluate_candidate_official28.py" in command
        assert "--limit 2" in command and str(SMOKE) in command
        time.sleep(300)
    smoke = json.loads((SMOKE / "smoke_completed.json").read_text())
    receipts = [
        json.loads(path.read_text())
        for path in sorted((SMOKE / "receipts").glob("*/*.json"))
    ]
    assert smoke["sequences"] == len(receipts) == 2
    assert smoke["physical_gpus"] == [0, 1] and smoke["workers"] == 2
    assert {item["physical_gpu"] for item in receipts} == {0, 1}
    assert smoke["frames"] == sum(item["frames"] for item in receipts)
    (CONTROL / "smoke_verified.json").write_text(
        json.dumps(dict(smoke=smoke, receipts=receipts), indent=2) + "\n")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", PYTHONPATH=str(
        ROOT / "runtime/task_utility_20261018"), OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", WANDB_MODE="disabled")
    command = [str(PYTHON), "-u", str(ROOT / "evaluate_candidate_official28.py"),
               "--checkpoint", str(CHECKPOINT), "--gpus", "0,1",
               "--workers-per-gpu", "1", "--condition", "miss",
               "--output", str(OUTPUT)]
    with LOG.open("wb") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (CONTROL / "evaluation_launch.json").write_text(json.dumps(dict(
        pid=process.pid, command=command, output=str(OUTPUT),
        checkpoint_sha256=sha(CHECKPOINT), workers_per_gpu=1,
        physical_gpus=[0, 1]), indent=2) + "\n")
    while process.poll() is None:
        time.sleep(300)
    assert process.returncode == 0, process.returncode
    completed = json.loads((OUTPUT / "completed.json").read_text())
    assert completed["sequences"] == 479 and completed["frames"] == 337353
    metrics = json.loads((OUTPUT / "metrics.json").read_text())
    assert set(metrics) == {"LasHeR-Miss", "RGBT234-Miss"}
    (CONTROL / "completed.json").write_text(json.dumps(
        dict(completed=completed, metrics=metrics), indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
