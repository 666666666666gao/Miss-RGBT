"""Verify actual candidate processes, shared validation inputs and finite updates."""
import datetime
import json
import math
from pathlib import Path

root = Path("/home/gaob/Miss-RGBT")
campaign = root / "reports/server28-candidates-20261030"
result = dict(observed_utc=datetime.datetime.utcnow().isoformat() + "Z", models={})
for selection in ["utility", "random"]:
    label = "batch10_" + selection + "_seed2027"
    launch = json.loads((campaign / label / "launch.json").read_text())
    run = root / "runs/server28" / label
    config = json.loads((run / "run_config.json").read_text())
    validation = json.loads((run / "validation_inputs.json").read_text())
    process = Path("/proc") / str(launch["training_pid"])
    command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "train_candidates28.py" in command
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    assert rows
    assert all(math.isfinite(row[key]) for row in rows
               for key in ["loss", "task_c", "task_m", "update_loss_mean"])
    recent = rows[-100:]
    speed = ((recent[-1]["seconds"] - recent[0]["seconds"]) /
             (recent[-1]["update"] - recent[0]["update"]))
    updates_per_epoch = config["samples_per_epoch"] // (config["microbatch"] * config["accumulation"])
    next_epoch_update = ((rows[-1]["update"] // updates_per_epoch) + 1) * updates_per_epoch
    next_epoch_seconds = (next_epoch_update - rows[-1]["update"]) * speed
    validation_path = run / "validation_history.json"
    result["models"][selection] = dict(
        launch=launch, validation=validation, updates=rows[-1]["update"],
        epoch=rows[-1]["epoch"], finite_recorded_losses=True,
        peak_allocated_bytes=rows[-1]["peak_gpu_bytes"],
        first_update_inputs=rows[0]["input_sha256_microbatches"],
        effective_batch=config["microbatch"] * config["accumulation"],
        first_update_seconds=rows[0]["seconds"],
        recent_seconds_per_update=speed,
        next_epoch_boundary_estimated_utc=(
            datetime.datetime.utcnow() + datetime.timedelta(seconds=next_epoch_seconds)).isoformat() + "Z",
        completed_epoch_validations=(json.loads(validation_path.read_text()) if validation_path.exists() else []))
assert (result["models"]["utility"]["validation"]["inputs_sha256"]
        == result["models"]["random"]["validation"]["inputs_sha256"])
assert (result["models"]["utility"]["first_update_inputs"]
        == result["models"]["random"]["first_update_inputs"])
result["paired_validation_and_first_update_hashes_equal"] = True
(campaign / "verified-launch.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
