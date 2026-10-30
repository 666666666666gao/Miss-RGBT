import datetime
import json
import math
from pathlib import Path

root = Path("/home/gaob/Miss-RGBT")
training = root / "logs/server28_a_full20_20261030"
evaluation = root / "logs/server28_a_evaluation_20261030"
launch = json.loads((training / "launch.json").read_text())
now = datetime.datetime.now(datetime.timezone.utc)
report = {"observed_utc": now.isoformat(), "models": {}}
for label, config in launch["configs"].items():
    receipt = json.loads((training / (label + "_launch.json")).read_text())
    process = Path("/proc") / str(receipt["pid"])
    command = process.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "miss_rgbt.train" in command and ("a_" + label + "_full20.json") in command
    rows = [json.loads(line) for line in (Path(config["output"]) / "steps.jsonl").read_text().splitlines()]
    assert all(math.isfinite(row[key]) for row in rows
               for key in ("loss", "task_c", "task_m", "update_loss_mean"))
    recent = rows[-100:]
    elapsed = recent[-1]["seconds"] - recent[0]["seconds"]
    seconds_per_update = elapsed / (recent[-1]["update"] - recent[0]["update"])
    remaining = (50000 - rows[-1]["update"]) * seconds_per_update
    report["models"][label] = {
        "pid": receipt["pid"],
        "live_command": command,
        "recorded_updates": len(rows),
        "last_update": rows[-1]["update"],
        "epoch": rows[-1]["epoch"],
        "all_recorded_loss_fields_finite": True,
        "recent_updates": len(recent),
        "recent_seconds_per_update": seconds_per_update,
        "estimated_remaining_hours": remaining / 3600,
        "estimated_completion_utc": (now + datetime.timedelta(seconds=remaining)).isoformat(),
        "last_update_mean_loss": rows[-1]["update_loss_mean"],
        "peak_gpu_bytes": rows[-1]["peak_gpu_bytes"],
    }
queue_launch = json.loads((evaluation / "launch.json").read_text())
queue_pid = queue_launch["pid"]
queue_command = (Path("/proc") / str(queue_pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
assert "evaluate_after_training28.py" in queue_command
report["evaluation"] = dict(
    pid=queue_pid, live_command=queue_command,
    status=json.loads((evaluation / "status.json").read_text()),
)
path = root / "reports/server28-training-progress-20261030.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
