"""Read-only checkpoint validation after actual repeated server reboots."""
import datetime
import hashlib
import json
import os
from pathlib import Path

import torch


root = Path("/home/gaob/Miss-RGBT")
records = {}
for label, epoch in [("a_utility_full20_seed2027", 10),
                     ("a_random_full20_seed2027", 10),
                     ("batch10_utility_seed2027", 9),
                     ("batch10_random_seed2027", 10)]:
    path = root / "runs/server28" / label / ("epoch%02d.pth.tar" % epoch)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    checkpoint = torch.load(str(path), map_location="cpu")
    assert checkpoint["epoch"] == epoch and checkpoint["optimizer"]["state"]
    assert all(torch.isfinite(value).all().item()
               for value in checkpoint["model"].values() if value.is_floating_point())
    records[label] = dict(
        checkpoint=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size,
        epoch=epoch, updates=checkpoint["updates"], keys=list(checkpoint),
        optimizer_lrs=[group["lr"] for group in checkpoint["optimizer"]["param_groups"]],
        config=checkpoint["config"])
    del checkpoint
report = dict(
    observed_utc=datetime.datetime.utcnow().isoformat() + "Z",
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    uptime=Path("/proc/uptime").read_text().strip(), records=records,
    scope="Actual CPU checkpoint deserialization, finite model tensors and optimizer state. "
          "No RNG/scheduler saved; continuation cannot claim bitwise equivalence.")
path = root / "reports/server28-reboot-checkpoint-audit-20261031.json"
assert not path.exists() or path.stat().st_size == 0
with path.open("w") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(report, indent=2), flush=True)
