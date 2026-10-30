"""Measure one real validation trajectory while GPU1 training stays live."""
import datetime
import json
from pathlib import Path
import time

from evaluate_candidate_val97_28 import initialize, sequence
from evaluate_val97_recovery28 import sha

ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-concurrent-evaluation-profile-20261030"
CHECKPOINT = ROOT / "artifacts/server25-data2/checkpoints/a_utility_full20_seed2027/epoch20.pth.tar"


def main():
    OUTPUT.mkdir(exist_ok=False)
    protocol_path = ROOT / "development/recovery_val97_20261030/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    spec = protocol["conditions"]["alternating_outage"]
    assert sha(spec["path"]) == spec["sha256"]
    masks = json.loads(Path(spec["path"]).read_text())
    item = protocol["sequences"][0]
    started = time.time()
    initialize(str(CHECKPOINT), "1")
    import torch
    receipt = sequence((item, protocol["data_root"], masks[item["name"]]["data"], str(OUTPUT)))
    torch.cuda.synchronize()
    result = dict(
        started_utc=datetime.datetime.utcfromtimestamp(started).isoformat() + "Z",
        seconds=time.time() - started, physical_gpu=1, receipt=receipt,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        checkpoint_sha256=sha(CHECKPOINT), source_sha256=sha(__file__),
        scope="Concurrent inference memory/throughput profile, first fixed val97 sequence. "
              "Not an official benchmark score or model-selection run.")
    (OUTPUT / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
