"""Exercise metric-verification helpers on completed development/official data."""
import datetime
import json
from pathlib import Path

from evaluate_val97_recovery28 import sha
from verify_recovery_campaign28 import verify_development, verify_official


root = Path("/home/gaob/Miss-RGBT")
development = root / "reports/server28-epoch05-tracking-health-20261031/utility_complete"
official = root / "results/server28_historical_a_20261030/miss_single_worker_per_gpu"
protocol = root / "development/recovery_val97_20261030/protocol.json"
development_metrics = verify_development(
    development, protocol, "cb7933450daaf2b14fb787728f73315d52404a168a1396e68bddc3ff46a5ea10")
official_metrics = verify_official(
    official, "c4aabefb495344b6d5e6e17655ac20fd51c4e2dec575843133096b319e047db3",
    "miss", json.loads((official / "metrics.json").read_text()))
result = dict(
    observed_utc=datetime.datetime.utcnow().isoformat() + "Z",
    verifier_sha256=sha(root / "verify_recovery_campaign28.py"), witness_sha256=sha(__file__),
    development=dict(path=str(development), trajectories_verified=97, frames=54646,
                     metrics=development_metrics),
    official=dict(path=str(official), trajectories_verified=479, frames=337353,
                  metrics=official_metrics),
    scope="Actual full existing trajectory/hash/metric/curve recalculations, no neural calls. "
          "Exercises shared helpers, not an end-to-end final-five selection/cleanup witness. "
          "Current recovered campaigns have not completed their official evaluations.")
with (root / "reports/server28-recovery-verifier-witness-20261031.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result, indent=2), flush=True)
