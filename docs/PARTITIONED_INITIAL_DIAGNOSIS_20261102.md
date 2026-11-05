# Initial Versus Epoch3 Task Loss

This diagnostic replays all32 fixed development batches (128 tuples) from
the running Utility replication. Every frame identity and augmented image/
annotation tensor hash equals the original manifest. The initialized model's
parameter hash equals the original initial_state.json. The epoch3 checkpoint
strictly loads and exactly reproduces the recorded three availability losses.
No optimizer update, new frame selection or official benchmark is involved.

| Search availability | Initial E0 + initialized Utility | Epoch3 | Difference |
|---|---:|---:|---:|
| Complete (11) | 3.252417 | 3.808033 | +0.555617 |
| RGB missing (01) | 5.764184 | 5.488136 | -0.276049 |
| TIR missing (10) | 3.976443 | 4.147695 | +0.171252 |
| Equal mean | 4.331015 | 4.481288 | +0.150273 |

Lower is better. These are task losses on the existing train882 internal
development tuples, not PR/NPR/SR/MPR/MSR or closed-loop tracking accuracy.
They are not val97 or a partial official benchmark.

The measured early behavior is uneven: RGB-missing task loss improves while
complete and TIR-missing task losses worsen. This contradicts an assumption
that current training already improves all availability states. It does not
prove convergence, a causal mechanism, overfitting or final benchmark harm.
Only epoch3 was replayed; the separately recorded epoch1-3 means increase.
The preregistered learning-rate drop and stopping policy remain unchanged.

For later P/F/X experiments, retain availability-separated diagnostics;
an averaged loss or RGB-missing improvement alone cannot substantiate the
complete-input goal. Recheck this same fixed-batch trend after the scheduled
learning-rate drop. Any later optimization change should be a separately
recorded experiment, not a silent intervention in the live replication.

Executed on GPU0 with250W cap; model evaluation took37.892seconds and peaked
at557701120 allocated bytes. The process exited0; no weights were written.
Epoch3 checkpoint SHA256:
9262d75051f55bc88ff304c18875addd5caec7d08d66ff4cfe471a95032da68e.

Raw report: reports/server28_partitioned_initial_diagnosis_20261102.json.
Entry: scripts/diagnose_partitioned_initial.py.
