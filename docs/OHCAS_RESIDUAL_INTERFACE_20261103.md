# OHCAS Residual Interface Contract

`research/history_stream/residual.py` defines an isolated feature-level
composition:

```text
H_out = H_native + (1 - availability) * Delta(H_native, H_history, log(1 + age))
```

The final projection is zero-initialized, so the complete adapter initially
returns the native tensor exactly, including the positional and block-transformed
content of a missing stream. After learning, the availability gate still
prevents this branch from changing a present stream. The caller supplies one
modality's native/history tensors and a binary availability column; the module
does not read the other modality or implement history retrieval.

`research/history_stream/check_residual.py` checks exact initialization parity,
the availability gate after making the residual nonzero, sensitivity to history
and age, and a finite nonzero gradient to the output projection. Run with the
project's Python 3.7 / PyTorch 1.12 environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
  conda-envs/miss_rgbt/bin/python -m research.history_stream.check_residual
```

This contract test does not integrate the adapter into `MissingTrackerModel`,
establish native image parity, validate a fresh history cache, or show tracking
improvement. OHCAS remains blocked on complete historical R frame provenance
and the registered M1 data/diagnostic gates. No active snapshot, training job,
official queue, checkpoint, or selection policy is changed by this prototype.

The exact local source text was executed in Server 28's Python 3.7 / PyTorch
1.12 environment with CUDA hidden. Result: `PASS`; `cuda_initialized=false`.
Source SHA256:

- `residual.py`: `7e4178c62f371380e970b63a7ca23abaee7fe13f51cf13aae4f33d38d5d25bcc`
- `check_residual.py`: `366684a19f7361bdcbf84a92b6efc156460ba08710b8b1bf80f9cf81f538de57`

This was an in-memory CPU contract check. The new files were not copied into
the live project directory or any frozen runtime.
