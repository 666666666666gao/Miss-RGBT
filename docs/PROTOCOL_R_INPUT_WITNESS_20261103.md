# R Input Prefix Provenance Witness

## Measured Evidence

The existing server Python 3.7 / Torch 1.12 environment executed
`scripts/replay_protocol_r_input_witness.py` successfully. It replayed four
native image-loader microbatches (16 returned tuples) on CPU in 2.146681
seconds, without initializing CUDA.

SHA256 of `template_images`, `search_images`, and `search_anno` exactly
matched every saved first-optimizer-update input hash: two microbatches for
R1 and four for R2. R1/R2 share microbatch size four, seed, dataset, workers,
and samples per epoch, but accumulate two/four microbatches respectively.

The replay used actual images, native sampling, crop rejection, transforms,
collation, and worker seeding. `FrameCapture` adds frame-identity metadata
after native processing without changing images, annotations, or random
draws. Rejected crop attempts clear their identities. Worker prefetch may
read additional images beyond the returned prefix.

The report binds both saved run configurations and the imported native
data, processing, sampler, transform, and LasHeR modules by SHA256. The
script checked those five modules against the frozen R source manifest.

| Artifact | SHA256 |
|---|---|
| `scripts/replay_protocol_r_input_witness.py` | `3d53e29f42fe512952a4560daf9d018008eff3cbf39b346df708f2849aebef09` |
| `reports/server28_protocol_r_input_witness_20261103.json` | `5f06801ef5e25f3f59410f7d7849b461589ca81044c45e0259a2aa6c1c5fcd42` |

Local and server copies of both artifacts have matching hashes.

## Limits

This establishes reproducibility and frame identities for the recorded
prefix only. It is not a full-run input manifest, proof of RNG continuity
across epochs, an OHCAS freshness-gate pass, training evidence, or an
official score. It does not repair the full historical R provenance gap.

Before accepting a complete reconstruction, audit main-process CPU RNG
through optimizer setup, training, validation, checkpointing, and epoch
transitions. Distinguish already-consumed inputs from future planned inputs.
The recorded hashes verify only the first update and cannot independently
certify later replayed batches. A metadata-only sampler omitting crop
rejection and augmentation RNG draws is insufficient.

The growing original A manifests also need completed-run verification.
Missing provenance is not a negative diagnostic and cannot close OHCAS.
No running source, seed, process, or environment was modified by this replay.

## Executed Invocation

```bash
cd /home/gaob/Miss-RGBT/runtime/protocol_r_20261103
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/home/gaob/Miss-RGBT/runtime/protocol_r_20261103:/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python -u \
/home/gaob/Miss-RGBT/scripts/replay_protocol_r_input_witness.py \
--output /home/gaob/Miss-RGBT/reports/server28_protocol_r_input_witness_20261103.json
```

The exclusive output path already exists; use a new path for a new observation.

## Live Scheduling Observation

At server time 2026-11-03 11:32:53 CST:

| GPU | Original training process | Utilization | Temperature | Power cap |
|---|---|---|---|---|
| 0 | R1, PID 3388846 | 100% | 82 C | 250 W |
| 1 | R2, PID 3395457 | 100% | 79 C | 250 W |
| 2 | A Utility, PID 1288280 | 100% | 77 C | 250 W |
| 3 | A Random, PID 1288281 | 100% | 74 C | 250 W |

R1/R2 were in epoch two at cumulative microsteps 8050/7700. Each epoch has
5000 microbatches. The estimated epoch-two boundary is 12:10--12:20 CST,
not the end of the twenty-epoch campaigns.

A official controller PID 1355742 and six-arm RPPM queue PID 3539938 were
live. RPPM status at 03:30:21 UTC was `WAITING_FOR_R`, awaiting both R
training verifications and both full official campaign receipts. Its
dispatcher checks every 300 seconds and uses free GPUs after the gates pass.
Each formal arm holds its GPU through ten full official evaluations.
Do not duplicate the queue or edit frozen `curriculum_p` / `protocol_r` files.

There is no new official score. Observed temperatures do not establish an
85 C hardware temperature limit.

## Completed Real-Image Epoch Transition Witness

`scripts/replay_protocol_r_epoch_boundary.py` exhausted one native epoch:
5,000 microbatches, 20,000 returned tuples, real images/augmentations/crop
rejections and two native workers. Every saved original first-update hash
matched. The main CPU RNG stayed equal to its post-prefetch state for all
5,000 returned batches, including the exhausted iterator.

It then compared two ways of reaching epoch two: consuming the complete
native image epoch, and advancing the installed loader's worker-base seed
plus its native batch sampler without reading dataset items. The next four
real-image microbatches had identical tensor hashes and frame identities.
Measured worker-base seeds were 5562934040544554675 and
5177384581399083067. Total execution took 1459.473608 seconds, CPU only.

The installed Torch1.12.1 source inspection found a main-process worker seed
and a RandomSampler seed per epoch; RandomSampler uses a local generator
after that. Native DropPath uses `x.new_empty(...).bernoulli_`, on the input's
device. Original validation uses `torch.random.fork_rng`, which restores CPU
state as well. The frozen training source must still be audited as a whole:
this witness did not execute the original CUDA training operations or read
their live RNG state. Original later-batch hashes were never recorded.

Server evidence remains in
`reports/server28_protocol_r_epoch_boundary_20261103/`:
`epoch01_input_manifest.jsonl`, `report.json`, and `completed.json`.
The publication copy of report.json is
`reports/server28_protocol_r_epoch_boundary_20261103.json`, SHA256
`1eba4e1cf437d7fa19d2c5e49b6035b17d4aef474cd71d62cde1ad244f51ce8b`.
The epoch-one manifest SHA256 is
`8d3166d345d2207be7599a75004a3f5f9e208672fb8516bedad9531d5b9c45ee`.

This advances epoch-boundary reproducibility evidence. It does not turn
the replay into an independently certified full original R manifest, or
authorize OHCAS. The active trainer sources/processes were not modified.
