# R Epoch-One Candidate Replay

## Launch and Scope

At 2026-11-03 11:44:34 CST, the isolated CPU replay was launched as PID
3914039 using the existing Python 3.7 environment. At 11:44:47 its parent
was PID 1, its nice level was 10, and registration plus a growing candidate
manifest existed. CUDA visibility is empty; native image loading uses two
workers and one thread per numerical library. Estimated duration from an
earlier 250-batch observation is 25--30 minutes.

The launcher receipt is
`reports/server28_protocol_r_epoch01_candidate_v2_20261103.launch.json`.
Remote output is
`/home/gaob/Miss-RGBT/reports/server28_protocol_r_epoch01_candidate_v2_20261103`,
with an adjacent `.log` file.

```bash
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python \
  /home/gaob/Miss-RGBT/scripts/launch_protocol_r_frame_manifest.py \
  --output-dir /home/gaob/Miss-RGBT/reports/server28_protocol_r_epoch01_candidate_v2_20261103
```

This invocation has already executed. Its output path is exclusive; do not
repeat it or overwrite existing artifacts.

The replay targets 5,000 native real-image microbatches / 20,000 returned
tuples from epoch one, which both R runs have already completed. It binds
native loader and installed Torch sources, checks both saved first-update
input prefixes, and checks the replay's main CPU RNG against two iterator
seed draws. Source hashes are checked again at completion.

Even a completed replay is only a candidate reconstruction. Original
trainer CPU RNG continuity and later input identities were not independently
recorded. It cannot certify all historical frame exclusions, pass the OHCAS
freshness gate, or count as training or an official score. The launch receipt
explicitly records completion and freshness as false.

An earlier foreground attempt in
`server28_protocol_r_epoch01_candidate_20261103` reached a logged 250/5,000
batches in 74.56 seconds. Only its newly created replay process and two
workers were terminated to move execution into a persistent session. Preserve
that partial directory; it is not a completed epoch. Original GPU jobs were
not stopped.

## GPU Scheduling

At 2026-11-03 11:44:02 CST, all four GPUs were 100% utilized:

| GPU | Training | PID | Temperature | Power cap |
|---|---|---|---|---|
| 0 | R1 | 3388846 | 82 C | 250 W |
| 1 | R2 | 3395457 | 79 C | 250 W |
| 2 | A Utility | 1288280 | 77 C | 250 W |
| 3 | A Random | 1288281 | 74 C | 250 W |

R1/R2 were still in epoch two at 11:44:47, cumulative microsteps
8,680/8,392. Their effective batches are 8/16 and their accumulation factors
2/4; optimizer update counts are not comparable progress measures.
The epoch-two boundary is expected around 12:10--12:20 CST, not full
twenty-epoch completion.

Existing A official controller 1355742 and RPPM queue 3539938 remain live.
The RPPM queue still awaits four R training/official completion receipts.
Its existing 300-second dispatcher runs six real-image smokes before the
six formal arms, with twenty epochs and ten full official evaluations per
arm. Do not duplicate the queue or modify its frozen sources.

No new official metrics were produced. Observed temperatures do not verify
an 85 C hardware temperature cap. Preserve all healthy jobs and checkpoints.
