# F Batch Profile

This profile measures synthetic resident tensors on RTX 3090 with the actual
E0 network and three full AdamW updates. It does not load images, use the
dataset, crop frames or produce a checkpoint or official score.

| Arm | Microbatch | Accumulation | Effective batch | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|
| single | 2 | 2 | 4 | 8.95 GiB | 9.42 GiB |
| none | 2 | 2 | 4 | 14.46 GiB | 15.12 GiB |
| shallow | 2 | 2 | 4 | 14.59 GiB | 15.24 GiB |
| deep | 2 | 2 | 4 | 14.57 GiB | 15.25 GiB |

The deep arm with microbatch=3 failed during backward with CUDA OOM:
20.46 GiB was allocated, only 21 MiB was free, and the allocator had
reserved 21.82 GiB. The failure was captured in
`reports/server28_feedback_f_deep_b3_oom_20261102.log`. Microbatch=4 was not
attempted because the measured double-pass microbatch=3 already exceeded the
24 GiB card.

For the smoke configuration, use microbatch=2 and accumulation=2. The
effective batch is 4 and the proportional learning rate used by the profile
is `5e-5`, with backbone learning rate `5e-6`. This is a preliminary
synthetic fit; actual image loading and augmentation can change memory use,
so the first image smoke must retain the same OOM boundary and record peak
memory before formal training.

Receipts:

- `profile_output/feedback_f_20261102/single_b2.json`
- `profile_output/feedback_f_20261102/none_b2.json`
- `profile_output/feedback_f_20261102/shallow_b2.json`
- `profile_output/feedback_f_20261102/deep_b2.json`
- `reports/server28_feedback_f_deep_b3_oom_20261102.log`

All successful profiles used checkpoint SHA256
`d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8` and
actual PyTorch `1.12.1+cu113`. The profile script is
`research/feedback_f/profile_batch.py`.
