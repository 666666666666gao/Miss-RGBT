# Final-Two-Block Low-Rank Pilot

## Question

Frozen head/fusion/context interventions have not passed their registered
gates. This does not establish that the backbone is the cause, but leaves
representation adaptation untested by those cached-output pilots.
Unlike the previous complete-input bypass, adapting the last two native
blocks can affect complete and missing inputs. Test whether equal-capacity
attention or MLP adaptation improves all three search-input states.

This is ordinary low-rank adaptation, not a novel paper module.
Read-only primary-source review: official microsoft/LoRA main commit
`c4593f060e6a368d7bb5af5273b8e42810cdef90`, `loralib/layers.py`,
UTF8 SHA256 `ea9fa6bd0930dfbf5af08440f6c71aeab3408cf07d3cca7620c86c45ea730797`.
Lines109-125 define frozen original weights, rank factors, alpha/r scaling
and zero output-factor initialization. Our hook adds a two-linear residual
with alpha=rank, no dropout or weight merging. No third-party file copied.
Source URL: https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py

The earlier raw/Hann candidate audit already found removing Hann harmful;
do not repeat or promote that policy. This pilot leaves decoding unchanged.

## Registered Design

- Server28 frozen E0 checkpoint SHA256
  `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
- Native zero-based blocks10/11 only. Attention arm adds rank5 to qkv;
  MLP arm adds rank4 to fc2. Each has30720 trainable parameters, shared
  across RGB/TIR and all availability states. Original backbone/fusion/head
  weights and buffers frozen; model eval mode, no stochastic depth/dropout.
  Matched capacity is not identical initialization across different shapes.
- Train882 only: existing705/88/89 sequence-name partitions, reused in prior
  pilots and thus not untouched independent sequences. New seeds49101/
  49102/49103,256 fitting/128 development/128 audit tuples. Exclude exact
  prior tuple identities, including the completed cross-stream pilot.
  E0 historical training exposure remains uncertified.
- First search, complete initial template; all3 dynamic-template states
  crossed with all3 search states for fitting, development and audit.
  Fitting2304 rows, development/audit1152 each. No patch masking.
- Cache float32 RGB/TIR tokens immediately before block10, GT boxes,
  native boxes/loss/IoU, identities and input tensor digests. Native tail
  replay must match all output maps exactly, batch16. No val97, RGBT234,
  official test frames or masks. Audit sampled only after both arms stop
  and select. Cache tokens are detached; adapting the earlier ten blocks
  is explicitly outside this experiment.
- Before training, measure real fitting-batch16 forward/backward memory
  separately for each arm, without optimizer updates. Choose the largest
  multiple16 dividing2304, at most256, below75% of the linear allocated
  memory estimate on the24GB card. The allowance covers allocator and
  matmul workspace; record actual peak allocated/reserved during training.
  No row duplication to inflate memory; no automatic OOM retry.
- Same batch and permutation per epoch for both arms. AdamW LR1e-4 at
  batch32, linear scaling with actual batch; wd1e-4, clip.1.
  Max20 epochs, LR x.1 from7, fresh stopping reference at7,
  patience3 without.2% development loss improvement, minimum10 epochs.
  Development score equally averages native task loss over all9 cells.
  Select minimum development loss among final5, ties earlier.
  Keep all checkpoints through verification.
- One optimizer seed per arm. This pilot cannot establish multi-seed
  robustness, full-image convergence or official benchmark gains.

## Gate And Interpretation

For each arm independently, after development selection evaluate audit once.
Use sequence-macro crop-IoU deltas versus cached E0, averaging the three
dynamic-template states separately for each search state11/01/10.
Bootstrap10000 sequence resamples, seed49301.

An arm advances only if mean delta >=.01 and95% lower bound >0 in all3
search states, with finite losses and unchanged baseline/absent baseline
gradients. Report the paired attention-minus-MLP effect over all9 cells
(seed49302) descriptively; do not require a post-hoc superiority threshold.
A pass authorizes a fresh train882 closed-loop diagnostic, not official
evaluation or a validated module. If neither passes, close this exact pilot,
do not retune its audit or enlarge training based on its test-like ranking.
Any improvement is ordinary adaptation until a specific mechanism and
additional independent evidence justify a paper claim.

## Execution

Check GPU2 free (<500MiB), temperature<85C, cap250W. One GPU with both
arms sequentially shares one immutable cache, minimizing duplicate work.
Warm-reuse existing2028 compute ledger; no environment/library change.
Estimate10-30minutes including cache and both arms; refine from measured
epoch times. Poll180-300seconds near expected milestones.
Full Complete +2-point, RGBT234-Miss88.1/64.3 and three-module goals remain.

## Result

Pending.
