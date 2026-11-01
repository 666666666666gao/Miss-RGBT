# Frozen Fusion Adapter Pilot

## Question and boundary

The preceding train882 joint-set diagnostic found no established useful
utility-top4 effect. Layer6 corrections traverse six additional Transformer
blocks before the fixed fusion and CENTER head. This motivates testing
whether a directly task-trained post-fusion correction can learn a useful
localization change without changing the baseline representation. It does
not establish that fusion is causal or that relocating a repair is novel.

This bounded optimization diagnostic is distinct from the rejected block
bypass and joint-utility hypotheses. It does not authorize those trainings.
No external implementation or library is introduced. Ordinary residual
adapters and mean-pooling controls are not claimed as contributions.

## Frozen design

- E0 epoch20 SHA256
  `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
- Reuse existing train882 name partition: indices divisible by5 are the177
  development names; all other705 names are fitting names. Split those177
  names into first88 for stopping/selection and last89 for a final audit.
  Never instantiate the val97 dataset.
- Cache256 fitting tuples (seed38101),128 stopping tuples (seed38102),
  and128 final-audit tuples (seed38103), first search per tuple only.
  The native training crop augmentation is frozen by these seeds; no
  sequence or frame is filtered by performance. Record names, frames and
  tensor hashes. The original E0 historical split exposure remains
  uncertified; independence describes this adaptation procedure only.
- Fitting: dynamic-template availability drawn once per tuple from
 11/01/10 with probabilities0.4/0.3/0.3, using a dedicated seed. Each
  tuple is used in both RGB-only and TIR-only search conditions.
- Stopping and audit: all nine dynamic-template/search11/01/10 states.
  Initial template is always complete; no patch dropout.
- Cache float32 fusion outputs with the exact frozen model in eval mode.
  Two arms share every cached feature, target, mask, initialization and
  minibatch order. Frozen baseline weights and BN buffers are unchanged.
- Local arm: availability-conditioned1x1 residual MLP, input fused768
  channels plus4 dynamic/search availability values, bottleneck32, final
  projection zero-initialized. Correction applies before the frozen head.
- Global control: identical parameter count/initialization, replace only
  the adapter's input feature by its spatial mean expanded to the same
  grid; the native base feature remains spatial in both arms. This tests
  whether the correction needs location-specific input. It is a mean-input
  control, not an inert sham and not an insertion-depth-matched comparison.
- Complete history11/11 bypasses correction exactly. Complete search
  after a missing template is measured separately and has no such guarantee.

## Optimization and stopping

Effective batch64, AdamW LR8e-4 (linear reference batch8/LR1e-4),
weight decay1e-4, gradient norm clip0.1, seed2027, eight updates per epoch
on the512 cached missing-search examples. Maximum20 passes, LR drop after
epoch6, fresh stopping reference at epoch7. Stop after three post-drop
epochs without0.2% relative development improvement, never before epoch10.
Development selection averages the six missing-search cells equally.
Select the best of the final five epochs by this development loss, ties
earlier. Preserve all pilot checkpoints until verification; no original
or official-tested checkpoint is cleaned. A cached-feature plateau is not
full-tracker convergence.

One GPU2 process performs caching then both matched arms sequentially.
Measure peak allocated/reserved GPU memory and actual update speed. This
small frozen-feature pilot has a finite512-example population; do not
duplicate samples or enlarge the batch solely to fill VRAM. Full training
batch/LR tuning is deferred until the diagnostic justifies full training.

## Registered advancement gate

Final-audit tuples are read for model comparisons only after both arms
stop and select their development checkpoint. Advance only to a fresh
train882 closed-loop test if all hold:

1. Local correction improves mean task loss at least1% relative to native
   in both missing search directions, averaging all three template states.
2. Sequence-clustered paired-bootstrap95% lower bounds are positive for
   local versus native in each missing direction and local versus global
   across the six missing-search cells.
3. Native initial-map replay and complete-history score/size/offset/box
   equality hold; whole baseline state hashes before/after match and
   baseline parameter gradients remain absent. All losses are finite.

These are training-objective diagnostics, not official PR/NPR/SR/MPR/MSR.
A failed gate rules out promotion from this pilot. Even a passed gate
does not prove a module, unseen-benchmark gain or approximate+2 Complete
improvement; complete-history output is intentionally unchanged here.

## Execution and evidence

`scripts/probe_fusion_adapter.py` runs inside the existing frozen runtime
with Python3.7 and Torch1.12. `scripts/launch_fusion_adapter_pilot.py`
checks GPU2 free memory, creates a unique log, registers source/protocol
hashes, and starts one detached worker. Source files remain fixed until
completion. No automatic restart.

Root output `reports/server28_fusion_adapter_pilot_20261102`; final
`report.json` and `completed.json` contain training histories, actual
update counts, retained final-five hashes, final audit rows and gates.
Large feature caches and small adapter weight files stay on server28.
Estimate5-15minutes provisionally; revise from cache/update timings.
Inspect near that window with180-300second polling, not continuous polls.

## Completed result

GPU2 PID4123633 completed normally in197.78seconds, launched
2026-11-01T23:23:25Z (2026-11-02 07:23 UTC+08). Both arms have50,080
parameters, identical initialization and cached data. Local stopped at
epoch18/144updates; global stopped at epoch14/112updates. Final-five
development selection chose local epoch15 and global epoch11, respectively.
All pilot checkpoints remain present. This is cached-head stopping, not
evidence of convergence of a full tracker.

Audit128 tuples cover60 actual sequences from the89-name audit partition.
Each arm has1152 matched audit records. Positive absolute gain below means
native task loss minus local task loss; bootstrap clusters by sequence.

| Comparison | Mean loss gain | Sequence-bootstrap95% interval | Relative macro loss improvement |
|---|---:|---|---:|
| TIR-only search01, averaged over template states | 0.068091 | [-0.022931,0.160346] | 3.367973% |
| RGB-only search10, averaged over template states | 0.053419 | [-0.013591,0.129761] | 3.807889% |
| Local versus global across missing-search cells | 0.010722 | [-0.041733,0.066307] | Not the registered ratio |

All three registered confidence intervals cross zero. The secondary
mean-of-tuple relative-gain intervals are positive, but that is a different
estimand and cannot replace the preregistered absolute-loss test.
Registered advancement gate=false. Do not launch full training or the
fresh closed-loop promotion from this pilot.

An additional observed limitation: after missing dynamic templates, complete
search losses increased from1.122942 to1.171483 (01_11) and from1.096826
to1.122120 (10_11), using the local arm's tuple means. Complete-history
11_11 remains exactly unchanged by design. A complete-history bypass does
not guarantee protection on recovery frames. These measurements neither
prove a general failure cause nor establish a contribution.

Baseline whole-state digest before/after matches
`4d64b467b82ff76963a8c32d13cb0e60980fef236df14546e7bceaddffdb6eb6`.
All baseline gradients absent; native cache/head replay and fully complete
score/size/offset/box parity passed. Peak allocated memory was approximately
800.4MiB(local) and800.8MiB(global), reserved1108MiB. Batch64 covered the
finite512-row fitting cache in8updates/epoch; no duplicated data was added
to inflate memory use.

`scripts/analyze_fusion_adapter_pilot.py --verify-server-files` recomputed
selection, stopping, bootstrap and gate arithmetic, checked split/pairing
and source/base/cache/selected/final-five SHA256 hashes. It passed. This is
saved-evidence verification, not independent neural replay or external review.
The producer's recorded source list does not archive every transitive import.

Small evidence files are under
`reports/server28_fusion_adapter_pilot_20261102/`. Large caches and weights
remain on server28. Report SHA256:
`2ca07ff702becc2aa26dc646d05e911810ca18a360b42b8b00e94a5e7a36584f`.
The byte-exact initial protocol is saved there as`preregistered_protocol.md`,
SHA256`52ea854a190fbf95cf84bebd01668c8f173eed8a2d0adefc2d6cca2f230be962`.
No val97, official test evaluation, baseline edit or weight cleanup occurred.
Official maxima and the unmet complete/RGBT234-Miss/three-module goals
are unchanged. Do not repeatedly retune this pilot on its audit sample.
