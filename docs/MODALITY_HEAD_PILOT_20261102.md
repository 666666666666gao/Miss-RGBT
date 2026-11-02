# Modality Head Routing Control

Previous target-position supervision failed. Fresh diagnostics showed larger
TIR-only geometry errors than RGB-only, but did not prove that shared-head
optimization causes them. Test that question with a capacity-matched control.
Conditional heads and random routing are ordinary controls, not novel modules.

Unchanged E0 checkpoint SHA256
d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
Reuse train882705/88/89 fitting/development/audit name partitions, new
256/128/128 tuple seeds42101/42102/42103. All9 dynamic-template/search
states, complete initial template, first search only. Verify no exact tuple
overlap with preceding fusion, response/geometry and head-supervision pilots.
Partitions reused across investigations are not untouched independent data.
No val97, official-test data or baseline edits.

Cache frozen fusion maps and verify native-head replay. Both arms use two
pretrained native CENTER head copies, all parameters trainable, BN eval.
The modality arm routes RGB-only search10 to head0 and TIR-only01 to head1.
The random arm assigns half of each of the six missing-search cells to
each head, using a fixed seeded permutation, independent of target values.
Evaluation random routing is similarly preregistered by seeded draws, not
selected from performance. This control is stochastic inference, not GT routing.
Each arm has identical capacity and1536 total fitting rows,768 per head.
Modality heads see3cells x256; random heads see6cells x128.
All complete-search states use frozen E0 head, including missing-template
histories. Verify all score/box/size/offset maps equal on those cached inputs.
This protection does not guarantee closed-loop recovery trajectory equality
because previous missing frames can change crops and template contents.

Original predicted-peak2GIoU+5L1+focal loss, native inference throughout.
Per-head batch64,AdamW LR8e-5,weight decay1e-4,clip0.1. Each head12updates
per epoch,24updates per arm. Identical per-head seeded order generation.
Max20epochs,LR drops after6;fresh stopping reference7,0.2% improvement,
patience3,min10. Selection minimizes six-cell native-inference development
loss among last5, ties earlier. Preserve all weights through verification.
Final audit only after both arms stop/select. No test-based cleanup.

Advance to a fresh train882 closed-loop test only if modality routing:

1. Improves sequence-macro IoU over E0 by at least0.01 in both missing-search
   directions averaged over template states, with positive95% lower bounds.
2. Has positive95% sequence-bootstrap lower bound versus random routing
   across the six missing-search cells.
3. Preserves cached complete-search maps bitwise; baseline full-state hash
   and gradients unchanged; copied-head BN buffers unchanged, finite training.

Use10000 bootstrap draws and retain all per-tuple records. Passing supports
only the next closed-loop test, not benchmark thresholds, three modules,
or complete approximately+2 points. Full-complete improvement requires
separate work; do not redefine the overall goal around missing-only gains.
Failure means do not expand this pilot or retune its audit samples.

GPU2 sequential arms, existing runtime, check actual occupancy first.
Expected3-8minutes based on preceding274-second pilot; inspect near
completion with180-300second intervals. Record actual allocated/reserved
memory; finite cached-head pilot is not full-model VRAM profiling.

## Completed result

GPU2 PID4186101 launched2026-11-02T00:03:50Z (08:03:50 UTC+08), completed
233.025seconds normally. Both arms stopped10epochs/240updates and selected
epoch6 from last-five6-10 using development loss. Each arm has12,948,234
trainable parameters(two heads),1536 fitting rows/768 per head,24updates
per epoch. Final audit128 tuples covered73 actual sequences.

Registered sequence-macro crop-IoU effects:

| Comparison | Mean gain | 95% sequence-bootstrap interval |
|---|---:|---|
| Modality versus E0,TIR-only | 0.003473 | [-0.006143,0.013821] |
| Modality versus E0,RGB-only | 0.012225 | [0.004128,0.022354] |
| Modality versus random,six missing-search cells | 0.000977 | [-0.001836,0.003703] |

RGB-only meets the preregistered direction-specific magnitude/CI test.
TIR-only does not, and modality-versus-random advantage is unestablished.
Overall advancement=false. Do not promote to full training/closed-loop
evaluation, retune this audit sample or claim a routing module. This does
not prove shared-head interference is absent; it fails to establish that
modality routing improves beyond the matched capacity/routing control here.
Both arms being trained separately can have different losses/clipping
scales and per-expert sample composition; no isolated gradient cause claim.
Random routing uses one fixed draw; robustness over routing seeds untested.

All complete-search cells11_11/01_11/10_11 routed to frozen E0, score/box/
size/offset maps bitwise identical on cached inputs. This protects static
inputs, not dynamic closed-loop trajectories after earlier missing frames.
It cannot yield approximately+2 full-complete improvement by itself.

## Verification and retained artifacts

Saved-evidence analysis verified actual705/88/89 partitions and exclusions,
fresh tuple IDs, cache/source/base/selected/final-five hashes; independently
recomputed both route schedules, per-cell balanced random assignment,
history/LR/stop/selection and all effects. All10 final-five two-head weight
files loaded finite, with BN buffers equal to E0. Initial head digests equal
the actual pretrained head state. Baseline whole-state digest unchanged,
parameter gradients absent. Recorded sources do not cover all transitive
imports, and there is no external scientific review.

Separate head construction and direct native inference replay passed
3456 cached audit rows(E0/modality/random), exactly matching loss/IoU;
complete-search output maps equal E0 throughout. No GT score map supplied.
This replays cached heads, not backbone/image sampling or full tracking.
Peak allocated bytes888,471,552/997,638,144, reserved1,094,713,344/
1,098,907,648. Finite cached pilot does not saturate24GiB; full-model
batch/LR profiling remains necessary if a mechanism clears its gate.

Report SHA256:
ab2f1aca04d047e4d92432e56bc9714687c8890133e0238e104d925e355c3790.
Small evidence:reports/server28_modality_heads_20261102/.
Initial protocol archived aspreregistered_protocol.md,SHA256
03452033ad65fb555103e0517019f1de972f40cf3b9546816f7bd5be94eff6bb.
All weights and large caches retained on server28. No val97, official-test
evaluation or checkpoint cleanup. Official scores unchanged; complete+2,
RGBT234-Miss thresholds and three supported modules remain unmet.
