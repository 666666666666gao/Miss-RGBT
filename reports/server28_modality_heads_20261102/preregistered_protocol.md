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
