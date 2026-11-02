# Asynchronous template source boundary

Official MPTrack repository: https://github.com/liulei970507/MPTrack
Inspected commit:70ae6b768a5fad920fd4bba7e170aaa3d75cc77d.
Read-only source inspection2026-11-02, no code/checkpoint/dependency copied
into Miss-RGBT or executed as a tracker.

Root LICENSE is MIT, copyright2026 liulei, Git blob
c6e96c378642e447f253ff228403665e93aa9eab. Tracker file:
lib/test/tracker/ostrack_st1_quadra_zfusetrans_3ioubranch_rgbtpos_regwcls_3head_specific2anotherall_wsq.py,
Git blob d42fb6b1602173461754af5b3ad517c0251955f6.
Git clone fetched objects but checkout failed for two unrelated experiment
YAMLs due Windows long paths. The inspected tracker/LICENSE are readable;
this is not a successful full installation or reproduction.

The initialize method creates separate RGB/TIR lists with initial,
synchronous, asynchronousRGB and asynchronousTIR template slots. Reliability
outputs from regression/classification branches determine update flags and
state. At tracker lines273-288, synchronous updates replace both stream
slots, while states1/2 replace the RGB/TIR asynchronous slot separately.
Lines314-329 contain another conditional branch with the same slot choices.
The inspected defaults enable temporal/state rules, set xizhi_update=False,
and combine regression/classification reliability; these source details are
not imported into our proposed control.

Consequently, separately retained sensor templates and asynchronous template
updates are already prior art. Our next frozen diagnostic asks only whether
the native CADTrack availability path benefits from preserving an observed
dynamic sensor template rather than masking it after an incomplete write.
It uses2slots and known availability with SAME native crops/write events;
it has no learned reliability network or extra template capacity.

This is an independently implemented policy control, not a novel method
claim. It cannot establish closed-loop efficacy from shadow predictions,
nor explain RGBT234-Miss without full official evaluation of a separately
registered candidate. No official-test ranking influenced its parameters.
