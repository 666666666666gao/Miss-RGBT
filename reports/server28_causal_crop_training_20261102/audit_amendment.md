# Prospective metadata-only audit amendment

Original worker PID877938 terminated at fresh_clips after both training
arms had stopped and selected, before any audit images or predictions.
Original 16 unique 96-frame clips are infeasible under the registered
partition, visibility and frame exclusions: inventory finds only12 eligible
sequences. The original protocol is incomplete, not passed.

Before reading audit predictions, fix the amended audit at24 unique64-frame
clips, seed58103, from the SAME89 audit names, same exclusions, valid positive
visible GT, IID .4/.3/.3 schedule after complete initialization. The inventory
finds exactly24 eligible sequences at64 frames (29 at48,12 at96). This is a
metadata-only feasibility decision, not selection using performance.

Keep both trained arms exactly as selected: native epoch11 from8-12, causal
epoch6 from5-9. Native stopped at12 (cap), causal at9 (registered patience).
No retraining, optimization, changed crop policy, new seed search or revised
threshold after outcomes. Preserve every original checkpoint and failure.

Run E0/native/causal complete and mixed closed-loop tracking on the same
24clips. No GT after initialization. Bootstrap10000 seed58301, same IoU
thresholds as original, now all24 sequences required. Both missing directions
causal-E0 mean>=.01 and lower95%>0; pooledmissing causal-native lower95%>0;
complete causal-E0 mean>=0 and lower95%>-.01.
An amended pass only authorizes a separately registered larger training study;
it cannot establish original96-frame success, long-horizon robustness,
official tracking improvements, or a novel module. Failure closes this pilot.

Save the immutable amendment snapshot and source/selection/inventory hashes
before launching resumed audit. Validate original launch hashes, frozen
producer source and both selection files. Resume audit only, never fitting.
GPU2 if idle<500MiB,temperature<85C,power250W; unchanged warm runtime.
Expected3-6min, observe180-300s. Verify saved boxes/GT/schedules/hashes and
firstclip native-image replay independently. No val97 or official tests.

Inventory first attempt failed JSON serialization on NumPy int64; its partial
file and log are preserved. Corrected inventory converts counts to Python
int, writes audit_inventory_v2.json, reads only metadata, no model/images.
