# Pairing verifier v2 executed evidence

Equal-length, unequal-length in either arm, and rejection of common input,
mask and development mismatches passed check_partitioned_pairing_v2.py.
Real saved six-epoch smoke verification passed again,24common microsteps,
both complete single-arm weight/source/stopping/selection checks unchanged.
No training or benchmark execution was used by this correction.
Checks report: reports/server28_partitioned_pairing_v2_checks_20261102.json.
Smoke receipt: reports/server28_partitioned_a_20261102_smoke/verification_v2.json.

At08:59:45UTCNov2, explicitly retired old waiting controllerPID1318842
after live command/status/source checks and before any epoch development
history or official results existed. Registered v2PID1355742. At09:00:09UTC
actual command live00:23,WAITING_FOR_TRAINING,all frozen-source hashes pass.
Original trainingPIDs1288280/1288281 remained live41:09 unchanged,GPUs2/3
74/75C and22710/22742MiB with250Wlimits. Original controller absent in ps.
No trainer/weight/data/source/optimizer/evaluation protocol was replaced.
New directory:
reports/server28_partitioned_a_20261102_official_queue_v2/.
Old superseded.json explicitly links the new controller receipt; old source,
registration,status,logs and files remain preserved.

Use the v2 directory/status/controller from now on. Terminal manual audit,
if needed, is verify_partitioned_a_v2.py --phase formal; do not manually
race the queued verifier or launch duplicate evaluation. Still no new
official score,convergence claim,validated module or goal completion.
The original amendment is frozen unchanged; this separate document records
execution evidence after registration.
