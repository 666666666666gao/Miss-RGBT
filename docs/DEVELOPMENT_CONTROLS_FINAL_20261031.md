# Development Controls: Final Results and Replication

## Availability History

Completed64 trajectories on16 development videos,7776 frames per arm/condition.
The two histories share E0 checkpoint, physical GPU assignment, block1 cycle
11/01/11/10/11/00, tracker and template rules.

| Full outage contrast | PR20 delta (pp) | SR delta (pp) |
|---|---:|---:|
| Availability history minus adjacent history | 1.806145 | 1.659243 |
| Paired video bootstrap95% interval | [0.114418,4.293027] | [0.372872,3.778192] |

SR improves14 videos and declines2. Blind SR delta2.197937pp
CI[0.784968,4.423532]; observed-after-first-blind SR delta1.566804pp
CI[0.276092,3.691328]. One video contributes16.172161pp SR improvement:
the mean must not be read as uniform benefit. Complete trajectories in both
arms and outage hold trajectories are bitwise equal across studies.

These same videos motivated the change. This is development evidence, not
fresh replication, official results, training-seed uncertainty or novelty.
Same-availability history also has a different temporal baseline; smoothing
and state matching remain confounded.

Candidate report SHA256:
`1e5899fd63b37690d08c715c827a893035c487c7e469f6ce806e194e70c0670c`.
Published report/comparison/completion are under `reports/state-motion-shortgap16-*`.
Finished evaluator/controller/analyzer sources were archived and checked
against launch/comparison hashes before the replication entry points changed.

## Protected Template Pair

Both arms completed192 updates, selectedepoch6 among epochs2..6, replayed
the selected branch and removed the other4 branch weights each.
Independent analysis verifies receipts, input identity, common initialization,
argmin selection, retained hashes and file absence.

| Dynamic template training | Selected mean six-cell task loss |
|---|---:|
| Complete | 4.363575024840732 |
| Mixed11/01/10 | 4.355297873883198 |

Mixed-minus-complete is-0.008277150957534, approximately-0.19%.
Six missing-search cells show3 decreases and3 increases; complete-history
11_11 output is unchanged. These selected losses reuse the selection tuples.
They do not establish tracking gains or independent generalization. No larger
GPU training campaign is justified by this small mixed result alone.

Complete retained SHA256:
`fe1bc6fb533ae81b2bc8e4e2cf1fc4005c10ee0b165fd70ff016b300838ea7ad`.
Mixed retained SHA256:
`15cad3fbd973ec1c674a76565961fbe42d39679a6affe01e7849dec78e23a434`.
Comparison SHA256:
`f4f28cb3a39fb2edd54c09be26804feb79da05c20601b1a84b7a4575f3184132`.
Published comparison/completion/independent analysis:
`reports/protected-template-pair-*`.

## Fresh Motion Replication

The next sequential development indices20..35 were registered at05:36:28Z,
before inspecting the preceding final comparison. The8611 frames are disjoint
from705 training videos,97 heldout videos and the preceding20 motion videos.
The registered assignment is4243/4368 frames onGPU0/1.

Run unchanged adjacent and availability velocity rules sequentially, each with
hold/velocity crossed with complete/outage:128 trajectories total,68888 frames.
Repeating hold and complete conditions provides equality controls.
No new thresholds, GT-based decisions or official masks are introduced.
The evaluator accepts only explicit development indices4..35 and the same
supported block1 availability schedule.

`launch_motion_replication28.py` calls the existing controller twice then the
independent paired comparison. Registration hashes and exact assignments are
checked at launch; comparison requires registration identity, equal cohort,
checkpoint/GPU and bitwise-equal controls. Controllers poll every300seconds.
Outputs:
`reports/server28-motion-replication16-{adjacent,availability}-20261031`.
Log `reports/server28-motion-replication16-20261031.log`.
Expected40-60minutes; launch verification is recorded in the handoff.

All four main trainings remain separate. Fixed20 controls save10/20 and
evaluate20 only. Adaptive batch10 campaigns perform the final-five selection
and cleanup workflow. Neither official target attainment nor three validated
paper modules has been established.
