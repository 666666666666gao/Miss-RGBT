# Candidate Epoch Nine Curve Audit

Evidence: `reports/candidate-health-epoch09-20261031.json`, downloaded from
the actual server inspector output while both trainers were running.
SHA256: `911808ac87ed25b3fb813918146ef0ce14f4f4838708387957886ba734d31eff`.
All recorded task/total losses were finite; paired validation input and
first-update input hashes match. This is loss evidence, not benchmark evidence.

| Epoch | Utility mean task loss | Random mean task loss |
|---|---:|---:|
| 1 | 5.041177 | 4.980912 |
| 2 | 5.119562 | 5.094764 |
| 3 | 5.001092 | 4.897613 |
| 4 | 5.050189 | 5.017014 |
| 5 | 5.095668 | 5.129561 |
| 6 | 4.968988 | 4.963056 |
| 7 | 5.069816 | 5.136228 |
| 8 | 4.963940 | 4.995458 |
| 9 | 4.979970 | 4.972420 |

Utility is lower on 3/9 matched epochs. These epochs are correlated and
are not independent trials. Utility's minimum is epoch8; random's is epoch3.
Latest losses are respectively 0.32293% and 1.52743% above those minima.
First-three/last-three means: utility5.053944/5.004576,
random4.991096/5.034702. No stable utility advantage or convergence follows.

At epoch9, utility minus random losses by 11/01/10 are
-0.001977/-0.048274/+0.072901: the TIR-only improvement coexists with worse
RGB-only loss. The mean alone obscures that tradeoff.

## Scope And Next Decision

The actual `candidate_validation28.validate` supplies complete availability
to every template and changes search availability to 11/01/10. It does not
measure closed-loop corruption of online-template history or localization
drift. The 128 cached tuples also do not establish full97-video coverage.
Do not use this curve to declare tracking convergence, a paper module,
or any PR/SR threshold attained.

Keep the registered scheduler and post-drop patience unchanged. Epoch11
resets the convergence reference after the epoch10 LR drop; with patience3
the earliest plateau stop is epoch14. The active controller then evaluates
each final-five checkpoint on all97 videos (54646 frames), complete and
alternating outage, and chooses one common checkpoint by mean macro SR,
ties earlier. Only after all five evaluations does it delete the other four
and run the selected weight on both official complete/Miss datasets.

This read-only audit confirms source logic and current process liveness;
it does not establish that those future stages have completed. The original
fixed20 controls save10/20 and are not the final-five selection campaign.
