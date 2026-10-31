# Feature Repair Scope Diagnostic

The existing LocalRepair gate changes only tokens with unavailable raw
sensor inputs. The transformer self-attention before layer6 can also
change the features of observed tokens in a stream containing missing
inputs. The independent modality streams share weights but do not mix
in the backbone `Block.forward`. We test the scope restriction directly
before adding a larger repair mechanism.

Use the first four already registered train882 development tuples,
seed27182 and original input hashes. E0 epoch20, zero-initialized
LocalRepair, eval mode, all parameters frozen, CPU four threads.
No optimization, new data split, val97/test score, or live source edit.

For each template-dynamic11/01/10 x search11/01/10 state, compare:

- Native missing-input features.
- Replace only unavailable token positions at layer6 with features from
  the identical complete-input forward.
- Replace the entire feature stream at layer6 if any template/search
  token in that modality was unavailable.

The third intervention should exactly reproduce complete teacher output,
because unaffected streams have identical inputs and the backbone streams
are independent. Check output-map and box hashes, unchanged whole-model
state, finite per-sample task losses and fully complete-history equality.
Report27cells, four tuples each, both search frames. Preserve full per-sample
losses; a teacher may have a higher supervised loss on individual samples.

Both replacement arms use privileged missing features. They are diagnostic
controls, not deployable repair methods or achievable performance bounds.
A gap between interventions suggests that restricting corrections to
unavailable positions leaves contextual effects, but does not establish
that learning to correct all tokens is beneficial. A learned candidate
would need the complete-history invariant, matched capacity/budget controls
and closed-loop evidence without privileged inputs.

Output `/home/gaob/Miss-RGBT/reports/server28-repair-scope-20261031`;
log is sibling `.log`. Exclusive output/log creation prevents overwrite.
Runner `audit_repair_scope28.py`, launcher `launch_repair_scope28.py`.
Estimate2-4minutes from the previous18-cell single-search40.9s diagnostic
and the larger two-search comparison here. Verify the actual PID once,
then inspect completion after180-300seconds. No additional GPU allocation.

## Completed Result

PID892597 completed normally in161.87seconds. All27cells/four samples,
original input hashes, frozen whole-model hash and finite losses passed.
The full affected-stream intervention reproduced complete teacher maps
and boxes bitwise in every state. Fully complete11_11 was identical in
all three arms. No weights were optimized or deleted.

| Dynamic/search | Native task loss | Missing-token oracle | Affected-stream oracle |
|---|---:|---:|---:|
| 11/11 | 2.411833 | 2.411833 | 2.411833 |
| 11/01 | 4.397563 | 2.454967 | 2.411833 |
| 11/10 | 2.261354 | 2.432919 | 2.411833 |
| 01/11 | 2.334019 | 2.388563 | 2.411833 |
| 01/01 | 4.483481 | 2.380780 | 2.411833 |
| 01/10 | 2.290711 | 2.428451 | 2.411833 |
| 10/11 | 2.328140 | 2.343326 | 2.411833 |
| 10/01 | 5.400487 | 2.382916 | 2.411833 |
| 10/10 | 2.219739 | 2.459045 | 2.411833 |

Replacing missing positions gets close to complete-teacher task loss;
its cell-mean difference from full-stream replacement is between
-0.068507 and+0.047212. This small probe does not support expanding
correction to all observed tokens. The sampled native RGB-only search
cells and complete-search/missing-template cells actually have lower
mean loss than the complete teacher. Exact complete features are not
a guaranteed beneficial supervised target.

Decision: keep the current repair scope while completing the matched
template-coverage training. Do not add unconditional feature reconstruction
or an observed-token repair branch from this result. Those are hypotheses
requiring larger independent development/closed-loop evidence, not
validated modules. The four reused tuples cannot establish general
modality preference or benchmark performance.

Report `reports/repair-scope-report-20261031.json`, SHA256
`162fa65f7959132eccefc8b79ff9e52d927c2f1b6e84c702233d3bedf4a20101`.
Completion/launch receipts have matching `repair-scope-*-20261031.json`
names. At12:00:57 UTC+08 original four GPU trainers and three controllers
were live; the matched CPU pilot869785 was live in arm1 epoch2/update59.
No training schedule or model source changed.
