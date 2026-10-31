# Batch10 Candidate Epoch Five

Verified on server28 at2026-10-31T08:02:19+08:00
(2026-10-31T00:02:19Z). Both original candidate trainers remained live and
entered epoch6. The fixed20 trainers on GPUs0/1 also remained live in
epoch7. All four GPUs are occupied.

Both candidates use identical128 cached val97 tuples, three availability
states, effective batch10 and LR1.25e-4. These losses are a convergence
proxy, not tracking PR/SR or a test result.

| Epoch | Utility mean task loss | Random mean task loss |
|---|---:|---:|
| 1 | 5.041177 | 4.980912 |
| 2 | 5.119562 | 5.094764 |
| 3 | 5.001092 | 4.897613 |
| 4 | 5.050189 | 5.017014 |
| 5 | 5.095668 | 5.129561 |

Epoch5 utility losses11/01/10=4.444279/5.805996/5.036729; random losses
=4.433926/5.900509/5.054249. Both are worse than their epoch3 means.
Utility is lower than random at epoch5, but the earlier ordering differs;
there is no stable advantage or convergence evidence.

The existing schedule drops LR after epoch10, resets the stopping reference
at epoch11, and stops after three stale epochs with relative improvement
threshold0.2%, earliest epoch14. Keep the running schedule unchanged.
After training exits successfully, validate the final five checkpoints on
all97 validation sequences in complete and alternating-outage conditions,
select one checkpoint by their mean macro SR, record cleanup, and use
that same checkpoint for both official complete and Miss evaluation.

Recent measured training speed is approximately3.2seconds/update for
utility and3.1 for random, about19updates/minute. From the08:00+08:00
snapshot, epoch6 boundaries are approximately09:45 and09:25+08:00;
validation/checkpoint overhead can shift these estimates. Observe near
those milestones at180-300second intervals, not every few minutes now.

Evidence is copied from each run's `validation_history.json` into
`reports/candidate-{utility,random}-epoch05-validation-20261031.json`.
The historical A Miss result remains separate; three of four requested
Miss thresholds, complete-input gains and three validated paper modules
remain unfulfilled.

## Epoch Six Observation

At2026-10-31T01:46:11Z (09:46:11+08:00) both original trainers,
tracking workers/controller and postprocessing queue remained live.
Utility finished epoch6/update12000 with losses11/01/10=
4.393904358/5.730497293/4.782562226 and mean4.968987959.
This is its lowest mean through epoch6, compared with epoch3=5.001092297.
Random epoch6 mean4.963055770 remains above its epoch3=4.897612979.
Utility-minus-random mean at epoch6 is+0.005932190; no stable advantage
or convergence follows. Neither stopping schedule nor LR was changed.

Epoch5 full tracking health evaluation continues independently, each
candidate complete condition at3/97sequences and15963/54646frames.
Do not confuse epoch6 tuple validation with epoch5 tracking evaluation.
Full candidate complete results are expected near10:40+08:00, pending
actual throughput. Utility history is archived as
`reports/candidate-utility-epoch06-validation-20261031.json`.
