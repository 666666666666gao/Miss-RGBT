**E2 experiment integrity audit — WARN**

Date: 2026-10-16. Fresh read-only reviewer; requested `gpt-6-astra` / `max`, actual host routing **unattested**. `review_independence: same-family`; `acceptance_status: provisional`.

All **10 E2 metrics and 93,966 per-sequence curve values reproduce exactly** from raw boxes and held GT. All 958 sequence runs, 674,706 box rows and 50,000 training updates verify. There are **0 blocking findings for the explicitly qualified result package**. The overall WARN preserves protocol-provenance, checkpoint-witness, single-seed and causal limits.

| Check | Status | Finding |
|---|---|---|
| A. GT provenance | WARN | Real external GT; author-toolbox/SCDT equivalence and full startup sealing unestablished |
| B. Scores | PASS | No output self-normalization; all scalar/curve differences 0 |
| C. Artifacts and claims | WARN | Closed records and numbers agree; checkpoint binary not independently inspected |
| D. Executed paths | PASS | Claimed scorers connected; nonzero KD logs are zero-weight diagnostics |
| E. Scope and attribution | WARN | Qualified one-seed controls; broad positive transfer and BN causality unsupported |
| F. Evaluation type | PASS | `real_gt` |

**A. Ground truth provenance — WARN**

All 713 external annotation files (245 LasHeR; 234 visible and 234 infrared RGBT234) match the earlier evaluation inventory and held package hashes. The 479 sequence names and 337,353 frames match exactly. Evaluation GT is loaded from dataset annotation text, never from tracker predictions or the complete-view teacher response.

All 1,967 actually held runtime-package files match the post-evaluation installed manifest. All 3,556 installed-manifest entries also match members of the supplied rgbt 1.0.1 wheel. The remaining 1,589 installed copies are not held/read by this reviewer. Public-release hash correspondence does not establish benchmark-author-toolbox or SCDT equivalence.

The collection/package/mask seal is post-evaluation. Archived prepare() checks mask and GT hashes before inference against the inventory, but this is not an independently captured seal of the complete installed package at launch. No network or upstream repository was queried in this review.

The separately bound archive has train 882/val 97/test 245 pairwise disjoint sequence names and 97 annotation hashes matching the val inventory. Its recorded observation precedes E2 launch by 3894.4561944 seconds. E2 training and campaign manifests omit split text and training annotations; this cannot certify original-E0 historical exposure or all runtime training-image bytes.

Evidence: [evaluate_model.py:29](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:29>), [evaluate_model.py:42](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:42>), [basedataset.py:9](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/dataset/basedataset.py:9>), [basedataset.py:77](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/dataset/basedataset.py:77>), [collection.json:2638](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/collection.json:2638>), [collect_e2_closed_20261016_v2.py:96](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/collect_e2_closed_20261016_v2.py:96>), [e2_closed_collection_verified_20261016.json:14](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/e2_closed_collection_verified_20261016.json:14>), [lasher.py:61](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/lib/train/dataset/lasher.py:61>), [val97_inventory.json:1950](<C:/Users/gb/projects/Miss-RGBT/reports/development_data_20261015/val97_inventory.json:1950>), [SCORING_PROVENANCE_FINDINGS_20261015.md:49](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/docs/SCORING_PROVENANCE_FINDINGS_20261015.md:49>).

**B. Score normalization and independent reconstruction — PASS**

Fresh reviewer-written scalar geometry and threshold calculations independently reconstruct all 10 E2 percentages and all 93,966 per-sequence curve cells. Maximum absolute score difference is 0 pp; maximum curve-cell difference is 0; no curve cell differs. The independent code imports NumPy only, never producer scoring code or a neural model.

No performance denominator is a prediction maximum/mean/minimum. NPR uses GT width/height plus1e-8. RGBT234 MPR takes the smaller center error to two real GT boxes, and MSR the larger IoU. IoU uses the union of box areas; these are geometric benchmark operations, not self-normalized performance.

Exact held semantics: float32 loading; ties-to-even prediction rounding; LasHeR GT rounding; LasHeR first prediction replaced by GT while retaining that frame in the denominator; nonpositive predicted dimensions copied from the previous box (zero occurrences here); inclusive-pixel geometry; PR/MPR<=20; NPR<=0.20; success IoU strictly greater than each of 21 thresholds from 0 to 1; arithmetic mean over those thresholds; equal sequence weighting rather than frame weighting. All four success-curve groups have exactly zero at threshold 1.

LasHeR contains 1,260 GT rows in 52 sequences with some coordinate<=0, including 533 with nonpositive size and 727 with positive size but nonpositive position. The scorer assigns distance/IoU=-1: those rows become PR/NPR successes and SR failures, and remain in denominators. The fixed macro precision contribution is 0.5568053864558151 pp relative to same-denominator failures. RGBT234 has 596 visible and 633 infrared nonpositive-coordinate rows, all with positive sizes, and does not use that override.

Evidence: [utils.py:30](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/utils.py:30>), [basedataset.py:19](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/dataset/basedataset.py:19>), [utils.py:117](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/utils.py:117>), [utils.py:121](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/utils.py:121>), [utils.py:161](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/utils.py:161>), [metrics.py:28](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/metrics/metrics.py:28>), [metrics.py:62](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/metrics/metrics.py:62>), [metrics.py:246](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/metrics/metrics.py:246>), [metrics.py:290](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/metrics/metrics.py:290>), [metrics.py:332](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/metrics/metrics.py:332>), [reconstruct_e2.py:1](<C:/Users/gb/.codex_tmp/e2_integrity_audit_reviewer_20261016/reconstruct_e2.py:1>).

**C. Artifact existence, completion, identity and claimed numbers — WARN**

Both E2 evaluations are closed with matching launch/exit PID records and exit 0. Each has 479 bbox files, 479 individual receipts, 479 aggregate receipts, 479 matching DONE log entries, eight ready workers on recorded physical GPUs 2/3,337,353 finite rows, and matching printed/file metrics. Total 958 sequence runs, 674,706 rows and 2,698,824 finite coordinates. Every bbox SHA and individual receipt matches; every first raw box equals initialization GT.

The fixed masks verify byte-for-byte and by per-sequence frame/state totals. All 52,493 noninitial both-missing Miss frames exactly hold the previous raw box. The original masks have 16 both-missing initial frames and 65 initial frames missing either modality; all 479 sequences are explicitly initialized with complete input. First frames and held frames are still scored.

There are 50,000 consecutive updates 1..50000, exactly 2,500 per epoch 1..20. All numeric fields are finite, elapsed time strictly increases, and 2,001 printed training records match steps.jsonl. Config, campaign, completion, archived configuration and parent checkpoint metadata agree. Campaign 115 source/config files, training 92 files and each evaluation 106 files all match archived bytes; collection 2,066 payload hashes also match.

All 60 comparison source values, 40 percentage-point deltas, 10 CSV rows and 10 six-decimal Markdown rows agree. Existing FT, availability and E3/E4 comparison files are bound by exact hashes to their prior audits; all 293 supplemental files have matching entries in those audits. Their old numerical reconstructions are reused explicitly, not claimed as fresh runs by this reviewer.

Both evaluations cite final checkpoint SHA f66799aaaa769f9cfde71ed05fecf02f9eff7be18e54518d7c1d20e8e76f9a3d. The successful collection transport and collector source support the parent CPU strict-load/finite-state/hash witness, but no original checkpoint binary is supplied or independently inspected here. Parent metadata has 250 model state keys, 39 BN buffers, 36 unchanged head running buffers, and a fusion counter 37500->637500; source-derived expected extra fusion calls are 600,000. No BN performance causality follows.

The first collector actually exited 1 with an assertion at embedded code line 37; its source incorrectly required physical_gpus==[2]. The corrected v 2 requires[2,3], consistent with the unchanged evaluation launch/run/worker receipts, and actually exited 0. The failure remains preserved. This was a collection expectation correction, not a neural replay or changed result.

The sealed tracker/README/HANDOFF contain the earlier 17:03 evaluation-running snapshot. Later closed records establish completion at2026-10-16T10:42:01.785385Z. Refresh live publication after review while preserving those historical sealed snapshots; their old running/pending wording is not evidence of phantom results.

Evidence: [status.json:2](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/records/status.json:2>), [e2_full20_complete_exit.json:3](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/records/e2_full20_complete_exit.json:3>), [e2_full20_miss_exit.json:3](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/records/e2_full20_miss_exit.json:3>), [evaluate_model.py:80](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:80>), [evaluate_model.py:128](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:128>), [steps.jsonl:1](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/training/steps.jsonl:1>), [steps.jsonl:50000](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/training/steps.jsonl:50000>), [completed.json:1](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/training/completed.json:1>), [comparison.json:797](<C:/Users/gb/projects/Miss-RGBT/reports/e2_20261016_v1/comparison.json:797>), [RESULTS.md:7](<C:/Users/gb/projects/Miss-RGBT/reports/e2_20261016_v1/RESULTS.md:7>), [checkpoint_metadata.json:438](<C:/Users/gb/projects/Miss-RGBT/reports/e2_20261016_v1/checkpoint_metadata.json:438>), [checkpoint_metadata.json:448](<C:/Users/gb/projects/Miss-RGBT/reports/e2_20261016_v1/checkpoint_metadata.json:448>), [collect_e2_closed_20261016.py:49](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/collect_e2_closed_20261016.py:49>), [collect_e2_closed_20261016_v2.py:49](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/collect_e2_closed_20261016_v2.py:49>), [collect_e2_closed_20261016_v2.py:54](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/collect_e2_closed_20261016_v2.py:54>), [e2_closed_collection_transport_20261016.json:2](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/e2_closed_collection_transport_20261016.json:2>), [e2_closed_collection_transport_20261016_v2.json:2](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/e2_closed_collection_transport_20261016_v2.json:2>), [EXPERIMENT_TRACKER.md:29](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/EXPERIMENT_TRACKER.md:29>), [HANDOFF.md:5](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/HANDOFF.md:5>).

**D. Executed versus unused paths — PASS**

All five claimed metric implementations are reached: evaluate_model registers the raw boxes and dispatches by key to LasHeR PR/NPR/SR and RGBT234 MPR/MSR, then writes scalar and curve outputs. The matching final stdout metrics, saved curves and independent raw-data reconstruction confirm result connectivity. Generic PR/SR, GTOT classes and plotting utilities in the package were not called for these results and support no additional evaluation claim.

E2 follows the dual complete/missing forward branch with repair=false. Its training_loss computes both response-KD diagnostics and the reliability gate but multiplies forward/reverse terms by configured 0.0. All 50,000 logged losses equal float32(task_c+task_m) exactly. Forward KD is nonzero in 50,000 rows; gated reverse KD and reliability are nonzero in 23,923 rows. Those diagnostic values do not demonstrate KD optimization or E5 efficacy.

Logs store only the last accumulation microbatch, not the full effective-batch mean. E0-FT instead executes only complete task supervision. E3 enables forward KD 0.1 with reverse 0; E4 additionally enables LocalRepair. Complete-view model outputs are detached training targets for KD, not evaluation GT.

The shared tracker uses one missing-aware inference forward, explicit template availability, the same Hann window and the same dynamic-template rule: every 5 effective track calls with confidence>0.65. Both-missing skips do not advance its track counter. No later BN or template-control source was used by this audit.

Evidence: [evaluate_model.py:133](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:133>), [lasher_dataset.py:80](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/dataset/lasher_dataset.py:80>), [lasher_dataset.py:94](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/dataset/lasher_dataset.py:94>), [lasher_dataset.py:115](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/dataset/lasher_dataset.py:115>), [rgbt234_dataset.py:83](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/dataset/rgbt234_dataset.py:83>), [rgbt234_dataset.py:116](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/dataset/rgbt234_dataset.py:116>), [train.py:89](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/train.py:89>), [train.py:107](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/train.py:107>), [objectives.py:41](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/objectives.py:41>), [run_config.json:6](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/training/run_config.json:6>), [tracker.py:36](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/tracker.py:36>), [tracker.py:53](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/tracker.py:53>).

**E. Scope, matched controls and causal claim ceiling — WARN**

Supported scope: one formal seed 2027, fixed final epoch 20 after an additional 20 epochs from the supplied E0 epoch 20 checkpoint, 400,000 sampled tuples, effective batch 8 (microbatch 4 x accumulation 2),50,000 optimizer updates, and complete/fixed-mask OPE on 245 LasHeR plus 234 RGBT234 sequences. These are sampled tuples, not 400,000 distinct examples or full unique trajectory coverage. No multi-seed uncertainty or unseen-test independence is established.

E2 versus FT: configs differ only in stage/output, all 92 training source hashes and all 106 evaluation source hashes match. Both use the same initialization, optimizer settings, logical tuple/update budgets, box-head-only BN-stat freeze, inference/template policy and fixed-final selection. E2 adds the missing-view task to the complete loss without dividing the sum by 2, samples masks and runs a second view. Thus the treatment includes extra supervision, loss scale, stochastic operation pattern and BN exposure; it is not an equal-forward-compute or pure-mechanism-isolation control.

E2 has 100,000 microbatches and 200,000 model-forward invocations, each processing 3 search frames:600,000 fusion calls or 2.4 million per-tuple search-view exposures. FT has 100,000 model-forward invocations/300,000 fusion calls/1.2 million such exposures. These are source-and-budget-derived counts, not measured FLOPs, timing parity or distinct data counts. E3 uses the same dual-view counts as E2; E4 adds repair operations and 398,466 parameters.

E3 versus E2: the 92-file training manifests differ only in train.py. The changed file adds the inactive FT branch/stage allowlist/import; after specializing away that branch, the active train AST is identical. Shared model/objective/data/tracker/YAML bytes are identical. The substantive config change is forward_weight 0->0.1; stage/output/source-version-file changes are provenance. Evaluation differences add selectable GPU mapping; actual E2/E3 use[2,3] and 8 workers with identical tracking/scoring logic. The four old 103-file evaluation manifests match their actual archived bytes. This supports a qualified single-seed marginal forward-KD comparison.

E4 versus E3: the actual training/evaluation source manifests match and configs differ only in repair true/false, stage and output. The residual is directly gated at missing positions; complete-input training can still alter shared parameters. This supports the observed repair-on/off treatment comparison, not a capacity/compute-matched module mechanism claim or a general benefit.

Both initial search-microbatch hashes match across FT/E2/E3/E4. Full template/mask/sample histories were not recorded, so exact tuple-by-tuple stochastic matching throughout training is not proven. The common BN policy does not equal identical BN state evolution, and head BN eval does not freeze affine parameters.

E2-FT improves all five Miss metrics, but complete-input E2-FT is positive only for LasHeR PR(+0.029960 pp) and negative for the other four. E3-E2 improves all five complete metrics but lowers all three LasHeR-Miss metrics and raises the two RGBT234-Miss metrics. E4-E3 raises LasHeR-Miss PR/NPR/SR and RGBT234-Miss MPR, lowers RGBT234-Miss MSR, and lowers every complete metric. All five E2 complete metrics remain below original E0. Only LasHeR-Miss SR exceeds its strict target:1/4 targets. General positive transfer, approximately+2 pp complete-input improvement, BN causality and three effective modules are unsupported.

The task documents disclose test-failure-driven research decisions, unresolved original-E0 val 97 exposure and no finalized independent RGBT234 development protocol. Preserve those qualifications. A later all-BN-freeze treatment cannot isolate missing-training positive transfer by comparing solely against FT with the original BN policy; a matched complete-only control is necessary for that causal claim.

Evidence: [run_config.json:18](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/training/run_config.json:18>), [train.py:30](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/train.py:30>), [train.py:56](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/train.py:56>), [train.py:89](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/train.py:89>), [objectives.py:47](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/objectives.py:47>), [model.py:34](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/model.py:34>), [cadtrack.py:27](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/lib/models/cadtrack/cadtrack.py:27>), [head.py:7](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/lib/models/layers/head.py:7>), [train.py:84](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/full20_closed_raw/source/miss_rgbt/train.py:84>), [run_config.json:9](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/full20_closed_raw/training/e3_full20_seed2027/run_config.json:9>), [run_config.json:6](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/full20_closed_raw/training/e4_full20_seed2027/run_config.json:6>), [model.py:11](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/model.py:11>), [model.py:20](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/model.py:20>), [vit_care.py:359](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/lib/models/cadtrack/vit_care.py:359>), [evaluate_model.py:61](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:61>), [evaluate_model.py:118](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:118>), [RESULTS.md:18](<C:/Users/gb/projects/Miss-RGBT/reports/e2_20261016_v1/RESULTS.md:18>), [NEXT_EXPERIMENT_PRIORITY_20261015.md:23](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/docs/NEXT_EXPERIMENT_PRIORITY_20261015.md:23>), [NEXT_EXPERIMENT_PRIORITY_20261015.md:46](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/docs/NEXT_EXPERIMENT_PRIORITY_20261015.md:46>), [NEXT_EXPERIMENT_PRIORITY_20261015.md:55](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/docs/NEXT_EXPERIMENT_PRIORITY_20261015.md:55>).

**F. Evaluation classification — PASS**

E2 and the bound FT/E0-availability/E0/E3/E4 benchmark comparisons use recorded-video tracking predictions and external dataset annotations: real_gt. Published sensor-deletion masks simulate missing inputs but do not replace GT or make the benchmark simulation_only. Training KD is model-generated supervision; it is not the reference used to score tracking. Mechanical CPU/source checks are not new tracking performance.

Evidence: [evaluate_model.py:42](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:42>), [evaluate_model.py:83](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:83>), [evaluate_model.py:141](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/scripts/evaluate_model.py:141>), [objectives.py:34](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_closed_raw/source/miss_rgbt/objectives.py:34>).

**Verified E2 metrics and treatment differences**

Percentages and percentage points; displayed values rounded to six decimals. Full precision and all source values are retained in the JSON report.

| Dataset / condition | Metric | E2 | E2 − FT | E3 − E2 | E4 − E3 |
|---|---|---:|---:|---:|---:|
| LasHeR | PR | 74.444219 | +0.029960 | +0.194469 | -0.252410 |
| LasHeR | NPR | 70.625478 | -0.033508 | +0.380788 | -0.370632 |
| LasHeR | SR | 59.251822 | -0.063694 | +0.290091 | -0.209629 |
| RGBT234 | MPR | 89.731440 | -0.746729 | +0.633535 | -0.474880 |
| RGBT234 | MSR | 65.181523 | -0.564314 | +0.396101 | -0.155487 |
| LasHeR-Miss | PR | 68.369839 | +7.164232 | -0.189048 | +1.036892 |
| LasHeR-Miss | NPR | 63.463823 | +6.465145 | -0.119295 | +0.937205 |
| LasHeR-Miss | SR | 54.431736 | +5.453639 | -0.069627 | +0.685131 |
| RGBT234-Miss | MPR | 84.771986 | +6.337192 | +0.313664 | +0.068810 |
| RGBT234-Miss | MSR | 60.542639 | +4.494876 | +0.330126 | -0.066192 |

Only LasHeR-Miss SR clears its strict threshold (54.4317357894 > 54.4). LasHeR-Miss PR 68.3698390449 does not exceed 69.3; RGBT234-Miss MPR/MSR 84.7719860973/60.5426388423 do not exceed 88.1/64.3. The complete-input goal remains unmet.

**Bound prior controls**

Old numerical reconstructions were not rerun: no discrepancy required it. All three comparison files match the hashes recorded in their prior audit inputs; all 293 supplemental files are also bound to prior audit hash entries. New E2–FT, E3–E2 and E4–E3 arithmetic and executed-source/config comparisons were checked independently.

| Prior audit | SHA256 of bound audit JSON |
|---|---|
| [e0_ft_20261016_v1](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/reports/e0_ft_20261016_v1/EXPERIMENT_AUDIT.json>) | `24cbf471bf3a599dedc0f1beab5dc76ae2e179f10631015e07c682d94f806f86` |
| [e0_availability_20261015_v1](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/reports/e0_availability_20261015_v1/EXPERIMENT_AUDIT.json>) | `a3353cdd4446289746b056d68669aee35318b261384ff84d4b9cdd237544ea00` |
| [full20_comparison_20261013_v1](<C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e2_audit_context_20261016/reports/full20_comparison_20261013_v1/EXPERIMENT_AUDIT.json>) | `7c83084fe7504f921d31cf9fb077bb9aebd7d0373f6d1cff1e2fe4e7558b7d79` |

**Claims and actions**

- **Protocol provenance ceiling:** Describe the exact held/public rgbt1.0.1 protocol and its invalid-GT/rounding rules. Do not label author-toolbox or SCDT equivalence established.
- **Checkpoint witness ceiling:** Keep hash/reload/BN statements explicitly attributed to parent CPU metadata; this review did not inspect the original binary.
- **Single-seed and unequal computational exposure:** Keep E2-FT and E3/E4 effects tied to this final-checkpoint treatment and logical budget. No general positive-transfer, BN-causality or three-module claim; use a matched complete-only control after changing BN/template policies.
- **Historical publication snapshots:** Update live tracker/README/HANDOFF with closed E2 results and this audit; retain sealed17:03 snapshots and failed collection transport unchanged.
- **Development independence unresolved:** Disclose test-informed research and unresolved original-E0 exposure; do not present reused test failures as independent development validation.

**Input preservation and execution boundary**

All **4,062 original sealed files** (128,811,226 bytes) and **293 supplemental files** (73,494,651 bytes) match before/after hashes: **4,355 experimental files, 202,305,877 bytes**, zero changes. The original seal itself is unchanged. The machine report enumerates all 4,355 path/hash pairs plus the original seal as a 4,356th integrity entry.

- Original `AUDIT_INPUT_SEAL.json`: `b074bf34595630223bffc83fe0cfaf4e6e8fd85b7365acb5500630b78b50aa75`.
- Supplemental review seal (292 files): `22b3f65053fc6df5a5cbf5aa3289d58f2e029fc2966f69837ec6cbc9b593facb`.
- Split-archive review seal (1 file): `e9b18690827e5920e71764ce0c402f3e71d843e153b5edc002c1edc065e84a56`.

Numeric execution used the requested local route: explicit uv-managed Python 3.12 launched `E:/anaconda/python.exe` as a subprocess; actual scorer runtime was Python 3.7.6 / NumPy 1.21.6, elapsed 35.493 seconds. The reviewer formulas used NumPy and did not import the producer scorer or any neural model. No installs, external model calls, SSH/network, GPU/process queries, training, neural replay or nested agents occurred. No current-main NN source was used.

The two public audit reports and private review helpers/records are the only writes. Full numeric outputs, per-sequence curves, source diffs, binding checks and before/after seals are retained in [the private review directory](<C:/Users/gb/.codex_tmp/e2_integrity_audit_reviewer_20261016>). The parent owns the outer delegation tool trace.
