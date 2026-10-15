# Full20 experiment integrity audit

**Overall verdict: WARN.** Numerical evidence passes; scientific/provenance judgment remains provisional.

Date: 2026-10-15. Fresh reviewer: /root/full20_integrity_audit. Requested model/reasoning: gpt-6-astra / max; same-family, acceptance_status=provisional. The spawn request is configuration evidence, not independent backend attestation. Trace: .aris/traces/experiment-audit/2026-10-15_run01.

No SSH, network, GPU/process polling, neural execution or checkpoint load was performed. Only this Markdown report and the accompanying JSON were written. Frozen RAW/source was audited; future-control worktree implementation was not substituted.

The independently recomputed **20 final metrics and 187,932 curve cells have exactly zero error**. Both actual E0 control archives and four epoch05 groups were also recomputed: **50 metrics, 469,830 curve cells, 4,790 sequence runs and 3,373,530 prediction rows** total. Actual-source execution and separately written numeric formulas agree exactly at every cell. All 11,869 audited local input files remained unchanged.

**E3 passes 0/4 and E4 passes 1/4 strict Miss thresholds; all ten complete-input metrics are below E0.** No three-component effectiveness, E5 performance, BN causal explanation or multi-seed stability is established.

## Evidence aliases

- PROJECT = C:/Users/gb/projects/Miss-RGBT
- RAW = C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/full20_closed_raw
- PKG = C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt
- BASE_RAW = C:/Users/gb/.codex_tmp/cadtrack_miss_ep20_20261013
- FIRST5_RAW = C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_evaluation_raw
- COLLECTION = C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013
- ORIGINAL_AUDIT_INPUTS = C:/Users/gb/projects/Miss-RGBT/.aris/traces/experiment-audit/20261013/temporary

References below use these exact aliases and original 1-based line numbers. The JSON embeds individual hashes, archive member hashes, recomputation results and checker sources.

## A-F verdicts

| Check | Verdict | Deterministic evidence |
|---|---|---|
| A. Ground-truth provenance and coverage | WARN | PASS |
| B. Score normalization and raw calculations | PASS | PASS |
| C. Existence, identity, training and reported results | WARN | PASS |
| D. Called metrics versus unused helpers and diagnostics | PASS | PASS |
| E. Scientific scope, causality and document state | WARN | PASS |
| F. Evaluation type | PASS | PASS |

Deterministic acceptance covers only the specified finite-file, source identity, hash and arithmetic checks. It is not cross-family semantic acceptance, original binary verification or official upstream authentication.

## Recomputed final scores

Percentages; fixed final epoch20; seed2027.

| Model | LasHeR PR / NPR / SR | RGBT234 MPR / MSR | LasHeR-Miss PR / NPR / SR | RGBT234-Miss MPR / MSR |
|---|---|---|---|---|
| E0 | 75.636503 / 71.772837 / 60.125714 | 90.677134 / 66.845228 | 63.472507 / 58.581851 / 50.583149 | 82.794815 / 59.574435 |
| E3 | 74.638688 / 71.006266 / 59.541913 | 90.364976 / 65.577624 | 68.180791 / 63.344528 / 54.362109 | 85.085650 / 60.872765 |
| E4 | 74.386278 / 70.635633 / 59.332284 | 89.890096 / 65.422137 | 69.217683 / 64.281733 / 55.047239 | 85.154460 / 60.806573 |

Evidence: PROJECT/reports/e3_full20_complete/metrics.json:2-13; PROJECT/reports/e3_full20_miss/metrics.json:2-13; PROJECT/reports/e4_full20_complete/metrics.json:2-13; PROJECT/reports/e4_full20_miss/metrics.json:2-13. Project reports match RAW byte-for-byte and were independently recomputed from raw boxes. E0 was recomputed from BASE_RAW/baseline_raw_results.tar.gz and COLLECTION/e0_complete_raw_results_20261013.tar.gz.

| Model | Miss metric | Strict target | Value | Target minus value (pp) | Exceeded |
|---|---|---:|---:|---:|---|
| E3 | LasHeR-Miss PR | >69.3 | 68.180790739 | +1.119209261 | No |
| E3 | LasHeR-Miss SR | >54.4 | 54.362108883 | +0.037891117 | No |
| E3 | RGBT234-Miss MPR | >88.1 | 85.085650422 | +3.014349578 | No |
| E3 | RGBT234-Miss MSR | >64.3 | 60.872765069 | +3.427234931 | No |
| E4 | LasHeR-Miss PR | >69.3 | 69.217682836 | +0.082317164 | No |
| E4 | LasHeR-Miss SR | >54.4 | 55.047239396 | -0.647239396 | Yes |
| E4 | RGBT234-Miss MPR | >88.1 | 85.154460187 | +2.945539813 | No |
| E4 | RGBT234-Miss MSR | >64.3 | 60.806573360 | +3.493426640 | No |

E4-E3 is +1.036892097 / +0.937205467 / +0.685130513 pp on LasHeR-Miss PR/NPR/SR; +0.068809765 / -0.066191709 pp on RGBT234-Miss MPR/MSR. All five complete-input E4-E3 differences are negative.

## Training and source evidence

| Model | Epochs / updates | Log rows | Logged epoch20 mean loss | Loop seconds | Original final checkpoint SHA256 (recorded) |
|---|---|---:|---:|---:|---|
| E3 | 20 / 50000 | 50000 | 5.752447447 | 103808.411170 | 01fa99822ab8936c02cd1f6223d0732ab450f2c942b199928b6de65903d87d5d |
| E4 | 20 / 50000 | 50000 | 5.728212461 | 100178.456448 | c1b1ba0ff60b854e07f970c171a44fc543b4f4349210295632309519317aa99a |

Each epoch has 2,500 updates. All logged numeric fields are finite and elapsed times monotonic. All 240 published epoch means match. Reconstructing logged loss from printed float32 components differs by at most 1.884e-6; exact log values were preserved. Logs describe only the last accumulation microbatch, not a full-population mean. E4 controller exit registration follows the E3 wait and is not a precise independent E4 completion timestamp (RAW/source/scripts/run_full_training.py:66-71).

The 92-file training manifests match Git 4c473fa5291b42cdd1969ea3d74161e92e142445. Across 104 archived source/YAML files, only scripts/validate_focal_normalization.py differs: Git d3658557019c1eea4dce7476dd49572170386cab1fce59ae17ce244ee4055c1c versus deployed cef46a7788dc7a055b0cff6cb7661278864497435d06bc7a677507b805627f0d. This auxiliary validator is outside the evaluated call path. The 87 baseline source/config files separately match Git 370f215814320d048a5c33974b64052126a8a1fd.

## Detailed checks

### A. Ground-truth provenance and coverage: WARN

- All evaluated targets are dataset annotation text files read by the recorded rgbt toolkit; none is derived from model predictions. The complete-view prediction is a detached training distillation target, not evaluation ground truth.
- Both original published-mask JSON files match the inventory SHA256. All 713 evaluated GT files, all 1,967 package/source/metadata files, the 245/234 sequence sets and 220,703/116,650 frame counts verify directly. The archived old E0 toolkit matches all 1,967 current package files byte-for-byte.
- Both-missing frames remain scored. The two final Miss runs each contain 52,493 noninitial both-missing frames with exact previous-box holds. All 479 sequence initial boxes per evaluation equal the annotation initialization. Inventory order and toolkit sequence-list order differ; membership is equal and all curves were compared in the toolkit's actual order.
- The exact installed-toolkit protocol is reproduced. An independently authenticated upstream official evaluator release/commit and annotation release manifest were not supplied; byte equality between local runtime copies cannot establish that stronger equivalence.
- LasHeR has 1,260 rows across 52 sequences with at least one GT coordinate <=0: 533 have nonpositive width/height, 727 have positive dimensions but nonpositive x/y. The toolkit makes these precision successes and success-rate failures without excluding them. Their fixed macro precision contribution is 0.5568053864558152 percentage points under this protocol. advancedredcup is fully present with 1,411 GT/prediction rows.

Evidence: RAW/source/scripts/evaluate_model.py:29-58; RAW/source/scripts/evaluate_model.py:79-99; PKG/dataset/basedataset.py:9-24; PKG/dataset/lasher_dataset.py:12-20; PKG/dataset/rgbt234_dataset.py:13-20; PKG/metrics/metrics.py:258-272; PKG/metrics/metrics.py:301-315; PKG/metrics/metrics.py:343-357; PROJECT/reports/e0/inventory.json:6-19; PROJECT/reports/e0/inventory.json:423; PROJECT/reports/e0/inventory.json:3206-3219; RAW/source/miss_rgbt/objectives.py:29-49.

### B. Score normalization and raw calculations: PASS

- No performance denominator is a model's own maximum, mean or scale. PR/MPR divide threshold-hit counts by sequence frame counts and macro-average sequences. NPR divides center coordinates by dataset GT width/height plus 1e-8 and reports threshold 0.20. SR/MSR use the mean over 21 success thresholds (0 through 1), not a separately integrated trapezoidal AUC.
- MPR takes the smaller center error to the two real modality annotations; MSR takes the larger IoU. IoU uses geometric union area. These are documented benchmark operations, not self-normalization.
- Prediction boxes are loaded as float32 and rounded to integers; LasHeR GT is also loaded through the rounded loader. Initial LasHeR predictions are set to GT; nonpositive predicted widths/heights would be replaced by the preceding box, but none occurs in the audited raw predictions. Precision uses <=; success uses >. Initialization and missing frames stay in all denominators.
- Reported percentages are score*100, deltas are percentage-point subtractions, and the separate relative-change statistic uses fixed E0 as the denominator while retaining raw values.
- Actual scoring-source execution and separately written NumPy geometry/threshold calculations are compared at every curve cell, with exact equality. Training focal normalization uses the GT heatmap positive count and batch size; masked reverse diagnostic averaging uses sample count.

Evidence: PKG/dataset/basedataset.py:19-24; PKG/utils.py:30-46; PKG/utils.py:117-184; PKG/metrics/metrics.py:23-44; PKG/metrics/metrics.py:57-80; PKG/metrics/metrics.py:241-277; PKG/metrics/metrics.py:285-360; RAW/source/scripts/evaluate_model.py:141-145; PROJECT/analysis/build_full20_comparison.py:101-110; PROJECT/analysis/diagnose_full20_results.py:67-75; RAW/source/miss_rgbt/objectives.py:18-26.

### C. Existence, identity, training and reported results: WARN

- Two actual closed training directories each contain 50,000 consecutive update records, exactly 2,500 per epoch for epochs 1-20; all logged numeric fields are finite and elapsed seconds strictly increase. Both configs match the frozen project configs, recorded checkpoint metadata and comparison records. The 240 epoch means match summary.json exactly.
- Each training source manifest has 92 files that match archived bytes and Git 4c473fa5291b42cdd1969ea3d74161e92e142445. The 104-file collected source set (103 Python files plus YAML) is byte-verified. Its sole Git mismatch is the disclosed, unevaluated scripts/validate_focal_normalization.py; active executed source has not been substituted with the future-control worktree.
- Four final evaluation groups each contain precisely 479 unique sequence receipts and 337,353 finite box rows; per-sequence receipt files match their aggregate receipts and box SHA256. Project report copies match archived bytes. All 3,974 archive file members match the extracted tree and its file set.
- All 20 final metric values, all final curves, 20 E0 deltas, 20 budget deltas, 20 relative changes, 40 CSV rows, eight displayed result rows and eight strict target comparisons verify. Both E0 controls were recomputed from their actual raw archives; epoch05 results are additionally recomputed to check the budget comparison.
- Recorded runtime evidence contains six matching successful launch/exit pairs, a completed controller and final training tails matching the full logs. This audit reads those closed records and makes no new process or GPU-state claim.
- Original E0/E3/E4 checkpoint binary bytes are not in the supplied local archives. Final checkpoint properties and BN buffers are independently checked for consistency of the existing server-generated metadata, but were not reloaded or independently hashed from original binaries by this reviewer. Of the comparison's 1,951 input hashes, 1,949 local files can be rehashed; the only missing entries are the two final checkpoint files.
- Training logs record the last accumulation microbatch at each update. Two initial search-microbatch hashes match across E3/E4, which does not prove all templates, masks or every later sample are identical. Original train image/init.txt bytes and per-sample execution traces are unavailable locally.

Evidence: RAW/training/e3_full20_seed2027/steps.jsonl:1; RAW/training/e3_full20_seed2027/steps.jsonl:50000; RAW/training/e4_full20_seed2027/steps.jsonl:1; RAW/training/e4_full20_seed2027/steps.jsonl:50000; RAW/source/miss_rgbt/train.py:58-72; RAW/source/miss_rgbt/train.py:77-120; RAW/source/miss_rgbt/tracker.py:14-21; PROJECT/reports/full20_comparison_20261013_v1/source_identity.json:2-11; PROJECT/reports/full20_comparison_20261013_v1/comparison.json:223-441; PROJECT/reports/full20_comparison_20261013_v1/checkpoint_metadata.json:2-40; PROJECT/reports/full20_comparison_20261013_v1/checkpoint_metadata.json:447-485; PROJECT/reports/full20/runtime_20261015_1100.json:237-364; PROJECT/docs/FULL20_RESULTS.md:23-27; COLLECTION/collect_full20_results.py:20-85.

### D. Called metrics versus unused helpers and diagnostics: PASS

- The full20 evaluator constructs LasHeR/RGBT234, registers all prediction files and calls each metric dynamically by key, then saves its actual scalar and curve. The E0 complete and Miss evaluators have the corresponding active scoring loops. Dataset methods dispatch to PR_LasHeR, NPR, SR_LasHeR, MPR and MSR; all five functions were executed on actual raw data by this audit.
- The legacy generic PR/SR and GTOT metrics, plotting functions and unused model helper modules are not evidence of additional evaluated datasets or deployed components. No claimed final metric relies on an unused scorer.
- Forward response KL has weight 0.1. Reverse KL and reverse_fraction are computed and logged, but reverse_weight is 0 in both completed configurations, so no reverse-feedback efficacy follows from nonzero diagnostic values.
- LocalRepair is enabled only in E4; its direct residual multiplier is zero at observed positions. Complete-input parameter changes still occur through shared training, and the repair computations are not themselves skipped on complete inputs. No deployed PointFlow/CDA/Mamba efficacy can be inferred from legacy files or the stale ViT factory docstring.
- The single divergent auxiliary focal-validation file is not imported or called by the training/evaluation path. CPU/native mechanical validation and analysis scripts are kept distinct from benchmark performance evidence.

Evidence: RAW/source/scripts/evaluate_model.py:128-147; RAW/source/scripts/evaluate_complete.py:153-174; PROJECT/reports/e0/evaluate_miss.py:151-172; PKG/dataset/lasher_dataset.py:63-121; PKG/dataset/rgbt234_dataset.py:53-121; RAW/source/miss_rgbt/objectives.py:41-53; PROJECT/configs/e3_full20.json:9-11; PROJECT/configs/e4_full20.json:9-11; RAW/source/miss_rgbt/model.py:20-24; RAW/source/miss_rgbt/model.py:67-70; RAW/source/lib/models/cadtrack/vit_care.py:45-63; RAW/source/lib/models/cadtrack/vit_care.py:354-361; PROJECT/reports/full20_comparison_20261013_v1/source_identity.json:4-10.

### E. Scientific scope, causality and document state: WARN

- The supported scope is two datasets, complete and fixed published-mask conditions, one seed (2027), two separately trained configurations and a fixed final epoch20 checkpoint per model. Each model received an additional 20-epoch/400,000-sampled-tuple training budget from the supplied E0 epoch20 starting point. It is not a multi-seed stability result, a complete CADTrack-paper reproduction or an unseen-test claim.
- E3 passes 0/4 strict Miss thresholds; E4 passes 1/4 (LasHeR-Miss SR only). All ten complete-input E3/E4 values are below E0, so the requested approximately +2 pp full-input gain is unsupported. E4-E3 is positive on the three LasHeR-Miss metrics and RGBT234-Miss MPR, but negative on RGBT234-Miss MSR and every complete-input metric.
- E3-E0 changes availability handling, extra supervised training, dual complete/missing task losses and forward KD together; it cannot isolate KD. E4 adds 398,466 parameters. Parameter/compute controls, E0-availability, complete-only E0-FT, E2 without KD/repair, E5 feedback and relevant multi-seed ablations are not results of this campaign.
- Epoch05 and epoch20 are independent fixed-budget runs from E0, with LR drop epochs 5 and 10 respectively; their difference is descriptive, not a pure duration effect or grounds to choose the best test checkpoint. All four conditions must remain attached to the same model/config/checkpoint.
- Source proves fix_bn covers only box_head and leaves fusion BatchNorm in training mode. The metadata's 37500->637500 fusion counter agrees with 50000 updates*2 microbatches*3 search frames*2 C/M forwards; 36 head running buffers are unchanged. These facts do not prove BN caused accuracy regression or that BN affine parameters were frozen.
- Train882/val97/test245 sequence names are disjoint in the archived/Git split files. Actual full20 source manifests omit data_specs text files, and sampled tuple coverage is not logged. Test failure analysis has already informed later research direction; future dataset-specific choices need an independent development protocol, including RGBT234, as current documents acknowledge.
- Current HANDOFF top summary and EXPERIMENT_TRACKER correctly mark full20 complete; dated and explicitly historical observations are not current-state failures. Some later undated HANDOFF sections still say formal training has not started (119, 121, 125), and call earlier metrics official (142); these should be date-labeled or qualified. NEXT_EXPERIMENT_PRIORITY line 5 is a plan step now fulfilled, not evidence of a pending run.

Evidence: PROJECT/configs/e3_full20.json:2-32; PROJECT/configs/e4_full20.json:2-32; PROJECT/reports/full20_comparison_20261013_v1/RESULTS.md:9-50; PROJECT/HANDOFF.md:3-25; PROJECT/HANDOFF.md:27-58; PROJECT/HANDOFF.md:119-142; PROJECT/HANDOFF.md:158-168; PROJECT/HANDOFF.md:203; PROJECT/EXPERIMENT_TRACKER.md:14-20; PROJECT/docs/NEXT_EXPERIMENT_PRIORITY_20261015.md:5-10; PROJECT/docs/NEXT_EXPERIMENT_PRIORITY_20261015.md:14-34; RAW/source/miss_rgbt/train.py:30-33; RAW/source/miss_rgbt/train.py:50-65; RAW/source/miss_rgbt/train.py:88-89; RAW/source/lib/models/cadtrack/cadtrack.py:27; RAW/source/lib/models/cadtrack/cadtrack.py:73-90; RAW/source/lib/models/layers/head.py:7-20; PROJECT/reports/full20_comparison_20261013_v1/checkpoint_metadata.json:414-443; PROJECT/reports/full20_comparison_20261013_v1/checkpoint_metadata.json:859-888.

### F. Evaluation type: PASS

- All final, epoch05 and E0 benchmark scores are real_gt: actual tracking predictions scored against supplied dataset annotation files. Published sensor-deletion masks simulate missing observations but do not replace real GT or turn these recorded-video benchmark scores into simulation_only.
- Teacher-response KL is model-derived training supervision, not the benchmark's reference. Synthetic focal arithmetic tests and a single-batch gradient/update probe are implementation diagnostics, not synthetic tracking performance, human evaluation or proof of E5 effectiveness.

Evidence: RAW/source/scripts/evaluate_model.py:42-49; RAW/source/scripts/evaluate_model.py:128-145; PROJECT/reports/e0/evaluate_miss.py:64-85; RAW/source/miss_rgbt/objectives.py:29-49; PROJECT/HANDOFF.md:152; PROJECT/EXPERIMENT_TRACKER.md:15-18.

## Explicitly unavailable original artifacts

- **original_binary**: /home/gaob/Miss-RGBT/checkpoints/e3_full20_seed2027/epoch20.pth.tar. No original binary is present in the audited local archives/expected project checkpoint path; this reviewer performed no checkpoint load or binary rehash. Epoch20/50000/config/state/BN properties are recorded server-extracted metadata.
- **original_binary**: /home/gaob/Miss-RGBT/checkpoints/e4_full20_seed2027/epoch20.pth.tar. Same limitation as E3.
- **original_binary**: /data2/zhouy/CADTrack-main/baseline/checkpoints/train/cadtrack/cadtrack/CADTrack_ep0020.pth.tar; alias /home/gaob/Miss-RGBT/checkpoints/pretrained/CADTrack_ep0020.pth.tar. E0 original binary is not in the supplied local audit archives. The report does not independently attest the actual binary, original baseline training history or upstream pretraining lineage.
- **upstream_official_artifact**: (No authenticated artifact path/version was supplied.). No authenticated original benchmark-author evaluator release/commit, authoritative annotation release manifest or upstream equivalence test corpus was supplied. Both old and new installed rgbt package copies are available and identical; those are not independent upstream authenticity evidence.
- **original_dataset_images**: /data2/wangwj/dataset/LasHeR-0428/trainingset; /data2/wangwj/dataset/LasHeR-0428/testingset/testingset; /data2/wangwj/dataset/RGBT234/RGB_T234. Image bytes and original LasHeR training init.txt annotation bytes were not available among audited inputs. Scoring GT and masks are available and verified. Original advancedredcup visible.txt/infrared.txt source annotations were not re-read; its 1411-row toolkit GT is available.
- **runtime_training_split_and_sample_attestation**: lib/train/data_specs/lasher_train.txt; lib/train/data_specs/lasher_val.txt. Git objects and the baseline source archive supply matching 882/97 split files, but the full20 start-time source manifest records Python files only. It does not hash these text split files or every training sample/mask/template. Equal first two search-microbatch hashes establish only that limited equality.
- **upstream_pretraining_binary**: (No authenticated artifact path/version was supplied.). Original DropMoE checkpoint/source lineage was user-described but not supplied as a separately authenticated artifact. Current training=False construction bypasses the residual OSTrack YAML pretraining field and directly strict-loads recorded E0; this does not prove original E0's initial pretraining lineage.

The actual old E0 toolkit, scoring annotations, both original masks, E0 raw predictions and curves were found locally and verified. These are not unavailable evidence. The two absent final checkpoint binaries are the only unrehashable entries among comparison.json 1,951 input hashes.

## Claim impact

- **C1 / supported_with_provenance_limit**: E3/E4 finished the fixed additional 20-epoch/50000-update runs and four full evaluations. Continuous logs, source/config identity, exits, 1916 receipts and 1349412 rows support the recorded run; original checkpoint binaries were not independently rehashed.
- **C2 / supported**: Reported final metrics, E0 deltas, budget deltas, strict target decisions and curves are numerically reproducible. All 50 metrics and 469830 curve cells across final, E0 and epoch05 groups agree exactly with actual-source and independently written numeric recomputation.
- **C3 / supported**: The exact installed scoring protocol is shared by E0 and E3/E4. 1967 old/new runtime package files match byte-for-byte; all 713 GT hashes and two mask hashes match.
- **C4 / unsupported**: The run is independently proven equivalent to the benchmark authors' official evaluator and original annotation release. No independently authenticated upstream release/commit/manifest or conformance evidence was supplied.
- **C5 / unsupported**: The complete-input approximately +2 pp objective or all four strict Miss targets are achieved. All ten complete-input metrics are below E0. E3 meets 0/4; E4 meets 1/4.
- **C6 / unsupported**: E3-E0 isolates forward-KD efficacy, or three modules are effective. Input handling, additional training and dual-view supervision are confounded; E5 is disabled and controls are absent.
- **C7 / supported_with_qualifier**: Adding E4 repair improves LasHeR-Miss in this seed and budget. E4-E3 PR/NPR/SR = +1.036892097/+0.937205467/+0.685130513 pp. Capacity/compute and multiple seeds remain untested.
- **C8 / unsupported**: E4 repair reliably improves both datasets and complete inputs. RGBT234-Miss MSR falls 0.066191709 pp; all complete-input E4-E3 metrics decrease.
- **C9 / supported_as_recorded_metadata_and_source**: Fusion BN statistics updated while head running buffers remained unchanged. Source and metadata agree on 600000 additional fusion calls and 36 unchanged head buffers per model. Raw tensor buffers were not available to this reviewer.
- **C10 / unsupported**: BN updating caused the complete-input regression. Counter/statistic changes alone cannot establish causality; no matched BN intervention result exists.
- **C11 / unsupported**: A pure duration effect is measured by epoch20 minus epoch05. The runs also differ in LR drop epoch and total exposure; test checkpoint selection would change the fixed-final protocol.
- **C12 / unsupported**: These test sets are wholly unseen development-independent validation. Documents disclose that test failure analysis informed research direction; RGBT234 lacks a specified independent development protocol.

## Action items

1. Retain and cite the fixed final-epoch20 results with the existing single-seed, real_gt, same-family/provisional and installed-toolkit qualifiers; retain all negative metrics and all four conditions.
2. If official-evaluator equivalence is claimed, add a fixed authenticated author implementation/release and conformance evidence for rounding, invalid GT, initialization, sequence averaging and threshold conventions. Keep the existing recorded scores unchanged.
3. Stronger independent checkpoint binding requires direct read-only hashes of the original E0/E3/E4 binaries, with checks against the recorded SHA256 values. Current metadata is evidence of the prior server read, not a new local tensor verification.
4. Before component/causal/stability claims, complete the planned E0-availability, E0-FT, E2, capacity/compute, BN intervention and E5/multi-seed controls. Keep thresholds and dataset-specific choices on independently specified development data.
5. Label undated prelaunch HANDOFF sections (119, 121, 125, 137) as historical and qualify the unverified 'official metrics' wording at 142; mark NEXT_EXPERIMENT_PRIORITY step 1 as completed. Preserve dated historical logs and failed attempts.
6. For future runs, record executed data_specs and scoring-package/GT/mask hashes at launch; do not reinterpret the present Python-only source manifest or two initial search hashes as complete sample provenance.

## Preserved failures and corrections

1. python - (PATH resolved E:/Scripts/python.exe): No pyvenv.cfg file Auditor inventory attempt did not execute. Retained; corrected by using pre-existing uv-managed CPython directly; no environment install.
2. Read inventory and package source line counts with Path.read_text() default encoding: UnicodeDecodeError: 'gbk' codec can't decode byte 0xa7 in position 156: illegal multibyte sequence Inventory read stopped at rgbt/utils.py. No inputs modified. Corrected all reads to UTF-8.
3. GT/mask checker v1: Traceback (most recent call last):    File "<stdin>", line 48, in <module>  TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'   Auditor used numpy.bincount on masks stored as binary float values. Corrected only the checker to cast the already verified binary expression to int64. No experimental data or implementation modified.
4. functions.exec JavaScript report-finalization orchestration: SyntaxError: Unexpected string The orchestration could not parse because Markdown backticks inside a JavaScript template literal were not escaped. No shell call ran and no file changed. The corrected finalization removes those template delimiters from generated Markdown strings.

The numeric exit codes for the first two attempts were not retained and are not invented. Mask-checker v1 exited 1; its full original code and corrected v2 code are preserved in the JSON. The finalization orchestration syntax error happened before any shell call or write.

COLLECTION/full20_collection_transport_sha_typo_failure.json retains an earlier returncode 1, empty stdout and AssertionError. Its filename labels a SHA typo; this reviewer did not recover the original failing hash literal. The successful later collector/helper hash/archive and every archived file were verified. E0 retains the stopped two-worker exit 143 and successful resumed exit 0, with all 479 final receipts. No failed attempt is silently converted into an original success.

## Artifact seals

- Closed archive: 48,998,868 bytes; SHA256 9b61557c67b05b796c0441b34636731cad8b1d586879b5eb53d42f819e32d37c; all 3,974 file members match RAW.
- Initial core-input tree: 9,900 files; SHA256 9dabde83002e146f49d48fcf6cd855edeaae9478d478e669f40873eeb960861a (unchanged).
- Additional old-toolkit/mask tree: 1,969 files; SHA256 e34b573814d04e2913403a72240d32d53dc47c60c0a2b1dea47b60d795ef965c (unchanged).
- Accompanying JSON SHA256: 7c83084fe7504f921d31cf9fb077bb9aebd7d0373f6d1cff1e2fe4e7558b7d79.
- JSON includes 11,869 local input hashes and 2,135 E0/baseline archive member hashes. Recorded binary checkpoint hashes are explicitly separated from hashes computed by this reviewer.
