**E0-availability experiment integrity audit — 2026-10-15**

**Overall: WARN. Numerical reproduction: PASS. Semantic acceptance: provisional. Blockers to retaining the qualified descriptive comparison: 0.**
A fresh same-family reviewer independently read the archived application/scorer source and reconstructed the results from existing trajectories and dataset GT. No network/SSH, image access, GPU/process query, neural execution, model construction or checkpoint load was performed. Inputs and author prose were preserved.

Requested route: native fresh `fork_turns=none`, `gpt-6-astra`, reasoning `max`; these are request metadata, **not backend attestation**. Reviewer: `/root/e0_availability_integrity_audit`; `review_independence=same-family`, `acceptance_status=provisional`. Request: `PROJECT/.aris/traces/experiment-audit/2026-10-15_availability_run01/001-availability.request.json:1`; route metadata: adjacent `run.meta.json:1`. Home `MEMORY.md` was not read; user-required session files/daily notes were read. All substantive findings below use primary artifacts.

Path aliases used in exact file:line references:

- `PROJECT = C:/Users/gb/projects/Miss-RGBT`
- `PRIVATE = C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013`
- `RAW = PRIVATE/e0_availability_closed_raw`
- `PKG = PRIVATE/first5_metric_package/rgbt`
- Private audit traces: `PRIVATE/e0_availability_independent_audit_20261015`.

**Reproduction and adverse results**

The independent script imports only standard-library modules and NumPy 1.21.6 under `E:/anaconda/python.exe`; it does not call the author aggregation helper or import the scorer/model. It checks 209,037 items, including 93,966 new curve cells and 93,966 original-E0 cells. Every cell matches exactly in the scorer sequence-list row order. All 20 raw scalar metrics reproduce with observed error 0 (acceptance bound 1e-12 percentage points). Four additional assertions bind original logs and new stdout metrics. Complete/Miss each cover 479 sequences and 337,353 frames; total 958 sequence-runs and 674,706 output frame rows.

| Condition | Metric | Original E0 | Availability | Delta pp |
|---|---|---:|---:|---:|
| LasHeR | PR | 75.636502887 | 75.636502887 | +0.000000000 |
| LasHeR | NPR | 71.772836806 | 71.772836806 | +0.000000000 |
| LasHeR | SR | 60.125713676 | 60.125713676 | +0.000000000 |
| RGBT234 | MPR | 90.677133921 | 90.677133921 | +0.000000000 |
| RGBT234 | MSR | 66.845228330 | 66.845228330 | +0.000000000 |
| LasHeR-Miss | PR | 63.472507063 | 61.314029292 | -2.158477771 |
| LasHeR-Miss | NPR | 58.581850562 | 56.806229726 | -1.775620837 |
| LasHeR-Miss | SR | 50.583148589 | 49.224669599 | -1.358478990 |
| RGBT234-Miss | MPR | 82.794814555 | 79.941202027 | -2.853612528 |
| RGBT234-Miss | MSR | 59.574435134 | 58.034248029 | -1.540187105 |

The strict Miss tests are `LasHeR PR > 69.3`, `SR > 54.4`, `RGBT234 MPR > 88.1`, `MSR > 64.3`: **0/4 pass**. These are four numerical tests, not independent experimental replications. All five negative deltas are retained. Targets and comparison reduction are explicit in `PROJECT/analysis/build_full20_comparison.py:14` and `PROJECT/analysis/build_e0_availability_comparison.py:54`.

Whole complete trajectories were independently read from the original raw tar and the new archive: **254/479 arrays and files exactly equal**; the other 225 have differences bounded by **2.6147972675971687e-12**, so they satisfy absolute tolerance 3e-12 with relative tolerance 0, but are not exact. All 46,983 complete-input curve cells and five scalar metrics are exactly equal. A float32 midpoint at one coordinate in `LasHeR/boy2buildings` makes **478/479**, not 479/479, rounded float32 arrays equal. This does not change any sampled curve cell. Source mapping arithmetic also differs in association (`RAW/source/lib/test/tracker/cadtrack.py:175` versus `RAW/source/miss_rgbt/tracker.py:47`); this is a plausible numerical mechanism, not a replay proof. The existing wording in `PROJECT/reports/e0_availability_20261015_v1/RESULTS.md:18` is accurate.

**A. Ground truth provenance: WARN**

713 external dataset GT files, both mask files, sequence lists and 257 instantiated attribute metadata files are locally present and match supplied inventory/collection hashes. All claimed 245 LasHeR plus 234 RGBT234 sequences are covered. Application code passes only first-frame GT into inference, uses available sensor paths thereafter, and holds both-missing frames. No generated reference or future-GT inference path was found.

LasHeR uses 245 GT files / 220,703 frames; RGBT234 uses 468 modality GT files / 116,650 frames. Mask state totals `[00,01,10,11]` are `[33743,56020,43390,87550]` and `[18766,24800,25651,47433]`. All **52,493** non-initial both-missing frames hold the preceding output exactly in both old and new Miss trajectories. Every new/original first output equals its external first GT. Full annotations are read in preflight for shape/hash verification, but the job passed to inference contains only `init_bbox`; tracking crops and template updates use previous predictions. Available-image duplication and pre-Transformer masking prevent the missing sensor image path from supplying subsequent frames in the audited code.

- GT and image upstream authenticity is not independently established by self-consistent hashes. Image bytes and checkpoint binaries were not supplied to this audit.
- The local package snapshot matches a post-evaluation installed-package hash observation; it is not a pre-launch upstream attestation.
- Complete first-frame initialization overrides masks in 65 sequences, including 16 whose published first-frame state is both missing; this is the explicitly recorded protocol, not an undisclosed exception.

Evidence: `RAW/source/scripts/evaluate_model.py:29`; `RAW/source/scripts/evaluate_model.py:49`; `RAW/source/scripts/evaluate_model.py:80`; `RAW/source/miss_rgbt/tracker.py:27`; `RAW/source/miss_rgbt/tracker.py:36`; `RAW/source/miss_rgbt/model.py:49`; `PKG/dataset/lasher_dataset.py:12`; `PKG/dataset/rgbt234_dataset.py:13`; `RAW/collection.json:2`; `RAW/collection.json:2182`; `RAW/collection.json:5740`.

**B. Score semantics and normalization: WARN**

Independent NumPy arithmetic reproduces all 93,966 new and 93,966 original per-sequence curve cells exactly, in installed sequence-file order; all 20 primary metrics have zero observed error. All 10 comparison rows/deltas and 4 strict-target decisions match. No model-output-based normalization is present.

The installed loader reads float32 and rounds predictions with NumPy `.round(0)` (ties to even). LasHeR GT also uses that rounded loader; RGBT234 GT does not round, though all supplied GT coordinates are integral. LasHeR sets the first prediction to GT and retains the frame in the denominator; it truncates to GT length and substitutes a preceding prediction if width/height is nonpositive. Here raw shapes already exactly match. PR/MPR use `distance <= 20`; NPR normalizes each centre by **GT** width/height plus 1e-8 and uses `distance <= 0.20`. Precision curves have 51 thresholds. IoU uses inclusive-pixel corner geometry; SR/MSR compare **strictly greater** than 21 thresholds from 0 to 1 and average all cells equally. RGBT234 MPR takes the smaller visible/IR centre error per frame, and MSR the larger visible/IR IoU. This maximum is across two external annotations, not normalization by a model-derived maximum.

- LasHeR invalid GT means any x/y/w/h <= 0 after float32 load and rounding. The installed scorer assigns distance -1, counting these frames as PR/NPR successes; SR assigns overlap -1, counting them as failures. 1,260 frames satisfy this rule, including one first frame. The denominator retains all frames.
- SR/MSR are unweighted means over 21 sampled threshold cells and sequences, not a trapezoidal integral or frame-weighted metric. NPR uses ground-truth width/height and representative threshold 0.20.
- Exact score/curve equality does not imply exact trajectory equality. Complete input has 254/479 exact arrays and bytes, maximum absolute raw difference 2.6147972675971687e-12; only 478 rounded float32 arrays are equal.

Evidence: `PKG/utils.py:30`; `PKG/dataset/basedataset.py:19`; `PKG/dataset/basedataset.py:77`; `PKG/utils.py:117`; `PKG/utils.py:121`; `PKG/utils.py:161`; `PKG/metrics/metrics.py:23`; `PKG/metrics/metrics.py:57`; `PKG/metrics/metrics.py:246`; `PKG/metrics/metrics.py:290`; `PKG/metrics/metrics.py:332`; `PROJECT/analysis/build_e0_availability_comparison.py:54`; `PROJECT/analysis/build_full20_comparison.py:14`; `PROJECT/reports/e0_availability_20261015_v1/comparison.json:137`; `PROJECT/reports/e0_availability_20261015_v1/RESULTS.md:18`.

**C. Existence and identity: WARN**

Verified 2,057 closed archive members against extracted bytes, 2,056 collection hashes, 115 archived source/config hashes, 106 per-run source bindings each, and 1,939 producer input hashes. Verified all 958 new plus 958 original bbox/receipt bindings, exact coverage, finite shapes, first-frame GT identity, logs and complete-trajectory comparison. Two new launches have matching PID/condition/checkpoint bindings and exit 0 records. Each new log has 8 worker-ready and 479 matching DONE lines; final stdout metrics equal metric files.

- The 250-state-key equality and zero-update claim are supported by packaging source and executed metadata, with consistent original/package hash strings; tensors were not loaded or independently compared by this reviewer.
- 1,967 local package files match the 3,556-entry remote manifest. The 1,589 extra paths are all GT-tree files and remain metadata-only.
- The two local collection/aggregation exit-1 events are preserved in a retrospective ledger. Separate contemporaneous native terminal receipts were not supplied; the remote transport exit 0 does not attest to those local failures.

Evidence: `RAW/packaging.json:2`; `RAW/source/scripts/package_e0_availability.py:23`; `RAW/source/scripts/run_attribution_controls.py:64`; `RAW/source/scripts/run_campaign.py:22`; `RAW/source/scripts/run_campaign.py:34`; `RAW/collection.json:123`; `RAW/results/e0_availability_complete_20261015/run.json:2`; `RAW/results/e0_availability_miss_20261015/run.json:2`; `RAW/records/e0_availability_complete_exit.json:2`; `RAW/records/e0_availability_miss_exit.json:2`; `PRIVATE/collect_e0_availability_closed.py:65`; `PRIVATE/e0_availability_package_coverage.json:2`; `PRIVATE/e0_availability_collection_failures.json:3`; `PRIVATE/e0_availability_collection_failures.json:15`.

**D. Executed paths and unused code: PASS**

Actual evidence identifies evaluate_model.main -> prepare/worker_init/sequence -> MissingTracker -> MissingTrackerModel -> unchanged CADTrack backbone blocks/head; evaluator then registers bbox files in rgbt.LasHeR/RGBT234 and calls PR_LasHeR, NPR, SR_LasHeR, MPR and MSR. Numeric execution is independently corroborated by every output curve cell. Optional repair is disabled, patch_keep is absent, and no KD/training call occurs on this inference path. Generic PR/SR, GTOT/RGBT210 metrics, cross_layer candidate elimination, alternative pretraining loads and training dataset classes are not the generators of the claimed scores.

- Static application-source review and consistent native output records are not an instrumented runtime trace or authentication of external PyTorch/timm/OpenCV binaries.
- The inherited evaluate_model docstring and run protocol say trained single model/single-view inference; for this checkpoint the explicit packaging/controller records establish zero-update evaluation of an already-trained E0.

Evidence: `RAW/source/scripts/run_attribution_controls.py:91`; `RAW/source/scripts/evaluate_model.py:61`; `RAW/source/scripts/evaluate_model.py:128`; `RAW/source/scripts/evaluate_model.py:133`; `RAW/source/miss_rgbt/tracker.py:14`; `RAW/source/miss_rgbt/model.py:28`; `RAW/source/miss_rgbt/model.py:67`; `RAW/source/lib/models/cadtrack/vit_care.py:57`; `RAW/source/lib/models/cadtrack/cadtrack.py:73`; `RAW/source/lib/models/layers/head.py:129`; `PKG/dataset/lasher_dataset.py:63`; `PKG/dataset/rgbt234_dataset.py:53`.

**E. Scope: WARN**

Supports a descriptive zero-update availability-aware inference-path control using one original E0 epoch20 checkpoint: two full conditions, 958 sequence-runs and 674,706 output frame rows. Complete five-metric equality and five adverse missing-input deltas are retained. Current RESULTS.md appropriately disclaims isolated module attribution and stronger conclusions.

- This is not a new training result, multi-seed robustness estimate or evidence of three effective paper modules.
- The intervention changes both unobserved patch-embedding handling and historical-template availability; it is not a one-line causal intervention.
- Metadata and launch files for E0-FT/E2 do not demonstrate their completion. BN or KD efficacy and E0-FT/E2 final metrics are outside this closed control.
- Recorded seconds include multiprocessing/evaluation overhead and missing-frame holds; they are not a matched per-frame deployment latency benchmark.

Evidence: `PROJECT/reports/e0_availability_20261015_v1/comparison.json:2`; `PROJECT/reports/e0_availability_20261015_v1/RESULTS.md:3`; `PROJECT/reports/e0_availability_20261015_v1/RESULTS.md:20`; `RAW/source/miss_rgbt/tracker.py:53`; `RAW/source/miss_rgbt/model.py:57`; `RAW/source/scripts/run_attribution_controls.py:97`; `RAW/records/e0_ft_full20_train_launch.json:2`; `RAW/records/e2_full20_train_launch.json:2`.

**F. Evaluation type: PASS**

Real-GT OPE on provided LasHeR/RGBT234 annotation copies. Published-mask sensor absence is simulated, but the reference remains independently stored dataset annotations, not synthetic/model/self-supervised targets. The classification does not authenticate upstream annotation/image/checkpoint provenance.


Evidence: `RAW/source/scripts/evaluate_model.py:32`; `RAW/source/scripts/evaluate_model.py:42`; `PKG/dataset/basedataset.py:9`; `PKG/dataset/lasher_dataset.py:12`; `PKG/dataset/rgbt234_dataset.py:13`.

**Evidence strength, execution history and limits**

The new raw archive is 21,839,791 bytes, SHA256 `866913d2025a9de4ebd4a354a5fd1e4254d4b12063f73bdce5e203e677594189`. All 2,057 members match the extracted files, and the 2,056 non-collection members match `RAW/collection.json:123`. The original complete and Miss tars contain 966 and 973 file members; their metrics/completed files equal the public baseline reports. Both originals reference the same 713 GT hashes and masks as the new control. All original/new trajectories and receipts were actually read, not inferred from script existence.

Complete evaluation launch PID 3796928 is recorded from 2026-10-15 03:40:32.043815Z to 04:33:26.561170Z; Miss PID 3934106 from 04:33:33.171895Z to 05:19:03.696132Z. Both have exit 0 records, 8 worker-ready lines, 479 DONE lines equal to receipts, and final stdout metrics equal to metric files. These mutually consistent archived records support actual execution; this audit is not an independent live runtime attestation. Launch/exit artifacts are under `RAW/records/e0_availability_{complete,miss}_{launch,exit}.json:1`.

The campaign source manifest covers 115 files, and each evaluator records 106 source hashes. The controller writes source/config hashes before evaluation and checks them after both conditions (`RAW/source/scripts/run_attribution_controls.py:64`, `:94`). This is distinct from the collector, which observed package/GT/mask/checkpoint hashes at **06:20:20.868506Z, after evaluation** (`RAW/collection.json:3`). Neither can be promoted to authenticated upstream provenance. All 1,967 supplied package files match current manifest entries, including required GT, scorer code, sequence lists and 257 attribute metadata files. The exact remaining 1,589 paths are all in `gt_file/`, and no claim of downloading/authenticating their bytes is supported.

Packaging source renames each key with `baseline.`, records 250 keys, epoch 20, repair false and zero optimizer updates, and checks tensor equality after serialization (`RAW/source/scripts/package_e0_availability.py:23`). Original SHA `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8` and packaged SHA `c4cc43454f80d838db0130f8664b36e73063aadb99a3a645ee12bfee1fd5c20e` agree across packaging, collection and run bindings. They remain **executor-observed hashes plus a packaging witness**: this reviewer did not possess/load the corresponding binaries or verify live tensor state. Original image bytes, full runtime-library bytes and upstream official-evaluator authentication were not provided.

The original Miss tar preserves an initial two-worker run ending **143 / Terminated**, with two completed sequences, followed by an eight-worker resumed run ending **0**, with 477 new DONE lines. All 479 combined log receipts equal the aggregate receipts. This interrupted run is not erased or misreported as a first-attempt success. Evidence: `baseline_raw_results.tar.gz::exit_code_initial2workers.txt:1`, `::run_initial2workers.log:9`, `::exit_code.txt:1`; the exact source resume logic is archived at `::evaluate_miss.py:207`.

Two later local failures are retained: the initial collector assumed the local package contained `gt_file/GTOT/SequencesName.txt`, and the initial comparison builder incorrectly averaged all precision thresholds. The erroneous statements are directly visible in `PRIVATE/collect_e0_availability_closed.py:98` and `PRIVATE/build_e0_availability_comparison_initial_failed.py:57`; the preserved initial-builder SHA matches its ledger. The ledger records exit 1 in both cases (`PRIVATE/e0_availability_collection_failures.json:3`, `:15`). The parent explicitly confirmed that these are **retrospective records** and that separate contemporaneous native terminal-receipt files were not saved. Its `observed_utc` is not independently authenticated failure time. The saved transport exit 0 proves remote collection success only. This audit did not rerun either failure as a substitute for historical evidence.

Auditor-only failures: the default `E:/Scripts/python.exe` failed with `No pyvenv.cfg file`; managed CPython 3.12 lacked NumPy; one optional-directory discovery returned exit 1 for absent candidates. All are preserved in private `auditor_failures.json`. Existing Anaconda/NumPy was then used without installation. The independent reconstruction, supplementary inspection and preservation rehash all actually exited 0.

**Claim impact and action items**

Retain the complete-input metric equality, all five adverse Miss deltas, 0/4 strict targets, 254 exact-trajectory count and measured maximum difference. Explicitly identify the installed scorer conventions when publishing absolute scores. Keep weight equality tied to its packaging witness and the two local failures tied to their retrospective record. The current descriptive result requires no further neural run for this audit.

Stronger claims remain unsupported: independently authenticated official evaluation; byte equality of all trajectories; locally authenticated bytes for all 3,556 package entries; isolated embedding/KD/repair or BN efficacy; three effective modules; repeated-run robustness; matched deployment speed; or completed E0-FT/E2. Resolving those limits requires the corresponding independent source evidence or separately scoped experiments. Presence of optional code/configurations or training launch records is not completion evidence.

**Preservation and reproducibility**

All **4,049** sealed input files have unchanged before/after SHA256 values, including both original tars, the new archive, raw sources, GT/masks, all trajectories/receipts/curves, comparison outputs, and review request metadata. Full per-file maps remain in the private audit directory:

- `input_seal_before.json`: `921a92d109e58e4f5dd3376789bb05a2d32bea63efc50a560a95d0346a0c57ea`
- `input_seal_after.json`: `dd70b7bf19c9ac2f2e120111f0a55f9055ccc0ad7be09af5353c33d9145f2d04`
- `run_independent.py`: `0bfe86ff1d4a9309605ec97d553f1338a5b6d6508b067a3ced9c0a9146971b08`
- `deterministic_results.json`: `9e71e763111077839d082577e916f21694661d9dd7e771ba4c8ff014b07d7958`
- Machine report `EXPERIMENT_AUDIT.json`: `a3353cdd4446289746b056d68669aee35318b261384ff84d4b9cdd237544ea00`

The machine report includes per-check evidence, exact rows, count/tolerance details, principal input seals and private trace hashes. Both final report hashes are stored in private `output_seals.json` after serialization and returned to the parent; an artifact cannot contain its own full-file SHA256 without self-reference. This report changes none of the original source, results or author claims.
