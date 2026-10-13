# First5 experiment integrity audit

**Overall verdict: WARN. Integrity status: warn.**

All four first5 evaluations are fully populated local result artifacts. All **20 reported scores and every saved per-sequence curve reproduce exactly** through the actual package's default dataset/metric entrypoints. The ten E0 control scores and their curves also reproduce exactly. I found no prediction-derived evaluation GT, self-normalized performance, phantom first5 result or reported-score mismatch. The preserved auxiliary source discrepancy does not enter inference or scoring.

The qualifications are material: actual checkpoint binaries are unavailable in the audited local bundle; the local `rgbt` package does not itself establish upstream author-reference equivalence; and single-seed, five-epoch results cannot establish full20 completion or three effective modules. Negative performance is not an integrity failure.

- Date: 2026-10-14. Reviewer: **gpt-6-astra**, reasoning **max**, fresh context.
- Review independence: **same-family**. Acceptance: **provisional**. The request records the configured model/effort; separate backend routing telemetry was not inspected.
- Scope: local project, supplied artifact caches and local Git objects. No SSH, live process/GPU query, neural inference, training or checkpoint loading.
- Only this report and its JSON companion were written. Exact input hashes, numbers and deterministic outcomes appear in the JSON.

The document/aggregation follow-up is **PASS**: current JSON/CSV/Markdown numbers agree with the previously verified results. Raw bbox/metric scoring was not repeated. The overall verdict remains **WARN**, with the same-family/provisional designation and unavailable checks retained.

`PKG` below means `C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt`; other relative paths start at the project root. File:line references identify the inputs hashed in the JSON.

## A. Ground-truth provenance: WARN

**The benchmark evaluations are real_gt.** New-model evaluation reads annotation files from `rgbt/gt_file`, checks them against frozen E0 hashes, and supplies only the first annotation box to initialization (`scripts/evaluate_model.py:29-58`). Later predictions use tracker state and available images, not later GT (`scripts/evaluate_model.py:79-99`; `miss_rgbt/tracker.py:27-60`). E0 complete/Miss use the same annotation source (`scripts/evaluate_complete.py:45-89`; `reports/e0/evaluate_miss.py:44-87`). Training labels come from dataset `init.txt` (`lib/train/dataset/lasher.py:68-74`). Teacher-response KL is a training loss, not benchmark GT.

I checked all **713 GT files**, their `(frames,4)` shapes and finite values against `reports/e0/inventory.json`. That inventory is byte-identical to its Git objects at `370f2158` and `58fe581`, before first5 results. All 728 original package files and 1,239 supplemental sequence/attribute files match their manifests and the prior local toolkit snapshot. The `advancedredcup` annotation contains **1,411 rows**; the sequence was not shortened to the erroneous 15-row alternate annotation described in historical notes. Evidence: `reports/first5_evaluation/metric_package_manifest.json:1-4`, `metric_metadata_manifest.json:1-3`; `PKG/dataset/basedataset.py:9-24,55-80`.

The actual evaluator is the recorded **rgbt toolkit**, invoking `LasHeR.PR/NPR/SR` and `RGBT234.MPR/MSR` (`scripts/evaluate_model.py:129-145`; `PKG/dataset/lasher_dataset.py:12-20`; `PKG/dataset/rgbt234_dataset.py:13-20`). Its local citation links do not establish which official author release/commit it equals. Independent upstream official equivalence is **UNAVAILABLE**, not a PASS, in this local-only audit.

The native LasHeR rule sets error to `-1` whenever **any GT coordinate is <=0**. Such frames count as PR/NPR successes and SR failures (`PKG/metrics/metrics.py:266-272,309-315,351-357`). There are **1,260 frames**, contributing **0.556805386456 percentage points** to macro PR/NPR relative to counting them as failures with the same denominator. This is common to every method and cannot be credited as learned gain. It is already disclosed in `HANDOFF.md:120`. Initial GT boxes are included in the denominator too; all frames are scored, but not every scored box is an unconstrained network prediction.

## B. Score normalization: PASS

No performance denominator uses prediction max/min/mean. Multiplication by 100 is percentage conversion (`scripts/evaluate_model.py:141-146`; `scripts/evaluate_complete.py:169-174`; `reports/e0/evaluate_miss.py:167-172`).

- PR/MPR: fraction within a fixed 20-pixel threshold, averaged equally over sequences.
- NPR: center error divided by **GT width/height**, reported at threshold 0.2 (`PKG/utils.py:117-158`; `PKG/metrics/metrics.py:327-361`).
- SR/MSR: mean success over 21 fixed IoU thresholds. IoU uses geometric union (`PKG/utils.py:161-185`).
- RGBT234 chooses the smaller center error/larger IoU across its two annotation coordinate systems per frame (`PKG/metrics/metrics.py:28-44,62-80`). This is not prediction-statistic normalization.

The toolkit rounds loaded predictions to integer coordinates (`PKG/dataset/basedataset.py:19-24`) and macro-averages sequences. The audit reproduced these conventions. They differ from the separate frame-weighted diagnostic statistics in the handoff.

## C. Existence, matching and provenance: WARN

Each first5 group has **479 bbox files, 479 individual receipts and 337,353 finite four-coordinate predictions**. Every receipt hash, individual/aggregate receipt, sequence set and frame count matches. All 24 copied report JSON files match raw files byte-for-byte. No predicted width/height is nonpositive, and all first boxes equal the applicable initial GT. The four recorded zero exits are present in `reports/full20/runtime_20261014_0240.json:118-141`.

After the actual sequence/attribute metadata was supplied, independent execution through default `LasHeR()` / `RGBT234()` and public metric methods reproduced **all 20 first5 scores and every curve element with absolute difference 0.0**. The two E0 controls passed raw-file/receipt checks and unmodified-core metric recomputation: another 958 bbox files, 674,706 frame predictions and ten exact scores. All six metric dictionaries match the runtime snapshot (`reports/full20/runtime_20261014_0240.json:378-469`).

All compressed archive hashes/sizes and all contained file bytes were checked without extraction or modification: evaluation **3,856** files; training **10**; metric package **728**; supplemental metadata **1,239**. The metadata archive is bound by `reports/first5_evaluation/metric_metadata_manifest.json:1244-1246`.

### Checkpoint identity: consistent records; binary verification unavailable

| Model | Recorded checkpoint | Recorded SHA256 |
|---|---|---|
| E3 | `checkpoints/e3_seed2027/epoch05.pth.tar` | `acd8543a0c68cbac66e62d6625571aa55c601a415652c4cb5535a4b950e81fa4` |
| E4 | `checkpoints/e4_seed2027/epoch05.pth.tar` | `edd94537b2697a4dbe0bf9e2d62c5653827ca1e2dc3da9bb7e6a8d09927b0f5c` |
| E0 | original `CADTrack_ep0020.pth.tar` | `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8` |

Each model's complete/Miss pair records the same hash. The controller fixes epoch05 (`scripts/run_campaign.py:81-90`), evaluation hashes the checkpoint (`scripts/evaluate_model.py:113-118`), and loading uses `strict=True` (`miss_rgbt/tracker.py:14-20`). These are consistent records, **not an independent rehash or metadata reload of absent binary bytes**. The loader does not itself assert epoch05/update12500. Actual binary identity and embedded epoch/update/config remain **UNAVAILABLE**. References: each `reports/e{3,4}_{complete,miss}/run.json:2-4`; `reports/e0/inventory.json:2-4`.

Per-evaluation manifests cover Python files only. Earlier campaign/E0 seals and both relevant Git versions agree on YAML hash `462a56bdba611aa60f1602c22f7d2794781caea55d9e17ba01294da0873aca1a`; the per-evaluation manifest does not seal YAML at worker load time (`scripts/evaluate_model.py:73,114-118`). This is a provenance limit, not evidence of an actual YAML change.

### Auxiliary source discrepancy: no material scoring effect

Direct comparison against Git objects shows:

- All four manifests are identical: **103** files each, of which **102** match `4c473fa`, current HEAD and local files.
- Only `scripts/validate_focal_normalization.py` differs: recorded/deployed `cef46a...7f0d`, Git/current `d36585...5c1c`. The supplied deployed snapshot matches the recorded full hash.
- The deployed helper adds one unused `CADTRACKActor` import and changes one comment. Removing that import makes its remaining AST identical. No audited project Python source imports the helper, and the campaign does not invoke it. Evaluation hashes script files without executing them.

Evidence: `reports/e3_complete/run.json:110`; deployed snapshot `reports/first5_evaluation/source_snapshots/scripts/validate_focal_normalization.py:7-17`; current helper `:7-16`; `scripts/evaluate_model.py:61-75,114-115`; `scripts/run_campaign.py:73-90`. This is **not an exact all-files Git snapshot**, but the difference does not enter prediction/scoring or invalidate reproduced values. The collection already states the exception (`reports/first5_evaluation/collection.json:52-57`).

Training has separate provenance: each 92-file source manifest matches **58fe5819c6e82770a5a36025924f5a05030ac7cc**, and stored configs equal that Git version as parsed JSON objects, not as file bytes. Current configs add `checkpoint_epochs` and `source_version_file`; the later train entry changes checkpoint saving/version-file lookup only. Executed first5 training must not be relabeled as the current full20 entry. Evidence: `reports/first5_training/{e3,e4}/source_version.txt:1`, `run_config.json:18-29`; `configs/e3.json:28-36`; `miss_rgbt/train.py:72,115-120`, plus the directly inspected Git diff.

## D. Dead evaluation code: PASS

All five claimed metrics are explicitly dispatched, and both scalar and curve outputs are recorded and independently reproduced (`scripts/evaluate_model.py:129-146`; `scripts/evaluate_complete.py:169-174,209-213`; `reports/e0/evaluate_miss.py:167-172,214-218`). No claimed first5 metric rests on an unused definition. Generic toolkit metrics for other datasets and plotting methods are outside these claims.

`analysis/build_full20_comparison.py` is a future aggregation path. It requires actual completed epoch20/50,000-update records, checkpoint hashes and four full evaluation directories (`:61-100`). `HANDOFF.md:28` correctly labels it prepared, not executed. No full20 result is accepted merely because this code exists.

## E. Scope and claim ceiling: WARN

Each model completed **5 fine-tuning epochs x 20,000 sampled examples = 100,000 examples**, microbatch4/accumulation2/effective batch8, **12,500 consecutive updates**, seed2027. Logs contain 2,500 updates per epoch with finite numeric fields; published epoch means reproduce exactly. Logged losses describe the **last accumulation microbatch**, not all training examples (`miss_rgbt/train.py:87-107`; `reports/first5_training/summary.json:2`). The first two search-microbatch hashes agree between models; this does not certify every tensor/mask for the full run.

Stored train/val lists have 882/97 unique sequences and are disjoint from each other and the 245 test names. They match both relevant Git versions. Source uses the train list and `init.txt` (`lib/train/dataset/lasher.py:61-74`). This is sampled training over train882, not exhaustive processing of all its frames.

| Dataset | Sequences per evaluation | Frames per evaluation | Conditions |
|---|---:|---:|---|
| LasHeR | 245 | 220,703 | complete; one fixed published Miss schedule |
| RGBT234 | 234 | 116,650 | complete; one fixed published Miss schedule |

Masks match their E0 hashes. For each new model, all **53,506** pre-missing prefix boxes including 479 initializations match across complete/Miss; all **52,493** post-initial both-missing frames hold the previous box exactly. The first raw mask is incomplete in 34 LasHeR and 31 RGBT234 sequences, but the documented protocol intentionally initializes both modalities. No later missing frame or sequence is dropped (`scripts/evaluate_model.py:53-58,79-91`).

The supplied 02:40 snapshot says full20 **RUNNING**, epoch1/update675 per model (`reports/full20/runtime_20261014_0240.json:238-362`). This is a dated observation, not live state checked by this audit. `EXPERIMENT_TRACKER.md:12-18` correctly separates first5 DONE, full20 RUNNING, E5 TODO and three modules UNPROVEN. Current `EXPERIMENT_TRACKER.md:20`, `HANDOFF.md:26` and `reports/full20/midpoint_observer_20261014.json:3-15` agree on the next single diagnostic at **2026-10-14 17:00 BJT / 09:00 UTC**, with recorded uv28232/python49524 and native session14398. The receipt still names the 02:40 snapshot as the last NN intake; observer registration is not a new training observation. Its `started_utc` is 03:13:48.017144 BJT, while the prose gives 03:13:47 for launch/birth; these recorded times are not identical, and this audit did not validate process birth or liveness. Older 23:10 observer paragraphs are historical. The previous audit's stale-observer criticism of tracker line20 no longer applies to the current version.

### Reproduced metrics (percent)

| Model / condition | LasHeR PR | NPR | SR | RGBT234 MPR | MSR |
|---|---:|---:|---:|---:|---:|
| E0 complete | 75.636503 | 71.772837 | 60.125714 | 90.677134 | 66.845228 |
| E3 complete | 74.388795 | 70.732830 | 59.200450 | 90.251132 | 66.118720 |
| E4 complete | 75.302106 | 71.670015 | 59.839741 | 90.294694 | 66.223281 |
| E0 Miss | 63.472507 | 58.581851 | 50.583149 | 82.794815 | 59.574435 |
| E3 Miss | 69.032091 | 64.017231 | 54.835854 | 85.224052 | 61.486632 |
| E4 Miss | 68.461011 | 63.372123 | 54.363025 | 85.613275 | 61.797040 |

Evidence: corresponding `reports/e*/metrics.json:2-13`; `runtime_20261014_0240.json:378-469`.

E3/E4 improve every recorded Miss metric versus E0, but **both are below E0 on all five complete-input metrics**. E4 minus E3 on LasHeR-Miss is **-0.571080 PR / -0.645107 NPR / -0.472829 SR pp**; on RGBT234-Miss it is **+0.389223 MPR / +0.310409 MSR pp**. Repair is not generally established as effective. Only E3 LasHeR-Miss SR passes its strict target: E3 exceeds **1/4**, E4 **0/4**, of PR>69.3, SR>54.4, MPR>88.1, MSR>64.3.

There is no E2 control for isolated forward KD, no equal-parameter repair control, and **reverse_weight=0** in both executed configs (`reports/first5_training/{e3,e4}/run_config.json:9-11`). Nonzero reverse fractions/KL are diagnostic, not reverse optimization (`miss_rgbt/objectives.py:41-53`). These causal distinctions are already specified at `docs/MODULE_EVIDENCE_PROTOCOL.md:7-11,55`. One seed cannot establish statistical stability. Test failure analysis has informed research (`HANDOFF.md:179`), so the tests cannot be described as wholly unseen development-external evidence.

### Current first5 documents and aggregation: PASS

The follow-up directly read the eight requested current files and checked the six source metric files against their already-audited hashes. The full20 helper and 02:40 runtime snapshot are also unchanged. An independent in-memory arithmetic check verified **20 JSON metric rows, their E0 percentage-point/relative-percent changes, 10 E4-E3 differences, all 20 CSV rows, eight strict-target decisions, and all 23 aggregation input hashes**. Numeric differences from the verified values are **0.0**. Markdown reports the same values rounded to six decimals: `RESULTS.md:9-35,41-63`; all 30 E0/E3/E4 values in `HANDOFF.md:13-15` also match. No raw prediction rescoring or production aggregator execution was repeated.

The new aggregator reads completed epoch05 records, checks 5 epochs/12,500 updates and the historical parsed configuration, and computes differences from E0 or E3 (`analysis/build_first5_comparison.py:23-58`). `relative_change_percent` divides by E0, not by the evaluated model's own prediction statistics. Its scope and the current reports explicitly retain the single-seed/first5 limits (`:60-62`; `RESULTS.md:67-72`; `README.md:3,19`).

One documentation error was found and corrected during this follow-up: `README.md:5` originally described historical configuration matching as byte-for-byte. E3 saved/Git files are **708/738 bytes**; E4 files are **707/737 bytes**. Their parsed JSON objects are equal, while their individual SHA256 hashes differ and are correctly recorded separately. The current README now states that distinction; the correction was directly re-read. Current hashes for all eight follow-up inputs and the earlier HANDOFF/tracker hashes are retained in the JSON. Former HANDOFF citations **15/107/166** are now **28/120/179**, respectively.

## F. Evaluation classification: PASS

| Evaluation | Classification | Claim boundary |
|---|---|---|
| E0 complete/Miss | `real_gt` | Recorded dataset annotation/toolkit protocol |
| E3 epoch05 complete/Miss | `real_gt` | One seed; five fine-tuning epochs; fixed masks |
| E4 epoch05 complete/Miss | `real_gt` | Same scope; repair has mixed Miss effects |
| Focal-normalization helper | `simulation_only` | Synthetic arithmetic, no benchmark performance |

The focal helper creates synthetic boxes/probabilities and labels its scope (`scripts/validate_focal_normalization.py:19-36`). Full20 aggregation is unexecuted and has no evaluation to classify. Teacher-response KL is a training objective, not unlabeled model-generated evaluation GT (`miss_rgbt/objectives.py:29-38`).

## Actions and claim impacts

1. Publish first5 only as **epoch05, seed2027, five fine-tuning epochs**, preserving complete-input regressions and mixed E4-E3 Miss effects. Artifact existence and qualified same-protocol Miss gains are supported; universal repair, complete-input improvement, isolated KD benefit and all-target success are unsupported.
2. Bind retained checkpoint bytes and saved epoch/update/config metadata to recorded hashes before claiming independently verified checkpoint identity. This report verifies available records and their metrics.
3. Name the **recorded rgbt-toolkit protocol**, including initialization and LasHeR invalid-GT behavior. Author-reference equivalence requires identifiable upstream provenance and was not verified here.
4. Retain both package manifests/archives and the deployed auxiliary snapshot. The initial metadata omission is **resolved**. Include actual inference configuration bytes in future run records.
5. Keep full20 completion and three effective modules unproven pending completed results and controls specified by `docs/MODULE_EVIDENCE_PROTOCOL.md`. This audit started or altered no experiment.

## Execution errors and disposition

The initial default `python` was a broken launcher (`No pyvenv.cfg`); `E:/python.exe` lacked NumPy. Deterministic checks used the existing `E:/anaconda/python.exe` (Python3.7.6/NumPy1.21.6), with bytecode writing disabled, without installation. These setup errors were not counted as metric passes.

The initial 728-file package lacked `lashertest.txt` and other metadata, so its default constructor failed. The supplemental actual-package 1,239-file metadata archive was then independently verified. Default constructors and all first5 scalar/curve checks subsequently passed (native session22716, exit0). Earlier unmodified-core execution verified all raw files and both E0 controls (session39493, exit0). Git/source checks completed with exit0 (session43178). The first combined report-writing command exceeded Windows command-length limits and was rejected before execution; the reports were then written separately. Checkpoint-binary and upstream official-provenance checks remain explicitly unavailable.

The follow-up checker initially expected a literal `pp` suffix inside Markdown cells; the actual table declares the unit in its introduction. That formatting assertion failed before any write, not because of a numeric discrepancy. After directly reading the literal table format, the corrected independent comparison passed (native chunk `df9d41`, exit0); unchanged helper/runtime hash checks also passed (`d933b1`, exit0).
