# Closed-record localization diagnostic audit

**Overall verdict: WARN — all numerical checks pass; one low-severity terminology warning, zero numerical blockers.**

The warning concerns the field name official_protocol_delta_pp. Its value is the archived installed-toolkit equal-sequence MSR delta; this audit does not authenticate the benchmark authors' upstream evaluator. The README and scope string already use the narrower installed/original-protocol wording.

Date: 2026-10-15. Reviewer: fresh native Codex agent /root/full20_localization_audit. Requested routing: gpt-6-astra, reasoning max. Routing is UNATTESTED_REQUEST_METADATA_ONLY; review_independence=same-family and acceptance_status=provisional. The request was absent at the initial read and was created later by the parent; the later request and metadata were read and sealed without backdating.

Scope: local closed artifacts only, independent CPU/NumPy arithmetic, no imported implementation functions, SSH/network, process/GPU inspection, neural execution, checkpoint loads, image reads or additional agents. Only the two DIAGNOSTIC_AUDIT reports were explicitly written. Original checkpoint/image binaries are unnecessary for this geometric arithmetic and were not independently verified.

Path aliases:

- PROJECT = C:/Users/gb/projects/Miss-RGBT
- RAW = C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/full20_closed_raw
- PKG = C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt
- OUT = C:/Users/gb/projects/Miss-RGBT/reports/full20_localization_diagnostics_20261015_v2
- COLLECTION = C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013

Trace: .aris/traces/experiment-audit/2026-10-15_localization_run01/

## Independent recomputation

| Check | Coverage | Maximum error |
|---|---:|---:|
| Quality groups | 63 groups / 441 numeric fields | 7.105427357601002e-14 |
| Joint-hit groups | 63 groups / 630 numeric fields | 1.0524914273446484e-13 |
| Crop groups | 54 groups / 540 numeric fields | 0 |
| Native scalar crop rounding | 1,695,978 constructions | 0 pixels |
| Original primary metrics | 30 scalars | 0 |
| Original curve values | 30 arrays / 281,898 cells | 0 |
| Historical E0 crop report | 198 numeric fields | 1.7763568394002505e-15 |
| Weighting summary | all scalars and four quartiles | 1.4210854715202004e-14 for summary scalars |
| Weighting CSV | 234 rows / 1,404 numeric fields | 2.842170943040401e-14 |
| README tables | 36 data rows / 208 numeric cells | 4.985033275062278e-07 |

Diagnostic and weighting arithmetic tolerance: 1e-10 in the saved units; integer counts, curve cells and trajectory equality require exact agreement. README values use six decimals, so tolerance is 0.000000500001. All three diagnostic CSVs match their JSON representations exactly: 180 rows and 2,466 fields.

The audit reads 2,874 prediction/receipt pairs, 2,024,118 prediction rows, 713 GT files and two masks. Per model/condition this is 479 sequences and 337,353 frames. The independent implementations use squared center errors, separate rectangle-intersection expressions, GT-relative size differences, math.fsum reductions, scalar Python crop primitives and explicit per-sequence primary metric reconstruction. Full checker sources and their hashes are retained in the JSON, including the failed document checker and its corrected version.

| Dataset/reference | Sequences | Raw frames | Valid nonfirst quality frames | Nonpositive-GT nonfirst exclusions |
|---|---:|---:|---:|---:|
| LasHeR / visible | 245 | 220,703 | 219,199 | 1,259 |
| RGBT234 / visible | 234 | 116,650 | 115,820 | 596 |
| RGBT234 / infrared | 234 | 116,650 | 115,783 | 633 |

Crop all-observed denominators are 185,671 LasHeR and 96,992 RGBT234 frames. Restored-current11 denominators are 54,822 and 26,353. “Valid” here means the explicitly declared positive-coordinate diagnostic convention, not an independent claim that every excluded image annotation is unusable.

The stored masks contain 34 LasHeR and 31 RGBT234 non-complete first states; execution initializes both sensors. After initialization, 33,737 LasHeR and 18,756 RGBT234 frames have both sensors missing. All 157,479 model-specific Miss outputs on those frames exactly hold the preceding box. Current-11 restoration groups include all subsequent complete runs, including the 21-plus lag group.

All 2,643 diagnostic input hashes and seven weighting hashes match. All 3,974 file members of the original full20 tar match RAW. Original E0 archives contribute 1,939 preserved member hashes. Before-first-missing complete/Miss trajectories agree exactly on 32,240 LasHeR plus 20,787 RGBT234 frames per model: 53,027 per model.

## A–F findings

### A. Ground-truth, mask and trajectory provenance: PASS

- 713 annotation files are separately stored dataset annotations, not predictions: 245 LasHeR GT files and 468 RGBT234 sensor GT files. All dataset-relative identities, hashes, shapes, finite values and GT reference ordering match the archived inventory. The inventory itself is byte-identical to the original E0 Miss archive.
- 479 unique sequences and 337353 frames per model/condition; all 2874 prediction/receipt pairs and 2024118 rows validate. E3/E4 individual receipts equal the collected receipt entries; all six first boxes equal visible initialization GT. Complete and Miss E0 inventories share GT bytes, names and lengths, while their environment prefixes and availability-state counts legitimately differ.
- Two mask files and all per-sequence and aggregate binary state counts match the inventory. Initialization forces [1,1], overriding 34 LasHeR and 31 RGBT234 stored first masks. Both-missing nonfirst frames total 52493; all 157479 corresponding E0/E3/E4 Miss boxes hold the preceding box exactly.
- Quality uses positive x,y,w,h for its chosen reference and excludes first frames, including both-missing holds in relevant quality groups. Crop support excludes first frames and both-missing holds, requires all reference GT positive, and accepts support from either sensor annotation. Restored_current11 means every current-11 frame after prior missingness, including lag 21-plus, not only the first return frame.
- These are verified local dataset-provenance records. No authenticated upstream annotation release, original image bytes, or original checkpoint binaries were inspected; none is required for this closed-record geometric arithmetic.

Evidence: PROJECT/reports/e0/inventory.json:3; PROJECT/reports/e0/inventory.json:7; PROJECT/reports/e0/inventory.json:9; PROJECT/reports/e0/inventory.json:18; PROJECT/reports/e0/inventory.json:3207; PROJECT/reports/e0/inventory.json:3209; PROJECT/reports/e0/inventory.json:3218; PROJECT/analysis/diagnose_full20_localization.py:101; PROJECT/analysis/diagnose_full20_localization.py:108; PROJECT/analysis/diagnose_full20_localization.py:112; PROJECT/analysis/diagnose_full20_localization.py:124; PROJECT/analysis/diagnose_full20_localization.py:135; PROJECT/analysis/diagnose_full20_localization.py:147; PROJECT/analysis/diagnose_full20_localization.py:165; PROJECT/analysis/analyze_paired_baseline.py:38; RAW/source/scripts/evaluate_model.py:42; RAW/source/scripts/evaluate_model.py:79; PKG/dataset/basedataset.py:9; PKG/dataset/lasher_dataset.py:12; PKG/dataset/rgbt234_dataset.py:13.

### B. Independent formulas, normalization and aggregation: PASS

- Independently reconstructed all 63 quality groups, 63 pairwise same-frame joint-hit groups and 54 crop groups without importing or running implementation functions. Independent forms use squared center error <=400, continuous rectangle intersections, abs(pred_size-GT_size)/GT_size*100 and math.fsum reductions. MAPE denominators are fixed dataset GT width/height, not model-derived maxima or means.
- All 1611 numeric diagnostic fields agree: maximum errors 7.105427357601002e-14 for quality, 1.0524914273446484e-13 for joint and zero for crop; floating tolerance 1e-10 in stored percent/pp units and zero integer tolerance. Own-hit cohorts are model-specific; each joint comparison uses the intersection of the two hits on the same valid group.
- 1695978 scalar crop constructions using Python math.ceil(math.sqrt(w*h)*4), round(center-side/2), half-open center inclusion and positive box overlap agree with the vectorized implementation exactly. This checks the nominal padded-crop rectangle, not image-clipped pixel visibility, appearance, resize interpolation or identity. Both native trackers use previous output state and the frozen factor 4.
- All 30 original primary metrics and 281898 curve cells were independently reconstructed for E0/E3/E4, both conditions, both datasets. Errors are exactly zero. Original package semantics include float32 loading, rounding predictions (and LasHeR GT), all-frame sequence denominators, initialized first frames, <= precision thresholds and strict > success thresholds. LasHeR nonpositive GT yields -1, counted by PR/NPR but not SR. RGBT234 chooses min center error/max IoU across sensor GT and does not apply the diagnostic positive-coordinate exclusions.
- All 234 RGBT234 rows match SequencesName.txt rather than asynchronous receipt order. All 21 MSR thresholds are retained. The equal-sequence E4-E3 delta is -0.06619170886234452 pp, the length-weighted diagnostic delta +0.5835527524340207 pp, and their difference 0.6497444612963652 pp. No original score or curve is renormalized. Length quartiles, stable ordering, counts 109/123/2 and all CSV contributions independently agree.

Evidence: PROJECT/analysis/diagnose_full20_localization.py:35; PROJECT/analysis/diagnose_full20_localization.py:42; PROJECT/analysis/diagnose_full20_localization.py:153; PROJECT/analysis/diagnose_full20_localization.py:162; PROJECT/analysis/diagnose_full20_localization.py:171; PROJECT/analysis/diagnose_full20_localization.py:189; PROJECT/analysis/analyze_search_support.py:11; RAW/source/lib/train/data/processing_utils.py:32; RAW/source/lib/train/data/processing_utils.py:37; RAW/source/lib/train/data/processing_utils.py:43; RAW/source/miss_rgbt/tracker.py:38; RAW/source/lib/test/tracker/cadtrack.py:83; RAW/source/experiments/cadtrack/cadtrack.yaml:83; PKG/dataset/basedataset.py:22; PKG/utils.py:30; PKG/utils.py:117; PKG/utils.py:161; PKG/metrics/metrics.py:28; PKG/metrics/metrics.py:57; PKG/metrics/metrics.py:69; PKG/metrics/metrics.py:75; PKG/metrics/metrics.py:79; PKG/metrics/metrics.py:266; PKG/metrics/metrics.py:309; PKG/metrics/metrics.py:351; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:22; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:33; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:41; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:50; OUT/sequence_weighting.json:3; OUT/sequence_weighting.json:13.

### C. Artifact existence, exact identities and report agreement: PASS

- All 2643 diagnostics.json input hashes and seven weighting input hashes match actual files. The original full20 tar has 3974 file members, all byte-identical to RAW. Both E0 tar archives were read directly; 1939 archive-member hashes are preserved. E0 source manifests each match all 87 local archived source/config files; E3/E4 run manifests each match all 103 archived Python source files.
- All three diagnostic CSVs exactly match their JSON string representations (180 rows, 2466 fields); all 234 weighting rows, 1404 numeric fields and original metadata order agree within 2.842170943040401e-14. README has 36 data rows and 208 numeric cells; all match six-decimal rounding with maximum error 4.985033275062278e-07.
- 198 fields of the historical E0 crop report reproduce, including restored and state/history groups; maximum error 1.7763568394002505e-15. Before-first-missing trajectory equality is exact for 32240 LasHeR plus 20787 RGBT234 frames per model, totaling 53027 per model.
- Original comparison.json and checkpoint_metadata.json match their tar-extracted copies exactly; 1949 of the comparison's 1951 recorded input hashes can be independently verified locally. The two final checkpoint binary hashes remain recorded metadata, not newly computed binary hashes. All original scalar metrics and curves remain unchanged.
- The historical failed v1 directory exists and is empty. Both author-recorded failures and three reviewer failures are preserved with their actual limitations. The initial trace request was absent at the first read and existed later after parent creation; its later presence is not backdated. Source and seven diagnostic-output seals are checked again at report finalization.

Evidence: OUT/diagnostics.json:2881; OUT/sequence_weighting.json:51; OUT/box_quality.csv:1; OUT/joint_center_hits.csv:1; OUT/crop_support.csv:1; OUT/sequence_weighting.csv:1; OUT/README.md:15; OUT/README.md:38; OUT/README.md:49; OUT/README.md:56; OUT/README.md:62; OUT/README.md:71; OUT/README.md:87; OUT/README.md:89; PROJECT/analysis/diagnose_full20_localization.py:196; PROJECT/analysis/diagnose_full20_localization.py:212; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:63; PROJECT/reports/full20_comparison_20261013_v1/checkpoint_metadata.json:2; COLLECTION/full20_localization_failures.json:2; PROJECT/.aris/traces/experiment-audit/2026-10-15_localization_run01/001-localization.request.json:6; PROJECT/.aris/traces/experiment-audit/2026-10-15_localization_run01/run.meta.json:10.

### D. Active analysis paths versus unused or unexecuted code: PASS

- The diagnostic main calls history_labels, errors_and_iou via box_values, support, group accumulation and JSON/CSV writers. The weighting main reads recorded MSR curves and computes the actual reported aggregates. The files and independently reproduced values support those offline analysis paths.
- The imported legacy analyze_paired_baseline.main/describe and analyze_search_support.main/counts are not invoked by the new diagnostic entry point. Generic PR/SR, GTOT scorers and plotting methods are not the scorers for the claimed two-dataset results. Their existence does not establish additional experiments.
- The archived evaluator dynamically dispatches PR/NPR/SR or MPR/MSR and stores their actual curves. This reviewer inspected neural source but did not import it or execute a model. Source for repair, BN or future A/B/C controls does not establish a new intervention, measured latency, a new inference run or component efficacy.

Evidence: PROJECT/analysis/diagnose_full20_localization.py:12; PROJECT/analysis/diagnose_full20_localization.py:35; PROJECT/analysis/diagnose_full20_localization.py:171; PROJECT/analysis/diagnose_full20_localization.py:223; PROJECT/analysis/analyze_paired_baseline.py:150; PROJECT/analysis/analyze_search_support.py:97; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:33; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:72; RAW/source/scripts/evaluate_model.py:128; PKG/dataset/lasher_dataset.py:80; PKG/dataset/rgbt234_dataset.py:115; OUT/README.md:83.

### E. Claim scope, conditioning and terminology: WARN

- The README and scope strings correctly describe a single-seed (2027), two-dataset, fixed-epoch20, complete/fixed-mask, frame-weighted diagnostic. They retain all negative primary results, distinguish own-hit versus joint-hit cohorts and disclose prior test failure diagnosis. No causal BN/repair efficacy or unseen-test generalization follows from these statistics.
- One actionable nomenclature issue: sequence_weighting.json and its producer use official_protocol_delta_pp. The verified quantity is the recorded installed-toolkit equal-sequence MSR delta. No authenticated upstream-author release/conformance evidence was supplied. The scope string already says installed-toolkit, and the README says original scoring protocol; the field name should be qualified or renamed without changing its value.
- Joint-hit means fix the same frames for each pair but still condition on both models' outputs. They do not estimate unconditional localization quality or a causal regression improvement. Crop support is likewise conditioned on model-specific previous predictions and input history; accepting either annotation is geometric support, not evidence present in an available sensor. Restored-current11 aggregates include long runs after a missing episode.
- The observations and direction changes are supported descriptively: own-hit IoU can decrease while joint-hit IoU rises; a smaller full-versus-Miss crop gap can coexist with worse absolute crop failure; length weighting changes the sign of the same recorded curves. None proves that sequence length, repair or BN caused the outcome, or that a wider search domain will improve identity tracking.
- Future parameter choices require independent development data as stated. This closed-record audit does not verify live queue state or the future 14:10:39 node mentioned in the README; that sentence is task context at authorship, not an audited runtime observation.

Evidence: OUT/diagnostics.json:2; OUT/sequence_weighting.json:2; OUT/sequence_weighting.json:5; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:54; PROJECT/analysis/decompose_rgbt234_sequence_weights.py:58; OUT/README.md:3; OUT/README.md:7; OUT/README.md:9; OUT/README.md:65; OUT/README.md:67; OUT/README.md:78; OUT/README.md:79; OUT/README.md:80; OUT/README.md:81; OUT/README.md:83.

### F. Evaluation classification: PASS

- Prediction trajectories are model outputs; their external references are the archived dataset annotation files. E0 predictions are a comparison model, never substituted as GT. No model-derived pseudo-reference, self-normalization or proxy reference is used.
- Published modality-deletion masks simulate observation availability on real recorded-video datasets. This does not turn dataset-GT scoring into simulation-only evaluation. The present outputs are real_gt descriptive diagnostics, not new benchmark inference runs, new primary benchmark scores, human evaluations or proof of causal module effects.

Evidence: PROJECT/analysis/diagnose_full20_localization.py:108; PROJECT/analysis/diagnose_full20_localization.py:124; PROJECT/analysis/diagnose_full20_localization.py:147; RAW/source/scripts/evaluate_model.py:42; RAW/source/scripts/evaluate_model.py:85; OUT/diagnostics.json:2; OUT/README.md:3.

## Weighting result retained as diagnostic

| Aggregation | E3 MSR | E4 MSR | E4 − E3, pp |
|---|---:|---:|---:|
| Original installed-toolkit, equal sequence weight | 60.87276506879558 | 60.806573359933246 | -0.06619170886234452 |
| Diagnostic, sequence-length weight | 60.55228297920111 | 61.13583573163513 | +0.5835527524340207 |

The difference is +0.6497444612963652 pp, exactly the independently evaluated sum of (length/total − 1/234) times each sequence delta at recorded precision. The original primary delta remains negative. There are 109 improved, 123 harmed and two tied sequences.

| Length quartile | Sequences | Frames | Length range | Mean delta, pp | Primary delta contribution, pp |
|---|---:|---:|---:|---:|---:|
| 1 | 59 | 7856 | 45–210 | -0.655441692048 | -0.165260939448 |
| 2 | 59 | 16464 | 211–375 | -0.770624162896 | -0.194302673551 |
| 3 | 58 | 27435 | 379–597 | 0.308936060388 | 0.076573895310 |
| 4 | 58 | 64895 | 598–4140 | 0.874667828714 | 0.216798008827 |

The quartiles use sequence length and stable original list ordering, not performance sorting. They describe this record only. The frame-weighted number still uses the original 21-threshold, rounded-box, all-frame per-sequence scores; it is not the separate continuous-IoU diagnostic and does not replace the primary MSR.

## Supported and unsupported claims

- C1 — **supported**: All 63 quality, 63 joint-hit and 54 crop groups are reproducible from sealed trajectories, GT and masks.
- C2 — **supported_with_conditioning_qualifier**: RGBT234-Miss E4 own-hit IoU falls while E3/E4 joint-hit IoU rises on the common valid subset.
- C3 — **supported**: The same MSR curves yield a negative sequence-equal delta and a positive length-weighted diagnostic delta.
- C4 — **unsupported**: The length-weighted diagnostic replaces the original equal-sequence primary MSR or rescues the primary result.
- C5 — **supported**: Before-first-missing complete/Miss boxes are identical on 53027 frames per model; historical E0 crop statistics reproduce.
- C6 — **unsupported**: The local installed scoring package is independently authenticated as the benchmark authors' official evaluator.
- C7 — **unsupported**: Geometric overlap proves visible target pixels, identity reliability or efficacy of a wider search intervention.
- C8 — **unsupported**: These observational cohorts prove repair, BN, sequence length or three proposed modules caused a gain.
- C9 — **unsupported_and_not_required_for_scope**: Original E0/E3/E4 checkpoints, original dataset images or full upstream annotation releases were verified by this audit.
- C10 — **supported_scope_statement**: No new neural result or current process state is established by these offline diagnostics.

## Action items

- A1 (low): Rename official_protocol_delta_pp to recorded_protocol_delta_pp, or explicitly document that it means the archived installed-toolkit protocol and does not certify upstream-official equivalence. Preserve the exact existing numeric value and primary scores.
- A2 (retain): Keep the present single-seed, frame-versus-sequence, fixed-sensor-reference, postselection and geometric-support qualifiers attached to reused figures/tables. Compare future controls on independent development data before making causal or generalized efficacy claims.

## Preserved failures

Author failures are preserved in COLLECTION/full20_localization_failures.json: the first archive reader omitted root-level receipts and raised KeyError; a subsequent error-recording one-liner failed to parse. The failed v1 output directory exists and is empty. Its pre-fix implementation bytes were not separately captured, so this audit does not invent their checksum.

- **Initial trace request read**: Get-Content: Cannot find path C:/Users/gb/projects/Miss-RGBT/.aris/traces/experiment-audit/2026-10-15_localization_run01/001-localization.request.json because it does not exist. Read-only inventory command. Parent wrote the request and run.meta.json after the spawn call; the later existing files were read and sealed. No retroactive availability or backend attestation is claimed.
- **Initial seal transport JSON parse**: SyntaxError: Bad control character in string literal in JSON at position 524288 (line 1 column 524289) The tool truncated the 5961-entry JSON output. The shell hash computation was read-only; the JavaScript decoder failed. Repeated the seal output with zlib/base64 and wrote the initial draft audit JSON, verifying exact equality of the two successful compressed seal maps. No source or diagnostic file was changed.
- **Reviewer document/provenance checker v1**: Traceback (most recent call last): /   File "<stdin>", line 68, in <module> / AssertionError Checker incorrectly asserted complete and Miss sequence_details dictionaries were identical. Real archives use complete-state counts and different environment path prefixes, while GT content hashes, names and frame counts agree. Corrected only the checker to compare dataset-relative GT paths/hashes and condition-specific state counts. Corrected run exited 0; no experimental inputs or implementation changed.

The failed reviewer document checker was a reviewer assumption error, not a data mismatch: complete versus Miss inventories intentionally have different state counts and absolute environment prefixes. The corrected checker verified matching dataset-relative GT hashes, frame counts and names, plus the appropriate condition-specific state counts. The original failed and corrected programs are both retained in checker_sources.

## Artifact seals and limitations

- Before/after file hashes checked: 5965, with zero changes.
- Tree SHA256: b5f7167ca7f898e4d0b144d63f4e8c69ffed2bcbf91e7ae69d6642076a489974.
- Tree hash encoding: UTF-8; sorted absolute POSIX path + TAB + lowercase SHA256 + LF.
- Original full20 tar SHA256: 9b61557c67b05b796c0441b34636731cad8b1d586879b5eb53d42f819e32d37c.
- The comparison manifest has 1,951 entries: 1,949 were rehashed from local files/archive members. The two epoch20 checkpoint entries remain consistent recorded metadata and were not loaded.
- Accompanying JSON SHA256: 11cf4b390a2223b04757bd02d1699ea7e08286ac5ffc7f9d9046a6b77164f9d9.

All audited local file hashes and 1,939 E0 archive member hashes are in the JSON. Selected input/output seals:

| File | SHA256 |
|---|---|
| PROJECT/analysis/diagnose_full20_localization.py | 183c5ab56bb22740fb7af2b836aca60266e2b2dc68ccfbf08708f75460745569 |
| PROJECT/analysis/decompose_rgbt234_sequence_weights.py | f5c7503e758fc0a8764d3b4be5229f0d0ef4278bc6915db9db17fcbcc956a6d0 |
| PROJECT/analysis/analyze_paired_baseline.py | e9a28f43fed4546e3177fdfff74de8c4010f932dec9c8e2920a6b777747e2260 |
| PROJECT/analysis/analyze_search_support.py | 956da2ec90d3b2a1bd4bcf91e5120ca3179bd4c829d3f64fffb26b00b74d9969 |
| OUT/diagnostics.json | 36a86849e88fd2fa4a675f9518575ee5aa618c0add569c11bee0360a64917668 |
| OUT/box_quality.csv | a1cf472f5830b33bd6cca4d31b26532e5b44fc905c6e1205318cf2ade56c9305 |
| OUT/joint_center_hits.csv | 1fa7c14e65d2689f5023186235573a3cb73467c672252fbe3f9b3c55f2779b46 |
| OUT/crop_support.csv | 0321c0fb4905e5794ea382458c5e607919eb77fac8a396574d63b1369edf07c6 |
| OUT/sequence_weighting.json | 908d5f663efbec83a463e40f29c82cc6cc60a7153a82a374157e2c373165f506 |
| OUT/sequence_weighting.csv | 10d2047793d971e655d2421e1dd57a513dd953059bfbc65ab05392cd2b08ef7a |
| OUT/README.md | 3c04fa83a7dd5b89fc5b363fda30472bd4d6af102a70f0d1299e7cf05884e184 |
| C:/Users/gb/.codex_tmp/cadtrack_miss_ep20_20261013/baseline_raw_results.tar.gz | 85b302508ce56b7e688c0f8c21f58a6dd07fd883ea8f57db5bf99a47bae3dc0a |
| C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/e0_complete_raw_results_20261013.tar.gz | eff683ac725fad93d5c3de02e0fc5a5dde46a081aae49f551dd707f901081951 |
| C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/metrics/metrics.py | 9ae628c82201a00002cbef8a62c78746eb26acbd8c226304a4cd78e9223d2382 |
| C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt/utils.py | e027df4694ca9cc23fad375780df27efc3a0b9c58b30021a4aeee17773b296f2 |

- Requested gpt-6-astra/max routing is recorded request metadata, not a backend attestation; same-family semantic acceptance remains provisional.
- No SSH/network, original image inspection, model execution, checkpoint load or process/GPU query was performed.
- Checkpoint SHA256 values are verified as mutually consistent metadata but the original binaries were not rehashed locally.
- Byte seals authenticate internal consistency of the supplied records; they are not independent certification of original dataset collection or benchmark-author releases.
- The before/after seal covers this review interval, not an unobserved historic file state. The failed author implementation's pre-fix bytes were not separately preserved and are not reconstructed as if they were.
- Full-tree hashed files include unused code and auxiliary annotations; sealing them is not a claim that every file was on an executed metric path.

The numerical checks may be accepted within this closed-record scope. The semantic review remains same-family/provisional; this report does not claim cross-family acceptance or backend attestation.

## Follow-up 2026-10-15: PASS — E/A1 clarification resolved

This is an appended narrow follow-up, not a replacement of the initial WARN. The initial verdict, A–F findings, numerical reconstruction, original seals, failures and same-family/provisional status above remain historical evidence.

- Started: 2026-10-15T04:25:30.953873+00:00.
- Seal and document checks: 2026-10-15T04:26:23.738457+00:00.
- Completed: 2026-10-15T04:27:44.270753+00:00.
- Reviewer: /root/full20_localization_audit; requested gpt-6-astra/max; routing remains UNATTESTED_REQUEST_METADATA_ONLY.

- The added README line 67 explicitly limits official_protocol_delta_pp to the sealed installed scoring package's original recorded protocol and rejects upstream-official authentication. This satisfies the explicit-documentation alternative in initial action A1; renaming the preserved field is unnecessary for this disposition.
- README line 89 accurately records the initial WARN with zero numerical blockers and retains same-family/provisional and non-efficacy/non-upstream-authentication limits. These are clarifications, not revised metrics or new model evidence.
- All 46 Markdown table lines (36 data rows, 208 numeric cells) are byte-identical after UTF-8 decoding; all numeric tokens across the full README are identical. No diagnostic or primary result changed.
- All 5965 original seal entries were rehashed; 5964 are unchanged. The only change is the declared README text edit. The four analysis sources and all six numerical diagnostic output files are byte-identical, so the initial independent numerical reconstruction remains applicable.
- The initial public audit files were byte-identical to the pre_clarification snapshots before this append. The initial WARN, A-F statuses, evidence, recorded hashes, failures and routing limits remain intact as history.
- Initial README evidence line numbers refer to pre_clarification/README.md. In the current README the inserted clarification shifts former lines 67 onward by two lines; the preserved initial evidence must not be interpreted as newly generated current line references.

Evidence: OUT/README.md:67; OUT/README.md:89; PROJECT/.aris/traces/experiment-audit/2026-10-15_localization_run01/pre_clarification/DIAGNOSTIC_AUDIT.md:95; the same snapshot:142.

The 5,965-file recheck has exactly one expected change: README text. The four analysis source files and six numerical diagnostic outputs retain their initial hashes. The current rechecked input-tree SHA256 is 401b372b3b3af47877eb9d66ce062bceaff5c8d3f83c739a9d1f06672ebfda1e. No second full arithmetic reconstruction was needed or performed.

Actual input SHA256 values at follow-up intake:

| Input | SHA256 |
|---|---|
| C:/Users/gb/projects/Miss-RGBT/reports/full20_localization_diagnostics_20261015_v2/README.md | 1d4930fe0f4ef021d7639ff94fdac0c96459806932bfa36521af1079126c9af2 |
| C:/Users/gb/projects/Miss-RGBT/reports/full20_localization_diagnostics_20261015_v2/DIAGNOSTIC_AUDIT.md | 5a4fbc45f3c63515dd5789efd00b0ac74525826306c64b785fb2e6a19f57365d |
| C:/Users/gb/projects/Miss-RGBT/reports/full20_localization_diagnostics_20261015_v2/DIAGNOSTIC_AUDIT.json | 11cf4b390a2223b04757bd02d1699ea7e08286ac5ffc7f9d9046a6b77164f9d9 |
| C:/Users/gb/projects/Miss-RGBT/.aris/traces/experiment-audit/2026-10-15_localization_run01/pre_clarification/README.md | 3c04fa83a7dd5b89fc5b363fda30472bd4d6af102a70f0d1299e7cf05884e184 |
| C:/Users/gb/projects/Miss-RGBT/.aris/traces/experiment-audit/2026-10-15_localization_run01/pre_clarification/DIAGNOSTIC_AUDIT.md | 5a4fbc45f3c63515dd5789efd00b0ac74525826306c64b785fb2e6a19f57365d |
| C:/Users/gb/projects/Miss-RGBT/.aris/traces/experiment-audit/2026-10-15_localization_run01/pre_clarification/DIAGNOSTIC_AUDIT.json | 11cf4b390a2223b04757bd02d1699ea7e08286ac5ffc7f9d9046a6b77164f9d9 |
| C:/Users/gb/projects/Miss-RGBT/.aris/traces/experiment-audit/2026-10-15_localization_run01/001-localization.request.json | 44c3b53fa99fb6751dc26ff3e07996a60d2f54bc862fc4acb18acfcca298d6b4 |

The public audit hashes in that table describe their pre-append state; the pre_clarification copies retain those exact bytes. Initial README evidence line references resolve to pre_clarification/README.md; the clarification shifts current README lines after the insertion by two.

Current accompanying DIAGNOSTIC_AUDIT.json SHA256: a21288f53d881d26f91abde9cd728a19c17f3cad87a6eb5e3abeb9379c175434.

Private reviewer trace artifacts:

- C:/Users/gb/projects/Miss-RGBT/.aris/traces/experiment-audit/2026-10-15_localization_run01/002-localization-followup.reviewer-check.py — SHA256 a29c28613675409cabb6cd906af025cccd556582eed40cc9eb4abe245d53363b.
- C:/Users/gb/projects/Miss-RGBT/.aris/traces/experiment-audit/2026-10-15_localization_run01/002-localization-followup.reviewer-evidence.json — SHA256 3a68be549a72995aefceca9dc987e6d0e43d66e3f6cedef938a6883d519adb63.

Retained limits: same-family/provisional review; installed-toolkit rather than authenticated upstream equivalence; model/input-conditioned descriptive cohorts; no causal efficacy or new runtime result; no image or original checkpoint verification. No SSH/network, process/GPU/image/checkpoint query, model import or neural execution was performed. No implementation, metric, diagnostic input, primary result or README was edited by this follow-up. No new reviewer failure occurred.
