# Recovery source review: local confidence, wider search, original-target verification

Date: 2026-10-30. Scope: two official repositories, source inspection only.

## Decision

**No implementation in this bounded two-repository review satisfies the complete request: permissive licensing plus an explicit separation of local confidence, expanded/global search, and verification of the original target.** KeepTrack supplies useful candidate-association control flow, but the checked repository has GPLv3 licensing and its identity state can reset or change. GlobalTrack supplies initialization-conditioned global proposals followed by initialization-conditioned proposal scoring, but does not implement a local-confidence recovery switch; permissive licensing of its tracker-specific code was not established. Do not copy either implementation into the proposed permissively licensed recovery component on this evidence. [K1, K3-K9, G1-G7]

The actionable outcome is a small, independently implemented **search-support and acceptance experiment**, not a verified identity module, a performance claim, or proof of module novelty. An expanded search can expose candidates outside the old crop; accepting the original target requires separate evidence. The supplied Miss-RGBT geometry evidence motivates the former and does not establish the latter.

## Scope and provenance

The research skill was read at `C:/Users/gb/.codex/skills/research/SKILL.md`. The main agent used its required background-research workflow; the research subtask was bounded to these two repositories and did not delegate further. No remote SSH/GPU activity, model execution, or edits to frozen code occurred during that source review. Two official repositories were fetched into local `work/` checkouts. Claims below come from the pinned files, not repository popularity, third-party summaries, or reported benchmark results.

| Implementation | Official repository and inspected commit | License evidence and reuse decision |
| --- | --- | --- |
| KeepTrack in PyTracking | `visionml/pytracking`, `7eb9e74bd3d40e29dbcec444902237da13de247b` (commit date 2024-08-03); README identifies the official implementation. [K0] | Root `LICENSE` contains GNU GPL version 3. No separate permissive grant was found in the inspected KeepTrack tracker/parameter files. Exclude code copying for the requested permissive component. [K1, K2-K9] |
| GlobalTrack | `huanglianghua/GlobalTrack`, `0c9bd9649ce6006323787e7d6e6a1b88c19eaf80` (commit date 2020-03-20); README identifies the official implementation. [G0] | No root license or tracker/module-specific license notice was found in the checked tree and files. The discovered Apache-2.0 license is inside bundled MMDetection and bears Open-MMLab's notice. It is not evidence of a permissive grant for GlobalTrack's own tracker/modules. Permission remains unestablished; exclude copying. [G1, G2, G3, G5, G7] |

Pins were obtained using `git rev-parse HEAD`; source ranges below use the actual checked-out file lines. These are reproducible inspected snapshots, not a claim that either project is currently maintained or that all historical revisions have identical licensing. Absence of a root license is a tree-inspection observation and therefore has a pinned tree link rather than a fictitious line citation. Dependency and checkpoint permissions were not comprehensively audited.

## Source facts: KeepTrack

1. **Crop-local evidence is computed before association.** `track()` samples around `get_centered_sample_pos()` using the current target scale, extracts classification scores, optionally multiplies them by the output window, then invokes candidate matching. Its default parameter file disables output windowing. Thus its score thresholds cannot be transferred directly to Miss-RGBT's Hann-weighted confidence. [K2, K3]
2. **Localization status and learned association are distinct operations, but association is conditional.** If the maximum response is below the candidate-extraction threshold, the code immediately uses ordinary localization. Otherwise it extracts local maxima/descriptors. With no previous candidates, or a frame-number gap greater than one, ordinary localization runs and the candidate collection is reset. With consecutive candidates, the learned matcher is used except for a single-candidate/high-score shortcut; match assignments feed `CandidateCollection.update()`. These branches do not implement mandatory independent verification on every accepted frame. [K4, K5]
3. **The expanded-search mechanism is historical scale recovery, not a whole-frame detector.** A `not_found` result calls `search_area_rescaling()`. That method averages a subset of larger recent target scales; the next frame still samples around the held center. It may enlarge the crop if the history contains larger scales, but neither guarantees enlargement on each failure nor sweeps the full image. Normal/uncertain localization takes the non-`not_found` position-update branch. Classifier updating excludes both `not_found` and `uncertain`. [K3, K6]
4. **Identity continuity is useful but is not an immutable original-target verification contract.** Matching propagates object IDs and can reject weak assignments to the selected object. However, the selected ID can switch to a candidate with a higher historical maximum, and reselection after disappearance uses recent classifier response. The association collection also resets after the gap described above. Therefore it would be inaccurate to describe this implementation as guaranteed verification against the first-frame target after arbitrary outages. [K4, K7-K9]
5. **The exported presence score is not a separate learned presence head.** It is derived from the maximum classification response, with a square-root adjustment in one identity condition. Advanced localization separately thresholds response and handles distractor peaks. Neither is evidence that Miss-RGBT's current local response is a calibrated probability of original-target presence. [K3, K5]

**Interpretation:** KeepTrack is a primary-source example of separating peak response from temporal candidate association and suppressing unsafe classifier updates. It is not a permissive source donor and is not the exact local/global/original-anchor architecture requested. [K1-K9]

## Source facts: GlobalTrack

1. **Initialization supplies a persistent query.** `init()` processes the first annotated image/box and calls `_process_query()`, which stores query features and the annotated query box. The ordinary update path does not refresh that query. [G3, G5]
2. **Search is global on each update, rather than triggered by low local confidence.** `update()` passes the current image through gallery transforms without a previous prediction box. The official evaluation script selects `BasicPairTransforms(train=False)`: its pipeline rescales, normalizes, disables random flipping, pads, bounds boxes, and converts to tensors, without a target-centered crop. The reviewed update path has no local tracker state or local-to-global switch. [G3, G4, G8, G9]
3. **Proposal generation and target-conditioned proposal scoring are separate source stages.** `_process_gallary()` modulates gallery features with the stored query for RPN proposals; `simple_test_bboxes()` extracts query and proposal RoI features, applies the RCNN modulator, and computes classification and box regression. The modulation explicitly combines query and gallery/proposal features. This is stronger source evidence of original-query conditioning than merely choosing the highest generic-object response. It is still a learned scoring mechanism, not a guarantee of instance identity. [G5-G7]
4. **Ordinary output does not require an explicit recovery acceptance decision.** Unless `return_all=True`, `update()` returns the coordinates of the highest-scoring detection. The checked test configuration sets the RCNN score threshold to zero. There is no separate abstention/presence decision in that wrapper. Do not transplant this top-1 rule as a verified recovery policy. [G3, G10]

**Interpretation:** The conceptual proposal-then-query-scoring split is relevant, but importing this detector stack would not be a minimal Miss-RGBT adaptation. The inspected code also fails the established-permissive-license requirement. [G1-G10]

## Adaptation hypothesis for Miss-RGBT

The following is a proposed experiment, not behavior established by the reviewed code. The starting facts are supplied task context, not a fresh audit of frozen Miss-RGBT: previous-box local crops, Hann-weighted response, fixed plus dynamic templates, RGB/TIR availability masks, previous-box hold when both modalities are missing, and offline evidence of lost crop support after outages. No learned presence or candidate-identity head is assumed.

**Hypothesis:** a recovery route with separately logged local evidence, wider candidate generation, and fixed-anchor acceptance may repair some failures caused by lost crop support while exposing false reacquisition. This requires testing; wider search alone cannot establish target identity.

Use only two operational modes, `LOCAL` and `RECOVERY`, with these explicit decisions:

1. **No current observation:** if RGB and TIR are both missing, preserve the existing previous-box hold and freeze dynamic-template updates. Record missing observation, not target absence. Retain an outage flag so that the first observable frame enters recovery even if a local distractor produces a high response. There is no reason to run visual search while both inputs are missing.
2. **Local evidence:** when at least one modality is present, retain native local prediction and separately record the response before Hann weighting. Use that response only as a crop-conditioned reliability proxy. The proposed trigger is either return from a both-missing outage or low local reliability, with the latter calibrated on validation data by available modality pattern. Do not label this value a learned presence probability or adopt upstream numeric thresholds.
3. **Wider candidate generation:** in recovery, evaluate a fixed, budgeted set of overlapping, native-resolution crops covering a region larger than the native crop. Define its actual support and inverse crop-to-image box mapping explicitly. Start with one fixed expanded region rather than adding an adaptive cascade; call it global only if the crop union covers the image. Preserve native crop sampling resolution rather than shrinking the whole image into one crop. Collect a small candidate set using existing response/regression outputs; these are proposals, not accepted targets.
4. **Separate acceptance:** re-evaluate candidate-centered crops against the existing fixed initialization template, using only modalities present now with an existing valid fixed template. Keep the dynamic template frozen during proposal generation and acceptance. Rank and accept using this separately recorded anchor-conditioned evidence, calibrated with target and distractor candidates on validation sequences; allow rejection of all candidates. If acceptance fails, hold the prior box and remain in recovery. Resume the existing dynamic-template policy only after acceptance.

The last step is deliberately only a **fixed-template verification proxy**: it reuses correlated tracker evidence, may score a similar distractor highly, and may reject a changed target appearance. It is not an independent learned identity head. A high fixed-template score must not be presented as proof of original identity. If wrong-anchor and distractor controls show little identity selectivity, stop the identity claim; a supervised candidate-verification head would be separate subsequent work, not something these repositories or existing geometry evidence have already supplied.

Do not synthesize a missing modality, use a stale modality as current evidence, or compare differently available modality combinations using an assumed common confidence scale. Initialize acceptance calibration for RGB-only, TIR-only, and paired observations using existing availability masks; report their sample counts and uncertainty. If the fixed representation required for an observed modality is unavailable, do not fabricate an identity score from it.

## Minimal controls and decision evidence

These are prospective controls; none were run in this review. Keep data splits, weights, missingness schedules, initialization, preprocessing, and evaluation sequence sets fixed. Select budgets and decision parameters on validation data, then freeze them before testing.

| Control | What it isolates |
| --- | --- |
| Native frozen tracker | Reference behavior and compute. |
| Native search plus recovery update freeze/hold | Whether template protection or abstention alone explains changes. |
| Wider search plus the same update freeze, accepting the best proposal | Effect and false-reacquisition risk of additional spatial support alone. |
| Identical wider candidates plus fixed-template acceptance | Incremental effect of acceptance, independent of candidate generation and its compute budget. |
| Same candidates with a shuffled/wrong initial target template | Diagnostic of whether the acceptance score contains original-target information rather than generic response strength. Keep correct and wrong anchors matched by modality availability. |

Include both outage-return and no-outage windows, genuine target absence/occlusion where annotations establish it, and similar distractors. Separate in-image targets outside the native crop from targets actually outside the frame; geometry alone cannot label absence. Report:

- Native and expanded crop support of the ground-truth target, plus proposal recall before acceptance; this separates search-support failure from proposal-generation failure.
- Acceptance correctness against annotated original-target boxes and, for a manually checked distractor subset, original-target identity. Correct-box overlap is a localization proxy, not a universal identity label.
- False acceptance, rejection of valid proposals, duration until a sustained correct reacquisition, and later drift/template contamination.
- Results by current RGB-only/TIR-only/paired availability and outage duration; report both-missing frames separately as observation-free holds.
- Extra crop count, processing latency, and total compute, including candidate-centered acceptance passes.

Define overlap criteria and the number of consecutive correct frames for sustained reacquisition before test evaluation. Use an offline ground-truth best-candidate selector only as an explicitly labeled proposal-set upper-bound diagnostic, never as a deployable gate. A raw-versus-Hann trigger comparison can be added as a focused diagnostic, keeping recovery candidates and acceptance fixed.

An informative failure is possible: expanded support improves coverage but original-target acceptance remains poor. That would support working on candidate identity, not claiming a successful recovery module. No accuracy, speed, robustness gain, threshold portability, or module novelty is established here.

## Pinned primary sources

All GitHub links below name exact commits. Line ranges refer to source files, not rendered-page line numbers.

- **K0:** [Official KeepTrack identification, README lines 86-95](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/README.md#L86-L95).
- **K1:** [Repository GPLv3 license, lines 1-11](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/LICENSE#L1-L11).
- **K2:** [Default crop and window parameters, lines 12-30](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/parameter/keep_track/default.py#L12-L30).
- **K3:** [Crop extraction, scores, matching and presence output construction, lines 125-153](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L125-L153).
- **K4:** [Matching trigger, gap reset, matching shortcut and collection update, lines 256-327](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L256-L327).
- **K5:** [Advanced localization, response thresholds and distractor handling, lines 370-435](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L370-L435).
- **K6:** [Position/update decisions and historical search rescaling, lines 159-235](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L159-L235).
- **K7:** [Candidate IDs and assignment rejection, lines 12-68](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/candidates.py#L12-L68).
- **K8:** [Selected-identity detection and possible replacement, lines 70-95](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/candidates.py#L70-L95).
- **K9:** [Disappearance cleanup and response-based reselection, lines 97-114](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/candidates.py#L97-L114).
- **G0:** [Official GlobalTrack identification, README lines 10-14](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/README.md#L10-L14).
- **G1:** [Pinned repository tree: no root license file](https://github.com/huanglianghua/GlobalTrack/tree/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80). Also inspected tracked filenames and searched `README.md`, `trackers/`, and `modules/` for license/copyright notices; none found there.
- **G2:** [Bundled MMDetection Apache-2.0 license and copyright, lines 1-15](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/_submodules/mmdetection/LICENSE#L1-L15).
- **G3:** [GlobalTrack initialization and ordinary top-1 update, lines 46-83](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/trackers/global_track.py#L46-L83).
- **G4:** [Official evaluation transform selection, lines 6-12](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/tools/test_global_track.py#L6-L12).
- **G5:** [Persistent query and RPN-to-RCNN inference flow, lines 285-310](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/modules/qg_rcnn.py#L285-L310).
- **G6:** [Query-conditioned proposal classification/regression, lines 224-275](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/modules/qg_rcnn.py#L224-L275).
- **G7:** [RPN and RCNN modulation, lines 34-85](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/modules/modulators.py#L34-L85).
- **G8:** [Gallery transform composition and image rescaling, lines 63-94](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/_submodules/neuron/neuron/data/transforms/pair_transforms/mmdet_transforms.py#L63-L94).
- **G9:** [Basic evaluation transform pipeline, lines 271-292](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/_submodules/neuron/neuron/data/transforms/pair_transforms/mmdet_transforms.py#L271-L292).
- **G10:** [Test proposal settings and zero RCNN score threshold, lines 87-98](https://github.com/huanglianghua/GlobalTrack/blob/0c9bd9649ce6006323787e7d6e6a1b88c19eaf80/configs/qg_rcnn_r50_fpn.py#L87-L98).
