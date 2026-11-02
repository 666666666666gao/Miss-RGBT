# Research Output Manifest

> Generated artifacts; full model evidence is distinguished from smoke checks.

2026-11-02 UTC: `scripts/diagnose_joint_gradients.py`,
`scripts/launch_joint_gradients.py`, `scripts/analyze_joint_gradients.py`,
`docs/JOINT_GRADIENT_DIAGNOSTIC_20261102.md`: originalE0 train-mode full
complete/auxiliary parameter-gradient diagnostic,64fresh tuples/16batches.
PID768965 completed37.670s,cosine .201456CI[.160851,.248108],negative0/16.
Gatefalse,no projection training. Server/local saved Gram/bootstrap/hash
verification passed;no independent neural-gradient replay or optimizer.
Evidence `reports/server28_joint_gradients_20261102/`.

2026-11-02 UTC: `scripts/diagnose_teacher_reliability.py`,
`scripts/launch_teacher_reliability.py`, `scripts/analyze_teacher_reliability.py`,
`docs/TEACHER_RELIABILITY_DIAGNOSTIC_20261102.md`: frozen E0 task/KD
gradient diagnostic on128fresh fittingtuples/1152rows. PID728997 completed
55.655s. Registered criterion=false;unreliable teacher gradient mean not
established negative,fused cosine .004752CI[-.058037,.066890],
projection .492476CI[.372922,.600634]. No reliability training promotion.
Server/local saved evidence/hash/bootstrap checked;1152row native-head
gradient replay exact,sharedmeasurement,notindependentimagereplay.
Evidence `reports/server28_teacher_reliability_20261102/`.

2026-11-02 UTC: `scripts/probe_window_training.py`,
`scripts/launch_window_training.py`, `scripts/analyze_window_training.py`,
`docs/WINDOW_TRAINING_PILOT_20261102.md`: matched raw/Hann regression
supervision in identical native heads. GPU2PID654595 launched04:29:02Z,
same frozen train882 feature caches,common Hann inference and delayed audit.
Prior crop evidence motivates an ordinary training-alignment control,
not a validated novel module or official score. Completed result below.
Initial run stopped at a wrong auxiliary source path during report creation;
failed worker source and log retained. Corrected module.__file__ lookup,
same seeds/configuration/gate,exclusive_v2 retryPID679342 launched04:40:09Z.
This retries a reporting failure,not an independent experiment or retuning.
CorrectedPID679342 completed185.454s,batch256/LR0.00032,both10epochs/
90updates,selected6 before delayed128tuple/66seq audit. Missing-direction
E0 gains pass,but Hann/raw advantage and complete/recovery protection do
not;advance=false. Server/local checks and3456row cached-head replay with
independent Hann decode passed;not independent image/backbone replay.
Evidence reports/server28_window_training_20261102_v2/. All weights retained.

2026-11-02 UTC: `scripts/probe_chronology_training.py`,
`scripts/launch_chronology_training.py`, `scripts/analyze_chronology_training.py`,
and `docs/CHRONOLOGY_TRAINING_PILOT_20261102.md`: matched native versus
chronological mask training using identical small native-tail adapters.
GPU2 PID589499 completed704.600s,both10epochs/120updates,selected6/9.
Original-order tensor digests verify exact fitting input pairing.
Audit128tuples/69sequences; all registered contrasts' intervals cross0,
advance=false. Server+local evidence/hash/stop/select/bootstrap verified;
cached E0tail1152+selected2304rows replay exact,not independent image replay.
Evidence `reports/server28_chronology_training_20261102/`.
Train882 only,delayed audit,no official score or validated module claim.

2026-11-02 UTC: `scripts/diagnose_template_chronology.py`,
`scripts/launch_template_chronology.py`, `scripts/analyze_template_chronology.py`,
and `docs/TEMPLATE_CHRONOLOGY_DIAGNOSTIC_20261102.md` register a frozen
train882 template-age/missingness binding diagnostic, including a paired
image+mask permutation control. GPU2PID538386 completed79.312s.
Either-sign diagnostic criterion supported for missing searches, not
improved policy or training efficacy. Saved evidence/3456boxIoUs/
bootstrap/source/protocol hashes verified; no neural replay.
Evidence `reports/server28_template_chronology_20261102/`.

2026-11-02 UTC: `scripts/diagnose_candidate_quality.py`,
`scripts/launch_candidate_quality.py`, `scripts/analyze_candidate_quality.py`,
`docs/CANDIDATE_QUALITY_DIAGNOSTIC_20261102.md` and
`reports/server28_candidate_quality_20261102_v2/` record verified fixed
native top8 candidate GT-oracle headroom. Gate passed; no deployable method.
Independent NumPy map replay checked294912boxes. Failed original launch
traceback retained. Follow-up `scripts/probe_quality_ranker.py`,
`scripts/launch_quality_ranker.py`, `scripts/analyze_quality_ranker.py`,
and `docs/QUALITY_RANKER_PILOT_20261102.md` record the matched
quality/response supervision pilot completed231.174s onGPU2.
Both10epochs/180updates,selected9/10 from final-five6-10.
Registered gate failed; saved-evidence and2304row cached-ranker replay
verified,9216candidate IoUs independently checked.
Evidence `reports/server28_quality_ranker_20261102/`.
No official metric or module claim; weights preserved.

2026-11-02 UTC: `scripts/probe_tail_lowrank.py`,
`scripts/launch_tail_lowrank.py`, `scripts/analyze_tail_lowrank.py`, and
`docs/TAIL_LOWRANK_PILOT_20261102.md` register a matched30720-parameter
attention-versus-MLP adaptation pilot in native blocks10/11.
GPU2 PID368156 completed584.557s,batch192/LR6e-4,both10epochs/120updates,
selected8/7 from final-five6-10. Both registered gates failed despite a
positive attention RGB-missing crop effect. Server file/weight checks and
3456cached-tail replay records passed; no official result/module promotion.
Evidence `reports/server28_tail_lowrank_20261102/`.
Read-only pinned TOOD review:
`docs/QUALITY_ALIGNMENT_SOURCE_REVIEW_20261102.md`; prospective only.

2026-11-02 UTC: `scripts/probe_cross_stream_context.py`,
`scripts/launch_cross_stream_context.py`,
`scripts/analyze_cross_stream_context.py`,
`scripts/check_cross_stream_layout.py`,
`docs/CROSS_STREAM_CONTEXT_PILOT_20261102.md`, and
`reports/server28_cross_stream_context_20261102_v2/` record completed
matched199104-parameter train882 cross-stream context training.
Both arms stopped10epochs/480updates,selected6 from final-five6-10.
Registered promotion failed; saved-evidence verification and2304-record
separate-process selected-adapter replay passed. Replay shares adapter/
measurement code and cached maps; not independent image/backbone replay.
No official results, val97, checkpoint cleanup or validated paper module.

2026-11-02 UTC: `scripts/diagnose_late_fusion_streams.py`,
`scripts/launch_late_fusion_streams.py`,
`scripts/analyze_late_fusion_streams.py`, and
`docs/LATE_FUSION_STREAM_DIAGNOSTIC_20261102.md` record a frozen-E0
train882 fusion-input stream ablation over128 tuples/117 sequences.
Saved evidence, source/protocol hashes, tuple exclusions and sequence
bootstrap were verified. The registered advancement gate failed.
No training, official benchmark, or module promotion.

2026-11-02 UTC+08: `scripts/diagnose_absent_position.py`,
`scripts/launch_absent_position.py`, `scripts/analyze_absent_position.py`,
and `docs/ABSENT_POSITION_DIAGNOSTIC_20261102.md` register a frozen-E0
train882 diagnostic of absolute position embeddings on unavailable search
streams, paired with a matched available-stream position-removal control.
The corrected run passed independent saved-evidence, bootstrap, source-hash,
and complete-parity verification, but the registered gate failed. No training,
official evaluation, or module promotion.

2026-11-02 UTC+08: `scripts/diagnose_burst_mask_rollout.py`,
`scripts/launch_burst_mask_rollout.py`, `scripts/analyze_burst_mask_rollout.py`,
`docs/BURST_MASK_ROLLOUT_DIAGNOSTIC_20261102.md`, and
`reports/server28_burst_mask_rollout_20261102/` record a frozen-E0 paired
train882 IID-versus-clustered outage rollout over24 clips. All3,024 tracked
rows, schedule/recovery labels, source/checkpoint hashes, box-IoU arithmetic,
and sequence-bootstrap intervals were independently verified. Effects did
not clear the registered gate; no training or official evaluation.

2026-11-02 UTC+08: `scripts/diagnose_absent_search_kv.py`,
`scripts/launch_absent_search_kv.py`, `scripts/analyze_absent_search_kv.py`,
`docs/ABSENT_SEARCH_KV_DIAGNOSTIC_20261102.md`, and
`reports/server28_absent_search_kv_20261102_v2/` record a completed frozen
train882 attention K/V diagnostic. Independent saved-evidence verification
passed; the registered improvement gate failed in both missing-search
directions. No training, official evaluation, or module promotion.

2026-11-02 UTC+08: `scripts/probe_modality_heads.py`,
`scripts/launch_modality_heads.py`, `scripts/analyze_modality_heads.py`,
`docs/MODALITY_HEAD_PILOT_20261102.md`, and
`reports/server28_modality_heads_20261102/` record completed matched
modality/random two-head training. Full promotion gate failed;3456 direct
native cached-head replay records exact. No official metrics/module claim.

2026-11-02 UTC+08: `scripts/probe_head_supervision.py`,
`scripts/launch_head_supervision.py`, `scripts/analyze_head_supervision.py`,
`docs/HEAD_SUPERVISION_PILOT_20261102.md`, and
`reports/server28_head_supervision_20261102/` record completed matched
native/target-position head training. Registered promotion gate failed.
Saved arithmetic/hashes and direct3456-row native-head replay passed.
No official benchmarks or contribution claims. All weights preserved.

2026-11-02 UTC+08: `scripts/diagnose_response_geometry.py`,
`scripts/launch_response_geometry.py`, `scripts/analyze_response_geometry.py`,
`docs/RESPONSE_GEOMETRY_DIAGNOSTIC_20261102.md`, and
`reports/server28_response_geometry_20261102/` record a completed fresh
train882 response/geometry/gradient diagnostic and saved-evidence hashes/
arithmetic checks. No training, official benchmark, or module promotion.

2026-11-02 UTC+08: `scripts/probe_fusion_adapter.py`,
`scripts/launch_fusion_adapter_pilot.py`,
`scripts/analyze_fusion_adapter_pilot.py`,
`docs/FUSION_ADAPTER_PILOT_20261102.md`, and
`reports/server28_fusion_adapter_pilot_20261102/` record completed paired
train882-only cached-head adaptation and saved-evidence/hash verification.
Registered advancement gate failed. No full-tracker training, official
evaluation or module promotion. All pilot and baseline weights preserved.

2026-11-02 UTC+08: `scripts/diagnose_joint_utility_sets.py`,
`scripts/analyze_joint_utility_sets.py`,
`docs/JOINT_UTILITY_SET_DIAGNOSTIC_20261102.md`, and
`reports/server28_joint_utility_sets_20261102/` record a completed train882-only
frozen-checkpoint diagnostic and saved-arithmetic verification. Registered
training gate failed; no training, official evaluation, or module promotion.

| Timestamp | Skill | File | Stage | Description |
|---|---|---|---|---|
| 2026-10-13 | /experiment-plan + /run-experiment | README.md | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | HANDOFF.md | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | env-spec.json | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | docs/ENVIRONMENT.md | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | docs/RELATED_WORK.md | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | refine-logs/EXPERIMENT_PLAN.md | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | refine-logs/EXPERIMENT_PLAN_20261013_v1.md | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/environment_validation.md | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/environment_witness.json | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/checkpoint_metadata.json | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/completed.json | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/evaluate_miss_initial2workers.py | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/evaluate_miss.py | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/inventory.json | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/metrics.json | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/results.csv | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/RESULTS.md | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/run_job.sh | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /experiment-plan + /run-experiment | reports/e0/source_sha256.json | implementation | 实际 E0、方案、环境；新训练效果待验证 |
| 2026-10-13 | /run-experiment | scripts/evaluate_complete.py | implementation | 真实核验及下一步入口，未宣称完整训练效果 |
| 2026-10-13 | /run-experiment | scripts/verify_baseline.py | implementation | 真实核验及下一步入口，未宣称完整训练效果 |
| 2026-10-13 | /run-experiment | EXPERIMENT_TRACKER.md | implementation | 真实核验及下一步入口，未宣称完整训练效果 |
| 2026-10-13 | /run-experiment | reports/method_native_validation.json | implementation | 真实核验及下一步入口，未宣称完整训练效果 |
| 2026-10-13 | /run-experiment | reports/method_native_validation.log | implementation | 真实核验及下一步入口，未宣称完整训练效果 |
| 2026-10-13 | /run-experiment | reports/method_native_validation_import_failure.log | implementation | 真实核验及下一步入口，未宣称完整训练效果 |
| 2026-10-13 | /run-experiment | reports/environment_validation.md | implementation | 真实核验及下一步入口，未宣称完整训练效果 |
| 2026-10-13 | /run-experiment | scripts/run_campaign.py | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | scripts/evaluate_model.py | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | scripts/validate_tracker.py | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | scripts/summarize_speed.py | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | reports/paired_speed_smoke.json | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | reports/baseline_integrity.json | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | reports/tracker_native_validation.json | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | docs/CAMPAIGN_20261013.md | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | docs/THIRD_PARTY.md | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | docs/CADTrack_LICENSE | implementation | 实测检查、冻结预算与来源；正式结果待完成 |
| 2026-10-13 | /run-experiment | reports/SOURCE_VERSION.txt | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /run-experiment | reports/campaign_launch/campaign.json | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /run-experiment | reports/campaign_launch/e0_complete_launch.json | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /run-experiment | reports/campaign_launch/source_sha256.json | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /run-experiment | reports/campaign_launch/status.json | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /run-experiment | reports/e0/missing_state_diagnostics.json | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /run-experiment | reports/e0/MISSING_DIAGNOSTICS.md | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /run-experiment | scripts/analyze_missing_states.py | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /run-experiment | HANDOFF.md | implementation | 实际启动、冻结源版本、CPU描述诊断；新最终结果待完成 |
| 2026-10-13 | /experiment-audit | reports/e0/missing_state_diagnostics_before_first_transition_fix.json | review | 初始化切换标签修正前真实历史；不覆盖原记录 |
| 2026-10-13 | /experiment-audit | reports/e0/MISSING_DIAGNOSTICS_before_first_transition_fix.md | review | 初始化切换标签修正前真实历史；不覆盖原记录 |
| 2026-10-13 | /run-experiment | scripts/validate_focal_normalization.py | implementation | 原生focal数学契约CPU检查；合成算术非模型指标 |
| 2026-10-13 | /run-experiment | reports/focal_normalization_validation.json | implementation | 原生focal数学契约CPU检查；合成算术非模型指标 |
