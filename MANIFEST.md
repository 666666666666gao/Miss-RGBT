# Research Output Manifest

> Generated artifacts; full model evidence is distinguished from smoke checks.

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
