# Miss-RGBT

基于实际 CADTrack epoch20 baseline，研究完整视图与缺失视图的共享模型训练。首版方案 A：独立缺失输入前向、局部任务特征补偿、C→M 响应蒸馏，以及待验证的选择性 M→C 反馈。

2026-10-13：E0 完整与缺失评估全部完成；E3/E4 的 5 轮首轮对照正在正式训练。按用户完整训练要求，另冻结 20 轮同预算配对 E3/E4；待首轮对照及其全量评估结束后执行。新方法最终成绩尚未完成，三个模块尚未被实验证明有效。

| epoch20 实测 | PR / MPR | NPR | SR / MSR |
|---|---:|---:|---:|
| LasHeR-Miss，245序列 | 63.472507 | 58.581851 | 50.583149 |
| RGBT234-Miss，234序列 | 82.794815 | — | 59.574435 |

337353帧全部计分，固定官方 masks，原始 bbox 与执行收据单独归档。[完整报告](reports/e0/RESULTS.md)与[交接文档](HANDOFF.md)记录来源、数据问题和待完成项。

同入口完整 E0：LasHeR PR/NPR/SR=75.636503/71.772837/60.125714；RGBT234 MPR/MSR=90.677134/66.845228，均覆盖全量479序列337353帧。严格目标为 LasHeR-Miss PR/SR >69.3/54.4，RGBT234-Miss MPR/MSR >88.1/64.3；完整输入检查相对本组 E0 约两点提升。

项目在 2025 服务器 `/home/gaob/Miss-RGBT`，只用物理 GPU2、3。独立 conda 环境见[环境文档](docs/ENVIRONMENT.md)。`lib/` 与 `experiments/cadtrack/` 保留实际 baseline，新增方法放在 `miss_rgbt/`。

```bash
cd /home/gaob/Miss-RGBT
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=2 conda-envs/miss_rgbt/bin/python scripts/validate_method.py
# 只读查看现有控制器状态，不重复启动。
cat logs/campaign_20261013_v1/status.json
cat logs/full20_20261013_v1/status.json
```

完整训练配置为 `configs/e3_full20.json`、`configs/e4_full20.json`：每项20轮×20000样本、有效batch8、50000次更新、seed2027、第10轮降学习率，统一评估末轮。只保留第10和20轮权重以控制已实际接近满盘的 `/home` 空间。`scripts/run_full_training.py` 在首轮控制器完成后顺序接续，单卡训练、两卡评估；排队和实际启动分开记载。权重、环境、数据、大体积原始结果不提交 GitHub。

[研究计划](refine-logs/EXPERIMENT_PLAN.md)和[近邻来源](docs/RELATED_WORK.md)说明对照及信息限制。保留原[baseline README](docs/BASELINE_ORIGINAL_README.md)。新代码未复制近邻论文实现；已保留官方 MIT 许可证，来源见 [代码说明](docs/THIRD_PARTY.md)。
