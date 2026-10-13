# Miss-RGBT

基于实际 CADTrack epoch20 baseline，研究完整视图与缺失视图的共享模型训练。首版方案 A：独立缺失输入前向、局部任务特征补偿、C→M 响应蒸馏，以及待验证的选择性 M→C 反馈。

2026-10-13：E0 两项缺失评估已完整完成；独立 conda 环境通过 CUDA 和独立文档验证。新方法处于真实 GPU 单步检查，尚无完整训练结果，三个模块尚未被实验证明有效。

| epoch20 实测 | PR / MPR | NPR | SR / MSR |
|---|---:|---:|---:|
| LasHeR-Miss，245序列 | 63.472507 | 58.581851 | 50.583149 |
| RGBT234-Miss，234序列 | 82.794815 | — | 59.574435 |

337353帧全部计分，固定官方 masks，原始 bbox 与执行收据单独归档。[完整报告](reports/e0/RESULTS.md)与[交接文档](HANDOFF.md)记录来源、数据问题和待完成项。

严格目标为 LasHeR-Miss PR/SR >69.3/54.4，RGBT234-Miss MPR/MSR >88.1/64.3；完整输入检查约两点提升。用户提供完整表不能代替同入口 E0 重测。

项目在 2025 服务器 `/home/gaob/Miss-RGBT`，只用物理 GPU2、3。独立 conda 环境见[环境文档](docs/ENVIRONMENT.md)。`lib/` 与 `experiments/cadtrack/` 保留实际 baseline，新增方法放在 `miss_rgbt/`。

```bash
cd /home/gaob/Miss-RGBT
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=2 conda-envs/miss_rgbt/bin/python scripts/validate_method.py
# 正式训练前需冻结预算、保存版本并确认 native 检查通过。
CUDA_VISIBLE_DEVICES=2 conda-envs/miss_rgbt/bin/python -m miss_rgbt.train --config configs/e4.json --max-steps 1
```

`--max-steps` 仅是 smoke。配置中的预算当前为候选，正式启动另写运行记录。权重、环境、数据、大体积原始结果不提交 GitHub。

[研究计划](refine-logs/EXPERIMENT_PLAN.md)和[近邻来源](docs/RELATED_WORK.md)说明对照及信息限制。保留原[baseline README](docs/BASELINE_ORIGINAL_README.md)。新代码未复制近邻论文实现；已保留官方 MIT 许可证，来源见 [代码说明](docs/THIRD_PARTY.md)。

