# Miss-RGBT

基于用户确认的“去掉CADTrack论文三个模块后训练得到的epoch20 baseline”，研究完整与缺失视图的共享模型训练。DropMoE是用户说明的CADTrack主干来源，本轮直接加载该baseline的epoch20训练权重，没有另载DropMoE原始文件。首版方案 A：独立缺失前向、局部任务特征补偿、C→M响应蒸馏及待验证的选择性M→C反馈。

2026-10-15：E3/E4各20轮、50000更新及四组全量评估已完成，[最终完整指标](reports/full20_comparison_20261013_v1/RESULTS.md)和[独立审计](reports/full20_comparison_20261013_v1/EXPERIMENT_AUDIT.md)已发布。E3严格Miss目标0/4，E4为1/4，仅LasHeR-Miss SR达标；完整输入全部低于E0。审计数值复算一致，语义结论为WARN、same-family/provisional，研究目标仍未实现。

2026-10-16：[E0-FT完整20轮与四套指标](reports/e0_ft_20261016_v1/README.md)已完成，审计WARN、数值复算一致；完整五项均低于E0，说明完整退化在仅完整继续训练时也出现，原因仍待归因。E2于17:03实际完成20轮/50000更新exit0，固定末轮全量评估GPU2、3已启动；下一必要节点18:39北京时间，此前不查询NN/GPU。长等待的本地观察进程丢失已记录，后续按绝对节点直接采集，不重启NN。详见[交接](HANDOFF.md)。[E0-availability](reports/e0_availability_20261015_v1/RESULTS.md)及E3/E4原指标保留；[资源采样](profile_output/attribution_20261015_1410/README.md)支持训练每卡一项。BN、A及模板控制源码均保持隔离，未正式启用。

模板更新已经在E0/E3/E4中启用。[隔离U0/U1/U2对照](docs/TEMPLATE_UPDATE_CONTROLS_20261015.md)及9项目标Python逻辑检查已准备，尚无NN/正式指标；按归因→BN→模板对照/有限调优→必要时A/B的次序，不改变现有训练。

| 原始E0 epoch20 实测 | PR / MPR | NPR | SR / MSR |
|---|---:|---:|---:|
| LasHeR-Miss，245序列 | 63.472507 | 58.581851 | 50.583149 |
| RGBT234-Miss，234序列 | 82.794815 | — | 59.574435 |

337353帧全部计分，固定公开缺失masks，原始bbox与执行收据单独归档。[E0报告](reports/e0/RESULTS.md)与[交接文档](HANDOFF.md)记录来源、数据问题和待完成项。[评分来源核查](docs/SCORING_PROVENANCE_FINDINGS_20261015.md)已将归档代码/GT对应到公开PyPI rgbt 1.0.1发行字节；这不等于数据集作者官方认证，与作者工具箱及SCDT实际评分代码的严格等价性仍未确认。既有分数及活动评分协议不变。

同入口完整 E0：LasHeR PR/NPR/SR=75.636503/71.772837/60.125714；RGBT234 MPR/MSR=90.677134/66.845228，均覆盖全量479序列337353帧。严格目标为 LasHeR-Miss PR/SR >69.3/54.4，RGBT234-Miss MPR/MSR >88.1/64.3；完整输入检查相对本组 E0 约两点提升。

项目在 2025 服务器 `/home/gaob/Miss-RGBT`，只用物理 GPU2、3。独立 conda 环境见[环境文档](docs/ENVIRONMENT.md)。`lib/` 与 `experiments/cadtrack/` 保留实际 baseline，新增方法放在 `miss_rgbt/`。

```bash
cd /home/gaob/Miss-RGBT
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# 在预定诊断节点只读查看；不重复启动现有队列。
cat reports/ATTRIBUTION_SOURCE_VERSION.txt
cat logs/attribution_20261015_v2/status.json
```

已完成的完整训练配置为`configs/e3_full20.json`、`configs/e4_full20.json`：每项20轮×20000采样元组、有效batch8、50000更新、seed2027、第10轮降学习率，固定末轮。科学执行版本4c473fa及实际部署源码已独立归档，不能把当前工作树当作当时执行字节。原E0/E3/E4权重仍在原位置；新增归因训练检查点写到`/data2/gaob/Miss-RGBT/checkpoints`，项目与环境仍在`/home/gaob/Miss-RGBT`。权重、环境、数据、大体积原始结果不提交GitHub。

[研究计划](refine-logs/EXPERIMENT_PLAN.md)和[近邻来源](docs/RELATED_WORK.md)说明对照及信息限制。保留原[baseline README](docs/BASELINE_ORIGINAL_README.md)。新代码未复制近邻论文实现；已保留官方 MIT 许可证，来源见 [代码说明](docs/THIRD_PARTY.md)。

冻结末轮的[定位、搜索区域及序列权重复核](reports/full20_localization_diagnostics_20261015_v2/README.md)已完成离线计算和独立数值复算；这不增加新的主指标或模块有效性结论。[LasHeR val97清点](reports/development_data_20261015/README.md)已完成，后续模块内部参数须在独立开发协议下选择；同数据集完整/Miss固定同一配置和权重。

[模块A隔离准备](https://github.com/666666666666gao/Miss-RGBT/tree/f6cead72d19a2544236cbcba2a7a86d2f15ce80f)已实现任务效用区域选择与补偿/细化，[真实数据CPU检查](reports/task_utility_cpu_20261015_v1/README.md)通过两次AdamW更新及状态/重载检查。当前main的科学源码保持冻结，A尚无GPU或完整20轮结果；B/C及三个有效贡献仍待验证。
