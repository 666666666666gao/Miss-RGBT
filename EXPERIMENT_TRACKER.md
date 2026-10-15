# 实验状态

| 实验 | 状态 | 真实范围 | 结果/依据 |
|---|---|---|---|
| epoch14 传输 | DONE | 2025、2028 | 三端 SHA256 一致，见 HANDOFF |
| E0 Miss epoch20 | DONE | 两数据集479序列337353帧 | reports/e0/metrics.json，退出0 |
| 独立环境 | DONE | 导入、真实 CUDA、独立原样文档命令 | reports/environment_witness.json；environment_validation.md |
| 新方法 native | DONE | 真实 train 数据、两模板三搜索、一次更新 | reports/method_native_validation.json；不是完整训练 |
| 新方法配对速度检查 | DONE | E3/E4，各25更新，有效 batch8 | reports/paired_speed_smoke.json；首update最后一个search microbatch哈希相同 |
| 推理适配器检查 | DONE | 10帧原生完整跟踪等价+2单缺失状态 | reports/tracker_native_validation.json；不是全量结果 |
| E0 完整输入 | DONE | 同权重、同 GT，两数据集全479序列337353帧 | exit0；LasHeR75.636503/71.772837/60.125714，23490.677134/66.845228 |
| E3/E4 首轮配对训练 | DONE，23:10已核验 | train882；5轮×20000样本，各12500更新；exit0 | 训练循环25992.220/25118.128秒；各92源码哈希匹配58fe581；见reports/first5_training |
| E3/E4 首轮全量评估 | DONE，10月14日02:40已核验 | 四组各479序列337353帧，固定epoch05；exit0 | 全部评估已完成，20指标/全曲线离线复算差0；审计WARN、same-family/provisional；详见first5_evaluation |
| E3/E4 完整20轮 | DONE，10月15日11:00已核验 | train882；20轮×20000采样元组，各50000更新；固定最终epoch20 | 两项训练及四组全量评估exit0；1916次序列运行/1349412帧；E3目标0/4、E4目标1/4；见reports/full20_comparison_20261013_v1 |
| 反向反馈 CPU 实现检查 | DONE，19:39退出0 | 固定E4 epoch01，seed2027，真实train单microbatch | report status=DIRECTION_AND_UPDATE_VERIFIED；不是E5性能结果 |
| E0同帧与历史损伤诊断 | DONE | 479序列，无新增NN；帧加权描述统计 | reports/e0_paired_diagnostics_v1；不是正式PR/SR或新方法成绩 |
| 最终20轮独立审计 | WARN；same-family/provisional | 实际原始记录与评分源码离线复算 | reports/full20_comparison_20261013_v1/EXPERIMENT_AUDIT.md/json；不外推统计稳定性或三个模块有效 |
| 归因短检查 | DONE，各2次真实更新 | E0-FT/GPU2与E2/GPU3；严格重载和三输入状态推理 | reports/attribution_preflight_20261015.json；不是正式训练 |
| 归因存储准入 | DONE | v1在NN前失败；只迁移两份本轮短检查权重，SHA核验 | reports/attribution_storage_20261015.json；新正式权重写data2，旧权重未动 |
| E0-availability | RUNNING，11:40:32启动收据 | 先完整，再Miss，各479序列/337353帧 | 控制器3796810，初始评估3796928；不是已完成结果 |
| E0-FT/E2-full20 | QUEUED | 各20轮/50000更新，等待E0-availability两条件结束 | 科学版本dfc989e；GPU2/3各一训练，固定epoch20全指标 |
| E5 选择性反馈 | TODO | E4有效后再做 | 未验证 |
| 三个有效论文模块 | UNPROVEN | 需要主结果和消融 | 不能据 native PASS 宣称成立 |

当前：原3294032和完整20轮控制器3623093均已完成退出，最终采集观察器90047及采集器80340均已exit0，原20轮无待消费观察器；新归因观察器见下文。E0冻结370f2158，5轮训练实际58fe581，20轮实际4c473fa及已披露辅助脚本差异。最终20轮全部指标已汇总，完整性能未提升；E5尚未正式训练，三个有效论文模块仍未证实。当前E0-availability完整评估已启动，E0-FT/E2仍排队；后续再做融合BN冻结对照。唯一新观察器native48256/python24136在14:10:39.950507北京时间采集，节点前不查询NN或重建观察器。首轮审计仍对应其历史输入快照，不作为最终20轮审计。

原始 native 的反向可靠比例为0，不能据该单步证明反向反馈实际发挥作用。E5尚未训练。


2026-10-13 时序源码核查完成（DiMP/SAM2固定提交，仅静态，不是新效果）。独立CPU反向探针已执行并通过真实可靠门、梯度方向、AdamW更新差异和严格重载检查，详见reports/cpu_reverse_probe_ep01_seed2027_v1。该历史源码核查未改变当时冻结队列；现最终20轮已完成并单独归档。来源笔记见docs/TEMPORAL_STATE_SOURCE_FINDINGS_20261013.md；执行约束和证据范围见docs/MODULE_EVIDENCE_PROTOCOL.md。
