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
| E3/E4 完整20轮 | RUNNING，10月14日17:00已核验 | train882；20轮×20000样本，各50000更新 | E3 880212/GPU2 epoch11/update25700，E4 880213/GPU3 epoch11/update26425；epoch10均已保存；固定epoch20四组评估待完成 |
| 反向反馈 CPU 实现检查 | DONE，19:39退出0 | 固定E4 epoch01，seed2027，真实train单microbatch | report status=DIRECTION_AND_UPDATE_VERIFIED；不是E5性能结果 |
| E0同帧与历史损伤诊断 | DONE | 479序列，无新增NN；帧加权描述统计 | reports/e0_paired_diagnostics_v1；不是正式PR/SR或新方法成绩 |
| E5 选择性反馈 | TODO | E4有效后再做 | 未验证 |
| 三个有效论文模块 | UNPROVEN | 需要主结果和消融 | 不能据 native PASS 宣称成立 |

原3294032已完成退出；E0冻结370f2158，5轮训练实际58fe581，完整20轮科学标签4c473fa未改。17:00唯一观察器已exit0，快照runtime_20261014_1700.json：20轮epoch11更新25700/26425，epoch10均已保存，最后日志有限；未产生最终成绩。用户要求不要频繁轮询，下一唯一观察器uv54936/python46272（17:02:52出生、native90047）仅在10月15日11:00作预计完成节点采集，节点前不查询NN。首轮指标及审计对应6878e91输入快照；后续运行观察不改其第一轮证据范围。

原始 native 的反向可靠比例为0，不能据该单步证明反向反馈实际发挥作用。E5尚未训练。


2026-10-13 时序源码核查完成（DiMP/SAM2固定提交，仅静态，不是新效果）。独立CPU反向探针已执行并通过真实可靠门、梯度方向、AdamW更新差异和严格重载检查，详见reports/cpu_reverse_probe_ep01_seed2027_v1。当前冻结训练及队列未改变；完整20轮最终指标尚未产生，预计10月15日产出四组全量结果。来源笔记见docs/TEMPORAL_STATE_SOURCE_FINDINGS_20261013.md；执行约束和证据范围见docs/MODULE_EVIDENCE_PROTOCOL.md。
