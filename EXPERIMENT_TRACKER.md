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
| E3/E4 首轮配对训练 | RUNNING，17时已核验 | train882；5轮×20000样本，各12500更新 | 15:45实际启动3442177/GPU2、3442178/GPU3；最终指标待完成 |
| E3/E4 完整20轮 | QUEUED，17:08已核验 | train882；20轮×20000样本，各50000更新 | 控制器3623093等待原5轮及其全量评估；固定epoch20四组评估 |
| E0同帧与历史损伤诊断 | DONE | 479序列，无新增NN；帧加权描述统计 | reports/e0_paired_diagnostics_v1；不是正式PR/SR或新方法成绩 |
| E5 选择性反馈 | TODO | E4有效后再做 | 未验证 |
| 三个有效论文模块 | UNPROVEN | 需要主结果和消融 | 不能据 native PASS 宣称成立 |

原控制器pid3294032/E0冻结370f2158；5轮训练实际58fe581；完整20轮冻结4c473fa，版本与具体源码哈希分别保留。首次Windows观察器SSH通信退出255，不是训练失败；后续独立SSH已核验实际运行。唯一下一只读节点23:10北京时间，Windows uv37852/python38240，节点前不查询远程NN进度。本地等候300秒，不重复启动。

原始 native 的反向可靠比例为0，不能据该单步证明反向反馈实际发挥作用。E5尚未训练。


2026-10-13 时序源码核查完成（DiMP/SAM2固定提交，仅静态，不是新效果）；反向梯度/AdamW/严格重载探针准备完成，仅Python3.7语法检查，状态PREPARED_NOT_EXECUTED。当前冻结训练及队列未改变。下一次远程指标采集仍为23:10 BJT唯一观察器；当前NN事实引用17:08。来源笔记见docs/TEMPORAL_STATE_SOURCE_FINDINGS_20261013.md；探针见analysis/validate_reverse_feedback.py，执行约束和证据范围见docs/MODULE_EVIDENCE_PROTOCOL.md。
