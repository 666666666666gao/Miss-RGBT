# 实验状态

| 实验 | 状态 | 真实范围 | 结果/依据 |
|---|---|---|---|
| epoch14 传输 | DONE | 2025、2028 | 三端 SHA256 一致，见 HANDOFF |
| E0 Miss epoch20 | DONE | 两数据集479序列337353帧 | reports/e0/metrics.json，退出0 |
| 独立环境 | DONE | 导入、真实 CUDA、独立原样文档命令 | reports/environment_witness.json；environment_validation.md |
| 新方法 native | DONE | 真实 train 数据、两模板三搜索、一次更新 | reports/method_native_validation.json；不是完整训练 |
| 新方法配对速度检查 | DONE | E3/E4，各25更新，有效 batch8 | reports/paired_speed_smoke.json；首update最后一个search microbatch哈希相同 |
| 推理适配器检查 | DONE | 10帧原生完整跟踪等价+2单缺失状态 | reports/tracker_native_validation.json；不是全量结果 |
| E0 完整输入 | RUNNING | 同权重、同 GT 全两数据集 | 2026-10-13 14:50:39 北京时间，pid3294054；尚无正式正常指标 |
| E3/E4 正式配对训练 | QUEUED | train882；5轮×20000样本，各12500更新 | E0完成后GPU2/E3、GPU3/E4；当前尚未开始训练 |
| E5 选择性反馈 | TODO | E4有效后再做 | 未验证 |
| 三个有效论文模块 | UNPROVEN | 需要主结果和消融 | 不能据 native PASS 宣称成立 |

控制器 pid3294032，冻结执行源码版本 `370f215a`（完整值见 reports/SOURCE_VERSION.txt 和启动收据）。首次只读检查节点2026-10-13 16:05:39北京时间，之后必要查询间隔240秒。不要重复启动控制器。

原始 native 的反向可靠比例为0，不能据该单步证明反向反馈实际发挥作用。E5尚未训练。
