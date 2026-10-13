# 实验状态

| 实验 | 状态 | 真实范围 | 结果/依据 |
|---|---|---|---|
| epoch14 传输 | DONE | 2025、2028 | 三端 SHA256 一致，见 HANDOFF |
| E0 Miss epoch20 | DONE | 两数据集479序列337353帧 | reports/e0/metrics.json，退出0 |
| 独立环境 | DONE | 导入、真实 CUDA、独立原样文档命令 | reports/environment_witness.json；environment_validation.md |
| 新方法 native | DONE | 真实 train 数据、两模板三搜索、一次更新 | reports/method_native_validation.json；不是完整训练 |
| 新方法配对速度检查 | TODO | E3/E4，同一有效 batch8 | 尚未启动 |
| E0 完整输入 | PREFLIGHT | 同权重、同 GT 全两数据集 | 正在准备；无实测正常指标 |
| E3/E4 正式配对训练 | TODO | train882，开发val97，预算待冻结 | 未启动，无新方法最终指标 |
| E5 选择性反馈 | TODO | E4有效后再做 | 未验证 |
| 三个有效论文模块 | UNPROVEN | 需要主结果和消融 | 不能据 native PASS 宣称成立 |
