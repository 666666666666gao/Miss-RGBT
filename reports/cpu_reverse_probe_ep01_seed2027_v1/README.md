# 反向反馈实现检查

2026-10-13 19:39:19（北京时间）在服务器 2025 的既有 conda 环境完成，退出码 0；CPU 四线程，CUDA_VISIBLE_DEVICES 为空。使用预先固定的 E4 epoch01 权重和 seed2027。

第 3 个真实 train microbatch 出现 1 个可靠样本：C 任务损失 2.839929，M 任务损失 2.663949，满足 `loss_M + 0.1 < loss_C`。未人工修改可靠门。

- M 响应目标无梯度，C 响应有非零梯度；梯度分解验证通过。
- 反向项直接更新共享 baseline；R 的直接反向梯度为零。
- 同一原生 AdamW 状态、同一输入比较 reverse_weight=0 与 0.02：接收参数 `baseline.decode_fuse_search.0.weight` 的最大更新差异为 4.433095455e-7。
- R 更新差异为 7.450580597e-9，记录为全局梯度裁剪带来的间接差异。
- 保存后严格重载，pred_boxes、score_map、size_map、offset_map 最大差异均为 0。

这是 CPU 单 microbatch 的梯度、优化器和重载检查。它不包含有效 batch8 累积训练，不是 E5 性能结果，也不证明反向反馈能改善完整或缺失指标。当前 E3/E4 正式配置的 reverse_weight 仍为 0。

完整数据、输入与源码 SHA256 见 [report.json](report.json)，进程与退出凭据见 [execution.json](execution.json)。探针生成的权重仅留在服务器，不提交 Git。
