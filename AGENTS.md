# Miss-RGBT 工作约定

- 项目位于 2025 服务器 `/home/gaob/Miss-RGBT`；只使用物理 GPU2、3。
- 不抢占其他项目，不读取或控制温度、功率。
- 独立 conda 环境：`/home/gaob/Miss-RGBT/conda-envs/miss_rgbt`。
- `lib/`、`experiments/cadtrack/` 保留实际 epoch20 baseline；新方法放在 `miss_rgbt/`。
- 优先最简单、直接、正确的实现，不增加没有证据的 fallback、异常吞噬或兼容层，不重构无关代码。
- 长任务先估计诊断或完成节点；节点到达后按 180～300 秒检查，不重复启动任务。
- GitHub 同步仅包含代码、配置、文档和小型真实报告；权重、数据、环境、原始大文件不提交。
- 不启用外部通知或 wandb。失败保留原始日志，不能把 smoke test 写成完整实验。
- 每次结果更新同步 `HANDOFF.md` 与桌面 `document/Miss-RGBT交接文档.md`，注明实际证据、待验证项和运行状态。
