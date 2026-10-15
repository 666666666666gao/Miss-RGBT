# 完整20轮结果汇总

2026年10月15日，正式20轮训练与四组全量评估均已完成，以下命令已在服务器实际执行并成功生成结果：

```bash
CUDA_VISIBLE_DEVICES='' conda-envs/miss_rgbt/bin/python analysis/build_full20_comparison.py \
  --project-root /home/gaob/Miss-RGBT \
  --output /home/gaob/Miss-RGBT/reports/full20_comparison_20261013_v1
```

脚本读取实际训练的 completed.json、run_config.json、source_version.txt、最终 epoch20 权重，以及四组评估的 run.json、completed.json、metrics.json 和 all_receipts.json；同时读取冻结 E0 完整/Miss 指标和序列清单。

执行前要求 E3/E4 各完成20轮、50000更新，且实际配置与冻结配置相同。每组评估必须覆盖479序列337353帧，使用对应模型的相同epoch20权重，逐序列帧数与bbox SHA256匹配。缺少任何必要文件会直接失败，不生成正式对照结果。

输出：

- comparison.json：完整数值、训练与评估记录、所有输入SHA256。
- comparison.csv：20个新方法指标值及其E0增益，保留未四舍五入数值。
- RESULTS.md：LasHeR/RGBT234完整及Miss的指标表，单独列出严格目标是否超过。

LasHeR报告PR、NPR、SR；RGBT234报告MPR、MSR。Miss严格目标为LasHeR PR>69.3、SR>54.4，RGBT234 MPR>88.1、MSR>64.3。增益使用百分点。

实际结果见../reports/full20_comparison_20261013_v1/RESULTS.md及同目录JSON/CSV。两模型各20轮/50000更新，四组全量评估共1916次序列运行/1349412帧。脚本读取服务器实际最终权重、训练记录及完整bbox回执，已成功验证全部范围与哈希；原始记录及实际部署源码已经独立归档。该汇总为CPU分析，不重新执行模型。

严格Miss目标E3达到0/4、E4达到1/4；完整输入均低于E0。E5尚未正式训练，不能据本轮宣称反向反馈有效或三个论文模块成立。独立审计状态以同目录EXPERIMENT_AUDIT.md/json实际文件为准。checkpoint_metadata.json给出服务器实际CPU读取证据；本机未下载原始权重字节，不能将其写成独立本地二进制验证。

完成后仍需独立实验审计、同预算E2/E5与容量对照，才能判断三个论文组件是否有效。单seed主结果不证明统计稳定性。
