# 完整20轮结果汇总

正式20轮训练与四组全量评估完成后，从服务器项目目录运行：

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

目前只有语法和真实E0读取检查完成，最终完整汇总尚未执行，不能把准备状态写成新方法指标。脚本在analysis目录，不加入冻结训练控制器，也不调用模型、修改训练或占用GPU。

完成后仍需独立实验审计、同预算E2/E5与容量对照，才能判断三个论文组件是否有效。单seed主结果不证明统计稳定性。
