# 模板更新固定权重对照

U0/U1/U2 在 E3-allBN、E4-allBN epoch20 权重上完成完整/Miss全量评估。U1 current 的 E3 完整轨迹与正式结果479/479 SHA一致；每个U0/U2任务均保存479条bbox、479条模板审计轨迹。

U1在缺失条件下优于U0和U2；U2仅在双模态同时可用时写入，降低了单缺失恢复效果。详细数值见[RESULTS.md](RESULTS.md)、[comparison.json](comparison.json)和[comparison.csv](comparison.csv)。
