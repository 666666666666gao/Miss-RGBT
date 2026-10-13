# 首轮5轮评估证据

这是已结束的E3/E4固定epoch05全量评估，不能代替正在执行的最终20轮结果。E3/E4各训练12500更新，四组各479序列、337353帧，seed2027，reverse_weight=0。

- 完整指标、原始E0差值、E4−E3差值和严格目标：[RESULTS.md](RESULTS.md)、[comparison.csv](comparison.csv)、[comparison.json](comparison.json)。汇总入口为`analysis/build_first5_comparison.py`，读取实际训练记录并核对配置解析后的JSON对象与58fe581提交中当时配置相等，分别记录两份文件SHA。保存记录与Git文件格式不同，不能称其原始字节相等。当前配置新增了保存轮次和版本文件字段，不应直接用当前配置替代当时记录。
- 四组评估元数据分别在`reports/e3_complete`、`e3_miss`、`e4_complete`、`e4_miss`；每组保留metrics、completed、run、all_receipts和曲线。
- 原始bbox/逐序列回执归档：服务器`/home/gaob/Miss-RGBT/logs/first5_evaluations_20261014_v1.tar.gz`，42689425字节，SHA256 `4eb2c98fae4eff6dbdbb01496bbd396e2eece3c04ec5848b7c270836e509de6d`。本地`.codex_tmp/miss_rgbt_campaign_20261013/first5_evaluation_raw`已解包并核验归档SHA，不提交大文件。采集详情见`collection.json`。
- epoch05权重SHA由服务器实际读取字节计算，并与四组run记录比对；原始权重留在服务器。本地未下载两份权重，不能称本地独立重新哈希权重。E3 `acd8543a0c68cbac66e62d6625571aa55c601a415652c4cb5535a4b950e81fa4`；E4 `edd94537b2697a4dbe0bf9e2d62c5653827ca1e2dc3da9bb7e6a8d09927b0f5c`。
- 实际评估环境rgbt包源码和713个GT文件快照：`metric_package_manifest.json`，归档SHA256 `f2468a146d2861a0693aad23d600ca1dae8f9862c6e7a32f146092acbd5e0dae`。序列名单和属性文件单独补齐在`metric_metadata_manifest.json`，归档SHA256 `800bb8b5503147cf5b7f65614df06be6cef066365f8fae085056abc575b34ec7`。来自同一实际conda包；补齐时重新核验原728文件，没有改GT或评分规则。本地包路径为`.codex_tmp/miss_rgbt_campaign_20261013/first5_metric_package/rgbt`。
- 评估源码记录103文件，其中102匹配4c473fa。唯一不同为辅助验证脚本，部署版增加一个CADTRACKActor导入并修改一行注释；实际字节快照在`source_snapshots/scripts/validate_focal_normalization.py`。证据见`source_difference.json`及`auxiliary_source.diff`。未覆盖正在训练的服务器源码，不能宣称103文件全部逐字节匹配Git。
- `EXPERIMENT_AUDIT.md/json`由experiment-audit技能要求的新gpt-6-astra/max审计代理生成。它可独立读取离线产物并复算，属于same-family、provisional语义审查；确定性哈希/指标检查与研究结论分开报告。

重建汇总表：

```powershell
uv run --no-project --python 3.12 python analysis/build_first5_comparison.py --project-root C:/Users/gb/projects/Miss-RGBT
```

当前E3/E4只是配对首轮证据。没有E2正式结果、容量匹配对照、多seed或E5正式反馈结果，不能据此声称三个模块都有效。
