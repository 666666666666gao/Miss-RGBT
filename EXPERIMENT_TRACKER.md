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
| E3/E4 完整20轮 | DONE，10月15日11:00已核验 | train882；20轮×20000采样元组，各50000更新；固定最终epoch20 | 两项训练及四组全量评估exit0；1916次序列运行/1349412帧；E3目标0/4、E4目标1/4；见reports/full20_comparison_20261013_v1 |
| 反向反馈 CPU 实现检查 | DONE，19:39退出0 | 固定E4 epoch01，seed2027，真实train单microbatch | report status=DIRECTION_AND_UPDATE_VERIFIED；不是E5性能结果 |
| E0同帧与历史损伤诊断 | DONE | 479序列，无新增NN；帧加权描述统计 | reports/e0_paired_diagnostics_v1；不是正式PR/SR或新方法成绩 |
| 最终20轮独立审计 | WARN；same-family/provisional | 实际原始记录与评分源码离线复算 | reports/full20_comparison_20261013_v1/EXPERIMENT_AUDIT.md/json；不外推统计稳定性或三个模块有效 |
| 归因短检查 | DONE，各2次真实更新 | E0-FT/GPU2与E2/GPU3；严格重载和三输入状态推理 | reports/attribution_preflight_20261015.json；不是正式训练 |
| 归因存储准入 | DONE | v1在NN前失败；只迁移两份本轮短检查权重，SHA核验 | reports/attribution_storage_20261015.json；新正式权重写data2，旧权重未动 |
| 冻结末轮定位/权重复核 | DONE，离线CPU | 63框质量组、63共同命中组、54crop组；不增加NN结果 | reports/full20_localization_diagnostics_20261015_v2；原主指标不变、初始审计WARN及后续限定保留 |
| LasHeR val97静态准备 | DONE，12:14只读快照 | 97序列/54646帧，99份split/GT哈希核验 | reports/development_data_20261015；未运行验证、生成mask或完成选参 |
| E0保存训练元数据 | DONE，15:06只读CPU加载 | epoch20及原SHA；保存配置路径、train/val统计，无完整cfg或split名单 | reports/development_data_20261015/e0_saved_training_metadata.json；不认证原E0的历史开发集独立性 |
| 评分器公开来源核查 | DONE，仅静态文本/发行字节 | 本地1967文件及评测后3556项清单对应rgbt1.0.1发行包 | docs/SCORING_PROVENANCE_FINDINGS_20261015.md；不认证作者工具箱/SCDT严格等价，不改原指标 |
| BN冻结CPU机械检查 | DONE，17.408秒退出0 | E3/E4真实模型13个BN子模块；allbn 39缓冲不变、affine更新 | 独立准备提交71f46eb；不是GPU任务前向或完整训练 |
| E3/E4-allbn-full20 | PREPARED，未启动 | 配置/预检/队列在独立分支；须先完成阶段1 | reports/bn_controls_preparation_20261015.json；当前活动科学源码未改 |
| A任务效用补偿 | ISOLATED_IMPLEMENTED，CPU原生检查exit0 | 原E0，新增596357参数；一个真实元组两次更新，39BN/主任务RNG/不可见输入/重载检查 | reports/task_utility_cpu_20261015_v1；执行源码a229afc、分支f6cead7；不是GPU或正式20轮成绩 |
| E0-availability | DONE，两项全评估exit0 | 0更新、完整/Miss共958序列运行/674706帧 | reports/e0_availability_20261015_v1；完整相等、Miss下降；审计WARN |
| E0-FT-full20 | DONE，07:48闭环，08:12审计WARN | 20轮/50000更新exit0；四套完整指标，958序列运行674706帧 | reports/e0_ft_20261016_v1；10指标独立复算误差0，单seed/协议等限制保留 |
| E2-full20 | 训练DONE，17:03 exit0；评估RUNNING | 20轮/50000更新，固定末轮完整评估GPU2、3已启动 | reports/attribution_runtime_20261016_1703.json；18:39下一观察，最终10指标待产出 |
| 模板U0/U1/U2 | ISOLATED_IMPLEMENTED，未NN/正式运行 | Python3.12/3.7各9项stdlib stub控制流检查通过 | reports/template_controls_preparation_20261015；23c6b46，仅准备，不改当前队列 |
| E5 选择性反馈 | TODO | E4有效后再做 | 未验证 |
| 三个有效论文模块 | UNPROVEN | 需要主结果和消融 | 不能据 native PASS 宣称成立 |

历史14:10证据（现由上方07:48闭环及07:42 E2记录更新；原61699已失效退役，当前仅隐藏43188/17:02）：原3294032/3623093已完成；新归因控制器3796810仍运行。E0-availability两项全评估均exit0，完整五项指标等于E0、Miss五项下降。E0-FT/GPU2与E2/GPU3于13:19实际启动，14:10分别2825/50000和1529/50000更新。原观察器48256已完成消费；唯一后续观察器61699/python65768于10月16日04:22:40.141988北京时间采集，之前不查询NN/GPU。科学源码保持dfc989e；BN准备分支71f46eb未正式训练，A/B/C有效性仍未证实。

原始 native 的反向可靠比例为0，不能据该单步证明反向反馈实际发挥作用。E5尚未训练。


2026-10-13 时序源码核查完成（DiMP/SAM2固定提交，仅静态，不是新效果）。独立CPU反向探针已执行并通过真实可靠门、梯度方向、AdamW更新差异和严格重载检查，详见reports/cpu_reverse_probe_ep01_seed2027_v1。该历史源码核查未改变当时冻结队列；现最终20轮已完成并单独归档。来源笔记见docs/TEMPORAL_STATE_SOURCE_FINDINGS_20261013.md；执行约束和证据范围见docs/MODULE_EVIDENCE_PROTOCOL.md。
