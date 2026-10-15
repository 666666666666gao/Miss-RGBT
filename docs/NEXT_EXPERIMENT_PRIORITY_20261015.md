# 冻结20轮之后的执行优先级

本顺序来自用户2026-10-14后续评审及执行指令。A已在隔离分支实现并通过真实数据CPU机械检查，B/C尚未实现；三者完整性能均未验证，不把mask、KD、当前残差MLP直接列为三个已成立贡献。

1. 已完成（2026-10-15）：冻结E3/E4各20轮/50000更新、固定epoch20和四套全量指标已归档、复算；审计WARN及限制另见完整报告。执行期间没有改变BN、缺失输入或优化设置。
2. 归因控制：固定E0权重的E0-availability；同起点20轮完整输入额外训练E0-FT；同预算E2双路task无KD/R。分清缺失输入处理、额外训练和双路收益。
3. BN单变量：当前冻结设置 vs. 全部BatchNorm2d设eval（包括融合层）。仍可训练affine参数；eval不等于冻结全部BN参数。梯度协调不会自动保护running mean/variance。
4. 稳定控制U上优先比较U/U+A/U+B/U+A+B。A研究任务效用补偿与已有弱证据细化，B研究证据状态及闭环恢复，正式结果支持后再列为结构贡献。
5. 最后比较C与普通加权、双向KD及标准梯度协调，记录实际AdamW增量和训练成本；不能只凭raw梯度点积保证完整性能。
6. 最终候选补关键消融、容量控制和多seed。除E0-availability无需训练，其余正式候选完整20轮并报告四套指标；短步机械检查不替代正式实验。

模块A隔离准备（2026-10-15）：[分支f6cead7](https://github.com/666666666666gao/Miss-RGBT/tree/f6cead72d19a2544236cbcba2a7a86d2f15ce80f)提供效用/相似度/随机同K/全位置四份完整预算配置。实际CPU执行源码a229afc、原E0、单个真实元组两次更新、八项单元检查、39个BN缓冲及RNG/重载检查已通过，见[报告](../reports/task_utility_cpu_20261015_v1/README.md)。只在独立prepared目录运行，没有改变活动main或归因队列，也没有正式GPU/20轮成绩。A反事实当前要求全部BN运行统计固定，U控制必须匹配；正式次序仍是归因→BN→A/B，不因预备检查通过跳过控制。

## 本地源码确认的归因差异

- E0单缺失复制可用图像进入两路（reports/e0/evaluate_miss.py:128–136）。新评估同样复制占位，但把availability传给MissingTracker；model.py:54–59在Transformer前将缺失patch embedding乘0再加位置编码。E0-availability用于隔离输入处理，E3−E0不能全部归因KD。
- train.py:30–33只对box_head内BN调用eval。融合模块在baseline另一子模块decode_fuse_search（lib/models/cadtrack/cadtrack.py:27）；conv默认freeze_bn=False，内部有BatchNorm2d（lib/models/layers/head.py:7–20）。model.cuda().train()后融合BN未设eval，C/M依次前向（train.py:88–89）。这确认冻结范围不同，尚未确认性能退化原因。
- R当前只在缺失位置直接注入残差；共享参数仍更新。B须区分crop内证据不足与需要新的真实观察；位置/尺度提交、模板写入和扩域观察分别归因，不能全部叫模板污染。

## 证据与报告口径

train882/val97确定门限、K和搜索预算；Test失败分析已参与方向形成，不能再称完全未见。主指标为LasHeR complete/Miss的PR/NPR/SR、RGBT234 complete/Miss的MPR/MSR。额外前向、采样、参数与延迟披露，同epoch不等于同计算预算。VTUAV/VTUAV-Miss未评估，不填数或称六套基准已超过。

按实际实现逐步补齐输入状态数量、恢复时延/失败/搜索支持、中心命中下IoU及尺度误差、修复效用与选择预算、身份/模板/位置错误、BN统计、反馈触发、实际更新、FPS/P95和显存；未实现诊断不能因列表存在称已测。

## Baseline来源更正

用户确认E0是去掉CADTrack论文三个模块后训练出的baseline；保留的[作者README](BASELINE_ORIGINAL_README.md)将三个模块列为Mamba-based Feature Interaction（MFI）、Contextual Aggregation Module（CAM）和Deformable Alignment Module（DAM）。用户说明DropMoE是CADTrack主干来源，原DropMoE文件暂未提供。本轮直接加载CADTrack_ep0020.pth.tar（SHA d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8），没有另加载DropMoE文件。当前YAML仍有OSTrack路径，但本轮build_cadtrack(training=False)后严格加载E0状态，不用该字段另载预训练；残留字段不能证明baseline最初训练的预训练文件来源。

## 用户授权的数据集专用模块配置

保持骨干和高层模块组合固定，后续允许按数据集调整模块内部参数或层数：例如修复宽度、模块内部深度、任务位置数量、门控/可靠性阈值及状态恢复预算。固定学习率、batch与正式训练预算；当前冻结E3/E4配置不动。repair_width是当前已有内部宽度配置；repair_layer是模块插入点，保持第6块不列入数据集内部调参；内部repair_depth和A/B效用参数尚未实现，不能仅因列入计划称代码已支持。

每个数据集的配置由独立开发/验证数据决定，同一数据集完整/Miss条件共用固定配置。结构参数改变需要相应重新训练与保存一致配置的权重；不能在评估时任意改变层数后沿用旧权重。分别披露参数量、前向数、延迟和训练成本，标注dataset-specific配置，区别于一套统一模型/配置。

当前已确认LasHeR val97。RGBT234尚未指定独立开发/验证集；正式对其单独选参之前，先核实可用开发数据和协议。公开Test已有失败分析需披露，不能把同一Test上的最优参数选择写成独立验证。E3的LasHeR最好与E4的RGBT234最好仍不是一套模型。

2026-10-15 12:14已完成[LasHeR val97只读数据清点](../reports/development_data_20261015/README.md)：97序列、54646帧，split序列名无交集；未生成开发mask或运行模型，不能记作验证完成。534行非正坐标与260行非正宽/高需在正式开发评分及状态监督协议中明确处理。当前[冻结末轮离线复核](../reports/full20_localization_diagnostics_20261015_v2/README.md)使用的是已参与失效分析的Test，只用于问题定位，不能据此选择每数据集内部参数。

15:06实际CPU读取原E0的保存元数据：只有cfg_file路径和train/val统计，没有完整cfg、序列名单及其哈希，不能据此认证val97未被原E0训练使用。当前适配的train882/val97名称独立已经核对，原始E0的历史训练划分独立性仍需分开说明，见同一开发数据记录。未改变开发划分、活动实验或选参规则。
