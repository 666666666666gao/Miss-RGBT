# 冻结20轮之后的执行优先级

本顺序来自用户2026-10-14后续评审及执行指令。A/B/C尚未实现或验证，不把mask、KD、当前残差MLP直接列为三个已成立贡献。

1. 先收齐当前冻结E3/E4的20轮/50000更新、固定epoch20和四套全量指标。禁止半途改BN、缺失输入或优化设置。
2. 归因控制：固定E0权重的E0-availability；同起点20轮完整输入额外训练E0-FT；同预算E2双路task无KD/R。分清缺失输入处理、额外训练和双路收益。
3. BN单变量：当前冻结设置 vs. 全部BatchNorm2d设eval（包括融合层）。仍可训练affine参数；eval不等于冻结全部BN参数。梯度协调不会自动保护running mean/variance。
4. 稳定控制U上优先比较U/U+A/U+B/U+A+B。A研究任务效用补偿与已有弱证据细化，B研究证据状态及闭环恢复，正式结果支持后再列为结构贡献。
5. 最后比较C与普通加权、双向KD及标准梯度协调，记录实际AdamW增量和训练成本；不能只凭raw梯度点积保证完整性能。
6. 最终候选补关键消融、容量控制和多seed。除E0-availability无需训练，其余正式候选完整20轮并报告四套指标；短步机械检查不替代正式实验。

## 本地源码确认的归因差异

- E0单缺失复制可用图像进入两路（reports/e0/evaluate_miss.py:128–136）。新评估同样复制占位，但把availability传给MissingTracker；model.py:54–59在Transformer前将缺失patch embedding乘0再加位置编码。E0-availability用于隔离输入处理，E3−E0不能全部归因KD。
- train.py:30–33只对box_head内BN调用eval。融合模块在baseline另一子模块decode_fuse_search（lib/models/cadtrack/cadtrack.py:27）；conv默认freeze_bn=False，内部有BatchNorm2d（lib/models/layers/head.py:7–20）。model.cuda().train()后融合BN未设eval，C/M依次前向（train.py:88–89）。这确认冻结范围不同，尚未确认性能退化原因。
- R当前只在缺失位置直接注入残差；共享参数仍更新。B须区分crop内证据不足与需要新的真实观察；位置/尺度提交、模板写入和扩域观察分别归因，不能全部叫模板污染。

## 证据与报告口径

train882/val97确定门限、K和搜索预算；Test失败分析已参与方向形成，不能再称完全未见。主指标为LasHeR complete/Miss的PR/NPR/SR、RGBT234 complete/Miss的MPR/MSR。额外前向、采样、参数与延迟披露，同epoch不等于同计算预算。VTUAV/VTUAV-Miss未评估，不填数或称六套基准已超过。

按实际实现逐步补齐输入状态数量、恢复时延/失败/搜索支持、中心命中下IoU及尺度误差、修复效用与选择预算、身份/模板/位置错误、BN统计、反馈触发、实际更新、FPS/P95和显存；未实现诊断不能因列表存在称已测。

## Baseline来源更正

用户确认E0是去掉CADTrack论文三个模块后训练出的baseline；用户说明DropMoE是CADTrack主干来源，原DropMoE文件暂未提供。本轮直接加载CADTrack_ep0020.pth.tar（SHA d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8），没有另加载DropMoE文件。当前YAML仍有OSTrack路径，但本轮build_cadtrack(training=False)后严格加载E0状态，不用该字段另载预训练；残留字段不能证明baseline最初训练的预训练文件来源。三个模块具体名称后续按原论文核对，不凭未执行分支名虚造。
