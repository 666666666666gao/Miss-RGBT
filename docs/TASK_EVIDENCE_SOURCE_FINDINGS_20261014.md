# 面向跟踪任务的补偿、模板可靠性与完整任务保护：第一方源码核查

2026-10-14。本文按已安装的 `research/SKILL.md` 和 `shared-references/local-codex-policy.md` 阅读第一方论文、作者仓库和指定本地文件。仅写本文及私有来源快照；未运行神经网络、查询 SSH/GPU/进程、修改科学源码/配置或评判实验完整性。full20 正在按既定方案运行；下述测试全部是其后独立进行的 **train/val 提案**，不改变当前任务。

**结论：**OSTrack 提供的是模板条件下的搜索 token 排序机制；MixFormer 的模板分数直接学习的是二分类样本标签；PCGrad 处理当前任务梯度的冲突。它们分别可以启发补偿位置、模板写入判断和训练更新设计，但任何一个都没有为本项目提供已验证的性能提升、完整输入测试精度保证或新颖性结论。[O-rank] [M-label] [M-loss] [P-code]

文中 **来源事实** 指已读代码/论文或指定结果文件；**适配提案** 是本项目尚未实施的设计；**未验证假设** 需要未来数据才能成立。

## 1. 来源版本、归属与许可

| 作者来源 | 本轮 `git ls-remote --symref … HEAD` 结果 | patch 作者日期 | 许可与论文 |
| --- | --- | --- | --- |
| `botaoye/OSTrack`，README 自述官方实现 | `main`：`33b5e12586216b7fd0e95d255bd01ba44cbec759` | 2023-08-03 19:41:02 +0800 | MIT，Copyright 2022 Botao Ye；ECCV 2022，arXiv **2203.11991v4**。[O-owner] [O-commit] [O-license] [O-paper] |
| `MCG-NJU/MixFormer`，README 自述官方实现 | `main`：`8b87088f45af02c614a277176c63181261e93de3` | 2024-02-28 17:50:34 +0800 | MIT，Copyright 2021 Multimedia Research；本文明确读取 **CvT online 路径**及 CVPR 2022 论文 **2203.11082v2**，不把后来的 MixViT/ConvMAE 变体混为同一实现。[M-owner] [M-commit] [M-license] [M-paper] |
| `tianheyu927/PCGrad`，论文链接的作者仓库 | `master`：`c5fbd7c856526373828074f06875230f7f3ee79e` | 2020-04-28 15:44:54 -0700 | MIT，Copyright 2020 Tianhe Yu；NeurIPS 2020，arXiv **2001.06782v4**；该快照的实现是 TensorFlow。[P-owner] [P-commit] [P-license] [P-paper] |

以上是本轮查询时的默认分支提交，不把提交日期写成检索日期。代码和许可证均按完整 SHA 下载。MIT 文本要求保留版权与许可声明；MixFormer SPM 导入的 `external/PreciseRoIPooling` 还有自己的 MIT 声明，Copyright 2018 Jiayuan Mao。本文没有复制任何外部实现进项目，也不对尚未检查的其他依赖作许可结论。[O-license] [M-license] [P-license] [M-roi-license] [M-spm]

## 2. 本地事实：当前修复在保护什么

本轮读取的八份本地源码/配置与本地 Git 提交 `4c473fa5291b42cdd1969ea3d74161e92e142445` 对应 blob 相同；以下 `L-*` 行级链接固定到该版本。这个比对仅用于确定本文所述代码版本。

- **输入与门控：**`miss_rgbt/model.py:49–70` 在进入 Transformer 前，以模态 availability 和 search `patch_keep` 乘掉不观测 patch embedding，再加位置编码。`LocalRepair:20–24` 从同位置 RGB/TIR 特征和 availability 生成残差，乘 `sigmoid(gate) × (1−available)`；这里的特征已经包含前面 Transformer 的上下文。它没有按目标相关性挑选 search token，也没有独立的模板质量标签。门控只限制当层的残差注入位置，后续 self-attention 仍可将修复信息传播到同模态的其他位置。[L-mask] [L-repair] [L-loss] [L-block]
- **完整路由的精确边界：**`model(..., availability=None)` 将全部模板与搜索的 availability 设为 1，故固定同一组参数时修复残差为零。可是 `train.py:88–97` 用同一个模型计算 complete/missing，再共同反传、更新；因此残差门为零不能保护训练后的 shared backbone/fusion/head。另需区分“本帧两模态恢复”和“全部输入完整”：历史模板仍可记录缺失 availability，恢复帧的模板修复仍可能生效。[L-default] [L-train] [L-history]
- **训练任务：**`objectives.py` 使用 `2×GIoU loss + 5×L1 + focal`；forward KD 将 complete response detach 为目标，但 missing 路径仍更新共享参数。`reliable = loss_m.detach()+margin < loss_c.detach()` 是当前样本损失比较；它既不是模板质量监督，也不是梯度方向保护。E3/E4 配置均为 `forward_weight=0.1`、`reverse_weight=0`；两者分别关闭/开启 layer-6 repair。[L-task] [L-loss] [L-e3] [L-e4]
- **模板写入：**`tracker.py:44–59` 用 Hann 加权后的响应最大值作为 confidence，在既定周期超过阈值时写入自身预测框 crop，并保存当帧 availability。模板 0 保留，`pop(1)` 替换后续模板。这个最大值没有被当前损失训练成“正确模板概率”。[L-history] [L-task]

指定的 first5 文件记录如下，单位为百分数，表内显示六位小数。四份 `run.json` 均指向各自 `epoch05.pth.tar`，每个条件记录 479 序列/337,353 帧；其中 LasHeR 245/220,703，RGBT234 234/116,650。这里只转录已闭合报告，不重算轨迹、不替代 full20。[E3C] [E3M] [E4C] [E4M] [E3C-run] [E3M-run] [E4C-run] [E4M-run]

| first5 报告 | LasHeR PR | NPR | SR | RGBT234 MPR | MSR |
| --- | ---: | ---: | ---: | ---: | ---: |
| E3 Complete | 74.388795 | 70.732830 | 59.200450 | 90.251132 | 66.118720 |
| E4 Complete | 75.302106 | 71.670015 | 59.839741 | 90.294694 | 66.223281 |
| E3 Miss | 69.032091 | 64.017231 | 54.835854 | 85.224052 | 61.486632 |
| E4 Miss | 68.461011 | 63.372123 | 54.363025 | 85.613275 | 61.797040 |

算术上，E4−E3 的 Complete 五项依次为 `+0.913311/+0.937185/+0.639291/+0.043562/+0.104561` 个百分点；Miss 为 `−0.571080/−0.645107/−0.472829/+0.389223/+0.310409`。这些数字支持把数据集和输入条件分开报告；**没有给出梯度冲突、错误模板写入或补偿位置不当的因果证据**。本轮不据这些 Test 数字选未来门限、token 预算或优化器参数。

## 3. OSTrack：用什么目标证据选择 search token

**来源事实。**`lib/models/layers/attn_blocks.py::candidate_elimination` 从 attention 取 **模板 query 行→搜索 key 列**的子矩阵 `attn[:, :, :lens_t, lens_t:]`。若给模板 mask，先只取被选模板行；随后对模板行与 heads 求均值，降序排序，保留 `ceil(keep_ratio×当前搜索长度)` 个 token，同时保存它们的原始位置索引。模板 token 本身保留；代码中的未保留 token 融合只是注释，实际没有执行融合。[O-rank]

标准 `vitb_256_mae_ce_32x4_ep300.yaml:54–56` 用 `CE_LOC=[3,6,9]`、每次保留比例 0.7、`CTR_POINT`。`ce_utils.py` 也支持 ALL、中心矩形和 GT_BOX。**配置中的 GT_BOX 是模板框，不是当前搜索帧的测试 GT**：训练 actor 传 `template_anno[0]`；测试 tracker 只在初始化时从给定 `init_bbox` 生成模板 mask，之后传相同模板/mask。中心点模式实际不需要该框数值。[O-config] [O-mask] [O-actor] [O-init]

ECE 真正改变后续网络长度：`CEBlock` 在 attention residual 后、MLP 前移除搜索 token；最后 `vit_ce.py:163–175` 按全局索引恢复位置，并给已删位置填零再还原二维结构。这是候选删除流程，不是一个已经实现的缺失模态补偿器。论文 §3.2 的依据是模板中心与候选的注意力相似度，使用它作相关性排序，并未监督成校准的目标存在概率。[O-block] [O-restore] [O-paper]

**适配提案：先只约束修复位置。**保留本项目完整 token 网格；从 layer-6 修复之前的 **missing 路由**读取当前可用传感器的模板→搜索注意力，以固定初始模板区域作 query，得到搜索相关性 `s_j`。在 train/val 固定的 `K` 个位置给 search 残差加一个选择因子 `q_j`：

\[
\Delta'_{j,m}=q_j\,(1-v_{j,m})\,\sigma(g_{j,m})\,\Delta_{j,m},
\qquad q_j=\mathbf{1}[j\in\operatorname{TopK}(s)] .
\]

这只是从 OSTrack 排序思想导出的 **补偿门控提案**，不称为复现 ECE。模板位置的现有修复规则可保持原定义，使首次对比只改变 search 补偿位置；完整路由的零残差性质继续由 `(1−v)` 决定。[O-rank] [L-repair]

实现边界必须来自现有接口，而不能直接搬运 OSTrack：

1. 本地 `vit_care.py::Block.forward` 用相同 attention/MLP 权重分别处理 RGB/TIR，返回两个特征张量，**目前不返回 attention**。`Attention.forward` 虽有 `return_attention` 参数，Block 没有启用；需有一次明确的接口改动才能得到合法的分数，不能假装现有 `out['attn']` 已可用。[L-block] [L-default]
2. 索引有 **1 个 CLS + 多个模板 + 搜索**；search 起点是 `1 + len(template)×N_z`，不能照抄 OSTrack 的单模板切片。原修复是 RGB/TIR 同位置配对，`forward_head` 要把完整 `N_x` reshape 成二维网格；分别剪掉两传感器不同位置会破坏这种对应。保留网格的补偿门控所需改动更少，也没有 ECE 的 backbone 计算缩减含义。[L-mask] [L-head]
3. 选择器只能读 **已乘 availability/patch_keep 后**的 student 特征或 attention；不能读 complete 路由、乘 mask 前的 embedding、隐藏图像或测试 GT。初始框和过去已经观测的模板是合法状态，模板的 availability 仍须保留。[L-mask] [L-history]
4. 当前训练的两个 patch 掩码独立采样，因此确实可能出现同位置两模态都被遮掉。可用传感器分支对这类位置的后层 attention 只代表从位置与其他可见 token 推断的相关性，不能称其为该位置的直接视觉证据。应在未来分层报告这一情况，而不是从 complete 分支补评分。初始模板的图像虽然固定，经过本项目双向 self-attention 的模板特征也已包含当次搜索上下文，并非恒定身份向量。[L-masksample] [L-mask] [L-block]

**未验证假设。**高相关性位置可能是更值得补偿的目标区域，也可能是相似干扰物；相关性排序不等于修复收益排序。若目标已在当前 crop 外，局部 token 门控没有新增目标观测。必须在 train/val 比较同一 `K` 的目标证据排序与随机位置控制，并记录目标覆盖、错误集中及实际开销；不能从 OSTrack 的速度或成绩外推本项目收益。[O-rank] [L-history]

## 4. MixFormer：学习到的“模板质量”到底是什么

**来源事实，需沿监督链读取。**

| 代码路径 | 实际含义 |
| --- | --- |
| `lib/models/mixformer_cvt/score_decoder.py:32–66` | learned score token 先 attend 搜索候选框的 PreciseRoI pooled 特征，再 attend 初始模板特征，经 MLP 输出一个 logit。结构利用目标条件，但结构本身没有规定质量标签。[M-spm] |
| `lib/train/data/sampler.py:216–250`，`get_one_search:280–298` | 正样本分配 `label=1`；负样本分配 `label=0`，视频优先取不可见帧并使用中心候选框，否则独立采样一个 search。独立采样函数没有显式排除原序列，不能把代码描述成严格验证过的不同身份采样。[M-label] [M-neg] |
| `lib/train/actors/mixformer.py:31–46,69–72`；`train_script_mixformer.py:89–92` | score 标签是 `data['label']`，目标函数是 `BCEWithLogitsLoss`；score 路径会把最终训练 loss 替换为加权 score loss。它没有把候选 IoU、未来帧收益或补偿误差当作 score 回归目标。[M-actor] [M-loss] [M-objective] |
| actor `43–46`；`mixformer_online.py:788–801` | 训练显式传入 `search_anno` 转换后的 ROI；推理没有 GT ROI 时，使用 box head 自己的预测框。对正样本而言存在 **标注框训练、预测框推理**的候选分布差别。[M-actor] [M-roi] |
| `base_functions.py:157–165`；online 配置 `TRAIN_SCORE=true` | 该阶段只训练 score 参数，冻结其余参数。论文也描述先训练 tracker，再训练 SPM；不能把 score 训练说成共同改善了其 backbone。[M-freeze] [M-config] [M-paper] |
| `lib/test/tracker/mixformer_cvt_online.py:135–168` | sigmoid 后分数须 `>0.5` 且高于当前保存的最高分才更新候选；到更新周期才提交候选。每个周期重设分数并把下一候选初始化为第一帧模板，因此它不是“每个低分帧都停写且永久保留上一模板”的同义实现。[M-update] |

所以，更精确的解释是 **学到正/负训练样本标签的目标条件分数，并将其用于挑选在线模板**。不能把它直接解释为“预测 IoU”“传感器质量”“该模板会让后续跟踪变好”或“0.9 等于 90% 正确写入概率”；后两种解释还需要符合部署分布的监督/校准证据。[M-label] [M-loss] [M-update]

**适配提案。**首先在未来 train/val 自身预测轨迹上记录现有写入候选、初始模板关系、Hann confidence、当时 availability 和 GT 匹配结果，检验现有分数是否能识别错误写入。若增加 learned score，需先明确定义它的标签：

- 若沿用 MixFormer 的二分类，论文表述应为目标匹配/正负分类；模态缺失本身不能当作负类，否则会拒绝合法的单模态目标。
- 若本项目要预测“该预测框是否适合作为模板”，可在 **train** 的真实预测候选上，以预先定义的 GT IoU/目标匹配条件产生二分类标签，并在 **val** 校准写入阈值。这是新的监督定义，不能借 MixFormer 的名称声称作者已经验证。
- 若标签声称“未来模板收益”，则需要相同历史下写入/不写入该候选的未来 train/val 反事实比较；当前框 IoU和目标存在标签都不能替代这一目标。

本地 `forward_tokens` 只返回搜索区的拼接特征，随后使用本地 fusion/CENTER head；SPM 需要同通道的候选 ROI 与初始模板特征，还依赖 PreciseRoIPooling 扩展。故它不是可直接挂到 `score_map.max()` 的权重或无成本替换。更换模板写入规则也会改变完整输入的状态轨迹，即使 repair 在所有可用位置为零；必须另做完整输入闭环验证。[L-mask] [L-head] [M-spm] [L-history]

## 5. PCGrad：可以约束什么，不能保证什么

**来源事实。**作者 `PCGrad_tf.py:28–68` 分别求各 loss 梯度；当内积为负，减去它在另一任务梯度上的负投影，最后相加；`76–95` 把结果交给被包装优化器。论文 Algorithm 1 对所有任务对称处理，没有指定“完整任务永远优先”。论文采用随机其他任务顺序；但这个固定代码版本在 `36` 行调用 `tf.random.shuffle(loss)` 后没有接回返回值，投影循环仍是 `range(num_tasks)`。因此不能把该快照直接描述为实现了论文全部随机顺序细节。[P-code] [P-optimizer] [P-paper]

论文 Theorem 1 在 **两个凸可微目标、总梯度 Lipschitz、足够小步长**下讨论总目标收敛，并允许两梯度反向共线时停在次优点；Appendix A.4 另分析强凸/光滑条件下具有特定步长的 heavy-ball 动量。它没有给本项目非凸网络、分组学习率、AdamW 与测试准确率作保证。[P-paper]

**从公式得出的有限结论。**令 `g_c=∇L_c`，`g_a=∇L_a`；`L_a` 可明确包含 missing task 和已加权 KD。对两个非零梯度，冲突内积 `c=g_cᵀg_a<0` 时，对称 PCGrad 的和记为 `d`，直接代数得到

\[
g_c^T d=\|g_c\|^2-\frac{c^2}{\|g_a\|^2}\ge 0.
\]

这只说明 **以同一标量步长做 `θ←θ−ηd` 的当前梯度一阶方向**对完整训练 loss 不上升；不是有限步后 loss 必然下降。反向共线时等号成立、对称 PCGrad 可完全停步。以上是本文对作者投影式的推导，不是本项目实测。[P-code]

若明确要优先保留 complete 方向，可以另提 **单向投影适配**：保留 `g_c`，只把 `g_a` 的冲突分量从 `g_c` 方向去除，再相加；对于 `g_c≠0`，其 raw 方向满足 `g_cᵀd≥‖g_c‖²`。这改变了作者的对称规则，必须称为本项目提案，不能直接标成原版 PCGrad 的结论。零 anchor 不能用该除法，也没有可保护的一阶方向；是否在共享参数上出现零 anchor，应由未来实际梯度记录回答。

**本项目的具体限制。**

1. **参数范围。**需要约束的是两个任务共享且实际相连的 backbone、fusion 和 head 参数。完整路由中 repair 参数的任务梯度为零，单独在 repair 参数上做投影没有 complete anchor；新模块自身可以保留辅助任务梯度。作者 TF 代码把 `None` 梯度跳过后再按 `var_list` 展开，不能作为任意参数连接关系下的 PyTorch 即插即用实现。[L-repair] [L-train] [P-code]
2. **梯度累积。**本地是 microbatch 4、accumulation 2，先累计再 clip/AdamW。若保护的是一次有效 batch 更新，应分别累计 `g_c`、`g_a` 后才作一次投影；逐 microbatch 投影再累加是另一算法，因为投影不是线性的。原来一次合并 `.backward()` 也不能恢复两份任务梯度，需要改变梯度取得方式和记录其训练成本。[L-e4] [L-train]
3. **实际优化器方向。**本地 backbone 学习率乘 0.1，其余组用主学习率，之后还有 AdamW 的历史动量、逐参数缩放和 weight decay。raw 欧氏内积非负并不推出实际更新 `g_cᵀΔθ≤0`。例如纯代数 `g_c=(1,1)`、`d=(−1,3)` 的内积为 2，但正对角缩放 `D=diag(4,1)` 后 `g_cᵀDd=−1`。这不是本项目测量，只说明为何必须检查实际更新；统一正比例 norm clipping 本身不改变方向，但不能消除后续优化器的区别。[L-train]
4. **损失/指标范围。**即使某个 train batch 的完整任务 loss 降低，也不能推出其他训练样本、val/test PR/SR 或长时模板轨迹改善。若把 C、M、KD 拆成三个任务，顺序投影还不构成所有约束始终满足的共同保证；首次提案宜明确 C 与“其余已加权目标”的两项定义，而非把每个 loss 项都命名为独立任务。[P-code] [L-loss]
5. **已有 reliability gate 不替代梯度检查。**missing 当前 loss 比 complete 小，不代表 reverse-KD 更新与 `g_c` 对齐；而 E3/E4 的 reverse 权重本来就是零。本轮不能以该 gate 的存在说明已经实现了 complete-task protection，也不能从 first5 指标倒推梯度冲突真实存在。[L-loss] [L-e3] [L-e4]

**未验证假设。**在未来 train/val 真正观测到辅助目标与完整目标的更新冲突后，投影可能是值得对比的训练干预；是否改善训练或跟踪精度尚无本项目证据。若只是测试时加修复 gate，训练期共享参数的变化仍需单独评估。

## 6. 未来最小验证顺序，不动 frozen full20

| 未来问题 | 先收集的 train/val 证据 | 独立对比及不能越界的解释 |
| --- | --- | --- |
| 模板证据是否选对补偿位置 | 在同一可见输入上保存修复前的 `s_j`、availability、所选原始位置；用 GT 仅离线统计目标区域覆盖。单独记录 patch 双遮挡和 crop 外目标。 | 当前局部修复 vs 固定 K 的任务选择；配一个同 K 随机位置控制。改变被 mask 的原图数值时，选择与输出须不依赖它；部署选择器不能访问 complete 分支。保持二维网格；不能声称减少 backbone FLOPs 或能从 crop 外恢复。 |
| 分数是否减少错误模板写入 | 在 tracker 自己的候选框上标注/评估，与官方 SPM 使用 GT ROI 训练的分布差别分开；按 complete/RGB-only/TIR-only/恢复状态报告校准、错误接受与正确拒绝。 | 先比较现有 Hann confidence 与目标条件分数的候选排序；再单独改变模板写入决策，保留原定位提交以隔离问题。标签定义、阈值及更新预算在 train/val 固定；完整输入的状态轨迹也必须测。 |
| 是否确有需要处理的训练冲突 | 在匹配有效 batch 上记录 `g_cᵀg_a`、两者 norm、各参数组以及实际 AdamW 更新的 `g_cᵀΔθ`；记录更新前后完整 train loss。 | 若有冲突，再比较原加权和与一个明确定义的投影方案；对称 PCGrad和单向 complete anchor 不能混名。既比较完整/缺失 val 全序列指标，也记录训练时间/显存。内积与单步 loss 是机制诊断，不是测试精度替代物。 |

任何新候选均应从约定的相同起点、数据划分与训练预算单独训练，先在 train/val 决定机制和阈值，固定后再做后续正式评估。first5 Test 数字已经是本轮问题背景，因此不能把它们称为对本轮设计完全未见的开发外证据。当前 full20 的源码、权重选择、阈值和运行日程均不由本文建议调整。

## 7. 可复查材料

私有缓存位于 `C:/Users/gb/.codex_tmp/miss_rgbt_task_evidence_sources_20261014/`。`source_receipts.json` 记录 22 份作者仓库文件、3 份 tree JSON、3 份 commit patch、3 份版本固定的论文 PDF，以及 20 份本地输入快照。22 份下载的仓库文件均按 Git blob SHA1 与作者仓库 API 对应提交 tree 核对，并记录 SHA256/字节数；本地八份源码/配置另绑定上述本地 Git commit。论文文本另存 `papers/*.txt` 供定位，论文语义仍由版本固定的 PDF 与源码负责。

四份指定指标快照的 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `reports/e3_complete/metrics.json` | `414701ee1b8450bdb691ffaecc87a62bc3998cea4ef2a601280e999703f4bd0a` |
| `reports/e3_miss/metrics.json` | `e898a4600b58d05337c0a93ed7a85300da0dd371be8ec115cd6b51d2a51c331d` |
| `reports/e4_complete/metrics.json` | `12c164321122f04994181b8639a3485f0629bf6fe8e84ae0b5b46b2ec8efe11f` |
| `reports/e4_miss/metrics.json` | `095fa6d2199471d164147af77f6488213c5a0952ffbdc6e2ef96a65cc8a7a444` |

本轮的来源文件匹配不等同于实验完整性审计。没有下载权重、执行外部作者代码或测试任何上述候选。

[O-owner]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/README.md#L1-L4
[O-commit]: https://github.com/botaoye/OSTrack/commit/33b5e12586216b7fd0e95d255bd01ba44cbec759.patch
[O-license]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/LICENSE#L1-L21
[O-paper]: https://arxiv.org/pdf/2203.11991v4
[O-rank]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/lib/models/layers/attn_blocks.py#L9-L74
[O-block]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/lib/models/layers/attn_blocks.py#L92-L103
[O-mask]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/lib/utils/ce_utils.py#L15-L65
[O-config]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml#L50-L64
[O-actor]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/lib/train/actors/ostrack.py#L53-L73
[O-init]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/lib/test/tracker/ostrack.py#L51-L85
[O-restore]: https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/lib/models/ostrack/vit_ce.py#L143-L178
[M-owner]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/README.md#L1-L3
[M-commit]: https://github.com/MCG-NJU/MixFormer/commit/8b87088f45af02c614a277176c63181261e93de3.patch
[M-license]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/LICENSE#L1-L21
[M-paper]: https://arxiv.org/pdf/2203.11082v2
[M-spm]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/models/mixformer_cvt/score_decoder.py#L9-L66
[M-roi-license]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/external/PreciseRoIPooling/LICENSE#L1-L21
[M-label]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/train/data/sampler.py#L216-L250
[M-neg]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/train/data/sampler.py#L280-L298
[M-actor]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/train/actors/mixformer.py#L31-L46
[M-loss]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/train/actors/mixformer.py#L66-L79
[M-objective]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/train/train_script_mixformer.py#L84-L92
[M-roi]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/models/mixformer_cvt/mixformer_online.py#L788-L803
[M-freeze]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/train/base_functions.py#L153-L168
[M-config]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/experiments/mixformer_cvt_online/baseline.yaml#L70-L86
[M-update]: https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/test/tracker/mixformer_cvt_online.py#L135-L168
[P-owner]: https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/README.md#L1-L7
[P-commit]: https://github.com/tianheyu927/PCGrad/commit/c5fbd7c856526373828074f06875230f7f3ee79e.patch
[P-license]: https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/LICENSE#L1-L21
[P-paper]: https://arxiv.org/pdf/2001.06782v4
[P-code]: https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/PCGrad_tf.py#L28-L68
[P-optimizer]: https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/PCGrad_tf.py#L76-L95
[L-repair]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/miss_rgbt/model.py#L8-L24
[L-default]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/miss_rgbt/model.py#L34-L47
[L-mask]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/miss_rgbt/model.py#L49-L77
[L-block]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/lib/models/cadtrack/vit_care.py#L14-L63
[L-head]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/lib/models/cadtrack/cadtrack.py#L73-L96
[L-task]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/miss_rgbt/objectives.py#L8-L26
[L-loss]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/miss_rgbt/objectives.py#L29-L53
[L-train]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/miss_rgbt/train.py#L50-L98
[L-masksample]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/miss_rgbt/train.py#L17-L27
[L-history]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/miss_rgbt/tracker.py#L27-L60
[L-e3]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/configs/e3.json#L1-L37
[L-e4]: https://github.com/666666666666gao/Miss-RGBT/blob/4c473fa5291b42cdd1969ea3d74161e92e142445/configs/e4.json#L1-L37
[E3C]: ../reports/e3_complete/metrics.json
[E3M]: ../reports/e3_miss/metrics.json
[E4C]: ../reports/e4_complete/metrics.json
[E4M]: ../reports/e4_miss/metrics.json
[E3C-run]: ../reports/e3_complete/run.json
[E3M-run]: ../reports/e3_miss/run.json
[E4C-run]: ../reports/e4_complete/run.json
[E4M-run]: ../reports/e4_miss/run.json
