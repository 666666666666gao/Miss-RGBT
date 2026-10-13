# 缺失模态蒸馏源码核查（2026-10-13）

核查时间：2026-10-13，UTC 09:32 后。范围：论文原文、作者仓库、CVF 官方补充源码，以及本项目相关文件的只读检查。**本次没有运行这些外部模型，没有修改 `miss_rgbt/`、`lib/`、`scripts/`、`configs/`、`experiments/` 或任何任务队列；以下效果建议均未实施。**

主要结论：SDSTrack 已有完整/遮挡共享模型的双路径训练；SMML 已有根据训练标签选择互教方向的设计；RobustSeg 会议版确实有公开源码，但关键反馈脚本没有实现论文所述的教师选择性更新。三者可帮助定义对照与实现约束，不能把机制组合直接写成新颖性或三个有效贡献。[SDSTrack §3.2](https://arxiv.org/html/2403.16002v1#S3.SS2)、[SMML PBC](https://arxiv.org/html/2507.07592v1#S3.SS2.SSS1)、[RobustSeg 反馈脚本](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/tools/train_deliver_mad50_proto_cur_ifv_dual.py#L77-L84)

## 1. 身份、版本和许可证

| 对象 | 已核实身份与源码版本 | 许可证状态 |
| --- | --- | --- |
| SDSTrack | Hou 等，*SDSTrack: Self-Distillation Symmetric Adapter Learning for Multi-Modal Visual Object Tracking*，CVPR 2024，26551–26561。[CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_SDSTrack_Self-Distillation_Symmetric_Adapter_Learning_for_Multi-Modal_Visual_Object_Tracking_CVPR_2024_paper.html)。`hoqolo/SDSTrack`：`822d9852fee66bdd6d9df4dcb679eebecd85e91c`，commit UTC **2024-06-30 04:13:58**。[commit](https://github.com/hoqolo/SDSTrack/commit/822d9852fee66bdd6d9df4dcb679eebecd85e91c) | [MIT](https://github.com/hoqolo/SDSTrack/blob/822d9852fee66bdd6d9df4dcb679eebecd85e91c/LICENSE) |
| SMML | Liang 等，*Semantic-guided Masked Mutual Learning for Multi-modal Brain Tumor Segmentation with Arbitrary Missing Modalities*，AAAI 2025，39(5):5137–5145，DOI `10.1609/aaai.v39i5.32545`。[出版方](https://ojs.aaai.org/index.php/AAAI/article/view/32545)。`banbooliang/SMML`：`d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f`，commit UTC **2025-07-11 01:47:41**。[commit](https://github.com/banbooliang/SMML/commit/d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f) | 固定版本完整 tree 未见 LICENSE；GitHub API `license=null`。不能记为已核实有开源复用许可。[tree](https://api.github.com/repos/banbooliang/SMML/git/trees/d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f?recursive=1) |
| RobustSeg 会议版 | **Jiaqi Tan、Xu Zheng、Yang Liu**，*Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation*，CVPR 2026，27665–27675。[CVF 正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.html)。官方 [supplemental.zip](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Tan_Towards_Robust_Multi-Modal_CVPR_2026_supplemental.zip) 包含 `RobustSeg_Source_Code/`，不是只有 PDF。ZIP 无 Git commit；HTTP Last-Modified UTC **2026-04-30 19:05:58**，SHA256 见下。 | ZIP 内 `RobustSeg_Source_Code/LICENSE:1–21` 为 MIT，copyright 2025 RobustSeg。[官方源码 ZIP](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Tan_Towards_Robust_Multi-Modal_CVPR_2026_supplemental.zip) |
| RobustSeg 当前可读 Git 仓库 | `RobustSeg/RMMSS`，`fda61fc603e0d110f96db79c52a531aa604f5fc6`，commit UTC **2025-11-20 02:25:06**。[commit](https://github.com/RobustSeg/RMMSS/commit/fda61fc603e0d110f96db79c52a531aa604f5fc6)。README 自称官方 RobustSeg；与 CVF ZIP 的 README 文本一致。[README:1–2](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/README.md#L1-L2) | [MIT](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/LICENSE) |

版本注意：`arXiv:2505.12861v1` 是含 representation regularization 的早期 RobustSeg；`v2` 标题变为 RMMSS，采用冻结完整/鲁棒双教师和 FSM。**不能把其 FSM 当作 CVPR 2026 反馈机制。**旧 `RobustSeg/RobustSeg` 链接本次返回 404，但不能据此说会议版“无源码”：官方附件及上表仓库均可读取。[v1](https://arxiv.org/abs/2505.12861v1)、[v2](https://arxiv.org/abs/2505.12861v2)、[旧地址 API](https://api.github.com/repos/RobustSeg/RobustSeg)、[CVF 附件](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Tan_Towards_Robust_Multi-Modal_CVPR_2026_supplemental.zip)

## 2. SDSTrack：共享双路径和完整输入能力的直接近邻

以下均是固定 SHA 的**静态代码行为**：

- `VisionTransformerCE.forward_features` 在 patch embedding 后产生干净与 masked 输入，循环复用同一 `blk`；在 **0-based `[3,6,9,11]`** 处保存融合特征。`apply_mask` 将 RGB token 以 0.3 概率置零，另一模态只在 RGB 未被遮挡的位置继续随机置零，保证同一 token 不同时遮掉两模态。它不是可直接替代整路传感器失效协议的实现。[vit_ce.py:130–139,210–245,331–347](https://github.com/hoqolo/SDSTrack/blob/822d9852fee66bdd6d9df4dcb679eebecd85e91c/lib/models/sdstrack/vit_ce.py#L130-L139)、[apply_mask](https://github.com/hoqolo/SDSTrack/blob/822d9852fee66bdd6d9df4dcb679eebecd85e91c/lib/models/sdstrack/vit_ce.py#L331-L347)
- `SDSTrackActor.compute_losses` 同时对两路使用 GIoU/L1/focal；总损失是 **`0.6*L_clean + 0.2*L_masked + 0.1*MSE_stage_mean`**。MSE 两端都没有 detach，特征收集处也没有 `no_grad`；这是双端可求导的特征一致性，不能当作“冻结教师的响应 KL”实现。源码中泛化的 `DISTILL_LOSS_TYPE='KL'` 设置不改变这里实际调用 MSE 的事实。[actors/sdstrack.py:85–133](https://github.com/hoqolo/SDSTrack/blob/822d9852fee66bdd6d9df4dcb679eebecd85e91c/lib/train/actors/sdstrack.py#L85-L133)、[特征收集](https://github.com/hoqolo/SDSTrack/blob/822d9852fee66bdd6d9df4dcb679eebecd85e91c/lib/models/sdstrack/vit_ce.py#L210-L245)、[train_script.py:64–73](https://github.com/hoqolo/SDSTrack/blob/822d9852fee66bdd6d9df4dcb679eebecd85e91c/lib/train/train_script.py#L64-L73)
- `get_optimizer_scheduler` 只训练名字含 `Adapter`、`multimodal` 的参数，以及低学习率的 `backbone.norm`；其余参数冻结。这提供了限制对预训练能力改动的具体实现。`infer=True` 跳过辅助 masked 路径，部署不需要另一条干净输入前向。[base_functions.py:171–181](https://github.com/hoqolo/SDSTrack/blob/822d9852fee66bdd6d9df4dcb679eebecd85e91c/lib/train/base_functions.py#L171-L181)、[vit_ce.py:210–251](https://github.com/hoqolo/SDSTrack/blob/822d9852fee66bdd6d9df4dcb679eebecd85e91c/lib/models/sdstrack/vit_ce.py#L210-L251)

**可迁移启发，尚非本项目效果证据：**保留完整输入任务监督，并明确限制哪些参数承担鲁棒性适配；不能仅凭多阶段一致性损失下降认定任务响应或完整输入提升。该文已覆盖共享完整/masked 路径，本项目不能把这层结构本身列为独有贡献。[论文 §3.2–3.3](https://arxiv.org/html/2403.16002v1#S3.SS2)

## 3. SMML：有标签的可靠互教，而非无条件“缺失教师更强”

**论文已核实：**PBC 用训练分割标签计算两分支逐 voxel CE，以较低 CE 的分支提供知识；两方向 KL 各按被选中的 voxel 数归一化。它不是置信度/熵单独决定方向，也不是固定完整→缺失。FRC 对训练标签生成的类原型关系做约束；这依赖分割语义类，直接搬到只有目标框的跟踪任务需重新定义。[论文 PBC 式(1)–(3)、FRC 式(4)–(9)](https://arxiv.org/html/2507.07592v1#S3.SS2)

**源码能核实的边界：**

- `train.main` 创建两个独立 `smml.Model` 和独立 Adam；数据集从 15 个非空模态子集随机选两个不同 mask，含全模态子集。不是本项目的一份参数、完整 C 与缺失 M 两次独立前向。[train.py:87–98](https://github.com/banbooliang/SMML/blob/d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f/train.py#L87-L98)、[datasets_nii.py:26–28,102–105](https://github.com/banbooliang/SMML/blob/d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f/data/datasets_nii.py#L102-L105)
- `kl_diverse_loss(fuse_pred1, fuse_pred2, target, ...)` 确实接收标签；`epoch > 100` 才把双向 KL 和 refinement consistency 加到总损失。但 **`utils/criterions.py`、`MAD` 等实现没有包含在该完整仓库 tree 中**，只有导入和调用。因此无法从公开代码确认其 PBC 的 target detach、精确归一化、空选择行为及实际梯度方向，也不能声称该仓库已直接跑通。[train.py:21–26,184–195,238–258](https://github.com/banbooliang/SMML/blob/d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f/train.py#L184-L195)、[完整 tree](https://api.github.com/repos/banbooliang/SMML/git/trees/d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f?recursive=1)
- `MaskModal.forward` 对各尺度模态特征置零；训练期 refinement 输入中的 `fuse_pred.detach()` 可核实，不能用这一处 detach 代替缺失的 PBC 实现证据。推理关闭 refinement，但仍执行两个学生并平均预测，不能引用为单模型推理成本。[smml.py:303–312,449–479](https://github.com/banbooliang/SMML/blob/d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f/smml.py#L449-L479)、[predict.test_softmax:183–201](https://github.com/banbooliang/SMML/blob/d40154138dc0e7cc0ab7e7413ee4e51332b3ca8f/predict.py#L183-L201)

**可迁移启发：**本项目用训练框产生的原生 `2*GIoU+5*L1+focal` 比较 C/M，属于任务级可靠性改写。`loss_M+0.1 < loss_C` 的 margin、整样本筛选、以及对全 batch 求均值都不是 SMML 的逐 voxel 原式，需要自己的消融；推理不得访问标签。[本地 objectives.py:8–49](../miss_rgbt/objectives.py)、[SMML 式(1)–(3)](https://arxiv.org/html/2507.07592v1#S3.SS2.SSS1)

## 4. RobustSeg：论文的完整能力保护与源码实际情况必须分开

### 4.1 会议论文主张

HPD 用跨模态类原型传递语义，用选定主导模态的 IFV 传递细节。反馈阶段用 **CE + 非主导模态 IFV**，冻结主导模态 encoder 和 fusion，只更新非主导 encoder；教师学习率为学生的 0.1 倍。主文 Table 4 中，无选择冻结的 CE+KL 反馈使学生 EMM/FULL 从 **49.10/61.26 降到 48.77/60.85**；选择冻结+非主导 IFV 为 **49.91/61.85**。这是分割论文对反馈受体及约束重要性的证据，不是 RGB-T 跟踪收益保证。[主文 §3.2–3.3、§4.1，PDF p.5](https://openaccess.thecvf.com/content/CVPR2026/papers/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.pdf#page=5)、[Table 4 / §5.3，p.8](https://openaccess.thecvf.com/content/CVPR2026/papers/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.pdf#page=8)

补充 PDF §1.3 / Algorithm 2 进一步描述：未选中的 encoder 在 `no_grad` 下前向，输出仍供融合，选中 encoder 才建图更新。这是**论文算法描述**。同份补充材料 §1.1 尚残留三模型/FSM 的描述，与主文存在版本混杂；不把它合并成额外的已验证机制。[官方 supplement，`supplementary.pdf` pp.1–2](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Tan_Towards_Robust_Multi-Modal_CVPR_2026_supplemental.zip)

### 4.2 已核实的源码行为和缺口

下表以 **CVF ZIP** 的 `RobustSeg_Source_Code/` 为根目录。为了方便行级访问，也给固定 Git 仓库链接：`dual.py` 在 checkpoint 部分比 ZIP 多一行，故其后 **Git 行号 = ZIP 行号 + 1**；本次 diff 证实二者差异仅为该 checkpoint 路径/注释，下面关键运算相同。

| 检查项 | 精确源码位置与实际行为 | 与论文/迁移的关系 |
| --- | --- | --- |
| 完整→缺失任务响应 | `tools/train_deliver_mad50_proto_cur_ifv_dual.py::main`，ZIP **158–180**：student 开启训练，teacher 在 `no_grad` 下处理完整 sample；类别维 softmax/log-softmax 做 KL，再加标签任务损失和 prototype/IFV 项。[Git:159–181](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/tools/train_deliver_mad50_proto_cur_ifv_dual.py#L159-L181) | teacher 目标在这个调用点确实断梯度。不是跟踪热图空间维 KL 的同一实现。 |
| **反馈 optimizer 的受体错误** | 同函数 ZIP **76–83,205–232**：`optimizer_2=get_optimizer(model,...)` 仍绑定 student，`scheduler_2` 也绑定第一个 `optimizer`；反向 loss 主要从 `model_t` 建图，却 step `optimizer_2`。`semseg/optimizers.py::get_optimizer:5–20` 只收集传入 model 的参数。[Git:77–84](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/tools/train_deliver_mad50_proto_cur_ifv_dual.py#L77-L84)、[反馈段](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/tools/train_deliver_mad50_proto_cur_ifv_dual.py#L206-L233)、[optimizer](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/semseg/optimizers.py#L5-L20) | **该 optimizer 没有任何 teacher 参数**。不能称此脚本已经实现教师弱模态更新；这是静态判定，没有运行或擅自修补。 |
| 反馈输入、损失和冻结 | ZIP **205–227** 把 student 设为 `eval()` 后重新用完整 `sample` 前向；`Seg.forward:91–103` 在 eval 保留全部模态。反馈实际写 `CE + KL*50 + UMD*10`，`lr_2=lr*.01`。这条调用路径没有论文的主导/非主导 encoder 冻结选择；调用的 backbone 以共享块逐模态计算并取均值融合。[Git 反馈段](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/tools/train_deliver_mad50_proto_cur_ifv_dual.py#L206-L228)、[Seg.forward](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/semseg/models/segformer/seg_block_select_UMD_zero_padding.py#L83-L122)、[backbone.forward_features:312–358](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/semseg/models/segformer/MixT_block_select_UMD.py#L312-L358) | 不是本项目“同批缺失 M 优于完整 C 时反向互教”的代码先例，更不是论文的 CE+非主导 IFV / 0.1 倍 LR 复现。 |
| detach 的真实作用 | `mmkd/IFV.py::CriterionIFV.forward:84` 仅调用 `feat_T.detach()`，未赋回变量，后续仍使用原 `feat_T`。调用方 `no_grad` 能隔断教师，但该函数自身并未做到 target detach。[IFV.py:81–113](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/mmkd/IFV.py#L81-L113) | 不可照抄为通用双向 loss 的正确 stop-gradient 写法；也不能据此反推所有调用都发生目标梯度泄漏。 |
| 公开代码不完整 | dual ZIP **23,27,31,33–35** 导入的 `tools.val_mm_mmkd`、`mmkd.UMD_zero_padding`、`RUMD_zero_padding_random`、`regularization`、`IFVMD_zero_padding`、`PGUMD` 均未出现在 ZIP / 固定 Git tree；公开的是另一个 `IFVMD_zero_padding_f.py`，没有把它擅自替换进调用。[Git imports](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/tools/train_deliver_mad50_proto_cur_ifv_dual.py#L23-L35)、[tree](https://api.github.com/repos/RobustSeg/RMMSS/git/trees/fda61fc603e0d110f96db79c52a531aa604f5fc6?recursive=1) | 本次不能验证论文所述 IFV modality selection 的可运行实现或性能；README 还写反馈训练未来发布。[README:31–36](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/README.md#L31-L36) |

另外，`PrototypeSegmentation.calculate_batch_prototypes:54–62` 先做类内 `mean`，随后又除以该类像素数；`PCUMD` 有重复同名定义，实际生效的是第二个。若未来重写原型损失，应按明确数学定义实现并核对尺度，不能默认这些细节已被论文验证。[Prototype.py](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/mmkd/Prototype.py#L36-L62)、[PCUMD 第二定义:61–100](https://github.com/RobustSeg/RMMSS/blob/fda61fc603e0d110f96db79c52a531aa604f5fc6/mmkd/PCUMD_zero_padding_random.py#L61-L100)

## 5. 对 Miss-RGBT 的最小后续方案（仅提案）

**当前事实，已只读核对：**C/M 是同一模型两次独立前向；响应 KD 对 target detach，比较的是恢复为 logits 后的**空间热图分布**；可靠性以训练框任务损失 `loss_M+margin<loss_C` 定义。`e3_full20/e4_full20` 都是 `reverse_weight=0`，因此这两项实验不能证明非零反向反馈有效。[train.py:87–91](../miss_rgbt/train.py)、[objectives.py:29–49](../miss_rgbt/objectives.py)、[E3 config](../configs/e3_full20.json)、[E4 config](../configs/e4_full20.json)

1. **任务响应传递：**先用同训练预算的缺失训练对照，检验现有 C→M response KL 的增益；不增设只为模仿分割语义原型的头。SDSTrack 的特征 MSE、SMML 的类别 KL 与本项目空间响应 KL 必须分别标注，不能写成直接复现。M 前向只使用实际可用模态和自己的中间特征；训练 target 可以来自 C，部署不能读取 C 特征。
2. **缺失位置修复：**保留当前 block6、仅缺失 token 的 residual repair 作为可单独归因的候选。完整输入 `available=1` 时，`LocalRepair.forward:20–24` 的 `(1-available)` 让其增量为零；完整输入约 +2 个百分点必须来自实际可学习的共享路径和训练结果，不能由这个恒为零的直接分支推导。[本地 model.py:20–24,67–70](../miss_rgbt/model.py)
3. **可靠反向反馈：**若后续启用，继续以训练标签判别“哪一个输出更好”，target 与判别 gate 均断梯度，保留完整输入任务监督。**先明确接收梯度的参数组：**当前 M→C loss 经完整 C 前向回传，修复器被零门控，因此梯度进入共享 baseline；若为了保护完整能力而冻结整个 baseline、只许 repair 更新，这一反向项会失去有效受体。需用真实梯度/参数更新验证受体，再靠完整/Miss 消融验证性能，不能把 gate 触发率或 loss 非零当作模块有效证据。[本地 model.py:20–24](../miss_rgbt/model.py)、[objectives.py:41–49](../miss_rgbt/objectives.py)、[optimizer 分组 train.py:60–64](../miss_rgbt/train.py)

建议的证据链是：同预算缺失训练 E2 → 加响应 KD 的 E3 → 加修复的 E4 → 加可靠反馈的 E5；再用完整方案去掉各模块检查交互。若把“可靠性筛选”列为反馈的核心机制，需增加等训练预算的无筛选反馈对照。**这些是待执行消融，不改变当前 frozen full20。**三项都必须在真实部署路径和同一规定评估协议下给出增益；增加训练本身的收益需由 E2 对照排除。

项目验收仍是 LasHeR-Miss **PR>69.3、SR>54.4**，RGBT234-Miss **MPR>88.1、MSR>64.3**，以及完整输入约 +2 点、三个有消融支撑的有效模块。这些是本轮任务给定目标，不是上述论文已证明能达到的结果。分割论文的空间蒸馏没有验证跟踪中模态恢复后的 crop/template 历史损伤；不得将本文建议表述成已修复该时序问题。

已读取的本地 E0 配对诊断报告显示：首次缺失前 53,027 帧 bbox 完全一致，经历缺失、当前恢复完整的帧仍有精度缺口；报告还提供了部分恢复帧搜索区域偏离目标的几何证据。该证据仍未把损伤唯一定位到裁剪或模板，也没有证明某项干预能恢复正式成绩。因此不能预设局部修复时目标始终位于当前 crop 内；任何时序恢复机制须另在 train/val 定位、验证，不据此修改已冻结的 E3/E4。本轮只读取该报告，未独立复算其原始预测。[本地 E0 诊断及限制](../reports/e0_paired_diagnostics_v1/README.md)

## 6. 复查入口与限制

全部外部快照仅保存于 `C:/Users/gb/.codex_tmp/miss_rgbt_source_research_20261013/`。`SOURCE_INDEX.json` 记录来源、commit、时间、文件数；各仓库 `retrieved_files.json` 记录 Git blob SHA / SHA256。三个 Git 仓库 tree 均 `truncated=false`；选择性下载文本 **SDSTrack 20 / SMML 18 / RobustSeg-RMMSS 18** 个文件，并逐文件核对 Git blob hash。没有克隆数据、权重或大资产。RobustSeg 官方 ZIP 仅提取文本源码和补充 PDF，保留原 ZIP。

- RobustSeg 主文：1,480,542 bytes，SHA256 `b582fdac5b46df50539a4664c3b09fc3e7f0041e218cdecc6ba9506d582e242e`。
- RobustSeg 官方 ZIP：6,791,928 bytes，SHA256 `88fc78746d40fdebdeabaa459f9415eb7ea33466a4f610c5a3a5e738a4be37fc`。
- ZIP 内补充 PDF：SHA256 `488576eeb37aab79fc39a8301944804585f5b8fac47d989566f1b96723e7c38f`。主文 pp.5/8 已渲染检查，确认 Fig.5 的冻结说明及 Table 4 的颜色/数值含义。
- 本文是源码研究，不是外部仓库运行复现或独立实验审计。SMML 的关键损失函数缺失，RobustSeg 的缺失导入与反馈实现差异已明确列出；没有补齐它们后声称作者结果可复现。
