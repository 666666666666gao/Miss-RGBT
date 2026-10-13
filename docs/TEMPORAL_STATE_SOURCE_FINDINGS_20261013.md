# DiMP / SAM 2：历史状态与搜索恢复源码核查

2026-10-13；仅静态阅读第一方源码/论文，没有运行模型、查询 GPU/远程训练或修改冻结代码。项目内仅写本文。任务给定 E0 的 5,536/54,822 恢复帧 crop 几何缺口未在本轮重算，也不代表干预可恢复的成绩。

**直接结论：记忆拒写不等于重定位。**DiMP 默认 `uncertain` 仍更新定位状态；SAM 2 的 no-object 帧仍写入记忆。SAM 2 观察完整图像，但仍依赖指定目标的 conditioning；全帧编码也不保证重新找到正确身份。[DiMP 状态][D-track]、[SAM 2 写入][S-write]、[conditioning 要求][S-require]

## 1. 固定来源与许可

| 第一方来源 | 核查版本、日期与许可 |
| --- | --- |
| [DiMP，ICCV 2019 论文][D-paper] → `visionml/pytracking` | SHA **7eb9e74bd3d40e29dbcec444902237da13de247b**；patch 作者日期 2024-08-03 21:15:11 +0200；**GPL v3**。[提交/日期][D-patch]、[许可证][D-license] |
| [SAM 2 论文 v2][S-paper] → `facebookresearch/sam2` | SHA **2b90b9f5ceec907a1c18123530e92e794ad901a4**；patch 作者日期 2024-12-15 16:47:17 -0800；主代码 **Apache 2.0**，可选 connected-components 另有 BSD 3-Clause。[提交/日期][S-patch]、[主许可][S-license]、[第三方归属][S-readme-license] |

仅阅读/引用，未复制实现到项目。DiMP 不能当作 MIT 代码直接搬入；本轮未审查项目与 GPL 的兼容性。下述 SAM 2 行为明确采用 **SAM 2.1 Hiera-L 配置**。[配置][S-config]

## 2. DiMP：定位状态和学习状态分别提交

`DiMP.localize_advanced:238–303` 先检查主峰，再排除目标邻域找次峰，结合峰值比例及其距旧位置的距离判别歧义。`dimp50` 实际设置：not-found 阈值 **0.25**，distractor/hard-negative 比例 **0.8/0.5**，`update_scale_when_uncertain=True`；独立 uncertain/hard-sample 绝对阈值未设置、默认负无穷。这些不是可直接移植的校准概率。[判别函数][D-localize]、[参数文件][D-param]

| 标记 | 位置、大小、搜索尺度 | 学习记忆/filter |
| --- | --- | --- |
| `not_found` | 跳过位置提交和 refinement，保持旧状态；默认仍输出保持框。 | 不写当前样本/预测框，不更新 filter。 |
| `uncertain` | **仍提交位置、进入 refinement；默认允许更新尺度/大小。** | 不写当前训练记忆，不更新 filter。 |
| `hard_negative` | 接受选定候选并 refinement。 | 写记忆，以 0.02 学习率立即优化一次；不是“拒绝负样本”。 |
| `normal` | 正常提交。 | 默认每个接受帧写记忆；每 20 帧优化两次，普通学习率 0.01。 |

依据为 `track:123–172`、`update_classifier:605–639` 和配置。[状态提交][D-track]、[学习更新][D-update]。即使把 uncertain-scale 开关设 false，`refine_target_box:703–711` 仍赋值 `target_sz`，只在该处分支控制 `target_scale`，不能称为冻结所有尺寸量。[赋值细节][D-refine]

`update_memory/update_sample_weights:430–482` 存的是自身预测框和特征；初始化样本不被后续替换，配置还保护其最低权重 0.25。正确报警可阻止污染，但高置信错误身份仍可写入；这是分类器样本库，不是 Transformer 模板的同一实现。[记忆与初始参照][D-memory]

**没有丢失后的自动扩域。**默认 288×288 输入、`search_area_scale=5`、未设 scale factors 时为单尺度；每帧以当前 `pos/target_scale` 采样。not-found 分支没有全图搜索或扩大区域。目标重新进入原区域时有机会恢复；持续在 crop 外时，停写记忆没有新增目标观测。[初始化][D-init]、[每帧状态][D-track]、[送入 backbone 的局部采样][D-sample]

## 3. SAM 2：明确表示无目标，但保留该帧记忆

| 路径与精确索引 | 固定配置下的实际行为 |
| --- | --- |
| `SAM2Base._forward_sam_heads:359–368` | **logit > 0 为存在；≤ 0（含恰好 0）为 no-object**，全部 mask logits 置 -1024。[mask 分支][S-mask] |
| 同函数 `392–403` | `fixed_no_obj_ptr=True` 且 soft 默认 false：无目标时原 pointer 乘零，以 learned no-object pointer **替换**。[pointer][S-pointer]、[配置][S-config]、[默认值][S-defaults] |
| `_encode_new_memory:699–724` | **仍融合图像和 mask 编码空间 memory**，无目标时额外加入 learned no-object embedding；不是清空/跳写。20p−10 的配置使无目标 mask 输入约为 -10。[编码][S-encode]、[实际融合][S-mem-fuse] |
| `propagate_in_video:602–614`；`_encode_memory_in_output:799–809` | 普通帧传 `run_mem_encoder=True` 并保存输出；写入条件没有 presence 过滤。[传播保存][S-propagate]、[写入条件][S-write] |
| `_prepare_memory_conditioned_features:525–584` | 默认 stride=1，取 conditioning 加 **t−1…t−6**；只跳过不存在的输出，**不按 presence/IoU 剔除**。多个 conditioning 时总帧数不严格限于七。[选择循环][S-select]、[默认参数][S-defaults] |
| `select_closest_cond_frames:19–61`；pointer `586–620` | conditioning 默认全保留（上限 -1），设上限后按时间邻近选；pointer 另读最多前 15 个非 conditioning 时间位置，加选中的 conditioning pointers。`only_obj_ptrs_in_the_past_for_eval` 只过滤 pointer，不能约束全部空间 conditioning。[帧选择][S-cond]、[pointer 索引][S-ptr-select] |

presence 是独立训练头；训练标签取 GT mask 是否有正像素，不是 mask IoU、响应最大值或模态 availability。误判存在仍会传播错误目标；误判无目标会抑制真实目标。[预测头][S-head]、[标签][S-loss]

**观察范围与身份条件必须同时满足。**loader 把完整 RGB 帧 resize 为 1024²，image encoder 处理整帧；所以旧 bbox crop 外但仍在图像中的区域可被观察。[加载][S-load]、[编码][S-image]。传播前要求每个对象已有 points/box/mask conditioning；之后可无新提示传播，但依靠已有身份记忆，绝非无条件全图找回。论文承认长遮挡、相似对象下会丢失并可能需要额外提示；这些人工修正不能进入无 GT 的 SOT 部署。[初始化要求][S-require]、[box 转提示][S-box]、[论文限制][S-limits]

额外代码限制：清旧记忆开关默认关闭；其调用名 `_clear_obj_non_cond_mem_around_input` 在已读 predictor/base 类中无定义，实际定义是另一名称 `_clear_non_cond_mem_around_input:956–973`。这是静态调用缺口，未执行/修补；不能把该可选人工修正路径当作已验证自动恢复。[默认值][S-predictor-init]、[调用][S-clear-call]、[定义][S-clear-def]

## 4. 最小迁移与验证边界（提案，未实施）

| 候选 | 必需的 train/val 证据 | 计算与限制 |
| --- | --- | --- |
| 分离模板写入与定位提交，保留初始参照 | 实际 crop、原始响应及次峰、预测框、模板来源/更新时间；检验能否在污染前报警，及是否误拒真实快速运动。只有 bbox 不能重建这些日志。依据是上面的 DiMP 分支。 | 若响应图已有，峰值判断约 O(HW)、无新增 backbone；保留更多模板可能增加 attention 成本。不要照搬阈值或把旧位置当真值。 |
| 若主张 crop 外恢复，另验证扩域观察及身份接受 | 先测候选区域覆盖，再测正确身份恢复/误切换；GT 只能用于训练与离线评估。依据是两套代码观察范围的区别，**不是现成有效模块**。 | 额外一个同尺寸视图意味着报警时多一次前向；报警率 p 时约 1+p 次搜索前向，非测得延时。缩放大范围会降低目标像素，完整高分辨率增加计算。 |

本项目现状经主 agent 本轮直接源码核查：[`tracker.py`](../miss_rgbt/tracker.py) 用 **Hann 加权后的 `score_map.max()`** 作为 confidence，按原 `UPDATE_THRESHOLD` 写模板；[`objectives.py`](../miss_rgbt/objectives.py) 为定位框和 focal 热图任务，没有独立 presence head。该最大响应不能当作已校准的 object-presence logit；需先用 train/val 标注检验和校准，不能从 Test 选择 gate 阈值。

SAM 2 的 RGB、mask 监督和 learned no-object 表示不能直接变成 RGBT/bbox 模块；模态缺失不等于目标缺席。**未做干预验证时，记忆门控至多是减少污染的假设；没有扩域观察与正确身份恢复证据，更不能称重定位有效。**

当前 Test 几何诊断是描述性问题依据，也促成了本轮时序源码调查；它没有用于调整冻结 E3/E4、full20 的阈值、范围、轮次或配置。未来候选机制、部署门限与预算须在 train/val 确定后固定，再评估正式 Test；公开 Test 的失败分析已参与方法开发，不能将它说成完全未见的开发外证据。部署只用当前实际可用模态、初始化和过去自身状态，不用完整分支特征、未来提示或 GT。分别消融写入门控、位置/尺度提交、扩域观察，并检查完整输入退化与实际前向/延时；本轮无效果或新颖性结论。

主 agent 复查了 17 份本地文件的 SHA256/本地 Git blob SHA1，以及 29 个行号链接的路径和范围；[来源收据](../reports/temporal_sources/source_receipts.json)随仓库保存。该检查没有验证远端 Git tree，也没有执行外部代码。

**复查收据：**源码索引即本文各固定 SHA 行级链接。private temp：`C:/Users/gb/.codex_tmp/miss_rgbt_temporal_source_research_20261013/`，选择性保存 DiMP 5、SAM 2 12 个文件及两份 patch；`retrieved_files.json` 记录时间/SHA256/本地 blob SHA1，`commit_dates.json` 留日期原件。API 返回 403 限流，故 SHA 用 git ls-remote 确认、文件按 SHA 下载；未做完整 tree 审计或声称 blob 经 tree 核验。无克隆、权重、外部运行。

[D-paper]: https://arxiv.org/pdf/1904.07220v2
[D-patch]: https://github.com/visionml/pytracking/commit/7eb9e74bd3d40e29dbcec444902237da13de247b.patch
[D-license]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/LICENSE
[D-param]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/parameter/dimp/dimp50.py#L12-L59
[D-init]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L52-L78
[D-track]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L103-L172
[D-localize]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L238-L303
[D-sample]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L305-L311
[D-memory]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L430-L482
[D-update]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L605-L639
[D-refine]: https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L697-L711
[S-paper]: https://arxiv.org/html/2408.00714v2
[S-limits]: https://arxiv.org/html/2408.00714v2
[S-patch]: https://github.com/facebookresearch/sam2/commit/2b90b9f5ceec907a1c18123530e92e794ad901a4.patch
[S-license]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE
[S-readme-license]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md#L163-L173
[S-config]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_l.yaml#L88-L118
[S-defaults]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L23-L96
[S-head]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam/mask_decoder.py#L236-L245
[S-loss]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/training/loss_fns.py#L234-L254
[S-mask]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L345-L403
[S-pointer]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L392-L403
[S-encode]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L678-L726
[S-mem-fuse]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/memory_encoder.py#L158-L181
[S-write]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L789-L813
[S-select]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L525-L584
[S-ptr-select]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L586-L620
[S-cond]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_utils.py#L19-L61
[S-predictor-init]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L22-L39
[S-propagate]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L586-L614
[S-clear-call]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L518-L528
[S-clear-def]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L956-L973
[S-image]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L704-L735
[S-load]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/utils/misc.py#L92-L101
[S-require]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L533-L539
[S-box]: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L195-L208
