# Miss-RGBT Focal Normalization Audit Addendum

日期：2026-10-13。审查者：**gpt-6-astra / max**；`same-family`，`provisional`。总体 **WARN**；本次归一化公式核查 **PASS**。原 `EXPERIMENT_AUDIT.md/.json` 保持原字节及原 WARN，本文件补充后续修订。输入快照时间：2026-10-13T07:16:30.091848+00:00。

本次完整读取修订后的 loss、train 入口与 CPU 算术检查脚本，核对保存结果、AST、SHA256 和 Git blob；没有新增 GPU/NN、数据集评估或训练状态查询，也没有重新运行 CPU 算术脚本。这里的“保存 CPU 结果”指读取既有 JSON，不能表述成审查者见证的新进程执行。

## W6：旧 Focal 归一化契约不完整，当前源码已修正

原生 `lib/utils/focal_loss.py:26` 汇总整个 forward batch 的正点数；无正点时使用负项总和（:30–33）。旧快照 `.aris/traces/experiment-audit/20261013/temporary/final_inputs/miss_rgbt/objectives.py:24` 却逐样本除以各自正点数，随后取均值。每样本一个正点时两者等价；B=2、正点数为 [1,0] 或 [0,0] 时，旧 focal 分量为原生的一半。它是损失尺度偏差，不能描述成已观察到的训练崩溃。

令 a_i 为样本的非负 focal 分子、p_i 为目标正点数、B 为 batch 大小，当前 `miss_rgbt/objectives.py:24` 使用：

```text
sample_i = B * a_i / max(sum_j p_j, 1)
mean(sample_i) = sum_i a_i / max(sum_j p_j, 1) = native focal
```

分母来自 GT heatmap 的正点计数，未使用预测值的 max/min/mean。当前 native head 将概率截断至 [1e-4, 1-1e-4]（`lib/models/layers/head.py:176`），因此本契约按实际 head 的输入域判断。此结论针对每次 forward 的 batch，不额外声称梯度累积等同于一次更大 batch 的全局归一化。

空 heatmap 是原有绘图函数可表达的结果（`lib/utils/heapmap_utils.py:85`、:93），新增脚本用框 [2,2,.2,.2] 构造它（`scripts/validate_focal_normalization.py:21`）。未查证历史 native batch、speed smoke 或正式训练中实际出现了多少这种目标，不能推断已发生的训练损害。旧 native loss 数值接近只支持那个记录样本；原报告的此项证据不能泛化为所有 heatmap 的数学证明。

## 保存的 CPU 证据与 A–F 检查

| 案例 | 正点数 | native | per-sample mean | 绝对差 |
|---|---|---:|---:|---:|
| one_positive_each | [1,1] | 164.86196899414062 | 164.8619842529297 | 0.0000152587890625 |
| mixed_empty_target | [1,0] | 335.012451171875 | 335.012451171875 | 0 |
| all_empty_targets | [0,0] | 297.9494934082031 | 297.9494934082031 | 0 |

以上数值直接来自 `reports/focal_normalization_validation.json:4`、:13、:22；差值计算及脚本 :33 的 atol=2e-5、rtol=1e-6 条件一致。

- **A — PASS（标注真实性）**：检查使用脚本 :20–25 的人工框和随机概率张量；:19 及 JSON :2 明确写明 CPU synthetic loss arithmetic，没有伪装数据集 GT。
- **B — PASS**：目标正点总数归一化，公式与原生全 batch focal 一致；这不是性能分数的自归一化。
- **C — PASS**：三个案例、键和值均实际存在。当前 objectives、train、检查脚本及结果四文件与 `58fe581` Git blob 逐字节相等。
- **D — WARN（执行证据边界）**：检查脚本 :30–36 有顶层调用、assert 和结果写入，保存字段与源码对应；本补充审查未复跑，范围内没有独立的该次 CPU stdout/exit receipt，因此不宣称新见证的运行 PASS。
- **E — WARN**：三个人工 CPU forward 数值案例不能证明模型成绩、数据集上空目标频率、梯度一致性或长期训练稳定性。历史 `method_native_validation.json:27` 的一次 update、:35 的 reverse_fraction=0，仍限于当时那次记录。
- **F — PASS（分类明确）**：分类为 **synthetic_arithmetic_validation / not_performance_evaluation**。它不是模型生成参考的 synthetic_proxy 跟踪评估，也不是 real_gt 成绩；原 E0 的 real_gt 分类不变。

## 版本及当前输入哈希

控制器/E0 的 `reports/SOURCE_VERSION.txt:1` 仍是 `370f215814320d048a5c33974b64052126a8a1fd`。正式训练独立版本由 `reports/FORMAL_TRAIN_SOURCE_VERSION.txt:1` 指向 `58fe5819c6e82770a5a36025924f5a05030ac7cc`；`miss_rgbt/train.py:68` 保存当前源码哈希，:72 保存正式版本字符串。文件存在与 Git 一致不等于训练已启动或完成；本次未查询运行状态。

| 文件 | 当前 SHA256 |
|---|---|
| miss_rgbt/objectives.py | `ea899e4727c2e1c85c26b1995deb2d96b59ea94b188e9953b1531511c9a6c24b` |
| miss_rgbt/train.py | `c16c5376045a80876bec084aa040deb65a65aa7ae4dce3daed8e76151d331ffb` |
| scripts/validate_focal_normalization.py | `d3658557019c1eea4dce7476dd49572170386cab1fce59ae17ce244ee4055c1c` |
| reports/focal_normalization_validation.json | `01d3a371a7ad6e10769fbb67fcf6a0e7db0af8466e211ce0c27d1d2a8681671d` |

旧 objectives 哈希 `7f00f945e48b1e44cce4e914746b1d0df8101d70ff3068022763aa6b16bae5bc`、旧 train 哈希 `611f972f82ecfbd5b3662295ca8973ee2cea500d66b29e3ec46b693924abaaa6` 与 `370f2158` 对应 Git blob 一致。原报告中的 105 个 campaign 哈希一致、103 个 Python AST 等检查属于其当时快照，不应直接移用于后续修订。新版 native GPU 验证未执行。

原报告 Markdown SHA256 为 `c5d66bf9955e3f34cbee88b68687c72311fa8bae42722e89bafba5638e90bc0a`，JSON 为 `82737b6e5c6a43c186716f24630c5276c35dc48687d8945d9320026077cc4ac0`；两者均与 `58fe581` 提交完全相等，未覆盖。

完整当前/历史输入哈希、源码原字节、对应 Git blob 及失败记录保存在 `.aris/traces/experiment-audit/20261013/temporary/focal_addendum_inputs_v2/manifest.json` 及同目录。审查者第一次采集误写 head.py 路径而实际退出 1；后经 rg 定位并纠正，重新采集退出 0。真实失败响应与原始脚本补丁留在编号工具 trace，不伪写成全程成功。

原 W1–W5、L1–L2 历史完整保留；本补充只支持 Focal 数学契约修订与源码版本归属，不增加模型正式成绩、模块有效性或已完成训练的主张。

