# Miss-RGBT Experiment Audit

日期：2026-10-13。审查者：**gpt-6-astra / max**，fresh reviewer；`review_independence: same-family`，`acceptance_status: provisional`。总体结论：**WARN**。确定性的 E0 文件、哈希、指标复算检查为 **PASS**；新方法证据仍限于 native/smoke，不能把本报告作为正式成绩或模块有效性证明。审查不阻断已授权实验。

本审查直接读取源文件、日志、原始归档与真实安装包标注；新增 GPU/NN 执行 **0**，训练状态轮询 **0**，审查者修改作者代码 **0**。正式 campaign 只检查已保存的启动文件：`370f215814320d048a5c33974b64052126a8a1fd`、2026-10-13 06:50:39 UTC、E0 complete launch；不推断此后的运行状态。最终输入快照时间为 **2026-10-13T07:02:52.092228+00:00**。

以下证据路径均相对项目根目录；`T/` 表示 `.aris/traces/experiment-audit/20261013/temporary/`。初始文件、修订文件及最终文件分别保存在 `T/inputs/`、`T/revision1/`、`T/final_inputs/`，对应清单含真实 SHA256。

## A. Ground Truth Provenance — PASS

E0 从安装的 rgbt toolkit 读取 LasHeR GT 及 RGBT234 visible/infrared GT；不存在从模型输出生成评估标签的路径。作业只携带第一帧 GT 作为初始化，后续跟踪循环只读图像、mask 和上一状态。证据：`reports/e0/evaluate_miss.py:64`、`:80`、`:122`；原双 worker 入口同样如此。完整 E0 入口位于 `scripts/evaluate_complete.py:65`；新模型入口还逐项核对记录的 GT/mask 哈希，见 `scripts/evaluate_model.py:34`、`:42`。

独立 CPU 检查了 **713 个 GT 文件**（LasHeR 245、RGBT234 468），其内容哈希同时等于 inventory、旧环境和新环境的对应文件。旧/新环境的 **15 个 toolkit Python 文件**也字节一致。479 个原始 bbox 第一行全部等于相应数据集原始初始化 GT。验证保存在 `T/remote_readonly_audit.stdout` 和 `T/archive_verification.json`。

这是对当前安装包及已记录标注来源的审查；未把安装路径等同于重新认证上游发布过程。论文比较行仍是论文报告，本审查未重新运行 SCDT。

## B. Score Normalization — PASS，附原生协议语义警示

实际指标分母是每条序列固定帧数，随后对序列等权平均；NPR 使用 **GT 的宽高**归一化坐标。MPR 对两套真实模态 GT 取较小中心误差，MSR 取较大 IoU。未发现除以预测自身最大值、最小值或均值来抬高分数的操作。标准 IoU 的面积并集不是这类分数归一化。

证据：`T/toolkit_old/rgbt/metrics/metrics.py:28`、`:62`、`:290`、`:332`；`T/toolkit_old/rgbt/utils.py:117`、`:140`、`:161`。预测由 toolkit 按 float32 加载并取整，见 `T/toolkit_old/rgbt/dataset/basedataset.py:19`；这与连续坐标描述诊断不同。

**需保留披露的原生规则：**LasHeR 中任一 GT 坐标 ≤0 时，toolkit 把误差置为 −1；PR/NPR 的“≤阈值”判断会自动计成功，SR 则计失败。实际涉及 **1260 帧**，其中 **533 帧**宽或高非正；相对同分母把这些帧计失败，PR/NPR 均含 **0.556805386456 个百分点**的固定贡献。证据：`metrics/metrics.py:266`、`:309`、`:351`（同上 toolkit 路径）。这不是作者新加入的规则，也不构成模型增益。最终 `reports/e0/RESULTS.md:30` 已披露，后续配对比较必须保持一致协议。

## C. Result Existence / Numbers / Status — PASS，历史记录错误已修订

本地原始归档实际存在，10,248,037 字节，SHA256：

`85b302508ce56b7e688c0f8c21f58a6dd07fd883ea8f57db5bf99a47bae3dc0a`

479 个 bbox 和479个独立收据全部核对 SHA256、名称、帧数、`N×4` 形状、有限性及合法 GPU 编号；合并收据有479个唯一序列。共 **337353 帧**，没有截短、跳过或遗漏。87 个 baseline 源码/配置同时匹配本地、原工程和归档清单；新项目 epoch20 权重实际重哈希为 `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`，1,236,922,039 字节。

| 数据集 | 序列 / 帧 | PR / MPR | NPR | SR / MSR |
|---|---:|---:|---:|---:|
| LasHeR-Miss | 245 / 220703 | 63.47250706301003 | 58.58185056215868 | 50.583148589078256 |
| RGBT234-Miss | 234 / 116650 | 82.79481455539279 | — | 59.57443513369343 |

独立调用复制来的实际 toolkit，**五项指标及所有逐序列曲线与归档误差均为0**；另写的向量几何实现也与全部曲线逐项一致。证据：`reports/e0/metrics.json:2`、`T/verify_archive.py`、`T/archive_verification.json`。这是已有预测的 CPU 复核，没有新增模型成绩。

初始双 worker 日志确有2个 DONE，退出码 **143**，同时保留 Terminated/semaphore 警告；续跑日志有477个 DONE、8个 worker，退出码 **0**。合计479条。2561.433269秒是续跑阶段计时，不能当作含前段的总耗时。证据：`T/baseline_original/run_initial2workers.log:5`、`exit_code_initial2workers.txt:1`、`run.log:511`、`exit_code.txt:1`（后三者同目录），以及 `reports/e0/completed.json:2`。

最终 tracker 将 E0 Miss 标为 DONE、完整 E0 标为 RUNNING、新方法正式训练标为 QUEUED，并明确尚无新模型完整成绩，见 `EXPERIMENT_TRACKER.md:6`、`:11`、`:12`。本审查对 RUNNING 的依据仅为保存的启动快照，不是一次实时状态确认。

## D. Evaluation Was Actually Called — PASS，按执行类型区分

完整逐行阅读了两个 E0 Miss 入口、`evaluate_complete.py`、`evaluate_model.py`、`validate_method.py`、`validate_tracker.py`、`summarize_speed.py` 和 `analyze_missing_states.py`。

| 入口 | 源码调用与实际证据 |
|---|---|
| E0 Miss 初始 / 续跑 | `evaluate_miss_initial2workers.py:212`、`evaluate_miss.py:215`调用评估；真实日志、框、曲线、指标和CPU重算闭合 |
| 完整 E0 | `scripts/evaluate_complete.py:210`有可达调用；保存启动记录存在，尚无本轮已审完整结果 |
| 新模型完整 / Miss | `scripts/evaluate_model.py:125`、`:137`、`:142`为完整可达路径；本轮只有源码审查，不宣称已经执行完毕 |
| native 方法检查 | `scripts/validate_method.py:70`调用训练损失并做反传更新；JSON 与 PASS 日志一致，范围为单步 |
| tracker 检查 | `scripts/validate_tracker.py:40`实际设计为10帧完整跟踪及2次单缺失前向；报告范围一致 |
| 速度 / 状态诊断 | `scripts/summarize_speed.py:8`、`scripts/analyze_missing_states.py:48`有实际输入与产物；审查独立CPU核对数值 |

没有把仅存在的评估函数当作新模型已取得成绩的证明。

## E. Scope — WARN / Claim Ceiling

E0 为一个 epoch20 checkpoint、两个完整测试集、各一套固定缺失 masks 的 OPE。它支持此精确协议的基线数值，不支持新方法改进、跨种子稳定性或三个模块有效的结论。

native 方法记录只有一批训练数据：batch2、两模板、三搜索、一次 AdamW 更新。记录确实报告四种完整输出差0、原生损失与逐样本均值约2.38e−7差、不可见 RGB 扰动响应差0、补偿输出层非零梯度与实际更新、同一模型对象严格读回后响应差0。证据：`scripts/validate_method.py:28`、`:39`、`:50`、`:64`、`:73`、`:88`；`reports/method_native_validation.json:18`。

这些证据没有覆盖：不可见 TIR 的扰动不变性、模板更新分支实际触发记录、独立进程恢复、所有参数均有非零梯度或完整训练稳定性。tracker 报告的2次缺失检查仅断言输出有限，见 `scripts/validate_tracker.py:46`；模板数量为2不证明更新分支发生过。

**反向反馈未被本次 native 激活证明。**报告 `reverse_fraction=0`、`kd_reverse=0`，见 `reports/method_native_validation.json:34`；E3/E4 的 `reverse_weight=0`，见两配置第10行。不能由该次 PASS 推出 M→C 已有有效梯度或增益。

E3/E4 speed smoke 分别有 **25 个真实更新记录**，短配置为1轮×200样本；速度为 **1.962929 / 1.960557 秒每更新**（扣除首更新），末次峰值17.2855 /17.4764 GB。5轮×20000样本、12500更新对应的 **6.815726 /6.807488小时是推算**。已独立读取已结束 smoke 的文件，未查询训练进度；证据：`T/remote_readonly_audit.stdout`、`T/auxiliary_verification.json`、`reports/paired_speed_smoke.json:3`。

描述诊断使用连续坐标、按帧加权并排除初始化，不是官方按序列 PR/SR。各缺失状态的样本构成不同；仅TIR得分较低不能单独证明因果性的 RGB 依赖。变化/稳定组的方向在两个数据集不同。修订说明准确保留了这些边界，见 `reports/e0/MISSING_DIAGNOSTICS.md:3`、`:16`、`:18`。

## F. Evaluation Classification — real_gt

| 证据 | 分类及可支持范围 |
|---|---|
| E0 Miss 指标 | `real_gt`，真实数据集标注上的完整基线 |
| 完整 E0 / 新模型评估入口 | 源码设计为 `real_gt`；完成成绩待实际产物 |
| 缺失状态描述诊断 | `real_gt`，非官方按帧诊断 |
| native / speed / tracker 检查 | 真实训练数据上的工程验证；不形成 benchmark 成绩 |
| 环境 witness | 环境可执行性检查，不是模型性能评估 |

固定 mask 是对观测条件的模拟，不会把真实 GT 变成 `synthetic_proxy`。C→M/M→C 的教师响应属于训练损失目标，未被当成最终评测 GT。未发现冒充 `real_gt` 的模型生成标签评估。

## Missing Information and Training Source

单路缺失的信息路径在已审源码中合法：`miss_rgbt/model.py:54`屏蔽不可见模态/patch，在任何 Transformer 前移除对应 patch 投影；实际 patch projection 是不重叠卷积，见 `lib/models/layers/patch_embed.py:20`。补偿仅输入 M 的特征和可用标识，完整位置的残差系数严格为0，见 `miss_rgbt/model.py:20`。C 的激活只出现在训练 KD 的停止梯度目标中，未输入 M 前向；可靠性来自训练标签，见 `miss_rgbt/objectives.py:29`、`:41`。

推理只加载一个模型，保存每个更新模板的真实可用标识，见 `miss_rgbt/tracker.py:18`、`:41`、`:53`。新/旧评估入口均只在首帧使用GT；双缺失保持状态。在原始结果中，**52493个初始化之后的双缺失框**与上一框完全相同。65条序列原 mask 的首帧不是完整态，但协议统一完整初始化；这是已声明的协议例外，不是悄悄访问后续缺失信息。

整模态缺失时，补偿覆盖该模态全部 token；当前没有“只恢复少量必要任务 token”的选择机制，也没有证明恢复真实完整特征。最终计划第31行及 HANDOFF 第91行已明确缩窄表述。

未在已审 E3/E4 路径发现有实证的训练阻断缺陷。新的 sampler 解码/GT 读取和任务损失没有原入口的异常吞噬，见 `miss_rgbt/data.py:13`、`miss_rgbt/objectives.py:8`；103个Python文件通过AST解析，启动清单105项与实际源码一致。train882、val97、test245序列名称两两不相交，见 `T/auxiliary_verification.json`。该判断限于源码及已有短运行，不承诺完整训练收敛或稳定。

## Historical WARNs and Verified Revisions

| ID | 原问题 | 复核结论 |
|---|---|---|
| W1 | 首update哈希仅记录最后一个 `search_images` microbatch，却写作完整有效update输入匹配；`T/inputs/miss_rgbt/train.py:82`、`T/inputs/scripts/summarize_speed.py:19` | 已改为 `matched_first_update_last_microbatch_input`；新版训练只新增完整首update的search哈希列表。templates/masks仍未被该证据覆盖。见 `scripts/summarize_speed.py:19`、`miss_rgbt/train.py:104` |
| W2 | 第一个 transition 对比原始mask[0]而非真实完整初始化；`T/revision1/scripts/analyze_missing_states.py:40` | 24/19条序列标签被修正，changed变为52383/29598。原始/修订结果均独立重算一致，逐状态指标未变；见 `scripts/analyze_missing_states.py:40`、`T/auxiliary_verification.json` |
| W3 | LasHeR无效坐标的原生PR/NPR自动成功语义未说明 | 1260帧及0.556805个百分点贡献已披露；保留相同协议。见 `reports/e0/RESULTS.md:30` |
| W4 | 原RESULTS称本地归档与报告同目录 | 已改为实际 `C:\Users\gb\.codex_tmp\cadtrack_miss_ep20_20261013\baseline_raw_results.tar.gz`。见原快照与最终 `reports/e0/RESULTS.md:28` |
| W5 | 中间文档短ID `370f215a` 与真实完整hash不符 | 修订为 `370f2158`；真实完整值与105项文件清单一致。见 `HANDOFF.md:74`、`EXPERIMENT_TRACKER.md:16` |

原诊断保留于 `reports/e0/*_before_first_transition_fix.*`；审查另保留初始及修订源码字节。不能把修订后的字段解释为早期运行已经记录了当时没有记录的信息。

## Preserved Failures / Trace

原作者 native 首次 import 失败 `ModuleNotFoundError: lib.train.loss`，见 `reports/method_native_validation_import_failure.log:1`；当前使用实际存在的 `lib.utils.focal_loss`（`scripts/validate_method.py:13`）。初始 E0 退出143和相关警告保留，不改写为成功。

审查自己的失败也保留：PATH Python 缺 `pyvenv.cfg`；另一个Python无numpy；首版远程校验器未将float mask转换为整数导致 `bincount` TypeError；首版辅助校验器把25步smoke配置误当成12500步正式配置触发assert。修正仅发生在审查者临时校验脚本，未修改作者训练代码。旧校验脚本、原stdout/stderr和真实非零工具响应均保留；后续CPU校验实际退出0。

真实执行请求/响应及会话终态存入 `.aris/traces/experiment-audit/20261013/*-tool.json`。工具本身截断的较大读取响应原样保存；全文输入快照、toolkit源码、原始归档解包和校验器代码补充其可复查性。远程CPU子进程的完整494943字节stdout、stderr与argv/退出码另存于 `T/remote_readonly_audit.*`。没有重建或捏造不存在的GPU执行证据。

## Claim Impact / Follow-up

- **支持：**精确记录协议下的E0两数据集数值；source层面的单路信息合法性；25更新速度测量；限于已测输入的native等价/有限性。
- **需限定：**完整训练耗时是推算；输入匹配只涵盖实际哈希对象；环境PASS、单步PASS、10帧PASS均不能扩大为完整训练/评估PASS。
- **未支持：**新方法正式涨点、三个有效模块、反向反馈实际有效、真实完整特征恢复或必要token选择。
- 后续沿已授权campaign取得实际完成文件，再审查新模型主结果和消融；继续固定GT/mask/toolkit规则，保留失败与负结果。

本审查的 same-family/provisional 语义判断与已完成的确定性复核分开记录；它不等于跨模型家族正式验收。

