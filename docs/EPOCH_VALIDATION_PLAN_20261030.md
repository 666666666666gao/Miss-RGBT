# Server28 候选训练：逐 epoch 验证、早停与最终选权重

日期：2026-10-30。本文为只读检查后的方案审查，不是执行回执。

## 结论

用户补充的最小方案可实施：独立 `train_candidates28.py` 复用冻结 model/objectives；每 epoch 用固定 128 个 val97 causal tuples 的 11/01/10 mean taskloss 作早停代理；停后对最后 5 个 epoch 分别跑全 val97 complete / alternating_outage，按两者 mean SR 选择一个共同权重，之后才正式测试。

必须补足：固定 tuples 的完整输入而不仅是随机种子；LR drop 后重置 patience；明确“相对改善”和 SR 定义；最终验证必须真正遍历全部 5 个权重；明确最后五个是预先限定的候选集合，而非全部历史最优。代理 loss 收敛不能称为跟踪性能收敛。

**时间计划：每模型每 epoch 代理验证先预留 1–5 分钟，尚无该代理的实测；最后 5 个权重的两条件真实跟踪外推约 5.5 小时，执行预算取 6.25–7.5 小时/模型。两个方法各独占 GPU2/3 同时进行时，墙钟预算仍约 6.25–7.5 小时，而不是减半。**

## 已核实事实

只读 SSH：`2028`，项目 `/home/gaob/Miss-RGBT`。检查时间约 2026-10-30 15:11–15:14 UTC，即纽约 11:11–11:14；服务器显示 UTC+8 的 23:11–23:14。

- GPU0/1 正式训练仍在运行，约 19.3 GB 驱动口径显存；未改动其源码、配置、进程或原有评估队列。
- 检查时 GPU2/3 已空闲。两步 batch5 profile 已结束；另发现已有八更新 profile，utility/random 都已完成，并覆盖完整/缺失、两个 search index 的探针。本文没有启动它们。
- 八更新峰值均为 `21118406144 bytes`，即约 **19.67 GiB / 20140.08 MiB**。3090 为 24576 MiB。剩余容量不能直接视作可同时加载验证模型的余量：该字段是 allocated 峰值，不包括全部 reserved、CUDA context 和其他进程。
- microbatch5 × accumulation2 为 effective batch10；LR 为 `1.25e-4`，backbone multiplier 沿用 0.1。相对原 effective batch8、LR `1e-4`，是预先指定的线性缩放候选，八更新通过不证明长期稳定或最优。
- 当前每 epoch 为随机抽样 `20000` 个训练 tuples，不是训练集逐帧遍历。batch10 为 `2000 optimizer updates/epoch`；20 epoch 为 40000 updates，较原 batch8 的 50000 updates 少，但抽样数量相同。
- 数据审计记录 train882 / 459435 帧、val97 / 54646 帧，当前名称划分无交集。val97 有 260 帧 GT 宽/高非正。原 E0 是否曾接触 val97 未获认证，必须保留此限制。
- 冻结训练入口无逐 epoch 验证/早停，只按 `checkpoint_epochs` 保存，目前配置为 10/20。独立入口是必要且足够的改动边界。
- 已有 val97 固定 32 视频帧 block 的 complete / rgb_outage / tir_outage / alternating_outage 掩码；首帧完整，单模态缺失沿用现有可用图像复制和 availability mask，双缺失 hold 且不调用 tracker。旧轨迹脚本只出诊断，不直接计算最终 SR，且 GPU 参数限制为 0/1，不能直接拿它的 CLI 在物理 GPU2/3 执行。
- 正式 `evaluate_model.py` 硬编码 479 条、337353 帧及测试集 GT/指标。不能用于候选 epoch 选择。

## 对补充方案的具体修正

### 1. 固定 128 tuples 的内容，不仅固定 seed

在训练前生成一次 manifest，utility/random 及所有 epoch 共用。覆盖全部 97 条序列：每条至少 1 个 tuple，剩余 31 个按预先固定顺序补齐；必须满足现有 sampler 的 causal、visibility 和 crop-valid 条件。

manifest 记录 sequence、template/search frame IDs、裁剪参数、处理配置和 hash。使用同一确定性预处理：禁用验证颜色抖动、随机翻转；保留必要的裁剪参数但将取值固定。不能只把 `split='val'`、seed 固定就假设每次输入相同，现有 validation loader 仍经过随机 causal sampling 和裁剪处理。

128 个 tuples 各跑三种当前 search availability：11、01、10。固定模板可用性约定，最简单是所有模板完整，仅当前 search 缺失；它只能代理“完整模板下当前观测缺失”，不模拟动态模板已被缺失污染。patch-level mask 不再随机增加，否则监控目标每次变化；用事先确定的同一设置，最小取无额外 patch mask。

用复用的 `task_loss`，不加 KD、utility auxiliary 或训练混合 loss；每个 tuple 内先平均 search frames，再平均三种状态，再平均128 tuples。分别输出三状态 loss。每条至少一个 tuple不等于序列等权：额外31个会增加相应序列权重，须把这一权重方案一并冻结。

RNG 隔离要覆盖 Python / NumPy / Torch CPU / CUDA，并避免推进训练 data-loader 的随机采样状态。预处理 manifest 可在独立 CPU 准备阶段完成。若在训练模型原位验证，执行 eval / no_grad，结束恢复训练模式并重新落实冻结 BN；确认 tuple 前向不会修改选择器缓存等非参数状态。不需要为了代理验证另起驻留 GPU 模型与训练抢显存。

### 2. 给 early stopping 一个没有歧义的定义

以下参数是用户预先指定的工程规则，不是已有数据证明的最优值：

- `min_epoch=12`；原 StepLR 在 epoch10 结束后 step，epoch11 起使用下降后的 LR；至少验证完 epoch11、12 两个 post-drop epochs。
- 原 LR schedule 保持固定；不再因代理波动临时调 LR 或 batch，否则变成新的候选搜索。
- LR drop 时重置 early-stop reference 与 bad-epoch counter；epoch11 的有效代理 loss 建立新 reference。
- 令 reference 为 `L_ref`。仅当 `L_e < L_ref * (1 - 0.002)` 时认定有效改善，并更新 reference、计数清零；否则 bad counter 加一。这样小改善可以累计跨过 reference，不能每次微小下降都更新 reference。
- 在 min epoch 和 post-drop 条件均满足后，`bad_count >= 3` 停；reference 在 epoch11 建立的规则下，第一次可能触发是 **epoch14**，不是 epoch12。若期望 epoch12 就可停，必须另定义降档前历史如何使用；不推荐。
- 非有限 loss、验证未完整完成是运行失败，不计作 plateau 或“已收敛”。训练 loss 只作健康监测，不能替代 val loss。
- 上限保留20 epoch；未触发 early stop而到达上限，记录 `budget_exhausted`，不能写“已收敛”。

固定128个样本可降低逐 epoch 噪声，但只覆盖很小的验证片段；11/01/10 没有双缺失 hold、长时间 drift 或恢复闭环。因此不要据此证明 outage recovery 已收敛。全 val97 tracking 是最终选择依据。

### 3. 滚动最后5个：明确收益和代价

停止于 E 后候选集合固定为 `{E-4,...,E}`；全部5个均需跑两条件的完整97条序列，不能先看测试指标、不能只验证 proxy 最好的一个。该规则节省存储和验证总量，但可能删除早期跟踪最好的 checkpoint，尤其 proxy 与 tracking 排序不一致时。

最小建议：坚持预先约定的最后5个选择规则，同时日志保存全程 proxy 最好 epoch 和它是否已被窗口排除；不要宣称找到了全训练历史的 tracking 最优。若需要保留历史 proxy-best 权重，应作为额外审计文件提前声明，不临时增加第6个正式候选。

最终 winner 定义：

`score(e) = [SR_complete(e) + SR_alternating_outage(e)] / 2`

SR 必须是明确固定的 success-curve AUC，而不是误用 `IoU>=0.5` 成功率。建议新 development evaluator 固定 overlap thresholds `0,0.05,...,1`，逐序列对有效帧计算 `IoU > threshold` 成功率，21点算术均值，再对97条序列等权平均；两条件等权平均。将它命名为 **val97 development SR**，不冒充正式 LasHeR/Miss 指标。

评分排除初始化首帧和非正宽/高 GT；剩余应为 `54646-97-260=54289` 帧/条件。双缺失帧只要 GT 有效，仍计入评分，不能为了跳过 forward 而从分母移除；x/y 非正本身不能排除。逐帧保留预测，不重初始化、不用当前 GT 构造 search crop。GT 只用于首帧初始化与离线指标。

并列分数用较早 epoch 决胜；同一 winner 用于 complete 和缺失正式测试。不得分别选两个权重，也不得根据正式测试结果返选 epoch。额外记录 PR@20、两条件各自 SR 和每序列结果作诊断，不增加事后选择规则。

### 4. 独立入口与清理的最小边界

`train_candidates28.py` 只增加固定代理验证、early stop、每 epoch 保存/滚动保留；不改冻结 model/objectives/tracker，不接入当前 GPU0/1 的 controller。新结果目录要与正式 runs/results 分开，并记录新入口和冻结依赖的源码 hash。

每个方法固定一张物理 GPU：utility→2，random→3；显式设置 `CUDA_VISIBLE_DEVICES`，不能 source 环境后遗留默认0/1。训练期间只跑代理；训练结束退出训练进程，再开始真实 tracker 验证，避免训练保留显存与新模型叠加。两个方法相同 batch/LR/样本预算、共享 tuple manifest 和最终评价规则，但各自按验证触发停止并报告各自更新数。

最后5个权重必须可被 `MissingTracker` 严格加载，保留 `model`、`config`、`epoch`、`updates`。若声称可断点续训，还需单独最新 resumable 状态的 optimizer / scheduler / RNG；现有只含 optimizer 的 epoch checkpoint不能自动视为完整精确续训快照。本方案不要求为此额外实现 resume。

全部候选两条件验证成功、winner 哈希和指标清单写入并验证可加载后，才清理本次候选目录中的其他权重。保留 winner、运行配置、每epoch代理日志、全部候选指标/轨迹、hash、停止原因和删除清单。失败或缺结果时不选 winner、不清理。当前正式权重与历史其他目录不在清理范围。本次只读任务未实施任何清理。

## 实测时长与预算

### 已有 GPU 跟踪实测

`run_gpu_consistency_cohort28.py`：8 条完整序列共4602帧；GPU2跑 complete，GPU3跑 alternating_outage，同时 GPU0/1 正式训练。用的是历史 utility epoch20，且有额外特征 hooks/诊断输出，不是当前候选新权重。

| 条件 | 8序列实测秒数 | 视频帧吞吐 | 按54646帧线性外推 |
| --- | ---: | ---: | ---: |
| complete | 177.7552 | 25.89帧/秒 | 35.18分钟 |
| alternating_outage | 156.3504 | 29.43帧/秒 | 30.94分钟 |
| 同权重两条件顺序运行 | 334.1056 | — | 66.12分钟 |

两条件总计109292视频帧。complete实际 post-init forward54549次；alternating_outage有7546个00 hold，forward47003次。没有 forward 的帧仍需输出与评分。

外推不是全97条的实测；不同图像分辨率、序列长度、I/O争用、新模块和初始化开销会改变速度。删掉诊断 hooks可能更快，但不据此承诺加速。每权重两条件预算75–90分钟；最后5个即 **375–450分钟（6.25–7.5小时）/方法**。两方法并行按相同墙钟预留；共享存储争用后应重估。若只让一方法使用两卡分别跑两条件，可能接近每权重35分钟，但会挤占另一方法，此方案不采用。

### batch5训练与代理验证预算

八更新实测剔除第一更新启动时间：

- utility：`(25.5262349-8.4862871)/7 = 2.4343秒/update`；
- random：`(25.1016641-8.2268414)/7 = 2.4107秒/update`。

以2000updates/epoch外推，约80.36–81.14分钟/epoch。短 profile不涵盖长周期 I/O/热状态，只作初始计划。128×3=384 tuple-state 评估，batch5约78个前向批次（末尾不足5也应计入，不能drop_last丢掉固定样本）。如果实现按每状态分批会是3×ceil(128/5)=78次，不能误计为128次。

**384-state代理验证尚未实测。**先预留1–5分钟/epoch；这是规划区间，不是测量结论。首次已授权运行时，记录包含取数、forward、任务loss、同步和写日志的独立wall time，并用它替换本估计；本次不启动计时任务。

| 场景 | 每方法初始墙钟估算，两个方法GPU2/3并行 |
| --- | --- |
| epoch训练 + 一次128-tuple代理 | 约81–86分钟，另加epoch保存时间 |
| 最早按上述规则停于epoch14，再验证最后5个 | 约25–28小时，另加未计入的保存/启动开销 |
| 跑满epoch20，再验证最后5个 | 约33–36.5小时，另加未计入的保存/启动开销 |

不要用前两update启动时长估算整轮；也不要把训练 batch 变大理解为 tracking 验证 batch 变大。tracking是逐序列状态递推，最小方案保持单方法单GPU串行序列，暂不增加多进程争抢显存。

检查长任务按阶段 ETA 安排，接近预计完成再看；需要监测时300秒间隔。八更新短profile无需沿用长训练轮询。验证开始后用已完成序列的帧数/耗时更新ETA，缺失 hold 与完整forward成本分开解释。

## 证据定位（服务器绝对路径，均为本次只读取得）

前缀 `R=/home/gaob/Miss-RGBT/runtime/task_utility_20261018`：

- `R/miss_rgbt/train.py:67–88,90–158`：train split、AdamW/StepLR、epoch/update语义、探针、保存；`R/configs/a_utility_full20.json`：原始训练配置。
- `R/miss_rgbt/data.py:40–57`：validation模式仍使用采样器和裁剪设置；`R/lib/train/dataset/lasher.py:61–84`：split文件与GT有效宽高规则。
- `R/miss_rgbt/tracker.py:14–22,28–62`：严格加载、初始化、native crop/Hann/模板更新。
- `/home/gaob/Miss-RGBT/prepare_val97_recovery28.py:24–73`：97条划分、固定mask、GT、历史暴露限制。
- `/home/gaob/Miss-RGBT/evaluate_val97_recovery28.py:37–44,74–120`：GPU限制、完整轨迹、诊断范围。
- `R/scripts/evaluate_model.py:29–58,105–151`：硬编码正式测试数据规模和指标，不可用于选epoch。
- `/home/gaob/Miss-RGBT/reports/server28-training-data-audit.json`：882/97、459435/54646、260无效val GT。
- `/home/gaob/Miss-RGBT/reports/server28-val97-gpu-consistency-20261030/{complete,alternating_outage}/completed.json`：177.75523114204407 / 156.35040044784546秒；对应`report.json`记录权重、来源hash及8条序列。
- `/home/gaob/Miss-RGBT/runs/server28/profile_batch5_probe8_{utility,random}/steps.jsonl`：八更新时长、显存、探针实际记录。

本次取得的关键SHA256：

| 文件 | SHA256 |
| --- | --- |
| `R/miss_rgbt/train.py` | `c74f079465024b1890213a9737f0179502ccdeb217e22501f2665dd778f0323a` |
| `R/miss_rgbt/data.py` | `78c18ee6a17cc1fbba8a1b345d901217bcc9c6f8348cd195c484b41c74a8dfdd` |
| `R/miss_rgbt/tracker.py` | `1d05274d2827de9a4e08044e5837fedaba971400162a72cb90e98c0d1679efa4` |
| `/home/gaob/Miss-RGBT/development/recovery_val97_20261030/protocol.json` | `71594b9155665ade076a1aff362b509f9bcbb3e6b76e66b24fb0ca1057853ef5` |
| `R/lib/train/data_specs/lasher_train.txt` | `27d39054b194f1b2e855c0bc1a1e23651f5e2b1cfef1e34d88074d71565df846` |
| `R/lib/train/data_specs/lasher_val.txt` | `ff5767e1e2522b87d31b898eb949949478c1dabc3b2cef2754a43c65e089016a` |

上述SR定义、early-stop细则及预算余量是本文建议，不是现有代码已经实现的功能。当前只读检查没有运行新训练、验证或清理，没有修改正式runtime。
