# Miss-RGBT 交接文档

更新日期：2026-10-13 19:51核验。当前状态：epoch20 完整/缺失 E0 全部完成；5轮E3/E4在GPU2/3训练中；20轮完整训练控制器存活并等待接续；CPU反向反馈实现检查完成。新方法最终指标尚未完成，研究目标未达成。

## 最新完整训练进度与接续

用户再次要求“做完整的训练拿完整的指标”，于19:51:15北京时间进行一次只读核验。原控制器3294032和完整20轮控制器3623093均存活。E3进程3442177实际CUDA_VISIBLE_DEVICES=2，update7050/12500；E4进程3442178实际CUDA_VISIBLE_DEVICES=3，update7300/12500。20轮队列当前仍是WAITING_FIRST_CAMPAIGN，尚未启动20轮训练。原23:10观察器继续采集预计首轮完成节点，不增加重复观察器。

完整20轮计划保持冻结4c473fa：E3/E4各从原epoch20预训练权重独立开始，20×20000采样样本、有效batch8、50000更新；固定最终epoch20，各自评估完整与Miss条件。每组包含LasHeR245序列220703帧、RGBT234234序列116650帧。输出PR/NPR/SR、MPR/MSR和同口径E0增益。按实测约2.0～2.1秒/update，当前首轮预计23时结束，接续其全量评估，再运行20轮训练与四组最终评估，预估10月15日上午到下午得分。实际核验见reports/full20/runtime_20261013_1951.json，最终指标仍待真实完成。

独立CPU探针19:36启动，19:39:19退出0，196.405秒，未占GPU。固定E4 epoch01和seed2027，第三个真实train microbatch有1个样本满足M任务损失2.663949+0.1<C任务损失2.839929。M响应目标断梯度、C接收非零梯度、梯度分解、同一AdamW更新差异及严格重载均通过；baseline接收参数最大更新差异4.433095455e-7，R直接反向梯度为0，R实际更新间接差异7.450580597e-9，四类输出重载差异均为0。报告见reports/cpu_reverse_probe_ep01_seed2027_v1。此结果仅验证CPU单microbatch实现，不能作为E5完整训练或性能收益；当前E3/E4 reverse_weight仍为0。

## 位置与范围

- 本地：`C:\Users\gb\projects\Miss-RGBT`。
- 桌面交接：`C:\Users\gb\Desktop\document\Miss-RGBT交接文档.md`，与本文件同步。
- 2025：`/home/gaob/Miss-RGBT`。用户已明确同意此位置，因 gaob 无权创建 `/home/gb`。
- GitHub：<https://github.com/666666666666gao/Miss-RGBT>。
- 只用物理 GPU2、3。其他项目的 GPU0、1 和环境不修改。

## 已完成的实际工作

1. 本地 `E:\CADTrack_ep0014.pth.tar` 已复制至 2025 和 2028 的 `/home/gaob`，三端 SHA256 一致：`cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731`，1,236,957,664 字节。
2. 在 2025 找到实际 epoch20：`/data2/zhouy/CADTrack-main/baseline/checkpoints/train/cadtrack/cadtrack/CADTrack_ep0020.pth.tar`。SHA256 `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`；元数据 epoch20、CADTrack、250 个 state_dict 项。后续从此权重预加载，不混用 epoch14。
3. 固定官方 masks，检查两数据集帧/GT 数与所有状态，完成严格加载和真实完整/单缺失前向检查。
4. LasHeR-Miss 245 序列220703帧、RGBT234-Miss 234序列116650帧，合计479序列337353帧全部完成，退出码0。
5. 初始双 worker 吞吐较低，保存已完成两序列与原日志后停止本次拥有的任务，续跑为四 worker/卡共8 worker；没有改变模型或丢弃序列。续跑阶段耗时2561.43秒，不能表述为含前段的总耗时。
6. 保存实际 baseline 的 `lib/` 与配置、权重/源码/GT/mask 哈希、每序列结果和执行收据。Git 只放代码与小型报告，原始大文件单独归档。

## 实测指标

| 权重 | 条件 | PR / MPR | NPR | SR / MSR |
|---|---|---:|---:|---:|
| epoch20 | LasHeR-Miss | 63.472507 | 58.581851 | 50.583149 |
| epoch20 | RGBT234-Miss | 82.794815 | — | 59.574435 |
| epoch20 | LasHeR完整，同入口E0 | 75.636503 | 71.772837 | 60.125714 |
| epoch20 | RGBT234完整，同入口E0 | 90.677134 | — | 66.845228 |

严格目标：LasHeR-Miss PR>69.3、SR>54.4；RGBT234-Miss MPR>88.1、MSR>64.3。相对本次 baseline 分别还需 +5.827493、+3.816851、+5.305185、+4.725565 百分点，并非仅需在 Miss 指标上提升两点。

用户提供完整输入参考：RGBT210 88.27/64.22；RGBT234 90.68/66.86；LasHeR 75.64/71.77/60.13。此行来源是用户，尚未在本入口复现。历史两个 LasHeR 结果目录重算分别75.200235/71.609387/59.881272和76.244410/71.991079/60.563321，均不完全等于参考值。因此完整输入“约两点提升”必须建立同权重、同入口、同 GT 的配对 E0，不混用目录。

## 发现的具体问题

- 旧代码硬编码 `/data3/wangwj/...`，本机实际数据/权重位于 `/data2`。采用独立配置与评估入口，没有修改原工程运行中的路径。
- 实际 baseline forward 没有使用原论文的 C/D 分支。以本次实际网络为 baseline，不能声称原论文架构完全复现。
- LasHeR `advancedredcup` 两个原始模态 GT 文件只有15行，而图像与 init.txt 为1411帧。评估改用 rgbt toolkit 的完整1411行 GT，并对全部序列做长度检查；没有截短或跳过该序列。
- 双缺失公共协议保持上一框；单缺失复制可用图像。所有帧仍计分，第一帧完整初始化。复制图像不是恢复真实另一模态，新方法必须使用明确可用标识。
- 原训练 actor 会吞掉 GIoU 异常并置零，原 trainer 有 fail_safe 恢复。新入口需暴露失败，不能将这种静默行为当作训练稳定性。
- 完整教师特征即使 detach，仍会泄漏 M 缺失模态的数值。方案 A 中 C 仅提供损失目标，M 前向独立。

## 方法与后续顺序

首选方案 A：共享 baseline、两路真实任务监督，一个局部残差补偿模块；先 C→M 任务响应蒸馏，再验证标签质量选择的 M→C。完整保留 `refine-logs/EXPERIMENT_PLAN.md`。

先完成独立 conda 恢复、GPU witness 和独立按文档检查；随后真实单步/梯度/保存重载/关闭机制等价与独立缺失推理，再测时间冻结正式训练预算。E0完整输入→E1 mask→E2双路→E3正向指导→E4补偿→E5可靠反向反馈。三个模块需有问题诊断、有效结果和消融，当前不能称为已验证贡献。

恢复与运行信息见 `docs/ENVIRONMENT.md` 和项目私有 `.aris/compute/2025.md`。后续每次修改需记录实际命令、source hash、训练预算、完整/缺失指标、负结果与待完成项，更新本文件和桌面副本，再推送 GitHub。

## 本次后续实施记录

独立环境已恢复至 `conda-envs/miss_rgbt`。真实 CUDA witness 与 fresh gpt-6-astra/max 按原样文档命令检查均 PASS（same-family/provisional，只证明环境）。版本 Python3.7.16、torch1.12.1+cu113、numpy1.21.5、timm0.9.12、OpenCV4.8.1、yaml6.0.1。

已实现 `miss_rgbt/model.py`、`objectives.py`、`data.py`、`train.py` 和带模板可用标识的 `tracker.py`：共享模型双前向、在第6块后对缺失位置进行局部门控残差、任务响应 KL、训练任务损失选择反向反馈。残差输出零初始化；新增398466参数。完整位置补偿严格为零。

真实 train 数据 native 检查：两模板、三搜索、batch2；完整输出四项与 baseline 最大差0；原生任务损失3.450153828与逐样本实现均值3.450154066一致；改变不可见 RGB 值时响应最大差0；一次真实 AdamW 更新、有限梯度、补偿参数实际变化、严格重载输出差0。GPU峰值8367684096字节，单步计算0.9103秒。这里只是检查，尚无完整新训练或涨点。

最初验证脚本错误引用 `lib.train.loss.focal`，在任何训练前报 ModuleNotFoundError；按已有实际文件修正为 `lib.utils.focal_loss`，原失败日志保留于 reports。新入口不吞掉解码或 GIoU 错误；保留原生无效随机裁剪拒绝重采样。

原始479序列 bbox/收据/曲线/日志归档已传回本机：`.codex_tmp/cadtrack_miss_ep20_20261013/baseline_raw_results.tar.gz`；本地与服务器 SHA256 相同 `85b302508ce56b7e688c0f8c21f58a6dd07fd883ea8f57db5bf99a47bae3dc0a`。项目 baseline87源码/配置和 epoch20 权重字节核验通过。该归档没有加入 Git。

下一步是 E3/E4 同输入速度检查与完整输入 E0；正式训练尚未启动。候选有效batch8采用microbatch4、累积2，需测实际峰值与耗时后冻结预算。模型构造后重置 RNG，避免新增补偿层初始化改变配对训练的数据/mask顺序。

## 首轮配对预算已冻结，尚待实际启动收据

E3与E4各25次真实更新已完成，分别55.74/56.31秒，峰值17.29/17.48GB。两个首update的最后一个search microbatch哈希相同；不据此宣称templates、mask或全部数据已逐项相同。新增正式入口保存首update全部search microbatch哈希。推理适配器在真实训练序列10帧完整输入上与同权重原生tracker bbox差0，两个单缺失状态输出有限；不是全序列成绩。

正式候选固定5轮×20000样本、有效batch8、12500更新、seed2027、末轮权重对比，不按Test成绩选轮次。先完整E0，然后GPU2训练E3、GPU3训练E4，再逐项全量完整/Miss评估。依据实测，训练约7小时；含E0及四组评估约11～14小时。控制器/子进程实际启动后需另补收据，当前不能称正式训练已开始。

官方CADTrack MIT许可证已核实并原样保留。Git设置保留原始文件字节，修正了忽略规则误排除lib/train/data的问题；87个baseline源码/配置Git内容逐项SHA匹配，原始服务器源码在评估结束后也与记录匹配。

## 实际启动记录：2026-10-13 14:50:39 北京时间

控制器实际pid3294032，完整E0子进程pid3294054。8个worker已在GPU2、3真实运行，启动核验各约9082MiB，CUDA绑定为2,3。冻结执行代码为370f2158对应完整版本，见reports/SOURCE_VERSION.txt及reports/campaign_launch；不是当前未来文档提交的代码版本。当前阶段是完整E0重测，E3/E4正式训练只是排队，尚未开始。不要重复启动。

首个只读进度节点为16:05:39北京时间。唯一Windows被动观察器uv pid2536，记录 `.codex_tmp/miss_rgbt_campaign_20261013/observer_owner.json`，本地每240秒等候，节点前不查询远程进度。原启动SSH会话27705因后台复合shell仍保留通道，不能把会话未返回当作失败或重复启动。

CPU缺失状态诊断已完成，正式指标与描述指标分开。仅TIR组在两数据集的按帧20像素精度低于仅RGB组；这受组内序列/难度差异影响，尚非RGB依赖的因果证明。变化帧与稳定帧的差异在两数据集方向不同，不声称统一的模态切换退化。见reports/e0/MISSING_DIAGNOSTICS.md。

反向反馈在native单步的可靠比例为0；这次没有验证它实际产生有效反向梯度或增益，E5仍待验证。完整E0和首轮新模型最终指标均尚未完成。


## 独立审查中已确认与修正的事项

审查独立CPU核对479个bbox/收据哈希、337353行、旧/新环境713个GT哈希和87个baseline源码。五项官方指标、曲线重算与原结果完全一致；这不代表新方法已达标。

原生LasHeR toolkit把GT任一坐标<=0的1260帧误差置-1，PR/NPR自动计成功，SR计失败；相对同分母计失败，PR含约0.556805个百分点的原生规则贡献。保留统一原生协议且披露，不能将其称作网络增益。

CPU诊断首跟踪帧的previous状态已修正为真实完整初始化[1,1]，影响LasHeR24条/RGBT23419条序列的切换标签；changed计数改为52383/29598，逐状态数值不变。原文件保留为before_first_transition_fix版本。RESULTS中的本地归档路径和冻结源码短ID笔误也已更正，执行源码完整值为370f215814320d048a5c33974b64052126a8a1fd。

当前整模态mask时补偿全部该模态token，是潜在特征补偿，尚无少量必要任务token选择机制，也没有证明恢复真实完整特征。三个研究模块的有效性仍待完整训练与消融。

## 正式训练前的focal数学对齐

已有FocalLoss以整个batch正点总数归一化。首版逐样本除以各自正点数只在每样本一正点时一致，故改为各样本numerator乘batch size再除max(total_positive,1)，其均值与原生focal一致，没有新增异常吞噬或恢复分支。真实GPU native/25步检查是该修订前证据；这项修订另用原生FocalLoss在CPU检验普通、混合空图、全空图三种算术输入，通过。合成算术测试不构成模型或数据集结果。

完整E0与控制器保持启动370f2158；修订仅用于尚未启动的正式E3/E4，单独记录FORMAL_TRAIN_SOURCE_VERSION.txt，每个正式训练目录保存版本和实际源哈希。不能把后续训练修订写成初始控制器原版本已执行，也没有修改已运行的模型训练。

## 完整训练要求与2026-10-13 17时核验

用户补充“做完整训练得到完整的指标”。原首轮5轮不是baseline的完整20轮预算，因此保留其正式对照及全量评估，另冻结20轮配对E3/E4，从相同epoch20预训练权重独立开始，不将旧5轮称为20轮完成或精确resume。

实际服务器核验：原控制器3294032在运行；E3训练3442177仅CUDA2，E4训练3442178仅CUDA3，均从15:45:39北京时间真实启动；约17:00时各约2100次更新，无最终成绩。训练源train.py SHA c16c5376、objectives.py SHA ea899e47，与58fe581修订一致。reverse_weight=0，日志中的非零reverse_fraction/kd_reverse仅为候选反馈诊断，不能称反向知识已参与优化。

完整E0全部479序列337353帧、exit0，于15:45:39完成。LasHeR PR/NPR/SR=75.6365028872/71.7728368056/60.1257136761；RGBT234 MPR/MSR=90.6771339213/66.8452283302。这是正式完整输入配对对照，用户参考值和旧目录不再用于计算本轮增益。

20轮配置：configs/e3_full20.json、e4_full20.json；每项20×20000个采样样本、有效batch8、50000次更新，固定seed2027，LR1e-4、backbone倍率0.1，第10轮降至0.1倍。原生LasHeR train882用于训练，val97仍留作开发；不是测试245训练，也不是穷举每一视频帧。C/M两路真实标签监督，forward_weight0.1，reverse_weight0；E4比E3只增加局部补偿。反向反馈E5需独立验证与完整消融后再认定有效。

当前/home实测可用34,191,867,904字节，故新完整训练仅保存epoch10/20，每步日志不删。对未来train入口只改变检查点保存条件和版本文件来源，不改前向、损失、优化器或采样计算；当前5轮进程已加载旧入口/config，继续按原5轮预算，不重启、不覆盖原源哈希。

20轮预计每项约28～30小时，两卡并行；之后4组完整/Miss全量评估约4～6小时。首轮尚余约6小时训练和约4～6小时评估，因此本次20轮最终成绩预计2026-10-15上午到下午，以真实速度修正。控制器排队收据与实际启动另记，不能仅凭配置存在宣称训练完成。最终成绩固定用epoch20，不根据Test选轮次。后续只在预计诊断/完成节点收集，必要轮询300秒。

旧16:05只读Windows观察器实际SSH退出255（banner exchange），原始first_checkpoint_intake.json保留；当前已独立SSH成功核验真实运行，旧观察器无活进程，不把观察器通信失败写成训练失败。

## 完整20轮接续实际排队：2026-10-13 17:06:34

`logs/full20_20261013_v1`控制器pid3623093，09:08:32Z实际仍活、状态WAITING_FIRST_CAMPAIGN；当前没有20轮子训练进程，不称已开始20轮。冻结完整训练科学/执行源码4c473fa，训练/配置/脚本部署逐文件SHA一致；源码解析Python3.7与配对预算检查通过，仅属配置与语法检查。初次语法检查多一个右括号，已在部署前修正，错误写入preparation_validation.json。

未来train.py仅增加按config保存指定轮次和选择版本文件；当下5轮进程保持其原58fe581代码。20轮的model.py、objectives.py、采样与optimizer计算没有修改。前述C/M都有真实task监督，E3/E4没有启用reverse_weight，因此不把reverse_fraction非零误称为反向指导有效。

完整E0原始479bbox/479收据及所有曲线与配置已归档回本地，13,708,364字节，SHA256 eff683ac725fad93d5c3de02e0fc5a5dde46a081aae49f551dd707f901081951；本地/服务器一致。Windows.codex_tmp为目录联接，Python实际路径D:\Program Files\UserCache\gb\codex_tmp\miss_rgbt_campaign_20261013\e0_complete_raw_results_20261013.tar.gz，可继续从C:\Users\gb\.codex_tmp\miss_rgbt_campaign_20261013访问。首归档检查误把E0的results目录当bbox目录，assert退出1；修正后479/479及归档SHA核验成功，原失败transport保留。没有因此改模型或重跑NN。

唯一下一节点观察器uv37852/python38240，实际启动17:09:59，计划23:10北京时间（15:10Z）只读收集；本地等待300秒，不在节点前查询NN训练/评估进度。不启动第二个观察器；原第一次2536/3232已退出。

结果索引reports/EXPERIMENT_STATUS.md、METRICS_SUMMARY.csv。所有新E3/E4完整/Miss指标尚待训练与337353帧评估；三个有效贡献和性能目标均未完成，后续须核验补偿提升，再做有选择的M→C完整训练及对应消融。

## 2026-10-13离线同帧诊断与源码核查

本轮没有新增NN、改变冻结模型/config或查询当前训练进度。离线逐项核对1675个输入哈希，对479序列完整/Miss预测配对，确认首次缺失前53027帧bbox完全一致。排除初始帧/无效GT后，LasHeR当前11帧的按帧精度20px差-8.9812pp，RGBT234-1.8629pp；真正经历缺失后恢复的54822/26353帧，差-14.2370/-3.3317pp。这是连续坐标、按帧描述，不替代正式PR/SR。

按原生4倍搜索区域公式，从上一框计算几何支持：LasHeR恢复帧有6035/54822（11.0084%）完整crop含GT中心而缺失crop不含；其中5536（10.0981%）完整crop与GT有交集而缺失crop无交集。RGBT234对应768（2.9143%）/684（2.5955%）。已发现具体的历史搜索区域偏移问题；没有读取图像，不把几何交集等同目标外观可见，没有证明GT重置/模板干预可以挽回多少部署指标。两模态返回后误差不必自动消失，局部补偿也不能默认crop仍包含目标。

报告reports/e0_paired_diagnostics_v1/README.md、paired_diagnostics.json、sequence_deltas.csv、search_support.json、SVG/PNG；离线工具在analysis/。首CPU运行误用bbox_sha256，实际旧E0为result_sha256，失败后按实际字段修正、退出0；没有加兼容fallback或重跑跟踪。

research技能要求的独立background研究由gpt-6-astra/max完成，仅源码研究，不是外部仓库运行复现/独立性能审计。SDSTrack、SMML、RobustSeg固定commit、许可证及精确调用位置均记录docs/OPEN_SOURCE_FINDINGS_20261013.md。已核实RobustSeg CVPR2026正式页、官方附源码ZIP与MIT；旧repo404但RobustSeg/RMMSS可读。该反馈脚本第二optimizer仍绑定student而非teacher，且冻结/损失/LR与论文不同，不能照搬后声称反馈已复现。SMML关键loss源码缺失、未见许可证；SDSTrack的双端特征MSE不是固定教师空间响应KL。

本项目RGB/TIR主干共享，R只在mask位置激活；C完整输入时R零门控。因此未来M→C梯度接收的是共享baseline，不是修复器；不能冻结全部baseline后期待该项有效。E5启动前需真实训练样本可靠性触发、独立梯度/实际参数更新和严格重载，再做完整训练。E3/E4当前reverse_weight0，三个贡献仍UNPROVEN。具体三项消融、参数/训练量控制和信息限制已写docs/MODULE_EVIDENCE_PROTOCOL.md，不按Test诊断调当前frozen配置。

唯一23:10节点观察器python38240/uv37852已再次CIM确认仍活；节点前无remote NN进度查询、无重复observer/训练启动。等待真实指标后再判断当前R能否缓解上述历史损伤。

## 2026-10-13 时序源码与反向原生探针准备
上一 goal turn 的实际进展为 da05cfb 的15份静态诊断/研究记录完成 GitHub、Desktop 和服务器逐文件 SHA 同步。本轮未查询远程 NN/GPU，不改冻结科学文件，已通过本地 CIM 再次确认唯一23:10观察器同 PID/出生时间仍活；当前远程训练事实仍引用17:08实况，不能将静态同步写成训练健康的新检查。

research 技能显式要求的后台任务完成两套时序源码核查：DiMP/pytracking 固定7eb9e74bd3d40e29dbcec444902237da13de247b，GPLv3；SAM2固定2b90b9f5ceec907a1c18123530e92e794ad901a4，Apache2。DiMP 默认 uncertain 仍更新位置/尺度，只停分类器记忆学习；not_found 才停定位提交，没有自动扩域。SAM2 presence<=0抑制mask并替换pointer，但空间memory仍编码保存，默认按时序读取；全帧1024编码依赖目标conditioning，不能视为无条件身份重定位。17份来源文件的本地SHA256/blob SHA1和29个行号链接已复查，未验证远端tree、未运行外部代码。笔记 docs/TEMPORAL_STATE_SOURCE_FINDINGS_20261013.md，收据 reports/temporal_sources/。

当前 Hann 加权最大响应并非经过presence标签校准的目标存在概率。模板写入、位置/尺度提交和扩域观察是不同干预，须在train/val分别固定/消融并计推理成本，不能把拒写记忆当成裁剪外恢复。Test失败分析已参与研究问题的形成，公开Test不能再被说成完全未见的开发外证据；冻结当前实验不变，未来门限/预算仅由train/val确定。

已准备 analysis/validate_reverse_feedback.py；仅通过Python3.7 AST/compile检查，尚未GPU/真实数据执行，reports/reverse_gradient_probe/preparation.json明确PREPARED_NOT_EXECUTED。第一次uv命令漏写python，启动阶段失败，改正后语法检查exit0；一次文档补丁上下文不匹配，在读取实际行文本后成功应用，没有改变科学文件。探针使用既有训练后E4 checkpoint、原生mask与最多八个固定seed训练batch；真实可靠门没有触发时只记未触发。触发后测M目标detach、C响应梯度、共享baseline反向梯度，对照同一已保存AdamW状态的零/0.02反向项单microbatch更新，并保存/严格重载。单microbatch机械检查不代替有效batch8完整训练，不加入当前队列，只能在冻结campaign完成且GPU空闲后运行。

反向项在C全模态路径不会直接给R梯度；但全局梯度裁剪可能因baseline梯度改变而间接改变R更新，所以探针分别报告直接梯度与实际更新差异。当前E3/E4继续reverse_weight=0，E5未实施正式训练，三个有效模块与最终目标仍未证明。full20预计10月15日得分、23:10唯一节点采集计划不变。

## 2026-10-13 19:22 CPU 原生参数路径检查准备
为在GPU冻结排队期间取得真实实现证据，反向探针增加显式 --device cpu（默认仍cuda），只改analysis/下独立检查脚本，不改训练/模型/配置/队列。CUDA模式仍须等冻结campaign完成与GPU空闲；CPU模式明确CUDA_VISIBLE_DEVICES为空、四个CPU线程，不是GPU失败后的fallback。
CPU预检11:21:59Z实际：64个逻辑核，242907279360字节MemAvailable，固定E4 epoch01文件1241526375字节存在；既有Python3.7.16/Torch1.12.1+cu113、环境spec哈希cc6cd075不变。32x32 CPU矩阵见证有限且非空，PyTorch optimizer.load_state_dict 源码实际deepcopy输入状态，配对CPU更新不需追加复制兼容逻辑。仅检查资源、已保存首轮权重metadata和CPU kernel，未查询当前NN进度/GPU状态。
最早epoch01仅供真实train样本上的梯度/AdamW/严格重载机械检查，不用于Test选epoch或代替最终epoch20评估。CPU与CUDA随机引擎不同，实际batch输入SHA和gate须记录。CPU4线程资源小于已知64核，预检可用内存远大于约17GB原生训练激活；预计探针数分钟至十分钟，最多8个固定batch，不改门限或人为翻gate。
此节记录的是启动前准备，尚无真实gate/更新/严格重载通过结果；GPU运行和三个有效模块仍未证明。预检reports/reverse_gradient_probe/cpu_preflight.json，CPU命令与限制见docs/MODULE_EVIDENCE_PROTOCOL.md。原23:10唯一训练采集观察器和20轮队列不变。
