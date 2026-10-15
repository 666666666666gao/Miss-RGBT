# Miss-RGBT 交接文档

更新日期：2026-10-15。冻结E3/E4各20轮及四组完整评估已完成，最终指标已实际采集汇总；研究目标仍未达成。完整结果已归档，独立审计已生成；当前归因队列正在运行，研究目标仍未达成。

## 当前归因队列与资源：2026-10-15 14:10已核对

14:10:41北京时间单次观测确认控制器3796810仍执行scripts/run_attribution_controls.py，科学版本dfc989e952014a6d5f2d22eb330428992489e175。E0-availability完整/Miss两项均已exit0；完整于12:33结束、Miss于13:19结束。13:19已实际启动E0-FT/GPU2（PID4063196）和E2/GPU3（PID4063197），各20轮/50000更新。14:10截面分别2825/50000与1529/50000更新，近期约0.917473/0.498578 updates/s；这是该节点的真实进度，不是持续实时状态。运行快照见reports/attribution_runtime_20261015_1410.json。

两项短检查实际exit0，各2次真实AdamW更新、212个状态项改变、严格重载通过；融合BN计数E0-FT增12/E2增24，36个头部运行缓冲不变。峰值分别9551390208/17284329984字节，单卡真实train序列完整/缺RGB/缺TIR推理都通过。见reports/attribution_preflight_20261015.json；不是正式20轮指标，也不用于正式初始化。

v1控制器在任何NN启动前触发磁盘准入失败，home实际只剩922402816字节。现将后续正式检查点保存到/data2/gaob/Miss-RGBT/checkpoints；项目、环境和评估仍在原home路径。仅两份本轮短检查权重经SHA核验迁到data2，原路径链接核验通过，腾出2473471054字节，home采集时可用3395690496字节。原E0与历史E3/E4全部留在原位；旧失败日志和v1清单保留，v2不覆盖它们。迁移收据见reports/attribution_storage_20261015.json。早先短检查启动器的字符串转义失败也保留，发生在远端代码执行前，修复后两检查均真实完成。

原观察器native48256/python24136在14:10完成且exit0，已消费；资源采样native7634也已exit0。新的唯一被动观察器native61699/python65768（parent64100）于14:23:13实际启动，下一必要节点为2026-10-16 04:22:40.141988北京时间，即按本次速率预计E0-FT结束前300秒；当前训练结束估计E0-FT约10月16日04:28、E2约17:11，届时还需固定epoch20全量评估。新节点前不再查询NN进度或GPU，不重建观察器。节点后只消费原61699；若仍临近结束，按180～300秒重查或重新估算。详见reports/attribution_observation_schedule_20261015_1410.json。

最终20轮独立审计状态：WARN，same-family/provisional。审计报告及输入SHA、复算范围见reports/full20_comparison_20261013_v1/EXPERIMENT_AUDIT.md/json。后续本交接的变化是实际运行与用户策略更新，不扩大该次审计的输入字节范围；科学结果、冻结源码、原始轨迹与训练记录不变。

## 评分公开来源与开发划分更正：2026-10-15 15:15

静态核查已完成，见docs/SCORING_PROVENANCE_FINDINGS_20261015.md。本地持有的1967个评分包文件逐字节匹配公开PyPI rgbt1.0.1 wheel，评测后安装包清单3556项路径/哈希也全部匹配；新增公开发行字节不扩大原审计当时实际持有的安装副本范围，不追认为启动前封存。评分代码可追溯至opacity-black/RGBT_toolkit，不能据此称数据集作者官方工具箱，作者网盘工具箱压缩包仍未取得；与SCDT实际评分实现严格等价尚未认证。已有指标、活动评分源码/环境及训练均未改动。

作者Protocol2固定名单为train979/test245；本地静态集合比较确认当前train882与val97合起来恰好等于作者train979，当前Test245集合与作者相同。882/97称项目沿用的开发划分；现行refine-logs/EXPERIMENT_PLAN.md已去掉“官方882/97”表述，历史冻结快照保留。用户授权的数据集专用内部参数配置继续有效，原E0历史训练暴露及RGBT234独立开发协议边界见下节。下一NN观察仍为10月16日04:22的原61699，本轮不提前查询或改调度。

## 模块A隔离实现与CPU实证：2026-10-15 22:39—22:41

正式归因/BN次序不改。独立分支prepare/task-utility-20261015，当前f6cead72d19a2544236cbcba2a7a86d2f15ce80f，已推送源码、四种选择策略的完整预算配置及真实CPU报告；未合入/部署活动main。实现TaskUtilityRepair：第6块后、16个区域中默认选4区64个搜索token，缺失补偿和已有观测的系数受限细化；只用本视图及初始模板中心/背景线索，残差零初始化。selector以单区域修正开/关定位损失差为训练目标，不读取C分支激活，不声称真实特征恢复。all/random/similarity/utility参数量相同，all的修正计算更多；数据集内部宽度/深度/K仍需开发验证，默认值未据Test选取。

实际CPU执行源码a229afc898ef98c2c63afc7c0b30ac1fc0c75e7c，106文件部署逐SHA核验，独立目录/data2/gaob/Miss-RGBT/prepared/task_utility_20261015_v1。PyTorch1.12.1+cu113、Torch两线程、OpenCV无工作线程、loader0，八项单元检查和约100.678秒native均exit0；原始E0 SHA仍d20c8641。一个真实元组（两模板/三搜索、microbatch1）重复两次AdamW更新，新增596357参数，总103734794。delta与selector梯度有限非零且参数改变；39个BN运行缓冲不变。初始化完整baseline、探针/无探针主输出及RNG、更新后不可见RGB扰动、保存到新实例严格重载的四输出误差均0；完整输入细化路径已有非零输出影响。第二次单样本探针相对效用6.1547937e-5只是微小局部干预值，不是任务性能提升。

实际源/日志摘要与范围见reports/task_utility_cpu_20261015_v1；检查权重epoch0/updates2、1243903793字节，SHA46040b331136c1b2f128cadcd0521928655cff8036619dc13135a94d4d35a43f，仅留隔离data2，不混作epoch20正式模型。原native66843已真实exit0并消费，不再等待或重启它。当前规范后续只补充全局裁剪耦合说明；NN源仍与实际a229afc完全相同，旧执行包保留。直接辅助梯度分离不保证AdamW步幅隔离或完整精度不降。

正式main训练入口/有效batch8、GPU原生检查、20轮/50000更新和四套全量指标尚未执行，普通投影容量对照及A/B/C性能仍待完成。main的miss_rgbt/lib/scripts/configs/experiments仍与dfc989e一致；当前GPU队列、环境及评分未改，没有提前读其进度。唯一正式观察器仍为native61699/python65768，10月16日04:22:40.141988北京时间才收取；期间新增CPU检查已结束，不再存在额外NN预检任务。

## 2026-10-15 本地离线定位复核与开发数据准备

15:06新增原E0保存元数据的只读CPU核验，原权重SHA仍为d20c8641。settings保存cfg_file路径、batchsize8和fix_bn=true，stats有train/val；未保存完整cfg或序列名单/哈希，net_info和constructor为None。因此当前train882与val97名称独立已确认，但val97未被原E0训练使用的历史独立性尚未认证。详见reports/development_data_20261015/README.md和e0_saved_training_metadata.json。没有NN前向、GPU或活动训练进度查询；数据集专用内部参数策略不变，尚未选参。

此段为12:30离线工作记录，当时不运行NN、不读实时训练日志或GPU状态，仅掌握11:40启动收据。后续14:10实际进度及新观察节点见顶节。两份CPU分析及完整表格见analysis/diagnose_full20_localization.py、analysis/decompose_rgbt234_sequence_weights.py与reports/full20_localization_diagnostics_20261015_v2/README.md；独立复核见同目录DIAGNOSTIC_AUDIT.md/json。初始审计WARN及原证据保留；字段含义限定已由同一审计员窄范围复核PASS（仅该项），表格数值与科学输出未变，见审计追加记录。same-family/provisional不变。

关键发现一：条件IoU的命中集合会变。RGBT234-Miss visible GT下，E4各自命中帧IoU低于E3，但同一93350个共同命中帧上，E3/E4为70.097636/70.207156%，差+0.109520pp。它仍是按两个模型输出选择的诊断子集，不能等同总体框质量或修复因果收益。

关键发现二：同一MSR曲线按原记录的序列等权汇总，E4−E3为−0.066192pp；仅把权重改为序列帧数时为+0.583553pp，234序列中109升、123降、2相等。帧数加权只作失效分布诊断，不能替代原主指标，也不能据此按Test长度选参数。机器字段official_protocol_delta_pp只指封存安装评分包的原协议，不表示已认证上游官方实现。

关键发现三：相对差距缩小不代表绝对恢复更好。LasHeR恢复帧中，E3完整crop有GT而Miss完全无交集的比例为4.861187%，小于E0的10.098136%；但绝对Miss无GT交集比例18.364890%，高于E0的17.514866%。E4绝对比例16.269016%。须同时保留完整能力、绝对失败和配对差距；几何相交不证明当前可用传感器中目标像素可见、身份可靠或扩域干预有效。

12:14只读核对LasHeR val97：54646帧，RGB/TIR文件数与GT行数逐序列匹配，首帧尺寸均正；train882/val97/Test245序列名无交集。GT中534行含非正坐标、260行宽或高非正，不能直接当作目标消失或静默删帧。原97份GT和两份split共99文件的私有归档251010字节，SHA256 f1966b4b874e7c04e5c005a604ba119bdf8f4d1c2b568b4430db1612dae0df56；Git只提交inventory与范围说明，见reports/development_data_20261015/README.md。未读图片、未生成开发mask、未评估验证模型或选参；RGBT234独立开发协议仍未定义。数据集内部配置选择继续遵守固定模块职责、插入位置和连接的约束。

## BN单变量对照的隔离准备（未正式训练）

独立worktree分支prepare/bn-freeze-20261015已提交并推送，科学准备提交71f46ebeedf1e80c992f34ce811795f8cd56ac7f。只为fix_bn增加可选范围开关；默认保留现有12个预测头BN冻结，开启时冻结全部13个BN的运行统计，affine与其他模块训练模式保持。当前主项目miss_rgbt/lib/scripts/configs/experiments仍与dfc989e相同，没有合入或部署到活动队列。

原环境、原E0初始化的CPU BN子模块检查17.408秒退出0：两模型各13个BN，默认仅融合计数+1，allbn的39个运行缓冲不变，全部BN affine有有限非零梯度并实际更新；其他模块仍train。没有图片、完整任务前向、GPU短检查或正式训练。完整记录及源码见[隔离准备分支](https://github.com/666666666666gao/Miss-RGBT/tree/71f46ebeedf1e80c992f34ce811795f8cd56ac7f)，本地小型元数据见reports/bn_controls_preparation_20261015.json。

E3/E4-allbn各20轮/50000更新配置、两步真实任务预检和正式全量评估队列已准备；入口要求阶段1完整结束，GPU2/3空闲，再独立从E0开始。新增启动清单纳入实际data_specs、882份训练init.txt、安装评分包/GT/masks。当前尚未执行GPU预检、正式BN对照或性能验证，不能据CPU检查归因完整性能下降。该准备记录早于14:10；当前实际进度与后续观察节点见顶节。

## E0-availability完整结果与并发决定

零训练对照已完成完整/Miss共958次序列运行、674706帧，原E0张量按封装见证不变。完整输入五项指标精确等于E0；479条完整轨迹中254条数组精确相等，最大bbox数值差2.614797267597169e-12，不能写成全部字节一致。

Miss结果：LasHeR PR/NPR/SR=61.314029/56.806230/49.224670，相对原E0为−2.158478/−1.775621/−1.358479pp；RGBT234 MPR/MSR=79.941202/58.034248，为−2.853613/−1.540187pp。独立审计状态WARN（same-family/provisional），见reports/e0_availability_20261015_v1/RESULTS.md及EXPERIMENT_AUDIT.md/json。该对照是整体可用性感知推理路径，包含缺失embedding和历史模板标识；不能细分成某一行的独立因果效应，不能据此声称KD/R或三个候选模块成立。仍等待E0-FT/E2完整指标和后续BN对照。

14:11:25～14:12:25一次60秒采样：GPU2已用11.19GiB/余12.50GiB，利用率均值85.17%；GPU3已用18.67GiB/余5.02GiB，均值99.17%。CPU64逻辑核区间均值35.48%，I/O等待0.626%，RAM可用均值233.99GiB。下一项双路训练CUDA分配峰值16.10GiB，超过两卡当前余量，故保持每卡一项训练（共两项），没有追加同卡任务或实施并发减速对比。当前GPU活动率不能单独证明FLOP/s饱和或并发一定变慢。原始采样与汇总见profile_output/resources_20261015_1410.json及profile_output/attribution_20261015_1410/README.md。

闭合归档21839791字节/2057文件成员，SHA256 866913d2025a9de4ebd4a354a5fd1e4254d4b12063f73bdce5e203e677594189，保存在/data2/gaob/Miss-RGBT/records及本地私有暂存。当前扫描评分包3556文件，先前本地1967文件全哈希匹配，包含713份被评分GT；额外1589个GT树文件只有远端哈希元数据，未全部下载认证。最初本地校验因假定全包覆盖而失败，核对路径后明确此范围；汇总初稿的精度曲线均值错误也由断言阻断，按实际指定阈值更正后才生成报告。原始失败和源文件均保留，没有重跑神经网络或改原始评分。

## 当前结果：冻结epoch20已全部完成

2026年10月15日11:00单次最终采集：控制器3623093为CAMPAIGN_COMPLETE且已退出，E3/GPU2、E4/GPU3各20轮、50000更新，四组全量评估均exit0。最后一组E4-Miss于10:22:42北京时间完成。每个模型完整/Miss共958次序列运行、674706帧；两模型共1916次、1349412帧。原native90047观察器与80340采集器均已exit0，不再等待或重复启动它们。原最终观察器均已消费；新的归因队列和下一被动观察节点见上节。

| 实验 | LasHeR PR/NPR/SR | RGBT234 MPR/MSR | LasHeR-Miss PR/NPR/SR | RGBT234-Miss MPR/MSR |
|---|---|---|---|---|
| E0原始epoch20 | 75.636503/71.772837/60.125714 | 90.677134/66.845228 | 63.472507/58.581851/50.583149 | 82.794815/59.574435 |
| E3最终epoch20 | 74.638688/71.006266/59.541913 | 90.364976/65.577624 | 68.180791/63.344528/54.362109 | 85.085650/60.872765 |
| E4最终epoch20 | 74.386278/70.635633/59.332284 | 89.890096/65.422137 | 69.217683/64.281733/55.047239 | 85.154460/60.806573 |

数值为百分数、单seed2027、固定最终epoch20。E3严格Miss目标0/4，E4为1/4，仅LasHeR-Miss SR超过54.4。E4的LasHeR-Miss PR距69.3仍差0.082317pp；RGBT234-Miss MPR/MSR距88.1/64.3分别差2.945540/3.493427pp。完整输入指标全部低于E0，约+2pp目标未实现。不能将epoch05或不同模型的最好项目拼成最终结果。5轮与20轮LR降阶分别在5/10轮，二者差值不等于纯训练时长效应。

实际比较输出见[完整指标](reports/full20_comparison_20261013_v1/RESULTS.md)、comparison.json/csv，训练小型元数据见reports/full20_training，原始逐步日志与全部bbox/回执和实际源码归档保留在私有full20_closed_raw及server logs/full20_closed_records_20261015_v1.tar.gz。归档48998868字节，SHA256为9b61557c67b05b796c0441b34636731cad8b1d586879b5eb53d42f819e32d37c。正式CPU汇总已经实际运行，逐序列哈希与帧数已核验；独立审计已生成，实际结论见同目录EXPERIMENT_AUDIT.md/json；不能将其中的限制解释为已验证消失。

最终检查点服务器实际CPU读取：E3 SHA256 01fa99822ab8936c02cd1f6223d0732ab450f2c942b199928b6de65903d87d5d，E4 c1b1ba0ff60b854e07f970c171a44fc543b4f4349210295632309519317aa99a，内部epoch20/updates50000/config解析值匹配。科学执行版本4c473fa，103个评估Python文件中102个匹配Git，辅助scripts/validate_focal_normalization.py的实际快照有单个已披露差异；它不被评估入口执行。实际源码及104文件清单已经归档，后续控制实验不得追溯覆盖此次执行证据。

实际BN缓冲证据：两模型融合层num_batches_tracked从37500增至637500，增量600000=50000更新×2microbatch×3search×C/M两次前向；36个预测头运行缓冲未变。此结果证明融合统计仍更新，不单独证明其导致完整性能退化。reverse_weight仍为0，本轮没有正式M→C反馈性能证据。

下一顺序保持：E0-availability无训练输入处理对照；E0-FT完整输入继续训练20轮；E2无KD/R双路任务20轮；再独立做融合BN冻结对照。随后验证A任务效用补偿与B证据状态恢复，最后判断C非对称互教是否超过标准对照。正式训练前须另记配置、source、实际启动及完整评估收据。截至14:10，E0-availability全量评估已完成，E0-FT/E2已于13:19实际启动；BN隔离准备版尚未正式运行。

用户已明确允许按数据集调整相同模块内部宽度、层数、修复预算和门限，保持模块职责、插入位置与整体连接关系。参数在train/val确定，披露参数量/计算变化；同一数据集完整/Miss使用同一配置和权重。目前没有独立指定的RGBT234开发划分，不据Test分数选择配置。当前E3/E4最终成绩沿用原始冻结配置。

## 历史进度记录（不代表当前运行状态）

### 历史：完整训练进度与接续

10月14日02:40:11单次采集确认：首轮控制器3294032已正常完成并退出，四组评估均exit0；完整20轮控制器3623093存活，阶段FULL20_E3_E4_RUNNING，02:16:45实际启动E3进程880212/GPU2、E4进程880213/GPU3，实际CUDA环境分别2、3，采集时均675/50000更新。快照reports/full20/runtime_20261014_0240.json。首轮5轮指标尚未全面满足目标，最终20轮结果仍待实际产出。

首轮5轮完整指标如下（百分数，固定epoch05，单seed；每组479序列337353帧）。完整表、相对E0差值和E4−E3差值见reports/first5_evaluation/RESULTS.md，独立复算审计另见该目录EXPERIMENT_AUDIT.md/json。

| 实验 | LasHeR PR/NPR/SR | RGBT234 MPR/MSR | LasHeR-Miss PR/NPR/SR | RGBT234-Miss MPR/MSR |
|---|---|---|---|---|
| 原始epoch20 E0 | 75.636503/71.772837/60.125714 | 90.677134/66.845228 | 63.472507/58.581851/50.583149 | 82.794815/59.574435 |
| E3 epoch05 | 74.388795/70.732830/59.200450 | 90.251132/66.118720 | 69.032091/64.017231/54.835854 | 85.224052/61.486632 |
| E4 epoch05 | 75.302106/71.670015/59.839741 | 90.294694/66.223281 | 68.461011/63.372123/54.363025 | 85.613275/61.797040 |

发现的问题：E3仅LasHeR-Miss SR超过目标54.4，其余三个主要Miss阈值未超过；E4四项均未超过。E4相对E3在LasHeR-Miss下降PR0.571080/SR0.472829pp，在RGBT234-Miss提升MPR0.389223/MSR0.310409pp。完整模态主要成绩均低于E0，未达到约+2pp目标。不能把局部修复称为两数据集均有效；reverse_weight=0，不能把日志诊断或CPU单批梯度见证称为正式反馈性能结果。

本轮完成20项指标汇总、10项修复配对差值及严格目标检查；实际配置以58fe581当时Git内容核验。四组bbox/回执归档42689425字节，SHA4eb2c98fae4eff6dbdbb01496bbd396e2eece3c04ec5848b7c270836e509de6d。本地权重字节未下载，checkpoint SHA来自服务器实际流式读取并与run对照。评估记录103源码文件，102逐字节匹配4c473fa；唯一不同的辅助验证脚本已保存实际部署快照及diff，没有覆盖正在训练的源码。实际runtime metric包15个源码文件、713个GT及缺少的序列/属性元数据已回传并SHA核验，评分规则未改。


23:10:11单次采集成功：E3/E4各5轮、12500更新，控制器记录exit0；原3294032仍存活，E3完整评估pid571805，实际CUDA_VISIBLE_DEVICES=2,3，22:59:02启动。20轮控制器3623093仍为WAITING_FIRST_CAMPAIGN并存活，尚未启动20轮训练。快照reports/full20/runtime_20261013_2310.json。

首轮完成元数据、源码、逐步日志归档SHA af6aceb3f0d5633d9432cc9bb042f4a5e67b7a446b15afac3b925710792546e3，1547796字节；两份实际源码清单各92文件逐个匹配58fe581执行版本。每份日志12500条更新连续完整，损失字段均有限，首更新两个search microbatch哈希一致。训练循环耗时E3 25992.219590秒、E4 25118.128007秒，约2.07938/2.00945秒每更新。日志统计只覆盖更新时记录的最后一个累积microbatch，不能称为全训练样本均值；顺序wait导致E4退出登记时间不能当实际结束时刻。元数据和摘要见reports/first5_training，原始逐步日志保留在SHA归档，Git仅发布小型记录。没有据此声称跟踪涨点或反向反馈有效。

用户明确“不要频繁轮询”，02:40和17:00观察器均已完成退出，未作节点之间的NN查询。下一唯一节点是10月15日11:00北京时间（UTC03:00），Windows uv54936/python46272于10月14日17:02:52实际启动并由CIM核验，native session90047。到时消费full20_final_intake/transport.json；节点前只等待，不查询NN进度。当前预计训练10月15日06:00～07:15结束，再顺序完成约3小时17分四组评估；最终得分仍须实际exit0、权重及回执验证。

完整结果汇总脚本已准备：analysis/build_full20_comparison.py。通过Python3.7语法及真实E0四套指标/数据范围检查；尚无新方法最终数值。训练与四组评估完成后按docs/FULL20_RESULTS.md执行，验证实际20轮/50000更新、固定epoch20权重、479序列回执和bbox哈希，输出JSON/CSV/Markdown全指标及严格Miss目标。脚本不进入冻结训练队列，不能据准备状态宣称完整结果完成。

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

## 历史：初始方法与当时执行顺序

首选方案 A：共享 baseline、两路真实任务监督，一个局部残差补偿模块；先 C→M 任务响应蒸馏，再验证标签质量选择的 M→C。完整保留 `refine-logs/EXPERIMENT_PLAN.md`。

先完成独立 conda 恢复、GPU witness 和独立按文档检查；随后真实单步/梯度/保存重载/关闭机制等价与独立缺失推理，再测时间冻结正式训练预算。E0完整输入→E1 mask→E2双路→E3正向指导→E4补偿→E5可靠反向反馈。三个模块需有问题诊断、有效结果和消融，当前不能称为已验证贡献。

恢复与运行信息见 `docs/ENVIRONMENT.md` 和项目私有 `.aris/compute/2025.md`。后续每次修改需记录实际命令、source hash、训练预算、完整/缺失指标、负结果与待完成项，更新本文件和桌面副本，再推送 GitHub。

## 历史：2026-10-13初始实施记录

独立环境已恢复至 `conda-envs/miss_rgbt`。真实 CUDA witness 与 fresh gpt-6-astra/max 按原样文档命令检查均 PASS（same-family/provisional，只证明环境）。版本 Python3.7.16、torch1.12.1+cu113、numpy1.21.5、timm0.9.12、OpenCV4.8.1、yaml6.0.1。

已实现 `miss_rgbt/model.py`、`objectives.py`、`data.py`、`train.py` 和带模板可用标识的 `tracker.py`：共享模型双前向、在第6块后对缺失位置进行局部门控残差、任务响应 KL、训练任务损失选择反向反馈。残差输出零初始化；新增398466参数。完整位置补偿严格为零。

真实 train 数据 native 检查：两模板、三搜索、batch2；完整输出四项与 baseline 最大差0；原生任务损失3.450153828与逐样本实现均值3.450154066一致；改变不可见 RGB 值时响应最大差0；一次真实 AdamW 更新、有限梯度、补偿参数实际变化、严格重载输出差0。GPU峰值8367684096字节，单步计算0.9103秒。这里只是检查，尚无完整新训练或涨点。

最初验证脚本错误引用 `lib.train.loss.focal`，在任何训练前报 ModuleNotFoundError；按已有实际文件修正为 `lib.utils.focal_loss`，原失败日志保留于 reports。新入口不吞掉解码或 GIoU 错误；保留原生无效随机裁剪拒绝重采样。

原始479序列 bbox/收据/曲线/日志归档已传回本机：`.codex_tmp/cadtrack_miss_ep20_20261013/baseline_raw_results.tar.gz`；本地与服务器 SHA256 相同 `85b302508ce56b7e688c0f8c21f58a6dd07fd883ea8f57db5bf99a47bae3dc0a`。项目 baseline87源码/配置和 epoch20 权重字节核验通过。该归档没有加入 Git。

历史（2026-10-13正式启动前）：当时下一步是E3/E4同输入速度检查与完整输入E0，正式训练尚未启动。候选有效batch8采用microbatch4、累积2，需测实际峰值与耗时后冻结预算。模型构造后重置 RNG，避免新增补偿层初始化改变配对训练的数据/mask顺序。

## 历史：2026-10-13首轮预算与启动前记录

E3与E4各25次真实更新已完成，分别55.74/56.31秒，峰值17.29/17.48GB。两个首update的最后一个search microbatch哈希相同；不据此宣称templates、mask或全部数据已逐项相同。新增正式入口保存首update全部search microbatch哈希。推理适配器在真实训练序列10帧完整输入上与同权重原生tracker bbox差0，两个单缺失状态输出有限；不是全序列成绩。

2026-10-13启动前冻结记录：正式候选固定5轮×20000样本、有效batch8、12500更新、seed2027、末轮权重对比，不按Test成绩选轮次。先完整E0，然后GPU2训练E3、GPU3训练E4，再逐项全量完整/Miss评估。依据实测，训练约7小时；含E0及四组评估约11～14小时。控制器/子进程实际启动后需另补收据，当前不能称正式训练已开始。

官方CADTrack MIT许可证已核实并原样保留。Git设置保留原始文件字节，修正了忽略规则误排除lib/train/data的问题；87个baseline源码/配置Git内容逐项SHA匹配，原始服务器源码在评估结束后也与记录匹配。

## 实际启动记录：2026-10-13 14:50:39 北京时间

控制器实际pid3294032，完整E0子进程pid3294054。8个worker已在GPU2、3真实运行，启动核验各约9082MiB，CUDA绑定为2,3。冻结执行代码为370f2158对应完整版本，见reports/SOURCE_VERSION.txt及reports/campaign_launch；不是当前未来文档提交的代码版本。当前阶段是完整E0重测，E3/E4正式训练只是排队，尚未开始。不要重复启动。

首个只读进度节点为16:05:39北京时间。唯一Windows被动观察器uv pid2536，记录 `.codex_tmp/miss_rgbt_campaign_20261013/observer_owner.json`，本地每240秒等候，节点前不查询远程进度。原启动SSH会话27705因后台复合shell仍保留通道，不能把会话未返回当作失败或重复启动。

CPU缺失状态诊断已完成，正式指标与描述指标分开。仅TIR组在两数据集的按帧20像素精度低于仅RGB组；这受组内序列/难度差异影响，尚非RGB依赖的因果证明。变化帧与稳定帧的差异在两数据集方向不同，不声称统一的模态切换退化。见reports/e0/MISSING_DIAGNOSTICS.md。

历史（2026-10-13早期native检查）：反向反馈在native单步的可靠比例为0；这次没有验证它实际产生有效反向梯度或增益，E5仍待验证。完整E0和首轮新模型最终指标均尚未完成。


## 历史：2026-10-13早期独立审查与修正

审查独立CPU核对479个bbox/收据哈希、337353行、旧/新环境713个GT哈希和87个baseline源码。五项当前安装评分包的指标、曲线重算与原结果完全一致（尚无独立上游官方等价证明）；这不代表新方法已达标。

原生LasHeR toolkit把GT任一坐标<=0的1260帧误差置-1，PR/NPR自动计成功，SR计失败；相对同分母计失败，PR含约0.556805个百分点的原生规则贡献。保留统一原生协议且披露，不能将其称作网络增益。

CPU诊断首跟踪帧的previous状态已修正为真实完整初始化[1,1]，影响LasHeR24条/RGBT23419条序列的切换标签；changed计数改为52383/29598，逐状态数值不变。原文件保留为before_first_transition_fix版本。RESULTS中的本地归档路径和冻结源码短ID笔误也已更正，执行源码完整值为370f215814320d048a5c33974b64052126a8a1fd。

当前整模态mask时补偿全部该模态token，是潜在特征补偿，尚无少量必要任务token选择机制，也没有证明恢复真实完整特征。三个研究模块的有效性仍待完整训练与消融。

## 正式训练前的focal数学对齐

已有FocalLoss以整个batch正点总数归一化。首版逐样本除以各自正点数只在每样本一正点时一致，故改为各样本numerator乘batch size再除max(total_positive,1)，其均值与原生focal一致，没有新增异常吞噬或恢复分支。真实GPU native/25步检查是该修订前证据；这项修订另用原生FocalLoss在CPU检验普通、混合空图、全空图三种算术输入，通过。合成算术测试不构成模型或数据集结果。

历史（2026-10-13首次训练前修订）：完整E0与控制器保持启动370f2158；修订仅用于尚未启动的正式E3/E4，单独记录FORMAL_TRAIN_SOURCE_VERSION.txt，每个正式训练目录保存版本和实际源哈希。不能把后续训练修订写成初始控制器原版本已执行，也没有修改已运行的模型训练。

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


## 2026-10-14 首轮结果复算及来源依据

experiment-audit技能要求的fresh gpt-6-astra/max审计代理已直接读取本地原始bbox、GT、回执和实际runtime metric包：四组20项指标与全部逐序列曲线复算差值0；两组E0控制的10项指标及曲线也复算一致。所有1916个新方法bbox/回执哈希、1349412行预测、713个GT、原包728文件及补齐元数据1239文件经离线核验。总体WARN，same-family/provisional；独立的本地权重字节核验和上游官方toolkit等价性未完成，单seed/5轮证据不足以证明20轮、普遍修复增益或三个模块有效。完整报告reports/first5_evaluation/EXPERIMENT_AUDIT.md/json；不改记录toolkit的GT无效坐标处理和评分口径。

research技能要求的独立源码任务已固定OSTrack 33b5e125（MIT）、MixFormer 8b87088f（MIT）、PCGrad c5fbd7c8（MIT）。文档docs/TASK_EVIDENCE_SOURCE_FINDINGS_20261014.md区分作者机制、适配提案及未验证假设；31份外部快照（22仓库文件、3tree、3patch、3PDF）、20本地快照及38个源码行号范围完成确定性检查，来源收据reports/task_evidence_sources。OSTrack候选排序可作为未来缺失分支自身目标证据的修复位置门控参考；MixFormer评分监督为正负二分类，不能当作IoU或未来模板收益；PCGrad raw梯度性质不保证实际AdamW更新或完整测试能力。尚未复制或运行作者模型，当前full20科学源码和配置不变。


## 2026-10-14 17:00 完整20轮中点

唯一17:00观察器native14398实际exit0，SSH transport0，快照09:00:12.181106Z：完整控制器3623093存活，阶段FULL20_E3_E4_RUNNING，E3进程880212仅CUDA2、E4进程880213仅CUDA3。E3最后记录epoch11/update25700/50000（51.4%），52975.124739秒；E4 epoch11/update26425/50000（52.85%），52998.046725秒。最近各3条日志损失字段有限；这里只核查已采集日志，不能据此称全部训练记录或最终任务性能已通过。

epoch10文件已保存：E3 1236735527字节、E4 1241526503字节。这次只采集其存在/大小/mtime，未做重载、评分或Test选轮；最终仍固定epoch20。完整训练completed记录尚不存在，完整模型最终指标尚未产出。原5轮回执与指标仅作历史对照。

按整体更新速度E3约2.0613、E4约2.0056秒/更新，余下训练约13小时55分/13小时08分；近期50更新速度给出的E3预计结束约07:10。四组历史实测评估约3小时17分，所以安排10月15日11:00一次结束节点检查。11:00观察器记录reports/full20/final_observer_20261014.json；采集时只能报告真实已完成范围。full20运行科学源码/配置冻结4c473fa未改，没有启用E5或新的修复/保护机制。

首轮独立审计对应此前6878e91发布输入快照，保留其哈希、实际复算及WARN范围；本段是新的只读运行观察，不把观察器注册、epoch10文件存在或配置计划称为最终指标。17:00原始快照reports/full20/runtime_20261014_1700.json。

## 2026-10-14后续用户澄清及资源评估

用户确认E0是“去掉CADTrack论文三个模块后训练出来的baseline”；DropMoE是用户说明的CADTrack主干来源，原文件暂未提供。本轮直接加载该删模块baseline的epoch20训练产物，不另加载DropMoE，不称原论文三个模块完整复现。权重路径和d20c8641 SHA未改，README已更正，优先级及来源边界见docs/NEXT_EXPERIMENT_PRIORITY_20261015.md。

应用户请求，23:57:26～23:58:26进行一次60秒资源采样：两卡RTX3090各24GiB；GPU2已用18.67/余5.02GiB、利用率样本均值95.5%；GPU3已用18.88/余4.81GiB、均值88.8%。CPU64逻辑核、区间均值43.97%，RAM可用224.35GiB，I/O等待均值0.228%。对比当前CUDA分配峰值约16.1/16.3GiB，余量不足再开同配置训练。保持每卡一个（共两项）任务，没有并发NN压测或已测多任务减速结论。原始JSON和采样文件记录见profile_output/；原11:00最终观察器未动，没有新增训练步数查询。

本地源码确认两项归因差异：E0复制可用图像 vs. 新模型复制占位后embedding显式availability屏蔽；fix_bn仅覆盖box_head，融合conv仍有未设eval的BN。尚未认定它们导致负结果。先完成冻结epoch20四套指标，再做E0-availability/E0-FT/E2和融合BN单变量；优先A+B，最后验证C是否超过普通互教/梯度对照。当前科学源码、配置和BN未改。

2026-10-15用户补充授权：在保持骨干和模块组合不变的前提下，可按数据集调模块内部参数/层数，而非学习率或batch。后续允许不同数据集的修复宽度/内部深度、选择位置数量和门控/恢复预算配置；独立验证选定后，同一数据集完整/Miss共用固定配置，记录对应权重、参数量及成本，不能把不同配置冒称同一统一模型。当前20轮仍不改。当前确认的开发集是LasHeR val97，RGBT234独立开发协议待核实；具体规则已补docs/NEXT_EXPERIMENT_PRIORITY_20261015.md。
