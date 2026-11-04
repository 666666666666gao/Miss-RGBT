# Miss-RGBT 交接文档

## 2026-11-03 09:50 CST - 四卡训练正常，六臂接续队列已注册

服务器28四卡现有任务继续运行：GPU0 R1 trainer3388846/controller3388832，
GPU1 R2 trainer3395457/controller3395402，GPU2 A Utility1288280，
GPU3 A Random1288281；A全量评估队列1355742正常等待。09:48核验温度
80/79/78/75C，功率上限均250W；这是实测及功率上限，不是硬85C保证。
R1/R2仍在epoch1，09:50约update1425/650；A两臂epoch10约19850/19925。
单步损失有限，不据此声称收敛或新增官方指标。现有训练与冻结源文件未改。

新持久队列PID3539938，实际命令research/curriculum_p/protocol_queue.py，
09:50:19确认WAITING_FOR_R，目录reports/server28_rppm_ostrack_20261103_queue。
等待R1/R2两份训练核验和两份官方完成凭据，再独立复算20次全量评估并检查
R1选中权重Complete五项均不低于E0-0.1pp。门槛失败即停止，不能替换协议。
不要重启退役P控制器2911513；不要再启动同名队列。新队列及其绑定的
research/curriculum_p和research/protocol_r源文件必须保持冻结。

门槛通过后，六臂fixed60/fixed50/progressive/shuffled/paced/paced+norm先各做
4个真实optimizer更新并严格重载；六臂全部通过再用空闲GPU0-3最多四项并行。
每臂20轮不早停、保留10及16-20轮，以固定val97内部任务损失选末五轮权重；
每张卡接着做该臂5权重×Complete/Miss全量评估，每项479序列/337353帧，
4workers/卡，独立复算；共60项，完成后核验六臂输入/掩码配对。
队列300秒轮询，卡释放后分配待办，不抢占健康任务；子任务失败停止新派发。

新增protocol_verify.py、protocol_launch.py、protocol_queue.py及检查器。
服务器CPU合成检查PASS：288step、36小型权重、240命令、12模拟调度；
未初始化CUDA。报告reports/server28_rppm_evidence_checks_v2_20261103.json，
SHA0aa8aaea313ef4cfe1883cfed8e190f4b42298c032d92cc9b845937e72c53905，
本地/服务器报告与绑定源文件哈希相同；现有R源检查通过。
这不代表实际完整权重核验、真实六臂smoke或正式评估已执行。
细节docs/RPPM_PROTOCOL_ENTRY_20261103.md。

R前置未完成期间不能承诺四卡始终无空闲。OHCAS仍因R历史全帧清单缺失受阻，
等待本身不能补回清单；未获数据门槛与真实诊断证据前不启动OHCAS训练。
环境原样复用，无重建、权重清理或新增官方分数。总研究目标仍未完成。

## 2026-11-03 00:14 CST - A组正式训练进度

只读 SSH 核验原 Utility/Random 训练进程 PID1288280/1288281 仍存活，均在
epoch5：Utility update8975，Random update9000。最新单步训练 loss 均为有限值，
仅是训练日志，不是收敛证据或官方指标。评估队列控制器 PID1355742 仍存活；
最近状态快照（00:10 CST）为 WAITING_FOR_TRAINING。尚无训练完成收据或新完整
数据集官方指标。

GPU2/3 温度为75/74C，功率241.60/244.72W，功率上限均为250W；GPU0/1空闲。
按约3.16秒/update估算，下一次每轮2000 updates的边界在01:05-01:15 CST左右。
保持现有训练和队列不变，在边界附近复核，并根据实际轮次耗时更新后续 ETA。

## 2026-11-02 09:00 UTC - Pairing-verifier correction; queue v2 active

Previous goalturn was verifiedwait. Foundrealcontractbug:oldverifierrequires
equal wholelogs despite independentlystoppingarms. Historicaldense20/native
vs14/dense proves stoppinglengthdifference occurs. No currentepochcomplete.
Newverify_partitioned_a_v2.py keepsallfullsingle-armchecks andrequiresexact
devinputs/commontrainingprefix IDs/tensor+maskhashes;reportsbothlengths,
nevercallslongersuffixpaired. Tests equal/unequal-bothorders andcommoninput/
mask/devmismatchrejection passed;real6epochsmoke reverified,24commonsteps.
No oldfrozenfilechanged,no trainer/optimizer/data/seed/stopping/selection change.

Oldwaitingcontroller1318842explicitlyretiredaftercommand/status/sourcechecks,
no devhistory/officialresults. Newv2PID1355742 registered08:59:45UTC.
At09:00:09UTCcommandlive00:23/statusWAITING_FOR_TRAINING;frozenhashespass.
OriginaltrainingPIDs1288280/1288281stilllive41:09,74/75C,22710/22742MiB,
250Wlimits. No trainingrestart/cleanup/benchmarkexecution.
ACTIVEqueue reports/server28_partitioned_a_20261102_official_queue_v2/.
Oldqueuepreservedwithsuperseded.json. Futureterminalaudituses
scripts/verify_partitioned_a_v2.py --phase formal,verification_v2.json;
do not race the automatedqueue. Full20officialruns unchanged.
Amendment/evidence docs/PARTITIONED_A_PAIRING_AMENDMENT_20261102.md,
checks reports/server28_partitioned_pairing_v2_checks_20261102.json.
Nextfirstepoch~18:05CSTNov2. Officialmetricsunchanged/fullgoalunmet.

## 2026-11-02 08:44 UTC - Live training prefix audit passed

Previous turn officialqueueregistration=progress. Separate readonlyaudit of
first800microsteps each:400loggedupdates/4000fittingtuples/703trainnames,
128devinputrecords,200utilityprobes. All identities/partition/exclusions/
causalorder pass. Entire pairedinput/mask/probe prefixes exact;allrecorded
trainerandqueued frozen sources unchanged. Logged total-loss reconstruction
maxerrorUtility1.2744451e-6/Random1.3189972e-6,alllossfieldsfinite.
Not rawimage/gradient/optimizer replay or convergence/effectiveness evidence.
No sampling/evaluation/parameter/rule change,GPUallocation or cleanup.
Reportserver28_partitioned_a_prefix0800_20261102.json SHA
07b6f720d0a25176619c72a03d5e09fe7d07a16667816760fc2c71dc134d577c.
~3.164/3.152s/update=>105min/epoch;revisedfirstepoch~18:05CSTNov2,
~25-36h for14-20epochs,official reporting afterward. Originaltrainingand
queuecontinue;no adverse evidence requiringintervention. No official result
or module validated;full goal incomplete. Inspectnearfirstepoch,notbefore
unnecessarily;180-300s live observations near milestone.

## 2026-11-02 - Full official continuation queue registered

Previous turn launched paired training=progress. Revalidated originalPIDs
1288280/1288281 live16:29:39CST,epoch1/update>=175,76/74C,250W. No full
epoch/new official score. No training/runtime/protocol changes.
New queue compiled and checked20full plans(no subset),strict native tracker
loading of both smoke weights,synthetic11/01/10images finite,parameter
digests unchanged/gradients absent. GPU0released;zero benchmark jobs run.
Checks reportserver28_partitioned_a_20261102_queue_checks.json.

Detached controllerPID1318842 registered08:36:53UTCNov2,verified actual
commandlive00:27/statusWAITING_FOR_TRAINING,CUDAempty/noGPUallocation.
300s process-identity polls,waitbothterminal,formalCPUverifier mustPASS,
thenUtilityGPU2/RandomGPU3 each5weights xcomplete/miss,4workers/GPU.
Eachrun479seq337353frames,isolated verify_official recomputesallreceipts,
box/GT/mask/sourcehashes andtoolkitcurves. No verify_development/main,
val97,subset,testselection,cleanup ortrainingrestart. Preserveallweights.
Twentyfullruns=9580trajectories/6747060frames. No future result claimed.
Remote reports/server28_partitioned_a_20261102_official_queue{.launch.json,.log,/}.
Queue source/protocol/inventory hashes frozen;do noteditthese pending files.
Docs PARTITIONED_A_OFFICIAL_QUEUE_20261102.md and PARTITIONED_A_QUEUE_EVIDENCE_20261102.md.
Nextfirstepoch~18:00CSTNov2,inspectnearboundary then180-300s livechecks.
Full goal/three modules incomplete;official scores unchanged.

## 2026-11-02 - Train882-only A replication launched

Previous table-only turn=no progress. Revalidated server28 idle and7340a53
clean. Added separate full-network A trainer;705/88/89train882 partitions,
prior+smoke frame exclusions,actual microbatch IDs/tensor/mask hashes.
No val97/official access,runtime/environment changes or automatic evaluator.
Both smoke arms6epochs/12updates verified,pairedinputs/masks/initial exact,
finite final5+selection/source/data checked,actual backbone/head/utility
updates,39BNbuffers frozen. Saved-evidence checks,not image/optimizer replay.

FormalGPU2UtilityPID1288280/GPU3RandomPID1288281 launched08:19:00UTC,
seed2028,batch5x2/LR.000125,20000tuples/epoch,max20/drop10/earlymin14.
Both original PIDs live16:21:22CST epoch1/update>=25;first50microsteps
input/mask pairs exact,74/73C,22710/22742MiB,250W,100%utilization.
Boundary partitions/879excludedsets/priorhashes structurally equal;dict key
order alone differs in bytes. All weights retained,no test-based cleanup.
~3.01s/update=>101min/epoch,24-34h;firstepoch near18:00CSTNov2.
Inspect near milestone,180-300s live checks,no restart on observation timeout.
After terminal receipts run scripts/verify_partitioned_a.py --phase formal
before final-five full complete/Miss reports. Development-only weight
selection;official final5 comparison is post-hoc,never cleanup criterion.
Remote reports/runs stemserver28_partitioned_a_20261102_formal.
Future freshsamplers excludeALLactive input_manifest.jsonl frames.
Protocol/results docs/PARTITIONED_A_*20261102.md. Official metrics unchanged;
full goal/three modules unmet. Implementation/evidence published as3cd8f23
on codex/server28-research.

## 2026-11-02 - Template attention information diagnostic verified/closed

Previous last-observedtemplate verifiednegative=progress. New frozenE0
GPU2PID1070949completed51.497s,128freshfittingtuples/117seq/1152state rows.
Excluded all prior tuple/burst/crop/image-training/last-observedclip frames
by samplingvisibilitymask only; images/physicalGT unchanged. Block6native
template0 foreground/background-query attention on same top8Hann geometry;
currentobservedstreams only,templateannotation known,currentGTscoringonly.
Native templates are contextualized,notindependentidentity verification.

FG-BG candidateconcordance complete+.015090CI[-.030979,.060233],
RGBmissing+.044031CI[.012752,.076987],TIRmissing-.018250CI[-.055731,.018017].
AllFG lowerCI>.5,butall3difference means<.05 and2CIcrosszero. Gatefalse.
FG*Hann fixedheuristic IoUdelta complete/RGBmissing/TIRmissing
-.018731/-.015895/-.007360,allCIcrosszero. Closeexactblock6signal;
no rankingtraining/layer/mask/weightsearch/controlpromotion/module claim.
Officialmetrics unchanged;overallgoals/three modules remainunmet.

Server/localv2 analysis verified all9216candidateboxes,1152maprows,
paircounts/bootstrap/21priorhashes/exclusions/source/base/cache/physical
visibility. IndependentNumPy9rawqkv cases maxattentionerror1.4117185e-9.
Firstbatch normalizedimages144nativeforward rows mapsbitwiseexact,
stateunchanged/gradientsabsent;9hook/nohookparitycalls. Sharednativenetwork,
notrawimagedecoder/fullinputreplay/externalreview. Firstverifierwrongly
required identicalhashdictkeys;independentonehas11additionalolderreports.
Allproducerhashesmatched. Correctedsubsetrequirement+recordall21hashes;
initialfailurelogkept,producer/gate/dataunchanged.

ReportSHA a7e8afa27171abd35ea53d963ae11d71f25c2f7b8145d4b2cef0528d81bb18eb.
WorkerSHA22cb23b74a388da0496860bd01c303a749a50ae4c9a146f545c4e9b2fc4de05b.
Docs:TEMPLATE_ATTENTION_EVIDENCE_20261102.md;reports/server28_template_attention_20261102.
Image/mapcachesserveronly,nooptimizer/weights/cleanup/val97/official.
GPUsidle38/37/42/37C,all250W. NextfreshsamplersmustexcludeALLnew
input_manifest template/searchframeids,notjustfirstsearch.

## 2026-11-02 - Last-observed template diagnostic verified and closed

PID1011087 completed351.701s,32fresh128frameclips/1088rows,377native
writes/165incomplete. Retained-native RGBmissing IoU+.008188
CI[-.003180,.021454],TIRmissing+.005576CI[.001102,.012710],
pooledmissing+.006703CI[.000943,.013224],complete-current+.001598
CI[.000318,.003017]. Retained-complete-only pooledmissing-.000440
CI[-.001530,.000563]. All32seq contribute. Promotion=false: insufficient
directional gains and no demonstrated matched-control superiority.
Close exactpolicy/control; do not promote complete-only control separately,
retune audit,expandtraining or claim a module. Smallpositive effects are
same-native-history one-step benefits,notclosedloop/official gains.

Server/local box/provenance/exclusion/bootstrap/source/checkpoint verified.
96native shams,731complete-history exactparityrows;firstclip native-image
replay34rows/18writes/templatehashes exact. Replay shares policyworker and
nativetracker,notfullimageaudit/independentneuralimplementation/externalreview.
ReportSHA b0e0cbb393f4e449dde5df845aa104898b7a5cf55cf8cd03a6a086d7bffb7963.
Nooptimizer/weights/cleanup/val97/officialcalls;GPUsidle38/37/45/37C,all250W.
Protocol/results docs/LAST_OBSERVED_TEMPLATE_DIAGNOSTIC_20261102.md.
MPTrack70ae6b7asynchronoustemplatepriorart documented,no sourcecopied.
Official best scores unchanged;overall objective/three modules unmet.
Future fresh-frame samplers must additionally exclude all32clips in
reports/server28_last_observed_template_20261102/selected_clips.json;
the recorded140eligible128frame count was BEFORE this new diagnostic.

## 2026-11-02 06:44 UTC - Last-observed template diagnostic running

Previous crop pilot verified/closed was progress,not goal completion.
GPU2PID1011087 launched06:44:26Z,live02:39/13of32clips,442rows/162writes,
2208MiB/67C/250W. FrozenE0,32fresh128frame fitting705clips,seed59101.
Excluded all prior tuple/burst/crop and image-training fit/dev/amendedaudit
frames; metadata preflight140eligible128frame fittingseq,175at96.
Native alone commits trajectories. At every4thcall+first3, shadow last-
observed per-sensor dynamic template vs complete-only-write control,
same nativecrop/writeevents,2slots andnooptimization. Native snapshots
restored;GTonlyinit/scoring. Both valid banks carryrealpastobservations.

Protocol docs/LAST_OBSERVED_TEMPLATE_DIAGNOSTIC_20261102.md snapshotSHA
6314346f0b969799e3a0ed834c2ff483f37eb9b55ade8756504e1b3d38eb0f89.
WorkerSHAf98e64eefea7d54451c90516ccfa493358f3e6d259a2885d55bcaccca74ca569.
Source frozen. Analyzer implemented/syntaxchecked, not resultverified.
Expectedremaining~4min at13clips;next180-300sobservation thenanalyze
--server-files --replay-first-clip ifcompleted. Reportroot
reports/server28_last_observed_template_20261102. Gate only authorizes
freshclosedloopdiagnostic,nottraining/official/module claim.

Read-only MPTrack70ae6b7 alreadyusesasynchronoussensortemplates;MITlicense,
sourceblobd42fb6b1602173461754af5b3ad517c0251955f6. No codecopied/installed.
Windowslongpath checkoutfailure forunrelatedYAML preserved;Gitobjects and
inspectedtracker/LICENSE readable. Priorartdoc records exactscope.
No val97/official/cleanup;officialmetrics unchanged;overallgoal unmet.

## 2026-11-02 - Full-image crop pilot closed after verified amended audit

Audit-only PID949348 completed346.833s; server/local analysis verified.
Original16x96 audit infeasible/incomplete; separately registered24x64
audit completed without retraining,all24sequences/9072totalrows.
Causal-E0 RGBmissing IoU+.005559 CI[-.008252,.020946],TIRmissing
-.003614CI[-.033199,.024765],complete-.002447CI[-.019433,.018133].
Causal-native pooledmissing-.002265CI[-.016638,.009103].
Amended promotion=false. Close exact pilot; no enlargement/retuning or
module claim. The oracle benefit did not transfer for this treatment;
not a universal rejection of trajectory learning.

Original native12/108selected11(final8-12),causal9/81selected6(final5-9).
All10final-five fullweights finite,BNunchanged,selected models each changed
151backbone+58head/fusiontensors. Actualmetadata seeded24clip schedules,
GT/visibility/cache/source/base/weightshashes and box/bootstrap arithmetic
checked. Separate-process firstauditclip378nativeimage predictions exact;
sharedtracker/crop,notfullaudit independent replay/externalreview.
ReportSHA9502da9b3fcbaf14caf5c604fe2a29a7b1d308afa8563f1bb1fac32728147a02.
Allweights/failures retained,GPUsidle38/38/46/37C,all250W.
No val97/officialruns;officialmetrics unchanged,fullgoalunmet.

Read-only pinned DenseCL/DINOv2 cross-crop source review documented in
docs/CAUSAL_CROP_PRIOR_ART_20261102.md. No code copied/installed, no new
module proposed as validated. Future audit feasibility must be checked
from metadata before fitting. Do not repeat this pilot on its audit cohort.

## 2026-11-02 06:22 UTC - Audit-only resume after metadata failure

Original PID877938 is terminal: both training arms completed and selected,
then fresh_clips asserted before any audit images/predictions were read.
Native12epochs/108updates selected11 from8-12; causal9epochs/81updates
selected6 from5-9 after patience3. Keep all checkpoints; do not retrain.
Metadata inventory under unchanged exclusions finds12 eligible96-frame
audit sequences,24 at64frames,29 at48. Original16x96 protocol infeasible,
not passed. Inventory initial NumPy JSON error and partial file retained;
corrected audit_inventory_v2.json saved, metadata only.

Prospective amendment docs/CAUSAL_CROP_AUDIT_AMENDMENT_20261102.md fixes
24x64 audit, same89names/exclusions/seed58103, same thresholds requiring24.
PID949348 onGPU2 launched06:22:25Z, live00:28,2206MiB/53C/250W.
No fitting/source/checkpoint changes. Amendment snapshot SHA
03f59489d10adc5dda14edcaf969fd2d8639a6f63583ed864fd0df05e6a71ac7.
Audit worker SHAa8e08824e3e70a8a76465d3400982b888fbb638e420490793f3d9602bd353b56.
Expected3-6min; observe180-300s then verify terminal result using analyzer
--server-files --replay-first-clip. No amended outcome yet. Original96-frame
success cannot be claimed; shorter audit is internal IoU, not official.
Official metrics unchanged; overall goal and three modules remain unmet.

## 2026-11-02 Matched full-image crop training launched

Liveverifiedworker19:20,nativecompleted12/108updates,selected11from8-12;
causalepoch7/63updates,stillrunning,noauditresult. Max12notconvergence.
NewGPU3firstfittingclip rawimage reconstruction passed:10acceptedpairs
bitwise,1rejectionreproduced,prediction/jitterboxes/templatehistoryexact,
independentNumPylabelmaxerror1.7149784e-7. Sharedtracker/crop/preprocessor,
notfullcache/optimizerreplay. Receiptinput_replay_first_clip.json,
verifier verify_crop_training_inputs.py SHA
080541c736588da0df8f3bce9a830e73fe8a9f8fbab16f3b1bdafc47ceaa18c3.
All32fittingvideoscontribute305pairs125/87/93states11/01/10;
47rejections46nativejitterinvalid/1predictioninvalid.
Nativepeakallocated22169819648/reserved22636658688bytes.
Independentanalyzernowalsoverifiesactualselectedbackbone/headweight
changes,allfinal5finiteweightsandBNbuffers. Notyet resultverified.

Previous causal-crop diagnostic was verified progress. NewGPU2PID877938
launched05:50:24Z,confirmedlive05:12/first nativeepoch1 completed62.687s.
32fresh96frame fittingclips,305accepted of352candidates,pairednative
GTjitter/predictioncrop images andsamefrozenE0 templatehistory.
Fulloriginal103Mparameter network retrained fromimage tensors,notfeature
caches. Same completebranch;auxiliarycrop differs. Both noKD/patchdrop.
Batch32/LR4e-5,backbone.1,9updates/epoch,17droppedremainder;observed
23350MiB/72C/250W. Fullimagestepbatch4 profilepeak3514857472bytes.
Native/casualarms sequential,max12/min8,LRdrop5,patience3 on8development
48frame nativecomplete/mixed trajectories;final5 selectiononlydevelopment.
Delayed16audit96frame clips,E0/native/causal nativeclosedloop,train882
705/88/89partitions;no val97/test/official masks. No officialscores yet.
Protocol docs/CAUSAL_CROP_TRAINING_20261102.md;output
reports/server28_causal_crop_training_20261102/. Analyzerdeployed,
syntaxchecked,notyetresultverified. ETA20-30min totalfromactualepoch.
Keep original/allfinal5weights. Ordinaryofflineexposurecontrol,
notonlinepolicyaggregation orvalidatedmodule. Fullgoal remains unmet.

## 2026-11-02 Causal crop exposure diagnostic verified

COMPLETED GPU2PID804850 in594.422s,48unique96frame fittingclips,
9120pairedrows. PreviousGTcrop-native IoU mixedmissing+.042196
CI[.022036,.066595],RGBmissing+.047694CI[.021744,.081731],
TIRmissing+.038413CI[.019240,.061804],complete+.032329
CI[.010975,.059321]. Advance=true only to separately registered matched
image-training experiment;oracle unavailable at inference,not a module.
Complete effect means not uniquely missing-modality evidence.
Posthoc fullycontained2696/2731missingframes delta+.039820
CI[.021298,.062868];shortclip phenomenon not just nominaltargetescape.
Server/local savedboxmath/nativehistory/exclusion/partition/schedule/
sham/bootstrap/hash checks passed;actualGT/visibility/source/base checked.
Separate firstclip190row native replay exact,direct shadow inversecrop
decode maxboxerror0;sharednetwork/crop,notfullreport imagereplay.
State/absent.grad assertions from producer. Temporary localrunner edit
reverted;finalsource/server/launch all ae3e517c90085b5e371f7dd99ec16041f0cb07ecd423dd7ffc862d7f7f7b7c53.
ReportSHAef0e7aa20a83097e265e2603d3b4ffdb328bc4c0c6aa30f6fe1e23e458be6a2c.
Nooptimizer/val97/official/cleanup;weightskept,officialscoresunchanged.
Next: matched fullimage training nativecrop versus causalpredictioncrop
with fixed capacity,completeprotection,train-onlydevelopment/audit and
measuredbatch/LR/finalfive. Notyet registered/launched. Fullgoalunmet.

GPU2PID804850 launched05:22:22Z; confirmedlive03:19 at16/48clips.
48fresh96frame train882 fitting clips, complete andIID mixed native
trajectories. Each shadow call changes only previous prediction crop to
previousGT; native template histories retained, shadow updates discarded.
First3frame native shams and firstframe initialization parity asserted.
OriginalE0 andavailabilitywrapper weight digests equal. Nooptimizer,
val97,official data or cleanup. Batch1 causal inference;~2206MiB/60C/250W.
RevisedETA~10min total;observe180-300s near completion.
Protocol docs/CAUSAL_CROP_EXPOSURE_20261102.md; output
reports/server28_causal_crop_exposure_20261102/.
Advance criterion: mixedmissing sequence-macro oracle IoU delta>=.02,
lower95CI>0,all48sequences. Pass only supports separately registered
matched image-training experiment, not deployment/module/official gains.
Launch history above superseded by completed verification.

## 2026-11-02 Full-parameter joint-task gradients verified

GPU2PID768965 completed37.670s,64fresh train882 fittingtuples,
16batches4,3searches,trainDropPath/allBN frozen,native availability+
patchmasks. Complete versus missing+.1KD raw parameter cosine:
mean .201456,batchbootstrap95[.160851,.248108],negative0/16.
Registered gatefalse;do not launch gradient projection/countmodule.
Does not rule out later conflicts,withinbatch cancellation or AdamW effects.
Server/local saved Gram additivity/cosines/finite values/partition/
exacttupleexclusion/masks/bootstrap/source/base/protocol checks passed.
No raw gradients/images saved,no neural replay;state/absent.grad checks
are producer assertions. No optimizer,val97,official evaluation or cleanup.
ReportSHAb0bf4d5fdd0df3e6d98b9e7b6a5deceb40b9a7ff82c6463fb0efc9d5242b7e41.
Protocol docs/JOINT_GRADIENT_DIAGNOSTIC_20261102.md,evidence
reports/server28_joint_gradients_20261102/. Official scores unchanged.
Complete+2,RGBT234-Miss and three validated modules remain unmet.

## 2026-11-02 Complete-teacher gradient diagnostic verified

FrozenE0GPU2PID728997 completed55.655s,128fresh train882 fitting tuples/
1152rows. Missing-search unreliable177rows/61seq,reliable591/114seq.
Unreliable fused-gradient cosine .004752CI[-.058037,.066890];
response-projection parameter cosine .492476CI[.372922,.600634].
Reliable fused .447424CI[.414153,.480243];paired57seq reliable-unreliable
gap .432528CI[.350958,.512308]. Registered diagnostic=false: no evidence
of negative unreliable mean at both required interfaces. Do not launch
reliability-filtered KD training or retune thresholds on these samples.
Higher teacher task loss alone does not establish harmful gradients.
Full-backbone parameter/optimizer effects not measured.
Server+local partitions/tuple exclusions/vector+dotnormcosines/bootstrap/
decision/protocol verified;source/base/cache hashes checked. Native-head
separate-process1152row gradient replay exact,shared measurement function,
not independent image/backbone replay. Producer parity/state/gradient-sum
assertions re-exercised by replay. No optimizer/val97/official/cleanup.
ReportSHAf46bbd0342729be6d3d8b15523908fcf7a6e3f26410b0352694b4614098382d6.
Protocol docs/TEACHER_RELIABILITY_DIAGNOSTIC_20261102.md;
evidence reports/server28_teacher_reliability_20261102/.
GPUs idle38/37/42/37C,all250W. Official maxima unchanged;fullgoalunmet.

## 2026-11-02 Matched raw/Hann supervision verified

Corrected679342 completed185.454s,batch256/LR0.00032,9updates/epoch.
Both10epochs/90updates,selected6 fromfinal5=6-10 before audit.
Audit128tuples/66seq:Hann-E0 RGBmissing+.037455CI[.020163,.056675],
TIRmissing+.043352CI[.018517,.072476];Hann-rawtraining+.000731
CI[-.006815,.009476]. Complete/recovery11_11/01_11/10_11 protection
failed;last has point-.006195. Registered advance=false despite E0 gains.
Do not expand/retune/promote Hann or raw control/countmodule.
This does not rule out all full-model alignment methods.
Server+local partition/tuple/boxmath/stop/select/bootstrap/protocol passed;
base/source/threecache/all10finite6474117paramweights/BN buffers checked.
Separate-process native heads3456rows exact,independent Hann decode;
sharedhead/taskloss,notindependentimagereplay. Producerstatechecks passed.
ReportSHA95200226fd20c340cf1e124fd3d704a9bed2ff9f52bfeee8f2cf287869c00612.
Evidence reports/server28_window_training_20261102_v2/.
All weights retained;GPUsidle38/38/51/37C,all250W. No val97/official/
cleanup. Official maxima unchanged;full goal remains incomplete.

### Reporting failure and corrected launch

Initial654595 terminated at source-manifest generation: helper was imported
from runtime/scripts,not root/scripts. Training/selection/audit caches
exist,but no report/completed receipt. Original log/source/weights preserved.
Worker now hashes imported module.__file__,same settings/seeds/gate,
exclusive_v2 output. Corrected GPU2PID679342 launched04:40:09Z,
confirmedlive00:37/fittingcache112of256. Follow this PID/output:
reports/server28_window_training_20261102_v2/. Do not restart on timeout.
Analyzer now also recomputes exploratory source-record window effects
and records its own hash. No original audit scores used to change settings.

### Initial launch

GPU2 PID654595 launched04:29:02Z. New256/128/128 train882 tuples,
same6474117parameter native CENTER heads compare raw-peak versus Hann-peak
regression; both evaluate using Hann,raw focal maps unchanged.
Backbone/fusion/BN frozen. Same feature caches,LR scaling,stop/final-five
selection; audit sampled only after selection. Output
reports/server28_window_training_20261102/. Protocol
docs/WINDOW_TRAINING_PILOT_20261102.md; dedicated analyzer deployed.
Estimate3-8min,180-300s milestone observations. Worker frozen after launch.
Prior candidate diagnostic's Hann-minus-raw crop IoU was significantly
negative in both missing directions,not complete; this is exploratory
evidence of a train/inference contract difference,not a deployable benefit.
No Hann removal,threshold tuning,val97 or official evaluation authorized
by this pilot. All weights preserved. Official maxima unchanged;goalunmet.

## 2026-11-02 Matched chronology training verified

PID589499 completed704.600s; both10epochs/120updates,final5=6-10;
development chose native6/chronological9. Audit128tuples/69sequences:
chronological-E0 complete+.003912CI[-.013950,.029749],
RGBmissing+.010976CI[-.008068,.029101],
TIRmissing+.009991CI[-.002026,.023430];
chronological-native-.003260CI[-.007479,.000038]. Advance=false.
Do not expand/retune audit/promote this exact intervention as a module.
This does not disprove all chronological or full-backbone training.
Server+local tuple/partition/inputpairing/boxmath/stop/select/bootstrap/
protocol verified. Base/source/fourcache/all10final5 finite30720parameter
weight hashes checked. Separate-process E0tail1152+selected2304rows exact,
sharedtail/measurementcode,notindependentimagereplay.
Producer baseline-state/gradient assertions passed. All weights retained.
ReportSHA645a7665abf1479a9620dd04fd9ffef6f70b498e6274802d360a48512999a0ce.
GPUs idle38/38/44/37C,all250W. No val97/official/cleanup; goal unmet.

### Original launch observation

GPU2 PID589499 launched2026-11-02T04:02:16Z, fresh256/128/128 tuples,
same original-order image/augmentation hashes checked across fitting arms.
Original E0 frozen; matched30720parameter attention adapters compare
newercomplete/olderdynamic against oldercomplete/newerdynamic.
Both evaluated on chronological train882 development/audit; delayed audit,
same memory-sized batch/scaled LR/stopping/final-five selection.
Output reports/server28_chronology_training_20261102/.
Protocol docs/CHRONOLOGY_TRAINING_PILOT_20261102.md; analyzer
scripts/analyze_chronology_training.py deployed and syntax checked.
Worker source frozen after launch. Estimate10-20min,180-300s observation
milestones. Do not restart on observation timeout; inspect actual PID/log.
No val97 or official evaluation. Official maxima unchanged; goal unmet.
First verified live observation elapsed5:09: nativeepoch3/update36,
batch192/LR0.0006,12updates/epoch,~20s/epoch,GPU22156MiB/73C/250W.
Exact fitting-input pairing passed; estimated6-9min remaining.

## 2026-11-02 Template chronology diagnostic verified

GPU2PID538386 completed79.312s,128tuples/118fittingsequences/1152rows.
Oldercomplete/newermasked minus newercomplete/oldermasked IoU:
search11-.005529CI[-.020217,.008644],RGBmissing-.018251
CI[-.035117,-.001065],TIRmissing-.015967CI[-.031889,-.000592].
Registered either-sign mechanism criterion supported. This is an observed
data-contract effect, not an improved policy or proven training cause.
Image+mask permutation sham mean-.000000234,allrowIoUdeltas<=1e-4,
maxboxerror.0000352263,maxmaperror.0141708255;notbitwiseinvariance.
Server+local temporal order/age gaps,exact tuple exclusions,3456boxIoUs,
bootstrap/source/base/protocol/report hashes passed. No neural replay,
image tensors/maps not saved. Baseline state/gradient checks are producer
assertions. ReportSHA20cb927876ec418518b6d196e859f6e861b756c26284ce994817ae9fbb94ae79.
Protocol/results docs/TEMPLATE_CHRONOLOGY_DIAGNOSTIC_20261102.md,
reports/server28_template_chronology_20261102/.
Next: register matched age-mask training with fresh train882 tuples,
same capacity/images/masks/updates, chronological evaluation for botharms,
complete and missing endpoints, then fresh closed-loop gate. Not launched.
No val97,official evaluation,optimizer update or cleanup. Overallgoalunmet.

### Original launch observation

Previous quality ranker failure was verified evidence, not a validated
module. Existing independent-anchor experiments already failed to show
stable identity acceptance, so do not repeat plain cosine/ranker pilots.
New code-level question: sampler emits [newer,older], first template
always complete; tracker retains [initial,recent], with recent availability.
Missingness is therefore bound to different template ages. Templates share
positional embeddings; image order alone is not a demonstrated cause.
Frozen E0 GPU2 PID538386 launched2026-11-02T03:45:39Z:
/home/gaob/Miss-RGBT/reports/server28_template_chronology_20261102.
Native [newer complete,older masked], chronological [older complete,newer
masked], and matched permutation sham[older masked,newer complete].
Fresh128tuples from705fitting names,seed52101,all9states; nooptimizer.
Last live elapsed1:13,112/128samples. ETA1-3min,check180s.
Protocol docs/TEMPLATE_CHRONOLOGY_DIAGNOSTIC_20261102.md,worker frozen.
SHAfa2a7433bf80e2d9685bb0cbefc3f92d1675f612803602085de38ea5072a9cc1.
ProtocolSHA7dd785e9b01306de603f7acafb642b3873e64be8179f3c1ce0f712638f49caa0.
On completion scripts/analyze_template_chronology.py --server-files.
Either-sign material effect is diagnosis only, not a better inference
policy; no automatic training/official evaluation. Overall goal incomplete.

## 2026-11-02 Matched quality ranker completed and verified

PID499198 finished231.174s. Both51585parameter arms10epochs/180updates;
qualityselected9,responseselected10 from final-five6-10, all weights kept.
Audit128tuples/66sequences/1152rows per arm. Quality-native IoU complete
+.004284CI[-.033706,.046220],RGBmissing+.022927CI[-.013764,.064605],
TIRmissing-.004925CI[-.039010,.028868];quality-response+.011954
CI[-.014616,.043682]. Registered gate=false; do not enlarge/retune/promote.
Oracle candidate headroom does not establish learned selection efficacy.
Server+local box arithmetic, pairing, partitions/tuple exclusions,
LR/stop/select/bootstrap/protocol checks passed. Source/base/3caches/
selected/all10final5 hashes verified; all10actual weights51585finiteparams.
Separate-process selected-ranker replay2304rows exact;9216candidate IoUs
independently checked. Shared ranker/evaluation code,cached features,
not neural image/backbone replay or independent scientific review.
ReportSHA686d472292a7eb8a72e3a7ee168818b880cb601a0a353824f129cb73d5628bb7.
docs/QUALITY_RANKER_PILOT_20261102.md,
reports/server28_quality_ranker_20261102/.
No val97/official evaluation/cleanup. Overall goal remains incomplete.
Next mechanism requires distinct fresh train882 evidence; candidate
identity discrimination from initial/dynamic templates is a prospective
question only, not validated by these negative results.

## 2026-11-02 Candidate quality headroom verified; original pilot launch

Frozen E0 GPU2 diagnostic PID460256 completed44.809s,128tuples/
121fitting sequences/1152rows. Native top8 Hann GT-oracle gain:
complete+.053037CI[.032385,.078354],RGBmissing+.078130
CI[.056767,.103489],TIRmissing+.079255CI[.052755,.109509].
Registered headroom gate passed. Not deployable/learnable/official metric.
Source/protocol/base/cache hashes, exact tuple exclusions, box arithmetic,
bootstrap/gate and actual fitting names verified server+local; independent
NumPy map replay1152rows/294912boxes passed. No neural image replay.
ReportSHAe774a5ff4b4cdacffae53c286c25bed0116d0feb6175a777f66a3061d4a35cda.
FirstPID425514 stable argsort API failure retained; corrected via NumPy
mergesort with same tie rule, no gate change. Evidence/protocol:
docs/CANDIDATE_QUALITY_DIAGNOSTIC_20261102.md,
reports/server28_candidate_quality_20261102_v2/.

Follow-up PID499198 on GPU2 launched2026-11-02T03:31:17Z:
/home/gaob/Miss-RGBT/reports/server28_quality_ranker_20261102.
Matched51585parameter quality/response-supervised rankers, same initial
state, fixed native top8 geometry+Hann prior. Fresh512/128/128 train882
tuples, existing705/88/89 names, delayed audit after selection.
Batch256/LR8e-4,18updates/epoch,min10/max20,drop7,patience3; retain final5.
Last live elapsed1:57 fitting384/512. Estimate3-8min,180-300s checks.
Protocol docs/QUALITY_RANKER_PILOT_20261102.md; worker source frozen.
On complete run scripts/analyze_quality_ranker.py --server-files --replay.
Pass only authorizes fresh closed-loop train882, not official/paper claim.
No val97, official evaluation or cleanup. Overall goal remains incomplete.

## 2026-11-02 Final-two-block low-rank pilot verified

PID368156 finished584.557s. Both arms10epochs/120updates; final-five6-10
retained, attention selected8,MLP7 before audit. Batch192/LR6e-4,
12updates/epoch; attention peak allocated17.39GB,reserved21.45GB,
GPU telemetry22156MiB/73C/250W. All epoch weights retained.
Audit128tuples/68sequences,1152rows per arm. Attention-E0 RGB-missing
+0.031107 CI[+.012872,+.050894]; TIR-missing+.008250
CI[-.003422,+.020444]; complete-search over template histories+.005012
CI[-.002023,+.014890]. MLP effects all cross zero. Neither passes
registered all-three-state gate. Do not enlarge/tune this exact intervention
or count it as a validated module. Attention-minus-MLP also crosses zero.
Server actual partitions/tuple exclusions, memory/LR/stopping/selection,
box arithmetic/bootstrap/source/base/cache/final-five hashes verified;
all10 final-five weight dictionaries finite and30720parameters.
Separate-process E0 cached-tail replay1152rows and selected-adapter2304rows
exact. Shared tail/measurement implementation, not independent image replay.
Local evidence analysis passed. Report SHA256
ca7092a9dc1f1ef51513f7245fb4582f7311b5deb84d93a4683a98b41ae730e2.
Full results docs/TAIL_LOWRANK_PILOT_20261102.md and
reports/server28_tail_lowrank_20261102/. Official maxima unchanged.
Read-only TOOD source review pinned1bda27c, no copied code or new training:
docs/QUALITY_ALIGNMENT_SOURCE_REVIEW_20261102.md. Next: preregister fresh
train882 fixed-candidate quality/ranking diagnostic before another learned
intervention. No val97, official evaluation or weight cleanup.

### Original launch observation (superseded by completion above)

Previous cross-stream result was verified negative evidence, not a module.
New registered question adapts native blocks10/11 rather than frozen
fusion outputs: attention-qkv rank5 vs MLP-fc2 rank4,30720parameters each.
Includes all9 template/search states and can alter complete-input predictions.
Ordinary low-rank adaptation; no novelty claim. Official microsoft/LoRA
source pinned/reviewed; no copied file or environment change.
GPU2 PID368156 launched2026-11-02T02:41:45Z, output
/home/gaob/Miss-RGBT/reports/server28_tail_lowrank_20261102.
Last live observation at elapsed1:55: train cache256/256,
development32/128; GPU2 2828MiB/66C/249.04W/100%,250W cap.
Estimate10-30min;180-300s milestone checks. Source stays frozen.
Protocol docs/TAIL_LOWRANK_PILOT_20261102.md, launch protocol SHA
6a08f79a9748a48306490cc0d80693b905fb8880845adc2ed698607874891328.
Worker SHA1ba7393ad0268e19d49d9411ec8305cde0806b1d973544ff404ef9755048e298.
Existing705/88/89 train882 name partitions reused, fresh256/128/128 tuples;
audit sampled after both arms stop/select. Native tail replay must be exact.
Data-dependent memory sizing, LR linear in batch, max20/min10/patience3
after epoch7 decay, select min9-cell loss among last5. Preserve weights.
Each arm gate: sequence-macro crop-IoU >=.01, lower95%>0 in all3
search states. Pass authorizes only fresh train882 closed-loop diagnostic.
On completion run scripts/analyze_tail_lowrank.py with --server-files
--replay, collect small receipts/reports, update desktop handoff and GitHub.
No val97 or official evaluation; full research goal remains incomplete.

## 2026-11-02 Cross-stream context pilot completed

GPU2 corrected PID318603 completed246.879s. Matched aligned/shuffled
199104-parameter adapters both stopped10epochs/480updates and selected6
from final-five6-10 before audit sampling. All weights/caches retained.
Audit128tuples/69sequences,1152rows per arm. Aligned-E0 crop IoU:
RGB-missing+.013811 CI[-.007264,.037425], TIR-missing+.008506
CI[-.017583,.031635]; aligned-shuffled+.001734 CI[-.017697,.019887].
Registered advancement=false. Do not expand/tune this pilot or run official
evaluation for it. Complete-search parity and producer frozen-state checks
passed. Separate saved-evidence arithmetic/source/protocol/file hashes
verified; separate-process selected-adapter replay2304records exact, using
the same adapter/measurement implementation, not independent image replay.
Local evidence analysis also passed. Report SHA256
21034836239b90595e1ed72addd176679dac7247013ad371ed79ea529d1fc2f8.
Protocol/results: docs/CROSS_STREAM_CONTEXT_PILOT_20261102.md;
reports/server28_cross_stream_context_20261102_v2/.
Initial pre-update layout-parity failure and its cache/log remain on server.
No val97, official metrics or cleanup. Full research goal remains active.
Next action must address a distinct evidenced mechanism on train882, not
retune this audit or count ordinary cross-attention as a validated module.

## 2026-11-02 Late-fusion stream diagnostic

Frozen E0 train882 fusion-input ablation completed on GPU2 in79.108s:
128tuples/117sequences/2304paired records. Zero absent-stream features only
before decode_fuse_search; matched control zeros available-stream features.
Absent-stream removal IoU delta RGB-missing -0.008314 CI[-0.029745,+0.014144],
TIR-missing +0.010427 CI[-0.002732,+0.024063]. Both intervals cross zero;
registered advancement=false. Do not train/tune/promote this ablation.
Complete producer parity/state checks passed; independent saved-evidence,
prior-tuple exclusions, bootstrap and source/protocol hashes passed on
server and locally. No independent neural replay; no image tensor hashes
or full output maps persisted. See docs/LATE_FUSION_STREAM_DIAGNOSTIC_20261102.md.
No val97, official evaluation, optimizer or weight cleanup. Official maxima
unchanged; remaining full goal is incomplete. Next: examine a matched
train-only cross-stream interaction, retaining spatial/template structure
and comparing a spatially shuffled context control before any long training.

## 2026-11-02 位置编码诊断更新

最新注册的 train882 冻结 E0 诊断已完成：移除缺失搜索流的位置编码后，RGB 缺失 crop IoU 变化为 -0.053025（95% 序列 bootstrap CI [-0.080336, -0.024700]）；TIR 缺失变化为 -0.006421（CI [-0.015715, +0.002530]）。晋级门槛未通过，不启动该方向训练，不用该样本调参，也不运行官方评估。Complete 输出一致、模型状态未变、无梯度；独立分析器及服务器源码哈希核验通过。

前两次运行在模型推理前因目录/schema 错误退出；第三次计算结果因 TIR 缺失组 treatment/sham 路由错误而作废；第四次正确按 availability 路由并完成。失败运行均保留，不纳入结论。完整协议及报告 SHA256 见[位置编码诊断记录](docs/ABSENT_POSITION_DIAGNOSTIC_20261102.md)。

官方完整指标没有变化：LasHeR-Miss 已超过 PR/SR 目标；RGBT234-Miss MPR/MSR 仍为 86.184864/61.660594，距离 88.1/64.3 目标 1.915136/2.639406 点；Complete 相对 E0 的约 +2 点目标未达到。此次不使用 val97，也未训练或清理 checkpoint。桌面交接补充见 `C:\Users\gb\Desktop\document\Miss-RGBT-absent-position-diagnostic-20261102.md`。

## 2026-10-30 分支保护控制实验

CPU8次真实更新完成，只训练现有LocalRepair，baseline参数与缓冲区哈希完全不变；
全11输入的score/size/offset图与预测框逐元素一致。8个训练/8个验证tuple；
验证01损失下降0.00284952、10上升0.00021687，无稳定收益或完整训练收敛证据。
不涵盖缺失动态模板历史，不作为新模块。详见docs/PROTECTED_ADAPTATION_AUDIT_20261030.md。
另记录普通RankNet排序基线的固定allRank源码与许可，未复制/安装/修改在跑训练。
16:15Z四路训练worker仍存活，候选1195/1224更新，首轮验证待完成。

## 2026-10-30 Global/coarse proposal audit completed

Next CPU audit4006613 is RUNNING: fixed native crop+5x5centers using
previous/initial box sizes=51crops, same46observations as global audit.
No current GT generates proposals or selects frames.4threads/chunk8.
Live17:13:15Z, first observation29.35s; estimate20-25min total, check near
17:35-17:38Z with180-300s spacing, not busy polling. Outputreports/
server28-native-grid-20261030 and siblinglog; docs/NATIVE_GRID_AUDIT_20261030.md.
All3924observedvalid fullcohort geometricunion restores909/1505 cropoverlaps,
744centers;previousscalealone888overlaps/728centers. Puregeometrynotquality;
initialscaleaddsonly21/16atdoubleproposalcount. Runningprotocolunchanged.
Neuraloutcomespending;4GPUtrainingjobsunchanged,250W allcards.

Follow-up CPU GT-crop diagnostic3991543 COMPLETE73.88s,45points135crops.
Native unsupported12: GTcenter+previous size reachesIoU>=.5 on6,
GTcenter+true size on10, GTcenter+coarse tile size on0. Existing global raw
oracle also0/12. Median target shortside searchpixels30.14/43.98/10.67.
Center/scale/context/Hann changes are confounded; no sole-cause or deployable
recovery claim. Next larger support must preserve resolution and separately
test acceptance. Raw reports/server28-oracle-crop-20261030 copied locally.
Draft GitHub PR1 created for codex/server28-research; main not merged.

Launch history: PID3991543 launched17:03Z; live17:04:10Z,
6/45points completed. Same fixed global-audit points; three GT-centered
crops retain previous size/use true GT size/use prior half-image tile size.
These use current GT and cannot be deployed or reported as recovery scores.
Also recomputes raw-response oracle from existing proposals. Output
reports/server28-oracle-crop-20261030; sibling .log; no GPU training changes.
Protocol docs/ORACLE_CROP_AUDIT_20261030.md. Wait for terminal receipts.

GitHub synchronization verified: branch codex/server28-research commit
f4b0f185888007dc76c108ad3d2e25677e28d44e. Working tree clean, remote ref
matches. model.py/objectives.py/task_utility.py SHA256 matches frozen server
runtime. This publishes existing running A implementation, not a new change
to active experiments. Default main was not merged.
At17:00:29Z fixed GPU0/1 utility7675/random8158 updates, epoch4, recorded
key losses finite; evaluation controller3463984 live and waiting.

Latest16:55:34Z: candidate utility/random epoch1 validation completed,
mean loss5.041177/4.980912, epoch01 weights exist, epoch2updates2062/2117.
One epoch is not convergence or PR/SR evidence. AllGPU250W temperatures
84/81/79/76C. Next epoch estimated18:46/18:33Z at current speed.
GitHub branch codex/server28-research carries server28 scripts/source/docs;
do not overwrite the frozen running runtime when syncing.

CPU fixed-proposal audit on46points/first8val sequences,11crops per point.
Whole/tile proposals restore GT crop support12/12 native failures but oracle
candidate IoU remains<.5 on all12. MeanIoU native/Hann-oracle/Hann-score/
raw-score=.565/.601/.547/.422; raw response has12 drops>.1. Coverage alone
does not recover identity/localization; no recovery module promoted.
Full details docs/GLOBAL_PROPOSAL_AUDIT_20261030.md; raw reports/
server28-global-proposals-20261030. Worker3946665 completed normally in
343.02s; all GPU training and frozen runtime unchanged.

## 2026-10-30 Selector objective control completed

Completed452.37s: regression stops45/select45, pairwise stops30/select26,
other4 final heads per objective cleaned with receipts. Heldout4tuples/state,
pairwise rawjointmean3.767578 vsregression3.774773/native3.769790.
Againstnativewins5/ties1/loses6;state10worse. Normalizedbenefitpairwise
0.0074%vsnative0.0681%;aggregation-sensitive,rankingaccuracy50.49%.
No promotion/fulltracker/benchmarkgain. Rawlabels/features/report/analysis/
selectedheads retained and copied locally; docs/SELECTOR_OBJECTIVE_CONTROL_20261030.md.
Latest16:31:14Z fixedutility6951/random7429, candidates1561/1601 firstepoch;
allrecordedkeylossesfinite,noepochvalidationyet. Allfourtrainingjobscontinue.

User explicitly authorizes single-GPU concurrent evaluation/data collection.
Candidate val97 and official evaluators already use4workers/GPU after the
assigned training releases memory; measure full-run occupancy/throughput.

CPU PID3918799 runs matched frozen-feature SmoothL1 utility regression and
ordinary pairwise ranking controls. Train16/development4 from train882;
heldout4 val97 sequences, one causal tuple/state,11/01/10,16regions.
Development chooses each objective's best final5, then heldout actual joint
task loss compares selected4/native/random/single-region-oracle4.
This is a small offline control, not a new module or benchmark improvement.
At16:27:56Z all48train and12development slates were labeled, heldout pending.
Output reports/server28-selector-objective-control-20261030; sibling .log.
Protocol docs/SELECTOR_OBJECTIVE_CONTROL_20261030.md.

At16:24:36Z GPU0/1 utility6786/random7254 of50000, finite losses;
GPU2/3 utility1409/random1444 of2000 first-epoch updates, finite losses,
22710MiB each. First epoch boundaries estimated16:50:21/16:48:05Z before
validation/save overhead. No epoch validation yet. Controllers and automatic
evaluation3463984 remain live. All four jobs continue; no frozen source edits.

## 2026-10-30 Wider-candidate audit completed

CPUfixed9crop audit completed35observations/315proposals in207.81s,
first2observed37stride perstate/sequence, initialtemplates only.
Centerbatch vsactualsingleCPU maxbboxdifference3.05e-5pixels.
MeanIoUcenter/oracle/Hann/raw/cosine/pilothead:
.708827/.723872/.722725/.688270/.575025/.605296.
Cosine/head harm7/6frames>0.1IoU; notpromoted. Only2sampledcenter
zerooverlap, neitherrestoredby9crops. Earlysamplinglimitsdriftclaims.
All3924GPUcohortobservedvalidframes geometric9cropunion restoresoverlap
150/1505unsupportedframes(9.97%),center116(7.71%).
Medianrequiredgridradius6=>169native-sizecrops, severelydominatedbylongseq.
No recoverymodule/closedloopgain established. See docs/WIDER_SEARCH_AUDIT_20261030.md.
16:05:12 UTC 核验：GPU0/1 固定训练更新6306/6745；GPU2/3 候选更新963/987，记录损失均有限。候选首轮边界预计16:48～16:51 UTC，尚无逐轮验证结果。自动评估控制器3463984仍在等待。

## 2026-10-30 Independent-anchor rejection evidence

IndependentCPU crop descriptors and cyclic wrong-anchor controls were tested.
Easywronganchors fail as identity evidence:92.3%lowIoU predictedcandidates
still favor ownanchor. Sameframe fixedoffset backgrounds beat even oracleGT
target crops often. Initial64stride aliases32frame outage schedule; explicit
37stride correction covers11/01/10 (57/22/23observedoutagerows), targettop1
84.2/68.2/73.9%. All raw experiments preserved; no thresholds selected.
FrozenMLPranker pilot trained24trainseq/dev8trainseq, heldout8valseq:
71/24/22groups; stops30/select26oflast5; heldout11/01/10=18/16/17of22
versuscosine18/17/16. No consistentgain; not deployed in tracker.
Selectedhead retained, other4 removed withreceipts.
See docs/INDEPENDENT_ANCHOR_AUDIT_20261030.md and reports/server28-anchor-ranker-pilot-20261030.
At15:49:48Z fixed0/1updates5924/6340, candidates2/3updates611/625, finite.
No overalltarget/module completion; wait for firstcandidateepoch validation.

## 2026-10-30 Larger-batch adaptive candidates launched

GPU2 utility controller3816049/train3816056 and GPU3 random
controller3816050/train3816057 now train effective batch10 (micro5/accum2),
LR1.25e-4. At15:24:15Z each had24 finite updates, identical cached128 val
input hash and initial training-input hashes,22710/24576MiB occupancy100%.
Separate trainer/evaluators preserve the original GPU0/1 frozen campaign.
Each epoch checks a fixed val97 tuple loss proxy; LR-drop reference resets
at11,3 consecutive <0.2% improvements trigger stop earliest14/max20.
Rolling final5 epochs all receive full paired val97 tracking; mean macroSR
selects one common checkpoint, then other4 are cleaned with receipts and
selected checkpoint receives official complete/Miss evaluation.
No official test selection. This is best of last5, not full training history.
Not yet evidence of convergence, target attainment or three new modules.
Pipeline smoke6short epochs/12updates and1sequence real tracking passed.
See docs/CANDIDATE_TRAINING_LAUNCH_20261030.md and
reports/server28-candidates-20261030/verified-launch.json.
Next useful milestone firstepoch about80-90min from15:23Z; controllers poll300s.

## 2026-10-30 GPU2/3 diagnostic and training update

User authorized physical GPU2/3; server AGENTS/compute ledger now allow0-3.
Fixed8 native GPU consistency diagnostic completed complete/GPU2 in177.76s
and alternating-outage/GPU3 in156.35s,4602 video frames each. Both released
their GPUs. Read-only context/response hooks preserved native tracker rules;
all8 common31-frame prefixes matched exactly. Low available layer6 similarity
AUC complete/outage=.478631/.765269, versus Hann=.564110/.835098 and
raw=.570248/.845034. This does not justify a universal trigger. Layer6
initial-template tokens are jointly contextualized, not independent identity
evidence. No threshold or recovery intervention selected; frozen runtime
unchanged. See docs/GPU_CONSISTENCY_AUDIT_20261030.md and
reports/server28-val97-gpu-consistency-20261030.

At15:07:17Z utility4866/random5217 of50000 were live with finite recorded
key losses; evaluator3463984 remains waiting300s. ETA October31 21:10/19:16
UTC (November1 05:10/03:16 Beijing). Overall research goal remains incomplete.

## 2026-10-30 28号机正式训练启动

2026-10-30独立CPU真实更新诊断完成：从历史A utility epoch20同一模型/AdamW状态出发，完整任务、完整+缺失、完整+缺失+正向KD各执行一次有效batch8更新，另8个train882元组只观察损失；耗时291.25秒，无GPU、无冻结科学源码修改。完整训练模式损失变化分别-0.001934886/-0.001088679/-0.001071393，额外样本变化-1.6868e-5/+5.1856e-6/+5.8413e-6。原始总梯度对完整梯度内积197.184/268.855/269.584，额外任务方向差值也为正，当前样本不能支持直接以负梯度冲突解释最终退化。实际改善受联合任务影响，但额外样本变化很小、未重复验证，不能称显著退化或模块有效。详见`docs/COMPLETE_TASK_UPDATE_AUDIT_20261030.md`及`reports/server28-task-interference-20261030/report.json`。暂不向当前训练添加梯度投影。

2026-10-30T12:58:40Z活进程核验：utility1612/50000、random1740/50000，全部已记录关键损失有限；评估控制器仍真实等待。最近100步2.369/2.200秒每更新，暂估美国东部10月31日16:48/14:28训练完成。25号机源目录只读复查仍仅A完整模态两组metrics，未发现遗漏的A Miss结果。

恢复诊断准备新增：`development/recovery_val97_20261030/protocol.json`及完整、RGB中断、TIR中断、交替中断四份固定mask，97序列/54646帧。每32视频帧切换预先固定状态，不读取测试mask或模型得分选周期；交替条件包含7546双缺失帧。独立入口`evaluate_val97_recovery28.py`可记录真实闭环框、Hann响应、动态模板真实替换、crop支持及离线GT IoU；没有向模型传入非首帧GT，也没有改冻结训练/评估入口。尚未执行此入口的神经网络诊断，不能称B模块已有效。

目标Python3.7语法/CLI检查通过；100组原生crop比较及100组原生IoU比较通过。首检查脚本误用IoU返回维度，按实际一维接口修正后通过，无fallback。宽高非正260帧保留轨迹、开发几何/IoU标签为空。协议限制、未认证的E0验证集暴露边界、实际验证范围见`docs/VAL97_RECOVERY_PROTOCOL_20261030.md`，两份实际报告位于`reports/server28-val97-recovery-preparation.json`及`reports/server28-recovery-geometry-validation.json`。

research技能要求的背景研究完成KeepTrack/GlobalTrack固定提交源码核查，结果见`docs/RECOVERY_SOURCE_REVIEW_20261030.md`。没有发现可直接移植且同时满足许可、局部可靠性/扩域搜索/原目标核验分离的完整实现；只借鉴机制问题，不复制不明许可或GPL实现。候选为有限扩域提案加独立固定初始模板接受，但当前只是假设，固定模板分数仍可能误认相似目标，需要错模板与同候选集对照，不作身份保证或新颖性声明。

按用户“迁移了之后开始训练”要求，2026-10-30 11:54:37 UTC（美国东部07:54:37）启动现有 A utility/random 完整20轮配对配置。控制器 PID3453475，utility/GPU0 PID3453478，random/GPU1 PID3453479。两组均已完成真实首次优化器更新，损失有限，首update两份 microbatch 输入哈希一致，峰值分配显存16652186112字节。不是仅创建配置或迁移 smoke。

固定预算：每组20轮×20000样本、microbatch4×累积2、50000次更新、seed2027；从相同 baseline epoch20 权重独立开始，冻结全部BN，reverse_weight=0，不因日志出现 kd_reverse 就称反向反馈已参与优化。沿用已有配置，本次属于28号机同配置训练，不是新方法改进或已达标证据。

- 运行副本：`/home/gaob/Miss-RGBT/runtime/task_utility_20261018`。
- 日志与启动/退出收据：`/home/gaob/Miss-RGBT/logs/server28_a_full20_20261030`。
- utility输出：`/home/gaob/Miss-RGBT/runs/server28/a_utility_full20_seed2027`。
- random输出：`/home/gaob/Miss-RGBT/runs/server28/a_random_full20_seed2027`。
- 配置：`configs/a_utility_full20.json`、`configs/a_random_full20.json`。
- 环境规格哈希81051614未变，运行源码哈希已保存于控制器日志目录。

保存epoch10/20，最终固定epoch20评价，不按Test选轮次。25号机已有结果和权重保留；本次尚无最终模型或四套完整/Miss指标。训练控制器只负责训练与退出记录。

随后已启动独立评估控制器PID3463984，等待两组训练均exit0及20轮/50000更新完成记录，再核对冻结源码哈希和GPU0/1空闲，依次运行utility完整/Miss、random完整/Miss四组全量评估。每组必须479序列/337353帧完成。等待间隔300秒，不自动重启失败任务。评估队列及收据：`logs/server28_a_evaluation_20261030`；结果：`results/server28_20261030`。

训练集文件/标注清点已通过：train882/459435帧、val97/54646帧，RGB/TIR数量与init.txt行数无缺项，划分无序列交集。宽高非正标注分别1314/260行，既有dataset可见性筛选将其排除；未据此修改数据或训练逻辑。此检查没有全量解码图像，也不能证明原E0未见val97。证据：`reports/server28-training-data-audit.json`。

2026-10-30T12:04:34Z实查活进程命令行：utility247/50000，random268/50000；已记录loss/taskC/taskM/update_loss_mean全部有限。最近100步每更新2.370/2.200秒，暂估美国东部10月31日16:49/14:28结束，届时才能启动全量评估。独立评估控制器仍在真实等待。快照：`reports/server28-training-progress-20261030.json`。研究证据及缺口：`docs/RESEARCH_STATUS_20261030.md`。

启动后约2分钟实查：utility update51/50000、125.85秒，random update56/50000、124.93秒；两卡利用率100%，显存约19332/19300MiB，日志损失有限。以启动早期平均速度估计训练约31～35小时，仅作暂估，后续按实测修正。启动SSH已断开，控制器和两组训练仍存活，后台运行确认。

## 2026-10-30 迁移到服务器28

日期：2026-10-30。

## 目录和运行入口

- SSH：`2028`，用户：`gaob`。
- 项目：`/home/gaob/Miss-RGBT`。
- Conda 环境：`/home/gaob/Miss-RGBT/conda-envs/miss_rgbt`。
- 物理 GPU：`0,1`。
- 最新 A 实验运行副本：`/home/gaob/Miss-RGBT/runtime/task_utility_20261018`。
- 原25号机 data2 实验资产：`/home/gaob/Miss-RGBT/artifacts/server25-data2`。
- 原25号机文件保留，未删除。

```bash
ssh 2028
source /home/gaob/Miss-RGBT/server28-env.sh
cd /home/gaob/Miss-RGBT/runtime/task_utility_20261018
python scripts/evaluate_model.py --help
```

需要 Conda 环境激活时：

```bash
source /home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/activate
```

## 验证和修正

- home 源目录103419个文件已核对清单，无文件缺失。
- 原 home 下模型、脚本、配置和权重242项 SHA256 校验通过。
- data2 全量归档14287个文件 SHA256 全部通过，文件数一致；项目总量约49GB。
- Python3.7.16 / Torch1.12.1+cu113 / NumPy1.21.5 / timm0.9.12 / OpenCV4.8.1 / PyYAML6.0.1。
- GPU0、GPU1 固定种子 CUDA 矩阵检查均通过，结果 `31.01445198059082`。
- utility/random epoch20 均严格重载通过，20轮、50000更新，模型张量有限。
- utility/GPU0、random/GPU1 各完成 LasHeR `10runone` 前两帧真实跟踪检查，输出有限。该检查不计作完整评估。
- 28号机 glibc2.43 加载旧 Torch 曾报可执行栈错误。只修改 `libtorch_cpu.so` 的 ELF GNU_STACK 标记，从7改为6，偏移124；未修改模型或损失。
- 库修改前 SHA256：`db0f9b299cf9144912fc46166c1f8ceb772391e5e22efd03a4a7cb0ba41284bf`。
- 库修改后 SHA256：`1a1de1d8fded88c2e708536c5c6a56ef3962df6dddbf1e5ef9f4de52403a6ce1`。
- 运行副本的评估参数由 GPU2/3 改为 GPU0/1；原始归档和 checkpoint 内配置保留历史路径。
- 环境规格哈希更新为 `81051614`，记录在 `.aris/compute/2028.md`。

## 数据状态

LasHeR 测试集使用 `/data/wangwj/dataset/LasHeR/testingset`，245条序列、220703帧，RGB/TIR 文件数量全部匹配原实验。训练配置指向 `/data/wangwj/dataset/LasHeR/trainingset`；未在本次迁移中重新核验全部训练图像。

LasHeR/RGBT234 缺失 mask 均与原实验 SHA256 一致。

RGBT234 已从25号机 `/data2/wangwj/dataset/RGBT234` 迁移至28号机 `/data/gb/RGBT234`，约7.8GB、234011个文件，全部 SHA256 与源端清单一致。运行副本 `reports/e0/inventory.json` 的数据路径已更新为 `/data/gb/RGBT234/RGB_T234`。

RGBT234 的234条序列、116650帧均已核对，RGB/TIR 数量无缺项。实际评估入口 `scripts/evaluate_model.py` 的完整模态和 Miss 数据预检均通过：479条序列、337353帧，标注哈希、标注形状和缺失 mask 均匹配。当前数据已具备完整评估条件。

校验记录：

- `reports/server28-rgbt234-sha-check.log`：234011个文件 SHA256 全部通过。
- `reports/server28-data-readiness.json`：两套数据均无不完整序列，mask 哈希匹配。
- `reports/server28-evaluation-data-preflight.json`：完整模态和 Miss 的全量数据预检。

utility 在物理 GPU0、random 在物理 GPU1 均完成 RGBT234 `afterrain` 前两帧真实跟踪，输出框有限。记录位于 `reports/server28-rgbt234-tracking-smoke-a_utility_full20_seed2027.json` 和 `reports/server28-rgbt234-tracking-smoke-a_random_full20_seed2027.json`，仅为迁移检查，不计作完整评估。

## 已有 A 实验状态

两项训练均已完成20轮、50000更新，完整模态评估已完成；目前未找到 A 实验 Miss 评估完成记录。

| 模型 | LasHeR PR/NPR/SR | RGBT234 MPR/MSR |
|---|---|---|
| utility | 75.100912 / 71.359539 / 60.004465 | 91.450712 / 66.197862 |
| random | 73.388636 / 70.127102 / 58.778041 | 90.297337 / 65.145811 |

以上是25号机已完成的原始结果，非28号机重跑结果。

本次未启动新的正式训练、正式评估或 GitHub 推送。



## 2026-10-19 清理记录

已删除四个不再引用的 smoke/preflight 测试权重目录，并删除旧的 `docs/HANDOFF_20261013_launch.md`。正式 E0-FT、E2、E3-allBN、E4-allBN 权重、A 当前训练目录和原始审计证据均保留。当前项目与桌面只保留这一份正式交接文档。
更新日期：2026-10-18。归因队列和BN冻结对照四项完整评估均已完成；研究目标仍未达成。

## 2026-10-18 RGBT234开发边界核查

服务器只读清点确认 `/data2/wangwj/dataset/RGBT234/RGB_T234` 只有234个序列目录和每序列标注，没有独立train/val/development目录或第二套名单。因此当前没有经过认证的RGBT234开发集。

当前RGBT234完整/Miss结果均冻结为正式诊断，不用测试分数选择修复宽度、层数、K、模板门限或训练轮次。后续若要做数据集专用配置，先补充外部开发数据或预先封存的序列级开发/保留协议；同一数据集完整/Miss必须共用选定配置和权重。详细边界见[RGBT234开发边界](docs/RGBT234_DEVELOPMENT_BOUNDARY_20261018.md)。


## 2026-10-18 模板更新 U0/U1/U2 固定权重对照

在 E3-allBN/E4-allBN epoch20 权重上，隔离推理入口完成 U0（关闭动态写入）、U1（当前每5次有效track且置信度>0.65）和 U2（仅双模态同时可用时写入）全量轨迹对照。每项479序列/337353帧；U1 的 E3 完整轨迹与正式评估479/479逐文件SHA一致。

| 模型-策略 | LasHeR PR/NPR/SR | RGBT234 MPR/MSR | LasHeR-Miss PR/NPR/SR | RGBT234-Miss MPR/MSR |
|---|---:|---:|---:|---:|
| E3-U0 | 71.086343 / 67.002180 / 56.608090 | 88.308356 / 64.783972 | 64.597406 / 59.443621 / 51.530383 | 82.165737 / 58.711050 |
| E3-U1 | 74.380137 / 70.563438 / 59.427986 | 90.576570 / 65.530565 | 69.337978 / 64.225884 / 55.175799 | 85.063210 / 60.325720 |
| E3-U2 | 74.380137 / 70.563438 / 59.427986* | 90.576570 / 65.530565* | 67.754340 / 62.603336 / 53.881587 | 84.830721 / 60.186831 |
| E4-U0 | 70.991487 / 67.057367 / 56.561364 | 88.508346 / 65.208093 | 64.118588 / 58.854460 / 51.095612 | 81.481812 / 58.570372 |
| E4-U1 | 74.082029 / 70.398986 / 59.207785 | 90.360653 / 65.350919 | 68.422283 / 63.305148 / 54.499880 | 83.975517 / 60.033254 |
| E4-U2 | 74.082029 / 70.398986 / 59.207785 | 90.360653 / 65.350919 | 67.452961 / 62.456182 / 53.650289 | 83.414958 / 59.607996 |

U1 在两个模型的缺失条件下都优于 U0/U2；限制单缺失帧写入会损伤恢复效果。U2 在完整输入上按策略逻辑应与 U1 相同；E4 聚合指标一致，E3完整值为逻辑等价推断（*），没有单独做完整轨迹SHA等价声明。模板更新是已有 CADTrack 行为的控制变量，不计为新论文模块；RGBT234-Miss 仍未达到88.1/64.3。


## 2026-10-18 02:58北京时间：BN冻结对照完成

E3-allBN与E4-allBN均完成20轮/50000次更新，分别从同一个原始E0 epoch20权重启动；训练exit0，epoch20权重严格CPU重载，39个BN运行缓冲逐项与E0相同。四项全量评估均exit0，每项479序列/337353帧。

| 模型 | LasHeR PR/NPR/SR | RGBT234 MPR/MSR | LasHeR-Miss PR/NPR/SR | RGBT234-Miss MPR/MSR |
|---|---:|---:|---:|---:|
| E3-allBN | 74.380137 / 70.563438 / 59.427986 | 90.576570 / 65.530565 | 69.337978 / 64.225884 / 55.175799 | 85.063210 / 60.325720 |
| E4-allBN | 74.082029 / 70.398986 / 59.207785 | 90.360653 / 65.350919 | 68.422283 / 63.305148 / 54.499880 | 83.975517 / 60.033254 |

E3-allBN在LasHeR-Miss的PR、SR超过69.3、54.4；E4-allBN仅SR超过54.4。两者RGBT234-Miss均未达到88.1/64.3，完整五项也没有达到相对E0约两点提升。E3-allBN比原E3提高LasHeR-Miss PR/SR约+1.157/+0.814pp，但完整LasHeR仍低于E0，不能把BN冻结称为完整性能恢复；它是归因对照，不是新贡献。

原始实验归档`bn_freeze_closed_20261018_v5.tar.gz`，SHA256 `725421555b99d2b701bcd53a1ad427104e05640797eef049524de561537fe709`，4015个成员。来源120文件、训练/评估exit0、每项479条收据、有限值、严格重载与39个BN缓冲已核对；本控制没有重新独立实现评分器，E2审计中的评分来源限制继续适用。

这次控制排除了“融合层BN运行统计持续更新”作为唯一解释，但没有解决RGBT234-Miss的系统性缺口，也没有证明缺失训练改善完整任务。下一步应固定E3-allBN作为LasHeR-Miss候选，按开发协议为RGBT234选择同一方法族内部参数，再做模板U0/U1/U2真实轨迹对照；A/B模块仍需正式训练和消融，C只有超过标准互教/优化器对照才能列贡献。


## 2026-10-17 09:14／09:22实况：BN对照过半，两项epoch10检查通过

原控制器1753089与E3/GPU2 PID1754012、E4/GPU3 PID1754013均仍存活，命令和物理CUDA绑定吻合。09:22截面E3为epoch11/update25033（50.066%）、E4为epoch11/update25702（51.404%）；所有已记录数值有限，更新连续，120份活动科学文件仍匹配。没有推理或Test指标，目标仍未完成。

09:14时E3尚差204步才到epoch10，未把文件不存在当失败；按204/0.490442754秒加60秒估计，09:22仅复查一次。E4在09:14、E3在09:22的epoch10保存状态CPU核验均为10轮/25000更新、全部模型状态有限、39个BN运行缓冲与原E0完全相同，融合BN affine已改变。没有NN前向/优化/Test选择，也未让短检查权重进入正式训练。E4未重复读取同一epoch10文件。

E3 epoch10 SHA `68b5e5d661841b3351adc702788895065cc0182d3cc426b7a264ae65c8aa7849`；E4 `3a5a7ee3ea7bf357072739a0fbafc516aa39ca60b4d39e55762dad221ac62950`。它们是中途实现证据，不是最终epoch20成绩、独立二进制审计或BN性能因果证明。

两卡占用仍为19093/19307MiB，09:22瞬时利用率79%/94%；继续每卡一个任务。近200更新实测速率分别0.486401659/0.500559358 updates/s，粗估训练结束E3 **2026-10-17T23:37:52.214791+08:00**、E4 **2026-10-17T22:51:23.904611+08:00**；后续仍自动完成四项完整/Miss评估。

**下一唯一绝对节点 2026-10-17T23:32:52.214791+08:00**：取较慢模型预计结束前300秒，不在节点前查NN/GPU/日志/进程，不新建本地长观察器；必要临近复查180～300秒。原BN/模块/mask/模板/训练预算和科学源码不改。见[中点](reports/bn_controls_20261016_v2/midpoint_20261017.json)、[补查](reports/bn_controls_20261016_v2/midpoint_followup_20261017.json)、[结束节点计划](reports/bn_controls_20261016_v2/end_observation_plan_20261017.json)。


## 2026-10-16 18:42闭环／19:08实际节点：E2审计完成，BN对照正式训练

E2完整评估于17:56:55、Miss于18:42:01北京时间exit0，阶段1控制器CAMPAIGN_COMPLETE。每项479序列337353帧，合计958次序列运行674706帧。最终20轮/50000更新，权重1236735527字节、SHAf66799aaaa769f9cfde71ed05fecf02f9eff7be18e54518d7c1d20e8e76f9a3d；父进程CPU严格重载250状态项且有限，原件保留data2。固定末轮没有Test选轮。

| E2条件 | PR/NPR/SR 或 MPR/MSR，百分数 |
|---|---:|
| LasHeR | 74.444219 / 70.625478 / 59.251822 |
| RGBT234 | 89.731440 / 65.181523 |
| LasHeR-Miss | 68.369839 / 63.463823 / 54.431736 |
| RGBT234-Miss | 84.771986 / 60.542639 |

fresh experiment-audit结果WARN、限定包0阻断项，same-family/provisional，请求Astra/max，实际路由未认证。10项指标和93966曲线值独立重算误差0；958轨迹/回执、674706行、52493双缺失保持、50000连续更新核对。4062原封存加293补充输入不变，补充旧控制重算明确复用既有审计。MD SHA619d5c1a2541a5c22ce2c49c89e39307c763ff4d6f977a346dcac4c0d2916692，JSON SHA5728424128757bd9e6499a6d6b7eff645a47fe5e66a769674d227c65cc26dda2；请求/响应/证据在.aris/traces/experiment-audit/2026-10-16_run02。评分来源/二进制独立检查/单seed限制见[完整报告](reports/e2_20261016_v1/README.md)。封存RESULTS中的“待审计”保留历史字节，README给出当前状态。

E2相对FT：Miss五项均提升，完整五项中四项下降；完整五项仍全部低于E0，严格Miss仅LasHeR-SR超过目标，1/4达标。E3−E2完整五项提高，但LasHeR-Miss三项下降；RGBT234-Miss两项提高。E4−E3完整五项下降、LasHeR-Miss提高、RGBT234-Miss MPR微升而MSR微降。FT/E2等元组/更新但不等前向计算、M损失总权重或融合BN曝光，不能将全部增益归给KD/修复，也不能声称缺失训练改善了完整能力。E2全部50000条loss等于float32(taskC+taskM)，KD权重均0。融合BN计数+600000不是原因证明。

E2原始归档25107850字节、SHA7621ac9d07039e67d5fb04b332149051d118355a09b0408a384d95bcb65989eb，2066份payload核验；在任何BN源码变更前归档旧115科学文件。首次收集器误要求GPU[2]，按真实[2,3]修正后收集exit0；失败和原记录保留，无NN重放。

BN源码最初656b140已提交推送；首次部署提取后，120文件核对发现一个旧CPU验证脚本的无用import/注释与Git不同，补齐唯一文件后120项全相符。v1控制器1752683随后在NN启动前遇到哈希函数Path/字符串类型错误；旧日志/目录留存，不记为训练成功。仅修正Path调用及新v2运行目录，实际科学执行提交**1f1929afe4de032f9726a210e53b6b22fb850eea**。未更改网络、模块、损失、mask或训练预算，没有fallback或吞错。

控制器1753089于19:05:22启动。E3/GPU2与E4/GPU3两项真实两步检查均exit0：39个BN运行缓冲不变、融合BN affine更新、严格重载通过；短检查权重不用于正式训练。19:05:55已分别启动正式训练PID1754012与1754013，均从原始d20c8641 E0开始，各完整20轮/50000更新。19:08:30实际截面为update70/72、已记录数值有限，120科学文件匹配、4445启动数据文本/GT/评分包/mask哈希已保存；不等于全图像封存。当前没有BN最终指标。

两卡当前19093/19307MiB占用、瞬时GPU利用率100%/89%，各约5GiB空闲，继续每卡一项；没有同卡并发压测或实测减速结论。环境spec与conda不变，无升级/安装，项目仍/home/gaob/Miss-RGBT，检查点在data2。当前冻结全部13个BN运行统计，affine与其他训练模式保留；只对比原E3/E4的一项统计策略。

**唯一下一节点：2026-10-17T09:13:58.786314+08:00**，按首70/72步速率估算两项epoch10保存后300秒，检查真实日志/权重BN缓冲并重估结束。当前粗估训练完成2026-10-17T23:11:48.848678+08:00，其后仍需四项全量评估，不当作已完成。节点前不查训练/GPU/日志/进程，不新建本地长观察器；必要近结束复查180～300秒。计划[JSON](reports/bn_controls_20261016_v2/observation_plan.json)，首次实况与失败历史见[BN记录](reports/bn_controls_20261016_v2/README.md)。A和模板方案仍隔离，不改当前冻结队列。缺失学习改善完整任务仍需同BN/模板的完整训练参照，不能以旧FT做唯一因果控制。


## 2026-10-16 17:03实际节点：E2完整训练结束，固定末轮全评估已启动

E2原PID4063197于17:03:41北京时间exit0；completed为20轮/50000更新、99866.805778秒，epoch20文件1236735527字节已保存。这次仅核对完成记录和文件大小，尚未做末轮权重SHA/严格重载或全指标验收。控制器3796810仍执行原队列，立即以物理GPU2、3启动E2完整输入全量评估PID1734928，采集时0条序列回执；之后仍需Miss全评估。没有改变缺失、BN、LR、batch、模板或其他设置。

原隐藏观察43188在到期后未找到PID，stdout仅有启动owner，未生成到点收据，实际退出码未知；保留原计划和文件，不把它记为成功或重启NN。到期后单次只读SSH实际exit0收得上述节点，见reports/attribution_runtime_20261016_1703.json。由于此前61699也发生长等待后观察丢失，后续不再创建辅助本地长等候进程，改为保存绝对节点、到点直接只读采集；不增加训练重试或后台轮询。

下一节点为18:39:26.125650北京时间（10:39:26.125650UTC），预计E2完整/Miss两项评估18:44结束。估计使用FT实际单卡完整6560.437171秒、Miss5529.748397秒之和除以2，得到6045.092784秒；只是当前估计，不是E2实测速率/扩展效率。节点前不查NN/GPU/日志/句柄，必要临近复查180～300秒；无新本地observer PID。计划见reports/attribution_observation_plan_20261016_1703.json。

阶段1全部完成、E2原始轨迹和源字节归档后，才部署已准备的BN单变量源码，先真实两步预检再两卡各一项完整20轮。原环境账本.aris/compute/2025.md及env-spec保持不变，复用现有conda，不新建/升级环境。A及模板对照仍保持隔离，尚无新方法成绩；本轮研究目标仍未达成。

## 2026-10-16 07:48闭环／08:12审计：E0-FT完整指标已完成

E0-FT训练20轮/50000更新exit0，固定最终epoch20；两项评估均479序列337353帧、exit0，共958次序列运行674706帧。完整评估06:13:56结束，Miss07:46:19结束，07:48实际核对完整回执。权重仍在/data2/gaob/Miss-RGBT/checkpoints/e0_ft_full20_seed2027/epoch20.pth.tar，1236735591字节、SHA0f276f65f360e8a744a76ae0f32bba2f41bbc62e8956b59c883e6f87485c57a8；父进程CPU严格重载250状态项、有限值检查通过。原始E0和FT权重均保留，不提交Git。

| 条件 | PR/NPR/SR 或 MPR/MSR，百分数 |
|---|---:|
| LasHeR | 74.414259 / 70.658986 / 59.315515 |
| RGBT234 | 90.478169 / 65.745838 |
| LasHeR-Miss | 61.205607 / 56.998678 / 48.978097 |
| RGBT234-Miss | 78.434794 / 56.047763 |

[正式完整记录](reports/e0_ft_20261016_v1/README.md)及CSV/机器表已生成。fresh experiment-audit请求gpt-6-astra/max，实际路由未认证，结果WARN、same-family/provisional、限定结果无阻断项；FT10指标/93966曲线值和两个基线20指标/187932曲线值独立复算误差0。958轨迹/回执、674706行预测、52493双缺失保持、50000更新和各源码清单均核对。6122份输入封存前后不变，原4055封存未改。审计输出SHA：MD4b37e13e746093af37020a25ffdffbb882def4ca341e1ff3e16d7d4f700e0110，JSON24cbf471bf3a599dedc0f1beab5dc76ae2e179f10631015e07c682d94f806f86；完整请求/响应/运行痕迹留在.aris/traces/experiment-audit/2026-10-16_run01，原始审查快照保留。

FT完整五项均低于E0，表明完整退化在仅完整继续训练时也出现，不能全归因mask/KD，仍不构成BN或优化器的因果诊断。Miss与原E0输入处理不同，应额外比较同availability路径：仅LasHeR-Miss NPR+0.192448pp，其余四项下降。四项严格目标0/4超过，完整约2pp与有效论文模块目标仍未达成。FT融合BN计数增加300000、36个头部运行缓冲保持；后续缺失训练正迁移比较须同时匹配BN与模板策略，如改成allBN不能直接拿原冻结策略的FT作唯一因果对照。

归档23505178字节、SHAe2e45f8d39645e8125115fdd6507fd40d0fb3f61848c682c65fee189dbdbbcb8；远端收集exit0，首本地收集因1967文件子集不含全部3556安装路径而exit1，原记录保留。仅本地续验全部2064份payload、1967实际持有文件及3556项公开wheel对应后完成，未重跑NN。评分清单是评估后核验，1589安装副本未独立持有，作者工具箱/SCDT等价性及独立二进制审计未完成。开发划分归档观察早于FT启动3894.443616秒，但不是启动时封存或原E0历史暴露证明。

E2最后实际07:42:54为epoch14/update32964/50000（65.928%），GPU3、原PID4063197，最近200记录有限，近期0.5032356 updates/s；预计17:07训练结束仅为估计。原隐藏观察37108/38200已交付收据，SSH0并确认进程已离开，不再重启。唯一下一隐藏观察器43188/parent39960，07:49:44已核验命令和出生时间，wait_attribution_e2_end_node_20261016.py持有17:02:08.621384北京时间（09:02:08.621384UTC）节点。此前不查NN/GPU/日志/CIM/句柄，不重复计时器；时区显示变更不平移绝对时间。到期先读既有attribution_e2_end_node_intake_20261016.json及transport，必要临近复查180～300秒。当前没有新的E2最终指标或额外NN任务。

当前科学目录仍dfc989e，次序保持E2全量归因闭环→BN→固定权重U0/U1/U2→有限调优→持续缺口下验证A/B，C需独立证据。保留baseline完整优势和两到三个贡献必须由正式结果证明，不用模块数量或逻辑检查替代目标。

## 2026-10-16 04:24实际节点：E0-FT训练完成，固定末轮评估已启动

单次只读采集确认控制器3796810仍执行原归因队列；E0-FT原PID4063196于04:24:23北京时间exit0，completed为20轮/50000更新、54309.540992秒，epoch20文件1236735591字节已保存。这次只采集完成记录和文件大小，尚未重载或核验该末轮权重SHA，不写成完整实验验收。随即启动固定epoch20完整输入评估PID1028715，只用GPU2，采集时序列回执0；完整/Miss最终指标尚未产出。

E2原PID4063197仍在GPU3执行，最后epoch11/update26975/50000（53.95%）、54323.384822秒；最近200条已记录数值字段有限、速率0.4972958 updates/s，预计训练10月16日17:16左右结束，仅为估算。E2日志kd_forward非零是诊断值，冻结配置forward_weight=reverse_weight=0；实际loss为taskC+taskM，不能因此称E2启用了KD。没有修改训练或推理配置。

原61699工具句柄返回Unknown process id，本地原PID65768/parent64100均未找到，deadline后也没有生成收据，实际退出码未知；保留原owner/script，不伪造原观察成功。只在到期后执行一次替代只读采集，transport/进程exit0，详见reports/attribution_runtime_20261016_0424.json；没有重启NN或控制器。这个实际观察故障是下一次改用独立隐藏进程等待的依据，不引入训练自动重试。

唯一新观察器为隐藏Python37108/parent36308，04:25:46已核验命令与出生时间；本地脚本wait_attribution_eval_node_20261016.py持有07:35:32.955781北京时间节点。根据先前双卡E0-availability完整3165.044476秒与Miss2719.443680秒，按单卡线性估计FT评估合计11768.976312秒，预计07:40结束，提前300秒观察。该换算不是实测扩展效率或已经完成的结果。节点前不查NN/GPU/日志/等待进程，不重复建观察器；到期读取其已写收据，必要近结束重查间隔180～300秒。计划与实际owner见reports/attribution_observation_schedule_20261016_0424.json。

次序仍为FT/E2全量归因闭环→BN→模板U0/U1/U2→有限调优→按持续缺口验证方法；A/B/C有效性及原研究目标未完成。main科学目录与dfc989e一致，当前每卡一个训练/评估角色，未增加同卡训练或读取温度功率。

## 模板更新对照与最新执行次序：2026-10-16 00:47北京时间

现有E0/E3/E4已启用CADTrack式模板更新，不重复添加。U0只关闭动态写入且保留两模板槽，U1维持每5次有效track且置信度>0.65，U2仅增加当前双模态可用条件；位置先提交、双缺失保持不计track调用。隔离分支prepare/template-controls-20261015（95e5fd1，实际执行源码23c6b46）已提供入口及逐帧更新诊断，见[说明](docs/TEMPLATE_UPDATE_CONTROLS_20261015.md)与[实际报告](reports/template_controls_preparation_20261015/README.md)。本地Python3.12与服务器原conda Python3.7.16各9项stdlib检查通过，仅stub神经网络输出/AST执行真实控制流；没有GPU、原生NN或四套新指标，未合入活动科学源码。外部sandbox试验包没有本地取得，不沿用其14测试声明。

最新顺序为现有E0-FT/E2完整训练及评估→BN单变量→固定权重完整重跑U0/U1/U2→按缺口有限调训练/推理或同模块内部配置→仍有持续系统性问题才做A/B正式方法，C需额外独立证据。不同时改所有因素，不为模块数量堆结构。每个正式训练候选仍完整20轮/四套10指标；推理控制不重训但必须完整跟踪，不能重算旧框冒充干预。

完整五项先不再系统性低于E0，再追求约2pp和原四项Miss阈值。缺失训练帮助完整能力必须比较同模板策略的E0-FT与缺失训练模型；推理标定收益分开报告。达标数据集配置另存，其他数据集另开内部配置；完整/Miss同数据集固定同权重/配置，开发边界和全部尝试披露。具体候选与预算口径见docs/NEXT_EXPERIMENT_PRIORITY_20261015.md。

本段只更新隔离准备与执行策略，最后正式进度仍是10月15日14:10，未提前读取服务器训练/GPU/句柄。原唯一61699于10月16日04:22:40.141988北京时间观察，时间与调度不变。当前科学目录仍dfc989e，A/B/C有效性和研究目标仍未达成。

## 历史归因队列与资源：2026-10-15 14:10已核对

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

\r\n
## 2026-10-30 CPU prefix diagnostic completed

Real CPU neural tracking passed all four fixed conditions on historical A
utility epoch20, boywalkinginsnow3 first257 frames;575.93s, CUDA disabled,
4 threads. Separate CPU port preserves native tracking/template rules.
Complete/RGB-outage/TIR-outage/alternating meanIoU:
0.796168/0.799209/0.766971/0.716680. Alternating had32 both-missing
holds and224 tracker calls. All finite/common-prefix/template/hold checks
passed; no observed target center outside crop or zero crop intersection.
No crop escape reproduced here; no recovery/module gain established.
Not full-val97, official benchmark scores, or CPU/GPU equivalence.
Report: docs/VAL97_CPU_PREFIX_AUDIT_20261030.md.
Evidence: reports/server28-val97-cpu-prefix-20261030.

Training snapshot13:13:37Z: GPU0 utility1990/50000, GPU1 random2144/50000,
logged key losses finite; evaluator3463984 waiting every300s. Training
and frozen runtime unchanged. Overall research goal remains incomplete.

2026-10-30T13:24:34Z: fixed first8 val97 full-sequence CPU comparison
started as PID3608927; complete versus alternating,4602 frames/condition.
Historical A utility20,4 CPU threads/CUDA disabled, expected80-95min.
Live process and actual frame256 log verified13:27Z. Outputs:
reports/server28-val97-cpu-cohort8-20261030, sibling .log.
Launch protocol: docs/VAL97_CPU_COHORT8_LAUNCH_20261030.md.
Do not restart while live; no cohort results established at this snapshot.
Prior optimizer-state alias concern was not reproduced on actual Torch1.12.1;
step/exp_avg/exp_avg_sq all distinct and saved values unchanged after update.

## Utility selector audit completed

Four fixed val97 search tuples/state,16 region interventions each for11/01/10.
Pooled Spearman predicted score vs single-region actual task-loss gain:
-.058/-.136/+.456. Native top4 normalized loss reduction:
-.020%/+.082%/+.356%, versus seeded random +.094%/+.110%/+.156%;
offline oracle +.326%/+.384%/+.554%. N=4/state with high variation;
only preliminary10 signal, no general selector efficacy. Do not change live
training or tune on this probe.
Report: docs/UTILITY_RANKING_AUDIT_20261030.md.

At14:39:41Z CPU fixed8 trajectory cohort completed normally, PID3608927,
elapsed4507s. Across3924 observed valid pairs:836 outage-only crop-support
losses,670 both-crop losses,6 complete-only losses (all6 in ab_pingpongball).
Rates38.38% outage vs17.23% complete; weighted paired IoU delta-.119095.
Only2/8 sequences have outage-only loss;6 have none. ab_pingpongball is
poor in both conditions and has26 invalid GT frames. Low-Hann score AUC for
no crop/GT overlap .837 outage vs .553 complete; raw .847/.559, not a
universal/calibrated trigger. Investigate response plus template consistency,
with threshold calibrated on train882. Not official metrics/module proof.
Report: docs/VAL97_CPU_COHORT8_LAUNCH_20261030.md.

## 2026-10-30 18:52 UTC server28 live state

Four RTX 3090s report target temperature85C and power limit250W; current
temperatures were84/80/79/75C on GPUs0-3. Existing jobs occupy all four GPUs.

Fixed20 utility/random training on GPUs0/1 remains live, epoch5; latest
observed log updates were10200/10600 and recorded loss fields finite. Its
evaluation controller is still waiting for both jobs to exit.

Batch10 utility/random candidates on GPUs2/3 remain live, epoch3, latest
observed updates4250/4375, finite losses, peak allocated memory
21118406144 bytes per process. Only epoch1/2 fixed128 task-loss validations
are recorded; epoch3 validation is pending. These are not tracking metrics
or convergence proof. Final selection still requires full val97 tracking
over the last five epochs.

Historical A utility20 LasHeR-Miss evaluation PID4138772 is live with one
worker per GPU0/1. Four of479 sequence receipts are present (`blkboy`shead`,
`boyshead9684`, `midredboy`, `broom`); no complete metrics yet. Preserve all
jobs and continue checks every180-300 seconds, nearer candidate epoch
boundaries estimated20:16-20:26 UTC from the18:49 launch receipt.

Snapshot detail: docs/SERVER28_LIVE_SNAPSHOT_20261030_1852Z.md. Overall
benchmark targets and three validated contributions remain unproven.

## 2026-10-30 19:01 UTC incremental check

All five monitored processes remained live. GPU0-3 temperatures were
84/78/79/76C with250W power limits. Fixed20 latest updates were
utility10400/random10775, epoch5, finite. Candidate latest updates were
utility4400/random4550, epoch3, finite; recent throughput3.207/3.096
seconds per update. Only epoch1/2 fixed128 task-loss validations exist.
Estimated next candidate boundaries are random20:16UTC and utility20:27UTC.
Historical A utility20 LasHeR-Miss evaluation PID4138772 reached5/479
sequence receipts; no complete metrics yet. Snapshot:
docs/SERVER28_LIVE_SNAPSHOT_20261030_1901Z.md.

## Candidate evaluator provenance

The candidate launch receipt contains the pre-`7085616` SHA for
`evaluate_candidate_official28.py`; the server/current branch now match at
SHA256
`bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
Commit `7085616` postdates launch and changes only GPU selection, worker-count
and limited-smoke CLI handling, not tracking or metrics. The unchanged
campaign controller invokes it only after last-five val97 selection/cleanup.
Current evaluator use will record its actual SHA in `run.json`. This
provenance delta does not justify restarting active jobs. Audit:
docs/CANDIDATE_EVALUATOR_SOURCE_AUDIT_20261030.md.

## 2026-10-30 19:10 UTC live check

All five jobs remain live. Fixed20 utility/random are at updates10625/10975,
epoch5, finite. Batch10 utility/random are at4600/4750, epoch3, finite;
only epoch1/2 fixed128 task-loss validations exist, and only epoch01/02
checkpoints have completed. Historical A utility20 LasHeR-Miss evaluation
PID4138772 has8/479 receipts and62941 completed frames, all on LasHeR-Miss.
No complete metrics exist. First8 receipt throughput projects about3.1h
remaining for the337353-frame evaluation, but this is rough because jobs are
sorted longest first. GPU0-3 temperatures84/80/79/73C; all power limits
250W and target temperature85C. Snapshot:
docs/SERVER28_LIVE_SNAPSHOT_20261030_1910Z.md.

## 2026-10-30 19:20 UTC live check

Fixed20 utility/random remain live at10775/11125 updates, epoch5, finite.
Batch10 candidates remain live at utility4775/random4900, epoch3, finite;
only epoch1/2 fixed128 task-loss validations are complete. Estimated next
random/utility boundaries are20:17/20:26UTC. Historical A utility20 Miss
evaluation remains live at12/479 receipts and79206 frames (11 LasHeR-Miss,
1 RGBT234-Miss; corrected from receipt paths). First12 receipt throughput gives a rough full-evaluation ETA of
22:08UTC. GPU0-3 temperatures were84/80/79/75C, limits250W. Snapshot:
docs/SERVER28_LIVE_SNAPSHOT_20261030_1920Z.md.

## 2026-10-30 19:24 UTC incremental

All jobs remain live. Batch10 candidates are epoch3, updates4825/4975, finite;
epoch3 validation is pending. Historical A utility20 Miss evaluation reached
13/479 receipts and82856 frames:12 LasHeR-Miss,1 RGBT234-Miss (`man7`).
No complete metrics exist. Rough remaining full-eval estimate is2h47m at
observed worker throughput. GPU0-3 are84/80/79/76C, target85C and power
limit250W. Snapshot: docs/SERVER28_LIVE_SNAPSHOT_20261030_1924Z.md.

## 2026-10-31 05:14 UTC live GPU snapshot

All four GPU campaigns remain live. GPU2/3 batch10 utility/random candidates
are at epoch4 updates6925/7125; finite step logs continue and peak allocated
memory is21118406144bytes each. Epoch4 validation has not appeared yet.
Recent throughput puts the utility epoch boundary roughly40minutes away;
check near that boundary rather than polling frequently. The registered
early-stop rule starts after the epoch10 LR drop and requires3 stale epochs,
so epoch4 is not a convergence decision.

Historical A utility20 official Miss evaluation PID4138772 remains live
with175/479 sequence receipts (103 LasHeR-Miss,72 RGBT234-Miss). There is
no completed metrics receipt. All GPUs remain under250W configured limits;
temperatures were83/79/77/75C at05:13UTC. Fixed20 GPU0/1 jobs are still
running. Do not restart or add GPU processes.

At05:46UTC all monitored trainers and historical evaluator were still live.
Batch10 utility/random were at epoch4 updates7525/7750; the random run was
about250updates from its epoch boundary and utility about475updates away.
This observed progress is slower than the prior 40-minute estimate; check
near the random boundary. Historical Miss receipts reached353/479
(179 LasHeR-Miss,174 RGBT234-Miss), still without final metrics. GPU0-3
temperatures were83/79/78/75C with250W limits.

## Protected six-epoch CPU pilot launched

PID61899 launched2026-10-30T19:31:47Z and was verified live at19:32:58Z,
CPU337%, RSS1.2GB. Source compiled under the existing Python3.7 environment.
Input manifest exists; initial reference losses precede the first updates.
Output reports/server28-protected-sixepoch-20261030, sibling.log.

Fixed six epochs/192 updates, effectivebatch4/LR1e-4, four CPU threads.
Baseline frozen/eval; existing LocalRepair only. Train882 is partitioned by
sequence (every fifth sequence reserved for development); dev16 uses fixed
validation transforms, heldoutval32 never selects an epoch. Finalfive epochs
2..6 selected by mean01/10 development task loss, ties earlier. Retain one
small branch snapshot after exact complete-output/baseline-state checks;
record size/hash before deleting the other four. Protocol:
docs/PROTECTED_SIXEPOCH_PROTOCOL_20261030.md.

Estimated45-75minutes from launch, pending measured update throughput.
No PR/SR, convergence, novelty, or new module benefit established. Original
GPU jobs remain live and unchanged. Historical19:20 snapshot attribution
corrected:12 total receipts included11 LasHeR-Miss plusRGBT234-Miss man7.

Protectedpilot61899 verified19:36:17Z: actualupdates1..3 finite, normsfinite,
availability01/10/01, losses5.872172/2.717501/4.297597. Update intervals
9.06/8.53seconds after~4min initialreferences. Manifest confirms705train/
177dev/97val sequences, pairwisedisjoint; dev16/val32cachedtuples.
E0 pretrainingvalexposure remains uncertified. Nextcombinedcheck20:11UTC,
near mainrandomcandidate boundary and potential CPUcompletion. Do not
restart while live; resultsstillpending.

## 2026-10-31 05:56 UTC historical A Miss evaluation completed

PID4138772 exited after writing metrics and completion receipts for all479
sequences/337353frames; the two dataset directories contain245 and234
sequence receipts. Runtime was12528.71seconds. Historical A utility20 scores:
LasHeR-Miss PR/SR68.565174/54.621376; RGBT234-Miss MPR/MSR84.964353/
60.511286. Against E0 Miss baseline the deltas are+5.092667/+4.038228 and
+2.169539/+0.936850pp. It clears LasHeR-Miss SR by0.221376pp, but misses
LasHeR-Miss PR by0.734826pp, RGBT234-Miss MPR by3.135647pp and MSR by
3.788714pp. These are not complete-input results and do not establish the
requested approximately+2 complete-input gain. This is the historical
utility20 checkpoint, not either active batch10 candidate. Exact metrics,
all receipts, curves and run manifest are archived in
reports/historical-a-miss-20261031/.

At06:02UTC the historical evaluation was no longer running. The four GPU
trainers remain active: fixed20 utility/random on GPU0/1 and batch10
utility/random on GPU2/3. Batch10 random has entered epoch5 at update8050;
utility remains epoch4 at update7825. The four GPUs are still occupied,
temperatures83/80/78/76C and limits250W. Do not infer GPU availability from
the completed evaluation alone.

At06:19UTC all four trainers were still live. Both batch10 candidates are
now in epoch5, utility/random updates8125/8375; the epoch5 boundary proxy
has not yet appeared. Each reported peak allocation21118406144bytes.
Fixed20 utility/random are in epoch6 at14300/14525 updates. GPUs0-3 remain
occupied; temperatures84/80/79/76C with250W limits.

At07:29UTC all four trainers remain live. Batch10 utility/random are at
epoch5 updates9450/9750, with no epoch5 validation yet; their next
boundaries are roughly11/5minutes away at observed throughput. Fixed20
utility/random are in epoch7. All GPUs remain occupied, temperatures
83/80/79/76C and configured power limits250W.

## 2026-10-31 00:02 UTC verified results and time correction

Server28 uses UTC+08:00. Earlier October31 entries labeled05:13..07:29
UTC used the server wall clock and should read UTC+08:00. For example
07:29+08:00 equals October30 23:29Z. Historical A Miss completed at
October31 05:55:32+08:00, equivalent to October30 21:55:32Z.
Older snapshots are historical, not the current running state.

AtOctober31 08:02:19+08:00 the original four trainers were live.
Batch10 utility/random both completed epoch5 and entered epoch6.
Epoch5 mean validation task losses=5.0956679235/5.1295612181; both
remain worse than epoch3, so no convergence or stable utility advantage.
Next epoch6 boundaries are approximately09:45/09:25+08:00 using actual
3.2/3.1seconds per update. Observe near milestones at180-300seconds.
Detailed evidence: docs/CANDIDATE_EPOCH05_20261031.md and paired reports.

CPU diagnostic552251 completed40.86seconds, four reused pilot development
tuples and9 dynamic-template/search combinations per arm. All6 missing-
search means decreased for the selected branch. Complete search with
missing dynamic templates changed by-0.000819/+0.000196, mixed signs;
complete-history outputs remain bitwise equal and baseline unchanged.
This bounds the equality claim; it does not prove general benefit/harm.
See docs/PROTECTED_HISTORY_DIAGNOSTIC_20261031.md and full JSON reports.
No GPU job added/restarted; frozen runtime unchanged. Full objective
remains incomplete, especially RGBT234-Miss and three validated modules.

## 2026-10-31 00:11 UTC full-val97 tracking health launched

Controller570241 started00:11:12Z /08:11:12+08:00; at08:11:45+08:00,
workers570254/570255 were live on GPU0/1, each2266MiB beside19290MiB
original trainers. GPU2/3 candidate training remains allocated22702MiB.
Use one evaluator per GPU; do not increase concurrency or restart jobs.

Full97 sequences/54646 frames each: GPU0 runs E0 complete, then utility
epoch5 complete/outage; GPU1 runs E0 outage, then random epoch5 complete/
outage. Reuse native validation sequence/metric functions unchanged.
This is a predeclared epoch5 development health check, not checkpoint
selection or official test tuning. Existing stopping/final-five selection
continues unchanged. No tracking result exists yet.

Budget3-4.5hours total per queue, pending measured speed. First meaningful
longest-sequence check around08:20+08:00; inspect near subsequent estimated
condition boundaries at180-300seconds. Output:
reports/server28-epoch05-tracking-health-20261031, sibling.log.
Protocol/launch evidence: docs/EPOCH05_TRACKING_HEALTH_20261031.md and
reports/epoch05-tracking-health-launch-20261031.json.

Paired analysis is prepared as analyze_epoch05_health28.py and compiled on
server Python3.7. Run only after six-cell completed.json exists: it verifies
582 trajectory hashes, recomputes all sequence/aggregate metrics and reports
paired effects plus10000-draw sequence-bootstrap intervals (one seed only).
At08:17:23+08:00 workers570254/570255 remain live with CPU/read activity,
no error or full sequence yet. First sorted video ab_bolster has8406frames;
expect its first receipt around08:23+08:00 using prior throughput. Next
check should be near that boundary, not from an empty log alone.

## 2026-10-31 00:25 UTC first tracking sequence verified

Both original E0 workers completed ab_bolster8406frames/8393scored in
about733s each. Complete PR20/SR65.864411/61.055982; alternating outage
38.782319/42.084390. Hashes and recomputed metrics match; first32 common
complete-input boxes are bitwise equal across GPUs0/1. This is one video,
not full-val97 or a candidate gain. All original jobs remain live.

Measured11.46frames/sec implies about79.5min per full54646-frame condition.
Next combined check09:25-09:35+08:00, near E0 condition completion and
candidate epoch6 validation; all three cases per GPU roughly12:10+08:00,
pending actual speed. Do not poll every few minutes before those nodes.
Receipt reports/epoch05-health-initial-verification-20261031.json.

Source review docs/COARSE_FINE_SOURCE_REVIEW_20261031.md pins EfficientLoFTR
07e9c140: fine matching uses retained higher-resolution features and yields
point correspondences, not target box scale/identity acceptance. No external
code, weights, environment change or GPU method run. Future recovery must
separately validate proposal scale/localization and target acceptance on
train882 development; these are not validated paper contributions.

At08:34:45+08:00 CPU postprocess613855 was verified live/sleeping, targeting
the original live controller570241. It checks at300seconds, then requires
six-cell completion and unchanged analyzer SHA before running paired
analysis. No new GPU load, retry, training change or weight selection.
Runner finish_epoch05_health28.py; log health-report-dir/postprocess.log.
Require paired_analysis.json AND analysis_completed.json before reporting
verified paired results. Receipt reports/epoch05-health-analysis-queue-
20261031.json. Next meaningful human/agent observation remains09:25-09:35
UTC+08:00; do not keep adding intermediate snapshot commits.

## 2026-10-31 01:30 UTC E0 full-val97 verified

Both E0 conditions completed97sequences/54646frames/54289scored around
09:24:41+08:00. All194saved trajectory hashes and GT hashes checked;
per-sequence PR20/SR and both aggregates recomputed within1e-9.
Complete75.879164/61.654017; alternating outage64.528129/52.346049.
These are val97 development references, not official Miss test scores.
Artifacts reports/epoch05-health-e0-{complete,outage}-20261031.json.

Same evaluator workers570254/570255 now run utility/random epoch5 complete
tracking from09:24:41+08:00. Original controller570241, postprocess613855
and all four trainers reverified live09:30:44+08:00 after local monitor
exit; no restart. Candidate full complete estimated10:40+08:00 pending
actual candidate speed; alternating outage follows in the same queues.
Random training epoch6 mean taskloss4.963055770 improves fromepoch5 but
is worse thanepoch3; nowepoch7. Utility epoch6/update11700 at09:30:44+08.
Do not infer convergence or candidate PR/SR gain before full receipts.

Full E0 paired availability analysis completed offline,194 trajectory
hashes rechecked,54289 valid scored frames partitioned00/01/10/11 as
7503/10086/8865/27835. Outage-minus-complete sequence macro PR20/SR gaps
are00:-35.038/-26.475,01:-16.923/-15.765,10:-10.942/-8.070,
11:-4.099/-2.937pp. Current complete frames retain a descriptive history
gap; template, crop/scale drift and timing are confounded, no causal
template-write conclusion or change. docs/E0_AVAILABILITY_20261031.md,
reports/e0-availability-analysis-20261031.json. No new GPU run.

At09:46:11+08:00 utility completedepoch6/update12000, mean4.968987959,
states11/01/10=4.393904358/5.730497293/4.782562226; lowest through6.
Random epoch6 mean4.963055770 remains worse than its epoch3. Neither
convergence nor stable utility benefit established. All original8jobPIDs
live. Epoch5 candidate complete tracking each3seq15963frames; next useful
full-complete node~10:40+08:00. No training/LR/selection changes.
reports/candidate-utility-epoch06-validation-20261031.json archived.

## 2026-10-31 02:32 UTC full candidate complete-input results

Verified all291saved complete-input trajectories and recomputed all97seq
metrics/model. E0 PR20/SR75.879164/61.654017, utility5=79.596578/64.472777,
random5=80.667388/65.321675. Utility minusE0 +3.717415/+2.818759pp;
random minusE0 +4.788224/+3.667658. Utility minusrandom -1.070809/
-0.848899pp, paired-video95% intervals cross0 for both. No demonstrated
utility-selector advantage; gains do not establish official complete/Miss
scores, unseen E0 data, training-seed robustness or three contributions.

Alloriginal8jobPIDs reverified live10:32:58+08:00; both candidates now
track alternating outage using sameepoch5weights. Utilityfirstseq done,
fulloutage stillpending; nextwindow11:15-11:40+08:00, refineactualspeed.
NoLR/train/selection/cleanup changes. Continuefinal5protocol.
docs/EPOCH05_COMPLETE_RESULTS_20261031.md and reports/epoch05-complete-
paired-analysis-20261031.json contain fullverifiedresults/intervals.

## 2026-10-31 03:34 UTC full epoch5 health complete

Six-cell tracking finished11:23:42+08, paired analysis11:24:33+08.
All582 trajectories/GT/metrics verified. Outage PR20/SR:
E0 64.528129/52.346049, utility5 70.868974/56.749288,
random5 71.457491/57.152398. Utility-E0 +6.340845/+4.403239pp;
utility-random -.588517/-.403110,95%video CIs both cross0.
No demonstrated selector benefit; fixed val97 outage is not official Miss.
Final analysis SHA7034c378efa3dd1430dd32dcd94e968aaa11832a3e333f8a95b922aa106f7c31.

CPU-only paired penalty analysis rechecked582 tracks. Utility-E0 reduction
in outage penalty +2.623430PR/+1.584479SR, CIs cross0; utility-random
+.482292/+.445789 also CIs cross0. State00/11 SR penalties did not improve
at point estimates; state01 exploratory benefit vsE0, not vsrandom.
No causal/template/motion claim or val-based tuning.
Report SHAfbda5dbf76cff4e812455b2202f3f97401bf5449318689f31dffa80452c5cfcb.
See docs/EPOCH05_FULL_RESULTS_20261031.md and matching reports.

At11:32:50+08 batch10 original trainers live, epoch8 updates14007/14466.
Epoch7 validation means5.069815951/5.136228281 worsened; keep original
LR/stop/final5 protocol. Next epoch8 boundaries roughly12:52random/
13:27utility+08, uncertain. At11:33:25+08 fixed20 original trainers live,
epoch9 updates20767/20806; evaluator3463984 waiting. Both batch10
controllers3816049/3816050 live. Health controller570241 and post613855
finished normally; do not restart them. Official goals, final5 common
checkpoint selection/cleanup and three modules remain incomplete.

## 2026-10-31 03:44 UTC matched history-coverage pilot

Launched CPU-only paired LocalRepair control PID869785, fourthreads,
initialreferences running/live347%CPU at11:44:50+08. Botharms freshE0,
sameinitialrepair/data/192updates, sixepochs, effectivebatch4 LR1e-4.
Onlydifference: complete vs independently sampled dynamictemplates
11/01/10 with existingformal probabilities. Botharms use samefinal5
sixmissingsearchcell devselection;9devcells measured. No val97/test
selection, newmodule claim, GPUallocation orlive sourcechanges.

Output reports/server28-protected-mixed-pair-20261031; siblings.log.
Runner probe_protected_mixed_pair28.py; launcher launch_protected_mixed_pair28.py.
Protocol docs/PROTECTED_MIXED_PAIR_PROTOCOL_20261031.md. Runtime/source/
checkpoint hashes verified before optimization. Initialmeasurements can
take severalminutes; firstupdates not yetverified atinitialobservation.
Roughfirstarm12:35-13:05+08, both13:25-14:25+08; refineactualthroughput.
Require comparison.json AND completed.json and perarmselection/cleanup.
Do not restart after observationfailure. Original4GPUtrainers+3controllers
verifiedlive11:39:26+08. Fullgoal remainsincomplete.

At11:49:44+08 sameCPU869785 live393%CPU3.22GiB, firstarmepoch1
10actualupdates finite loss/norm; update10loss4.892799616 norm.190860614.
Recent7.7-9.1s/update; originalepoch1inputhashespassed. No gainclaim.
Next meaningfulcheckfirstarm12:35-13:05+08; avoidimmediatepolls.

## 2026-10-31 04:01 UTC repair-scope diagnostic completed

CPU892597 completed161.87seconds,27cells x4reused train882devtuples,
native vscomplete-feature replacement atlayer6. Fullaffectedstream
replacement reproducedteacher bitwiseallstates; frozenmodelhashpassed.
Missingpositiononly loss vsfullstream differs-.068507..+.047212 percell.
Teacherloss2.411833; nativeRGBonlymeans2.219739..2.290711 arelower,
whileTIRonlynative4.397563..5.400487 improvewithoraclefeatures.
Complete-search/missingtemplate nativealso lowerthan teacher here.
No universalteacherbenefit orneedforobservedtokenrepair established.
Keepcurrentrepairscope andcomplete mixedtemplate pair; do notadd
unconditionalfeatureMSE ornewmodulefrom4samples.

Report repair-scope-report-20261031.json
SHA162fa65f7959132eccefc8b79ff9e52d927c2f1b6e84c702233d3bedf4a20101.
docs/REPAIR_SCOPE_PROTOCOL_20261031.md hasfulltable andlimits.
Pinned FlexTrackV2 source30a5ff1 reviewed: masksencodedfeaturesafter
encoder, unlikeourpremasking; BMR per-samplefusionmask doesnotdirectly
handle tokenlevelhistory. Noexternalcode/weights integrated, nopinned
licensefilefound. docs/FLEXTRACK_SOURCE_REVIEW_20261031.md.

At12:00:57+08 alloriginal4GPUtrainers+3controllerslive; CPU869785
arm1epoch2/update59live. Nextmeaningfulfirstarm12:35-13:05+08,
batch10epoch8boundaries~12:52random/13:27utilityuncertain.
Officialthresholds/final5selection/cleanup/3modules remainpending.

## 2026-10-31 04:18 UTC blind-motion closed-loop control

Offline saveddev4tracks showCVimproves00SRall3blindvideos, butCPUoutputs
arenotGPUoutputs. RanmatchedsameGPU0holdvsCVcomplete/outage16trajectories,
3352video frames; evaluator925753finished237.57s andreleasedGPU.
Independent analyze_blind_motion28.py verified16bbox/trace/GT hashes,
allmetrics, complete/no-blind/prefixbitwise equality andfrozenmodel.
Fulloutage PR/SR89.324968/70.910351 ->91.134491/72.246885
=+1.809524/+1.336535pp. Blind80frames3seq+19.791667/+14.087302;
observedafterblind128frames2seq0/+.186012. Mostbenefit duringgap.
All3fulloutageSRdeltaspositive(.136054/2.278245/2.931839);
no-blindbike unchanged. OrdinaryCVcontrol, noofficial/noveltyclaim.

ReportSHAfaead2d14bc4580fb2efba9caac3365a422245058ff234bb3c276ee7b80ea4a5,
analysisSHA07e20f91cb1d79b2f7069536942c53382094afb46ce0e72ed132a8ca19086794.
docs/BLIND_MOTION_PROTOCOL_20261031.md hasoffline/closedloop distinctions.
OC_SORT pinned8462e7e source/MIT inspected; recoveryvirtualpath usesnew
measurement, doesnotforecastfuturemeasurement. Noexternalcodeintegrated.
At12:12:34+08 original4trainers andCPU869785live. OriginalLR/final5/
selection unchanged; fullgoal remainsincomplete.

## 2026-10-31 12:32 UTC+08 - Next16 motion control launched

Controller967085, GPU0 evaluator967093, GPU1 evaluator967094 verified live.
Four original trainers remain live; CPU869785 complete-template arm at
epoch6/update166 at12:31:58. No convergence or official-score claim.
Fixed unchanged CV versus hold on development indices4..19:16 full videos,
7776 frames per arm/condition,64 trajectories total. Same-GPU paired methods;
GPU0 eight videos3942frames, GPU1 eight videos3834frames.
Output reports/server28-blind-motion-next16-20261031; controller polls300s.
Budget20-35min from12:32; inspect near12:52-13:07, not early repetitive polls.
Require report.json and completed.json plus both independent worker analyses.
Code compiled with server Python before launch; frozen runtime unchanged.
Launch receipt reports/blind-motion-next16-launch-20261031.json;
protocol docs/BLIND_MOTION_NEXT16_PROTOCOL_20261031.md.
Official targets, convergence, last5 selection/cleanup and three validated
paper modules remain unfinished.

At12:37:29 UTC+08 CPU869785 remained live after first-arm update192;
final development/selection receipts were not yet present. Both batch10
trainers are in epoch8; no convergence evidence or schedule change.
Independent analyze_protected_mixed_pair28.py is deployed, and its
initial/current measurement checks passed on actual data. Run its main
only after paired comparison.json AND completed.json exist. No automatic
analysis queue has been launched. Full audit/second-arm results pending.

## 2026-10-31 12:42 UTC+08 - Gap duration mismatch and first CPU arm

Published-mask structure audit found effective blind frames33737/220703
LasHeR and18756/116650 RGBT234. Blind interval median1,p954,maximum15/12.
Current32-frame development gaps are longer than every official blind gap;
do not extrapolate motion stress-test gains to official scores.
Registered same16 videos/block1 follow-up in
docs/OFFICIAL_MASK_STRUCTURE_20261031.md. Do not modify in-flight motion
sources before current independent analyses finish. Short-gap run not launched.
Raw interval report remains server/local outputs, SHA4be7ac472affdd0c6ee023b3882556431cd74c4d7db7b0595f2f994d0d3d9c04.

CPU complete-template arm finished: selectedepoch6 score4.363575024840732;
retained repair_epoch06.pth SHAfe1bc6fb533ae81b2bc8e4e2cf1fc4005c10ee0b165fd70ff016b300838ea7ad.
Independently verified final5 argmin, retainedhash, exactly4deletedweights,
onlywinnerremains. First-arm reportSHA00664097b8fd98b64047a15023c6bd9b5d2203feb9078d87d16f46563e6f8f40.
CPU869785 live12:42:26, preparing second mixed-template arm. Both-arm
comparison/analysis and tracking benefit unproven. Main goal incomplete.

## 2026-10-31 12:49 UTC+08 - Short-gap continuation queued

PID997655 verified live waiting on real controller967085 with300s polls.
It requires the prior final64-track report/completed and independentworker
analyses, archives3sourcefiles withlaunchhashchecks, then deploysthestaged
frame-block1files and launches the same16 fullvideos on GPU0/1.
Shortgap GPU run has NOT started at12:49:56; oldworkers967093/967094live.
Serverrootmotionfiles remainold while GitHub nowdescribesstagednextversion.
Staging development/shortgap_staged_20261031; logreports/server28-motion-shortgap-continuation-20261031.log.
Crossduration analyzer analyze_motion_gap_contrast28.py deployed/compiled,
requires bothfinalruns; no resultsyet. One-frame versus32 changes switching
frequency too, not isolated durationcausality.
CPU869785 actualmixedarmepoch1/update25 at12:48:21; firstarmcomplete.
Fullgoalandofficialtargets remainunachieved.

## 2026-10-31 12:55 UTC+08 - Long gap verified, short gap live

Next16 completed64tracks. Complete84.909989/69.541361 same botharms;
outage hold69.109374/55.843400 -> CV70.264971/56.690159 PR20/SR.
Delta+1.155597/+.846760; bootstrap95CI[-.915736,3.678837]/
[-.891109,2.836747], bothcross0. SR6wins6ties4losses; noofficial/moduleclaim.
ReportSHAd0f7e15e2633924a2738661af1f383122af3ca2ee6e91856fe862021335bb933.
New intervalgeometrydiagnostic35gaps: blkskirtwoman gap352-383 blindSR
+10.565476 yet48observedframesafterSR-66.369048; firstrecovered384 target
notinCVcrop,34/48CVcropsunsupported versus0/48hold. Actualrecoverycrop
failure, notprovenidentity/confidencecause orvalidatednewsolution.
Docs BLIND_MOTION_NEXT16_RESULTS_20261031.md.

Continuation997655 completedoldaudit/archive/deploy thenexitednormally.
Shortgapcontroller1006350 started12:54:45, children1006355/1006356
verifiedlive12:55:22 each2266MiB. Oldsourcesarchivedinlonggap/verified_source.
Shortgapoutput reports/server28-blind-motion-shortgap16-20261031;
inspectnear13:14-13:25 then300spolls. Do noteditactivehashrecordedsources.
Batch10random nowepoch9,epoch8val4.995457771; utilityepoch8update15524.
CPU869785 mixedarmrunning. Fullgoalstillincomplete.

## 2026-10-31 13:03 UTC+08 - Velocity audit and result collection queued

audit_motion_estimates28.py verified endpointformulaall35longgaps.
All35 inputstatepairs11->11, so nocrosssensorpairsexplaintheseestimates.
blkskirtwoman gap352 lastobservedcentererror5.203668 grows479.839829
atblindend; holdingthesamelastCVbox90.852464. Othercasehyalinepaper928
GTrecentvelocity0,0 butpredicted-19.464630,-9.828613 withalreadylost
observedbox. Mixedcauses; nofittedthreshold/cap/newmodule.
Shortblock1 velocityinputstates10->11 byschedule, so crossduration
contrastalsochangesobservationstatepair, notjustgapduration.

Postanalysiscontroller1020925verifiedlive13:02:55, waitingactual
short1006350 andCPU869785 independentlywith300spolls. Aftereachprocess
exitrequirescompletion; short->contrast+estimateaudit;CPU->pairedreceipt
analysis. Logreports/server28-development-controls-analysis-20261031.log,
receiptsibling.json. Analyzerhashesfixed; donotedituntilqueuedanalysisdone.
Shortworkers1006355/1006356stilllive; CPUmixedarmepoch2update64.
Goalremainsincomplete; noofficialscores/threevalidatedmodulesyet.

At13:05:32 UTC+08 fourtrainers, short1006350/1006355/1006356,
postanalysis1020925 andCPU869785 allverifiedlive. Shortalreadyinvelocityarm;
CPU mixedepoch3/update88. Nojobrestart. At13:06:20 pairedCPUinitialJSONs
exactlyequal; matchedepoch1sixcelllossdifference mixed-complete-.002659345,
epoch2+.005247972, mixedsigncells. Noconsistentbenefit/promotionevidence.
Receipt reports/protected-template-paired-progress-20261031T0506.json.
Waitregisteredfullsixepoch/final5checksandshortgapcompletionnear13:14-13:25.

Expanded privilegedpastGTvelocitydiagnostic audit_past_gt_motion28.py ran:
34/35gaps,1047validframes,11videos. Exclude blkskirtwoman160 invalidpastGT.
Same-lastCVholdblindPR/SR22.897727/19.192145; measured27.642045/23.395788;
pastGTvelocity34.668561/25.305961. PrivilegedSRvsmeasured+1.910173 but
6wins1tie4losses, notupperbound/uniformsolution/closedloopresult.
Report reports/blind-motion-next16-past-gt-audit-20261031.json.
At13:08:27short54/64tracksfinishedandworkerslive; nofinalresultthen.

## 2026-10-31 13:19 UTC+08 - Short-gap result and availability history control

Shortgap64tracksverified. Holdoutage80.858060/65.136455 ->adjacentCV
81.693045/66.032080; paired+.834986/+.895625, CIs[-.679692,2.966112]/
[-.177558,2.483089]cross0. ReportSHA9a076d9b77cd9c1b0c81a9ecec9e649f2b6e72a6e77cc1850a987900b29649a9.
Post1020925finishedshortcontrast/estimateaudit, stillwaitingCPU869785.
CrossgapSR-effectdifference+.048866CI[-1.982117,1.833271], noestablished
durationadvantage; completebitwiseequalacrossdurations.

Newoffline availabilitymatchedvelocity1287validblindframes16videos:
adjacentPR/SR80.809015/65.654678 ->matched81.473186/66.201395.
SR+.546717CI[.170715,1.007952],12wins1tie3losses. Offlineonly;
sameavailabilityalsomeanslongertimeinterval,soconfoundnotisolated.
Motivatesmatchedclosedloopcontrol, notnovelmodule.

Archivedshortsourcesinshortoutput/verified_source beforedeployment.
Newstatehistorycontrol1044705started~13:17, children1044714/1044715
verifiedliveeach2266MiB; GPU0/1oneworker, alloriginaltrainerscontinue.
Outputreports/server28-state-motion-shortgap16-20261031.
ETA13:37-13:47; actualcomparisonwaiter1049221verifiedlive13:19:32,
300spolls thencomparison.json verifiesbothcompleteandholdoutagebitwise
equalityacrossruns. Do noteditactiveevaluator/controller/analyzerhashes.
docs/AVAILABILITY_VELOCITY_CONTROL_20261031.md containslimits/protocol.
Reused16videos, notfreshreplication/officialscore/validatedmodule.

## 2026-10-31 13:25 UTC+08 - Lifecycle scope verified

Actualfixed20configscheckpoint_epochs[10,20], evaluate_after_training28.py
usesonlyepoch20. They DO NOT performfinal5selection. Final5requirement
isimplementedbybatch10adaptivecandidatecampaigns, notallfourruns.
Noepoch16-19exist/willbesavedbyunchangedfixedconfigs;donotclaimotherwise.
Training/controllers/validation sourcehashescandidatematchlaunch;official
evaluatorknownprovenancechangealreadydocumented20261030audit.
Archivedexactlivefixedqueueevaluate_after_training28.py inGitHub without
changingservercopy, SHAb0f91914a219109f679424e3692c21ace5a66a16ac7730c6ea010783fba3f912.
docs/TRAINING_LIFECYCLE_AUDIT_20261031.md andreportreceiptclarifyallstages.
Futurecandidatefinal5eval/selection/cleanup/officialscoresstillpending.

## 2026-10-31 13:32 UTC+08 - SAMURAI source review

PinnedSAMURAI76ba195984892b0d1e3db5d9c9f90bb62175680a, inspected
modeling/sam2_base.py,utils/kalman_filter.py,rootApache2license.
MotionmeanIoU+predictedmaskqualityselectsmasks; filteredhistoricalmemory
stillforciblyincludeslatestframe. Notstricttrustedmemoryguarantee;
doesnotrestoremisingCADTrackcropcoverage. Filterdt1mustadvancevideotime,
notourneuralcallframe_idwhichskipsblindframes. DonotcopySAMquality
thresholdstohannresponseormislabelordinaryKF/memorygatingasnovelmodules.
Noexternalcode/weights/depsintegrated; docs/SAMURAI_SOURCE_REVIEW_20261031.md.

At13:32:10 CPU869785 completedsecondarm192updates, stilllivefinal
measurement/selection, nopairedcompleted/analysisyet. State1044705,
1044714/1044715 andcomparison1049221live. Preservequeues/sourcehashes.

## 2026-10-31 13:46 UTC+08 - Final development controls; replication

CPU paired384update pilot independently verified complete. Both arms select
epoch6, each retains one branch weight and deletes4 with receipts.
Complete loss4.363575024840732 vs mixed4.355297873883198 (-0.19%),
mixed cell effects, reused selection tuples: no larger GPU campaign justified.

Closed-loop availability-minus-adjacent PR/SR +1.806145/+1.659243pp,
95% paired video CI [0.114418,4.293027]/[0.372872,3.778192].
SR14wins2losses; one video contributes16.172161pp. Complete and outage
hold controls bitwise equal. Reused16videos, state/temporal baseline
confounded; no official or novelty claim. Final report/comparison and CPU
analysis archived under reports/. See docs/DEVELOPMENT_CONTROLS_FINAL_20261031.md.

Fresh development indices20..35 registered before final comparison:
8611frames, GPU0/1 4243/4368. Finished sources archived/hash-checked.
Replication1090396 started13:43:43; adjacent controller1090400,
workers1090441/1090444 verified live13:46, each2266MiB.
All4 main trainers verified live; batch10 utility epoch9/update16500,
random epoch9/update17050. Replication repeats both4-condition studies
on fresh16,128tracks total, then automatic paired comparison.
Log reports/server28-motion-replication16-20261031.log; outputs
reports/server28-motion-replication16-{adjacent,availability}-20261031.
300s controller polls. First arm expected14:04-14:14, final14:24-14:44
UTC+08. No restart after SSH observation timeouts.
Full target and three modules unproven; main final-five workflow pending.

## 2026-10-31 13:56 UTC+08 - Motion identifiability audit

Exact1290blindframes/study reconstruction verifies state-matched velocity
equals ordinarylag2 at every blindframe,maxerror0. Includes3invalidGTframes
excludedfromearlier1287accuracyrows. Fresh16replication cannot distinguish
state matching andlag2 under identical schedule. It remains running for
ordinary-rulegeneralization; no newmoduleclaim.
Audit:reports/motion-identifiability-20261031.json;
docs/MOTION_IDENTIFIABILITY_20261031.md.

CPU equal-lag probe runner1109528/worker1109529 launched13:55:35, runner
verifiedlive13:56with36realcalls22.98s. 576forwards, fixedsavedcomplete
cropgeometryandinitialtemplates, prior11/01/10 versus current11 atlag1/2.
NoGPUs,tuningorofficialdata. Doesnotisolateconsistencyfromcomplete-quality.
Outputreports/server28-fixed-crop-motion-factorial-20261031/logsibling.log,
automaticindependentanalysisafterprobe. ETA14:02-14:04;preservequeuedhashes.
FourmaintrainersandGPUreplication1090396/1090400/1090441/1090444
verifiedlive13:50. Noofficialthresholds/threevalidatedmodulesyet.

## 2026-10-31 14:07 UTC+08 - Equal-lag results and replication queue

576-call CPUprobe completed303.75s, independent384rowanalysispassed.
Originalcurrent11 lag2 reducesvelocityerror acrossall3priorstates;
current11vsprior10effectuncertain. 3x3posthocfactorialusesalready-saved
frame2/3etc outputs,64validtriplets,zeroaddedcalls. Same-statepriorbest
macrovelocityerrorineachcurrentstate. Current01/10 same-minus-completeprior
velocityerror -2.912099/-1.035978px/frame,95%videoCIbelowzero.
Thuscompletepriorqualityaloneinsufficienthere; no closedloop/officialclaim.
Detailsdocs/MOTION_IDENTIFIABILITY_20261031.md; reports/fixed-crop-motion-*.

OldCPUsourcesarchived/hashverifiedbeforeaddingpathargs. Freshfactorial
queue1129716launched14:07:20,verifiedlive14:07:46. Waitactual1090400
every300s, requirefreshadjacentcompletionhash, repeat576callsandanalyses.
Outputreports/server28-fixed-crop-motion-replication16-20261031,
logsibling.log. ETA14:18-14:25subjectreferencecompletion. Preservequeued
sources. No activeGPUexecutionchanges or newGPUallocations.
At14:07:46adjacentbothworkers32/32,controllerstilldoingfinalchecks;
availabilitynotyetstarted. All4maintrainerslive.

ReverifiedhistoricalA Miss68.565174/54.621376 and84.964353/60.511286,
onlyLasHeR-MissSRabovegoal. Centralresearchstatusupdatedsostalehistorical
liveentriescannotbeconfusedwithcurrentstate. Fullobjectiveunfinished.

At14:12:13adjacentfresh64-track reportfinal/verified. OutageholdPR/SR
79.572046/65.009895 vsadjacent90.016246/71.278588;large delta
+10.444200/+6.268693butCI[-.283139,25.678533]/[-.240559,15.576421]
cross0, SR10wins6losses;2videostaildominates,no stablegainclaim.
Availability1132262/workers1132272/1132273verifiedlive,started~14:08:47.
CPUfreshworker1138583+runner1129716live14:13:40,108/576calls65.01s;
expect14:18-14:22automaticanalysis. Batch10utility9/17000random9/17575.
DetailedfreshreportsandlimitsinMOTION_IDENTIFIABILITYdoc. Fullgoalactive.

## 2026-10-31 14:35 UTC+08 - Final replication and offset-control launch

FreshCPU576calls/338.26scomplete,analysesarchived. TIRstateeffectreplicates,
RGBsame-minus-completepriorvelocity -.104311CI[-.717432,.481283]uncertain.
FreshGPUlag2-adjacentPR/SR -2.712765/-1.301565pp,CIcross0; SR10wins1tie5losses,
fallenbikeitself -31.014006SRdominates. Controlsbitwiseequal. Do notpromote
oldpositive16mean. Finalfreshreportsinreports/motion-replication16-*.

Posthocpastoffsetauditlowersfresh01centererror3.204725pxbutfresh10
0.096662pxuncertain;notprospectivefornewcorrection. Builtstandalone
parameterfree offsettracker+native/calibration-sham/corrected control
onnewdev36..39 fourvideos2155frames24tracks;00hold,noblindvelocity.
Correctionappliedbeforeclip/templatewrite;extra2callsatfullframes.
CPU7-framebodywitnesspassedlimitedequality/frozenchecks; fullGPUnotyet.

Controller1173352started14:33:00;1173436/1173438live14:34:55each2266MiB,
besideoriginaltrainers;all4maintrainerslive. Assignment1334/821frames.
Outputreports/server28-state-offset-control4-20261031,logsibling.log.
300spolls,ETA14:55-15:15;do notedit4queuednewsources.
Registrationimmutablehash99d6eab430e203a13b2c4dfec5584a41d97a967423fb0c2e9fc4376caf0dbdb2.
Independentanalysisverifiescausaloffsettrace+allmetrics/equalities.
docs/STATE_OFFSET_CONTROL_20261031.md givesprotocolandlimitations.
Fullobjectiveunfinished,nonewofficielscores/threevalidatedmodules.

## 2026-10-31 14:56 UTC+08 - Offset control final and fresh16 launch

Four-video offset control completed24tracks, independently verified and
downloaded with matching final/worker/source hashes. FinalSHA
c522e6a898ebfc2f88a145bc41c9c317afc1c380efbddf397f1c36f35dccf32f.
Outage nativePR/SR76.109299/60.194491 ->corrected76.889868/60.844880,
delta+.780568/+.650389pp;95CI[-1.951681,4.105839]/[-1.360381,3.278505]
cross0,SR2wins2losses. Allcomplete/shambitwiseequal,pastoffsetsverified.
Notstablegain,officialresultorvalidatedmodule.

Archived4originalsourcesbeforeCLIcohort/outputchanges;trackerunchanged.
Freshregistereddev40..55,16videos13951frames96tracks,loads6968/6983.
Controller1212195started14:55:51;workers1212202/1212203actualrunning
14:56:10withGPUallocationbeside4trainers.
Outputreports/server28-state-offset-replication16-20261031,logsibling.log.
RegistrationSHA68d5a2fad5fb69663ddc03185a1061e7edd3f6aac3aecda412d795deab29f1f2.
Estimate16:30-17:00,300spolls;preserve4newsourcesuntilfinalanalysis.
Noalgorithm/parameterchanges,nosequencefiltering,6158-framevideo retained.
SourcescompileactualserverPython;final4reportsandregistration/launchinGit.

DSRTrackpinnedd9e23e08e5b98ede129e4512fc89220d68d307ca source review:
anchor-guidedrepair+volatilityqueuealreadyexist,notDCTrackcodeidentity
proof;no rootlicensefile,selectedheadersApache2only;coordinatecontract
concern(updaterwidth/heightvsevaluatorcornerboxes)static,notruntimeproof.
Noexternalcode/dependency/weightsintegrated;docDSRTRACK_SOURCE_REVIEW_20261031.
Candidateverification06:51:48Zutility9/17727random10/18303,finite,
sharedinputsidentical. Estimatednextboundaries15:06/16:20UTC+08.
NoLRdropconvergenceyet,earlieststop14;fullobjectiveunfinished.

## 2026-10-31 15:04 UTC+08 - Saved offset mechanism audit

NewCPUofflineaudit_offset_mechanism28.py completed,zero modelcalls.
FinalreportSHA44deca28fb1b641c06f930ff798e40f320e8dfd5528b81be6e8cfe461c95ea9c,
downloadhashmatches. Validatesinput/worker/trace/GThashes and recomputed
allpervideoPR/SR/framecounts against authoritative receipts1e-10.
Samecorrectedhistory immediate TIRcenter50.781617->48.854224px
delta-1.927394CI[-3.689640,-.383854],butPR/SRintervalscross0.
RGB47.453297->47.236450delta-.216847CIcross0,PR/SR-1.086957/-.182500pp.
WholetrajectorystateSRdeltas11/01/10/00=.567341/1.669953/-.932571/1.456278,
allintervalscross0. bikefromnorthwholegain4.396941despitebothimmediate
stateSRdeclines;cannotattributeallgaintoimmediatespatialcorrection.
Writedecisionschange4leftredflagframes/3bikeframes,nottemplatecausalityproof.
DocSTATE_OFFSET_MECHANISM_20261031.md. No parameter/state-gate changes.
At15:01:23actualall4trainers+1212195/1212202/1212203live;SSHtimeout
recoveredwithoutrestart. Utility9/17905random10/18485finite latestlosses.
Fresh16stillpending,nextmeaningfulfinal16:30-17:00;preserve4sources.

## 2026-10-31 15:23 UTC+08 - Size-error evidence and queued fresh geometry control

Offlineaudit_offset_geometry28.py reuses32fixedcropvideos,zero modelcalls.
ReportSHA9f556ba28a3258aa412c7ec3eab324262dbe8b694b680d035eabfe703967345f.
Pastsize-ratioaddedtocentercorrectionSRoriginal/freshTIR+6.324405/+7.589286,
CI[2.008929,12.276786]/[2.827381,13.467262],13/14winsoutof16.
RGB+1.413690/+.148810;freshCI[-.446429,.818452]uncertain.
Bothcohortsalreadyinspected,syntheticboxesunclipped,noclosedloopclaim.
DocSPATIAL_CALIBRATION_CONTROL_20261031.md.

Newspatial_calibration_tracker28.py extendsisolatedoffsetcontrolwith
pastcompleteframewidth/heightratios;scalesaroundrawcenterbeforeclip/
templatewrite,allothernativepoliciesunchanged. No fit/cap/gate/weights.
Fourarmsnative/sham/center/geometry,same3callsatcompleteincalibratingarms.
Seven-frameCPUtrackbodywitnesspassesnative/sham,completeall,
newcenter/oldcenterbitwiseequalandunchangednetwork;realnonidentityscales
exercised. NotGPUconstructor/fulltrajectoryproof.
WitnessSHA9fc2f23a9c7405556b10babc31236cc0675ff2cc27db23df047593753a7f7987.

Queuedcontroller1256793registered15:21:59andactualalive15:22:52waiting
actual1212195. Freshdev56..59,4videos836frames32tracks,loads396/440.
rightbhindbike-gai265/leftwhitebike199/whitegirlwithumbrella131/manbikecoming241.
Outputreports/server28-spatial-calibration4-20261031,logsibling.log.
RegistrationSHA2127191883035a6c3721a4de4a55651c8c8d5e6a3ea8e178c825035a0fd93251.
NOgeometryGPUworkersyet. Requirespriorfinalverifiedreport/free>=3000MiB,
then~10-20minplus300squeuedelay. Independentanalyzerchecksall32metrics,
pastoffset/ratio/identityprior/clip/holds/callcounts,controls/hashreceipts.
Preserve2newsourcesAND4activecenter16sources/sharedhelpersuntilanalyses.
At15:22:52all4trainers+1212195/1212202/1212203and1256793actualalive.
NoGPU2/3extrajob,notemperature/powerchecks. Fullobjectiveunfinished.

## 2026-10-31 15:36 UTC+08 - Primary-paper correction

Actual local DCTrack PDF SHA256:
ac899ca27264de01218dd7c86ee4b89dddbf218844dcf14d0697d672890cd715.
Withdraw unsupported DCTrack-to-DSRTrack repository identity and feature
disentanglement claims. Actual sections3.3-3.5 specify max-intensity>=5
availability, image substitution, IPT/MMCITrack/DAM4SAM instantiations,
past-state linear extrapolation blended with Kalman, and buffered replay.
The intermediate STD/SIAMTFA/DiMP/future-interpolation account is also
unsupported by these bytes. See docs/DCTRACK_PRIMARY_REVIEW_20261031.md.
Do not use generic routing or motion prediction as our novel module.

Four trainers and center16 workers verified live15:33:50; center16 parent
1212195 and queued geometry1256793 verified live15:36:34. No job restarted.
Latest batch10 utility/random updates18525/19150, both epoch10, logged
losses finite. No new validation/convergence/official threshold evidence.
Next meaningful random10 boundary~16:20, center16~16:30-17:00 UTC+08.
Registered sources untouched; geometry still awaits predecessor completion.
Full objective remains incomplete.

## 2026-10-31 Epoch9 paired-curve audit

Full live inspector evidence archived as
reports/candidate-health-epoch09-20261031.json, SHA256
911808ac87ed25b3fb813918146ef0ce14f4f4838708387957886ba734d31eff.
All recorded losses finite; paired inputs equal. Utility lower than random
only3/9 epochs. Epoch9 means4.979970/4.972420; utility state deltas11/01/10
-0.001977/-0.048274/+0.072901. Best proxy epochs8/3, no stable advantage.
Validation uses complete templates and cannot represent missing-history
closed-loop behavior. Doc docs/CANDIDATE_EPOCH09_CURVE_AUDIT_20261031.md.
Actual live15:39:20 all4trainers, bothcandidatecontrollers, fixedevaluator,
center16parent/workers and queuedgeometry. Inspector utility/random
updates18618/19225epoch10; projected boundaries16:54/16:20 UTC+08.
Postdrop reference resets11; earliest patience stop14. Controller's final5
full97 selection/cleanup/official stages verified in source, not yet executed.
No active sources changed. Official attainment and three modules incomplete.

## 2026-10-31 Calibration age coverage gap

New CPU-only official-mask audit: reports/calibration-age-20261031.json
SHA4cda9fa314e59d52ddc2264c7bd181fddb97b979d6ab6499d2caf4122da8a8d5.
Both masks match frozen inventory; independent vectorized recomputation
matches147248calibrated single frames. No GT/model calls/GPU or fitting.
Official medianvideoage LasHeR01/10=493/279,RGBT234=141/128;~94%age>1.
Block1development age always1 after calibration. Beforefirst calibration:
LasHeR01/10=954/699frames,RGBT234=96/815. Rawmask0 notalways11 but actual
evaluator initializes bothimages, doesnotcalibrate, begins inferenceframe1.
Framepooledstats canbedominated by longvideos;pervideocounts retained.
Doc docs/CALIBRATION_AGE_AUDIT_20261031.md/source audit_calibration_age28.py.
Next promotion now requires independentlonggapclosedloopcontrol even if
current center16/geometry4positive. Do not infer stale-cache failure or add
decay/expiry/gates from distributionalone. Preserve currentregisteredjobs.
15:49:26UTC+08 all4trainers+1212195/1212202/1212203/1256793actualalive.
No finalcenter16/geometryresult yet. Next random10~16:20+08. Goalincomplete.

## 2026-10-31 16:00 UTC+08 - Long-gap control queued

New isolated run_calibration_gap28.py + launch_calibration_gap28.py.
Same frozen E0/native/sham/center/geometry on freshdev60..63,4videos2384frames:
boyaftertworunboys458,twopeople_0109 450,10rightblackboy333,leftbottle1143.
Complete/block1/block32/block128,64tracks,loads1143/1241GPU0/1. No gates,
decay/expiry/weight changes. Gap interaction not isolated cache-age causality.
Short333video hasnoRGBonlyblock128;only1143video reachesblindblock128.
CPU7framebodywitness exit0 equality/frozenpass;block32/128first7allcomplete
so NOT longgap/CUDA/fulltrackproof. WitnessSHA
050a54f4f770ce7a638b9adca160a0c7e1e3c4077e80c9bd9964a63e22a3dd7e.
Controller1320176registered16:00:19andactuallylive16:00:45waiting1256793.
RegistrationSHAe9260c3a032c2bc3395904d974066fb479639318020707a18aeee1b150f7a3b7.
NewsourceSHA9c1ae50b7a9efbad10d83480fe2e8db786d6d1a7512280c4a55ec5b53a56c06e.
No gapGPUworkeryet. Requires geometry4finalhash/controls thenfree3000MiB
eachGPU0/1;poll300s,estimate60-120minafterrelease. Outputreports/
server28-calibration-gap4-20261031,siblinglog. All4trainers+center16workers+
geometryqueueactuallylive16:00:45. Do notmodifynewsource/shareddependencies
untilanalysis;originalactive sourcesunchanged. SeeCALIBRATION_GAP_CONTROL doc.
Nextmeaningfulrandom10~16:20,center16~16:30-17:00,utility10~16:54 UTC+08.
Officialtargets,final5selection andthreevalidatedmodulesremainincomplete.

## 2026-10-31 17:00 UTC+08 - IMPORTANT: reboot and recovery

Original processes ALL TERMINATED after actual repeated reboots16:24/16:28.
Current boot8212e607-b051-467d-a468-86436b0b702e, unchangedthrough16:59.
CPUverified4checkpoints restored in independent runs/server28-recovery-20261031.
Controllers49952/49953/49955/49956;trainers50004/49999/50013/50008 GPU0/1/2/3.
Fixedbothfromepoch10,Utilitycandidatefrom9,Randomcandidatefrom10. Actual
updateshundredsafterrestoration16:59,finite losses,VRAM19346/19346/22732/22806.
OriginalRNGnotstored,NOTbitwisecontinuation;newmatchedepochseedstreams.
Originalfilespreserved;rolling/finalcleanupremoveonlyrecoverylinks,global
oldnonselectedweightcleanupstillpending. All4nowgetfinal5full97selection
andthenofficialscores. Fixedruns20/candidatespostdroppatienceunchanged.

Newrandom10validationmean4.9969308004(states4.2394807264/5.9841034710/
4.7672082037),worse9mean4.9724204093;Utility10interrupted,notcompleted.
Recoverydocdocs/REBOOT_RECOVERY_20261031.md;checkpointaudit/recoverylaunch
archivedreports. Firstauditbecame0bytesaftersecondreboot;replacementfsyncs.
Noassertedcause/no temperaturepowerqueryorchange.

Evalrecovery72348workers72352/72353live17:00:21. Reusehash/metricsverified
complete savedtracks,unfinishedvideosrestartfrominitialization;oldpartial
outputsunchanged. Newcenteroutputserver28-state-offset-replication16-recovered-20261031.
Thenregisteredgeometry4/gap4new-recovereddirs;unchangedoriginalsources.
Preserveallold/newregisteredsourcesuntilanalyses. No finalresultyet.
Nextcandidateepochboundaries~18:15-18:40UTC+08subjecttoactualspeed.
Fullobjective/threevalidatedmodules/officialtargetsremainincomplete.

Recovery reverified17:08: all four controllers/trainers and evaluation
controller72348/workers72352/72353 alive. At17:09 center16 has82/96
complete tracks (GPU0 24/30, GPU1 58/66; video counts intentionally unequal
to balance frames). Includes hash/metric-verified reused tracks and fresh
whitegirl2 outage. Long leftconergirl6158 corrected tracks still pending.
Estimated center final17:45-18:15 + up to300s queue wakeup, provisional.
Detailed recent metrics docs/LATEST_TRAINING_METRICS_20261031.md;
candidate Random10 worse than9, no convergence or official new score.

## 2026-10-31 17:22 UTC+08 - Compensation versus selection intervention

New isolated CPU4thread audit101337 launched~17:18; analyzer queue105825
uses300s polling. Both actual alive17:21:40. Utility16/128 samples completed
all4arms/3states in185.28s, native tail bitwise witnesses passed. No result
claim yet. Two matched candidate epoch6/update12000 checkpoints, because
Random5 was already removed by rolling retention; different from epoch5
full97 tracking. Exact existing128 cached inputs SHA13105d... reverified.
Same pre-repair features/context within each model: native4/off/random4/all16;
full templates/both searches/states11/01/10, no fitting/no weight writes.
Independent analyzer checks receipts/hashes/shapes/random draws/means;
producer checks unchanged full model state after restoring selector RNG.
No PR/SR, novel-module or retrained-ablation claim. Between-model effects
also include jointly trained trunk/correction. Doc
docs/CANDIDATE_REPAIR_INTERVENTION_20261031.md; launch/queue receipts inreports.
Keep4new audit sources/registered runtime unchanged. Expected~18:05-18:15
full CPU result; current GPU sources/queues unchanged. Goal incomplete.

## 2026-10-31 17:35 UTC+08 - Independent final-result verification queued

New read-only verify_recovery_campaign28.py checks970 final5development
tracks, common validation-selected checkpoint, four removed recovery links,
958 official complete/Miss tracks, source/GT/mask hashes and independently
recomputed scores/curves. Reports baseline+2pp deltas and strict Miss targets.
No original weight deletion or three-module/convergence certification.
Metric helper actual witness: existing full97 epoch5complete79.596578/64.472777
and historical479 officialMiss68.565174/63.576593/54.621376,
84.964353/60.511286 reproduced. Final5end-to-end witness still future.
Launch17:34:41; actual17:35:18 waiting128343/128344/128345/128346
for fixedU/fixedR/batch10U/batch10R, poll300s on real controller cmdlines.
Doc docs/RECOVERY_RESULT_VERIFICATION_20261031.md, queue receipt inreports.
Preserve verifier/launcher/sharedhelpers hashes until all finish.
GPU1 center16completed66/66 withreceipt;GPU0 24/30stilllongleftconergirl,
overall90/96 at17:29. All4trainers andCPU101337 live17:27,
CPUUtility48/128 at552.61s. No finalcenter/CPU/officialnewresult.

## 2026-10-31 17:52 UTC+08 - Center16 complete, geometry actually running

All96 tracks independently verified including6158leftconergirl; exact
complete/sham/past-only controls passed. Report246af77cac7b3272aa5adff9599839e75612250b9d5eed263f4203c3bf233fab
matches server/download/completion. Outage nativePR/SR86.550590/71.114156,
corrected87.301072/71.472439;delta+.750482/+.358282,95intervals
[-.140864,1.691871]/[-.326380,1.071918],SR11wins5losses. Notstablebenefit,
notofficialscore/module. Complete90.656936/73.839150 identicalallarms.
Unchangedmechanismaudit zero calls reporta5f42c576ccf5afec3bbdf35f7aee7f071d3efd0aa6107dc5a09f59befd2e748
downloadmatch. ImmediateTIRSR+1.256158CI[.063631,2.555410],RGBSR+.162236
CIcross0;RGBPR+2.160315CI[.142191,5.306046]. Posthocnotstate-gateproof.
14/16templatewritedecisionschange,notisolatedcausaltemplatemechanism.
Docdocs/STATE_OFFSET_REPLICATION16_RESULT_20261031.md;reportsarchived.

Geometryworkers151970/151971launch~17:49:44,actualalive17:51:25 and
fullnativeTRACKreceiptsalreadywritten. Originalregistration/cohort/source
unchanged. Outputserver28-spatial-calibration4-20261031-recovered;then
registeredgap4unchanged. ETAgeometry18:00-18:15+300swakeup,provisional.
All4trainers+CPU101337/105825live17:52:16. CPUUtility128done1428s,
Randomongoing;nofullpairedresult. Preserveallregisteredsources.
BoundedPMAalignmentcodepriorartreview recordsimport/package/license
uncertainty,notrunnabletransplant/nointegration;FlexTrackwrapperredirect
andPAFCabstractonlyscopeindocs/CALIBRATION_PRIOR_ART_BOUNDARY_20261031.md.
Officialthresholds/3modules/final5/globalweightcleanupremainincomplete.

## 2026-10-31 18:04 UTC+08 - Geometry completed, gap workers running

All32 fresh geometry trajectories independently verified; report
96be9ee99255d28ee98ccafb01f844c82cc49666ec074d0676fd8f2a1046242e
matches server/completion/download. Complete allarms94.507576/77.629789.
Outage native93.983586/75.107497,center93.226010/74.745948,
geometry93.068182/75.233551. Geometry-native PR/SR-.915404/+.126055pp,
95intervals[-1.893939,0]/[-.117244,.394120];SR2wins2losses.
Geometry-centerSR+.487603CI[.082036,1.199495],4wins, but not demonstrated
gain over native. Do not promote as a validated module or official result.
Docdocs/SPATIAL_CALIBRATION_RESULT_20261031.md; final/completion archived.
Controller72348advanced to gap4; actualworkers168995/168996alive18:01:45.
Outputreports/server28-calibration-gap4-20261031-recovered,64tracks across
fresh4videos2384frames and4arms xcomplete/block1/32/128. Launch links
verifiedgeometrySHA. No source/parameterchanges. Gap changes observation/
templatehistories, not isolated causal cacheage. Keep short/unexposedstates.
Estimate~19:00-20:00+08 plus300squeue fromoriginal60-120minbudget,
reviseusingactualthroughput. Fourtrainers+CPU101337/105825alive18:01:10.
CPUfullpairpending. Latestcomplete usermetrics snapshot17:59:33 updated
locally; candidatesUtility9/Random10cachedvalidation unchanged. No new
currentcampaignofficialmetrics or convergence claim. Fullgoal incomplete.

## 2026-10-31 18:08 UTC+08 - CPU full pair completed and corrected verification passed

Producer101337 completedbothmodels128tuples in2760.96s at18:04.
Original analyzer failed: hardcoded2searches versus actual frozenYAML3,
templates2. Producer didprocessall3; original launch "bothsearchframes"
wording incorrect. Original sources/report/failurelog preserved. New
analyze_candidate_repair_corrected28.py reads hashverifiedYAML3search/2template,
checksall3searches/randomdraws/source/checkpoint/means and exited0~18:07.
ReportSHA070f5f10f4147e6c036442d3cc7d80b4c3e93865b60e51a7103dcf215666d939;
analysisSHA0631845222c32de202b59f1f054185111af4afc1e5f18b56dc627eaf1571fa9b.
Completion/download/codehashesmatch. OriginalCPUproducer/queue areterminal.
Docdocs/CANDIDATE_REPAIR_INTERVENTION_20261031.md now fullresult+correction.
Utility native-off11/01/10lossdelta+.000105/-.000058/-.007465(~.156%RGB);
RGB62wins66losses. native-random4RGB-.006720but59wins69losses.
All16notuniformlybetter. U-R native+.143197/-.027140/-.098227,off
+.143354/-.026861/-.091905; cannotattributejointmodeldifferences toselector.
NotPRSR/retrainedablation/official/novelmodule. No activeparameterchanges.
Gap64running; final5/official/3modules/convergence/globalcleanupincomplete.

## 2026-10-31 18:19 UTC+08 - New validation and original-weight cleanup queue

RecoveredUtility10validationmean5.026235371828079,states11/01/10=
4.387995824/5.847460255/4.843250036, worseUtility9. Random11mean
5.032911457121372,states4.348451704/5.879302219/4.870980449,worseRandom10.
No convergencedeclaration;epoch11postdropreset/earliestplateaustop14.
DownloadedfullhistoriesSHAa54d7580.../c3222577...matchserver. Latestmetricsdoc
now includesUtility10/Random11,trainingupdate snapshotstill17:59:33.

Newcleanup_original_weights28.py actualfixturewitnessremoved4temporary
files,preserved4selected+1unrelated,source2685ec98c14d09d565709e1a8aa476c8b6f202fdf68946eb843e7c96c5acf7ab.
Queue197828registered18:17:15andactualalive18:17:57;300spolls,CPUonly.
Registration3cc1a95c53ed920f018ed06cd677f33ee3870ab85b644d1b2e5f2e7f65e1ea0b
captures12originalepochweights fromfourcampaigns. Waitsallfourfinal
independentverifications,checksselectedcommonweights+CPUauditcompletion,
thenexactpath/size/hashinventorybeforedeletion. Keeps4selectedrecoveryweights;
doesnotselectontest,claimtargets/modules,orcleanunrelatedexperiments.
Approvedplan/perfilefsyncedjournal/completionreceipts;realdeletionpending.
Originalverifiersreadonlyreportsremainunchanged;newcleanupreceiptseparate.
Docdocs/ORIGINAL_WEIGHT_CLEANUP_20261031.md. Preserveallsourcesandregistered
originalpaths; donotrepurposethemfornewjobswhilepending. Fullgoalactive.

At18:20:59 gap35/64fulltracks:GPU0 9/16 and17396/33776registeredcalls
infinishedtracks;GPU1 26/48 and17780/37544. Actualelapsed~21min projects
final18:40-18:55+08includingparent300sdelay,reviseearlier19:00-20:00.
CandidateU11/update20181,R12/22222;nextvalidation~19:30-19:50+08.
Do notpollrepeatedlybeforethosemilestones;allregisteredvideosretained.

## 2026-10-31 18:34 UTC+08 - Observed-copy ordinary control running on CPU

Newaudit_observed_copy28.py compiledPython3.7andPID220773launched;
registration18:31:17actualalive18:31:53at360%CPU/RSS~1GiB. Nofinalresult.
FrozenE0,nooptimization/GPUallocation;seed1702,batch4,16tuplesfromexisting
705trainpopulation(disjoint177dev/97heldout),valmodetransforms. Initial
templatecomplete,dynamic/search11/01/10ninecells,all3searches. Twoarms:
nativeorcopyotherobservedstreamfeaturesonlytounavailablepositionsatblock6.
Nosyntheticcompletefeatures/teacher/learnedgate/GTselection. GTconstructs
ordinaryofflinecropsandscoresloss;notclosedloop. Checkfullhistorybitwise
equality,unchangedobservedpositionsatintervention,frozenmodel,finiteoutputs.
Separatearithmetic/token-count/source/hashanalysisafter18fullarm-cells.
NotPRSR/novelmoduleevidence,andnocurrenttraining/registeredGPUsourcechanges.
Source13ee19f1f82b0d8966c0ab2c2a63039a4b56063c25d1fa470231d2294127b431;
registrationcdf8f129b1352a3e9b4127dca2d1a2ef591dfdbce566ebc690f3eb90ec4de40d;
input52a5ac3b48bea5224dfbf46f53276b718041e1159fd889d1d5a9be0c37a5c7c9.
Outputreports/server28-observed-copy-20261031,docOBSERVED_COPY_CONTROL.
Preservesourceuntilfinalanalysis;provisional18:36-18:46thenrefinefirstbatch.
READMEtopnowexplicitOct31recovery/newmetricsinsteadstaleepoch1snapshot.
Gapfinal18:40-18:55,candidatevalidation19:30-19:50+08;goalactive.

## 2026-10-31 19:05 UTC+08 - Completed controls and mechanism evidence

Observed-copy CPU completed360.60s,18arm-cells/16tuples/all3searches.
Six of eight noncomplete cells worsen; reject unconditional observed copy.
Frozen-state/observed-position/complete equality passed.
Report014156689a47ab34cc750fc55e4b9caa9215019fb30daaadab8a42790658e229;
analysis188752b5ab8326a1c214b4865650165ed224d0b1ded7d9fa075e689514d51a43.
Downloaded hashes equal completion receipts. No PR/SR/module claim.

Gap controller72348/workers168995/168996 terminal at18:46:36, final64/64
verified. Report2ff1cb19cb250aeb4cc6f98d6b440de1b6f32633c246206cfb23375617c61ebd.
Geometry-native SR block1/32/128 +4.083916/+.949828/-4.805974;
intervals[-1.532030,11.896272]/[.493879,1.251953]/[-12.499397,2.405965].
Block128 history/exposure differs; not isolated cache-age effect.
New audit_gap_mechanism28.py executed zero model calls/all64traces:
reportae238e608c5f05739012810f542a529161d9e1375e388645a48e3960f359c8d3.
Same geometry-history block128TIR center-raw SR-1.046317,
scale addition+2.264695; RGB-.313900/+.591450,3exposed videos.
Cannot attribute whole regression to scale staleness alone; no gate selected.
Details/full tables in CALIBRATION_GAP_CONTROL_20261031.md.

CMNeXt official DELIVER pinnedb3a76c33a7b22098b7285d845f22c910abc5cb16
source/blob/license review completed, no integration/execution. Sigmoid
residual weighting then elementwise max, not modality argmax; FRM updates
both streams; no inspected availability-token/complete-identity contract.
DocCMNEXT_SOURCE_REVIEW_20261031.md; ordinary selection/fusion prior art.

Latest metrics snapshot18:44:59: fixedU11/27499,R12/27593,
candidateU11/20740,R12/22791; all recorded five loss fields finite.
New fixedRandom11validationmean4.965231 states4.222112/5.883677/4.789905.
No convergence/current official results/final5selection/three modules proven.
Candidate next validation~19:30-19:50+08. Original cleanup197828live18:46:36;
registered12originalweights protected pending independent final checks.

19:07:02 verified all4trainer cmdlines,4verifiers128343..128346 and cleanup197828
live. FixedU12/28072,R12/28169,candidateU11/21249,R12/23306.
New fixedUtility11validation4.894813754906257,states11/01/10=
4.155396767/5.794388473/4.734656025; fixedRandom11mean4.965231249729793.
These are batch8 proxies, not batch10 PRSR/convergence. Download/server
history hashes match: U47a29672de633aad726b4ed8072a037d37405b9ad28e28bc074f5b64349fd4ef,
R4df5c6adfbaa9eff8bf9b2a1b3138cf8d9b4196064ef4dd6f2331f999cc72d41.
Next candidate boundaries approximately19:37-19:45+08 using latest update rate.

## 2026-10-31 19:20 UTC+08 - Prospective scale-only component control

Newrun_scale_only28.py follows completedgapentrypoint without modifying
sharedtracker/runtime/registeredpredecessors. Dev64..67sequential4videos881frames,
472/409GPU0/1. Native/sham/scale-only/center+scale xcomplete/block32/block128
=48tracks. Allvideos<384frames: block128onlyTIR, noRGB/blindcoverage,
keepallvideos/nooutcomefilter. No learned gate/expiry/weights/GTcorrection.
CPUactualbodywitnesspassed7framescomplete/block1,2nonidentityscaleframes,
sham/completeequality/zerooffset/frozenstate;notCUDAfullevaluation.
Witness0d3fb33841f505c8b1340340c82336e3be75f5cad9cb48af08dfbc2cd1ec6d05;
sourceacf6a56e84cbfdfa406ef7904664627ea52e899f3933e37e25afe2de2e182086;
registrationc8335c3a96ba57e2ad3fbc3a96c3f764e15f64304095af0f55079c358a51ed10.
Controller268017launched19:18:14actualalive19:18:33;children268025/268026
actualGPUprocesses2124/2080MiB besideall4trainers, nofulltrackreceiptyet.
Outputreports/server28-scale-only4-20261031,sibling.log.
ETA19:30-19:45+08thenrefineactualcalls;300spollsnearcompletion.
DocSCALE_ONLY_CONTROL_20261031.md. Preserve newsource/shareddependencies
untilfinal48traceverification. Notofficial/modulebenefit/convergence.

## 2026-10-31 19:40 UTC+08 - Completed component control

Scale-only control finished48trajectories, report SHA256
df6a9be1f1ce6ec5dad1b81b980357b1b639f29be7c3fc3d5de0d8ca064b9cf7.
Native/sham, scale-only, geometry PR20/SR:
complete75.146496/57.352125 for all;
block32 57.257201/44.689179,52.145743/39.656658,58.213389/44.977624;
block128 57.049005/44.075706,57.443327/43.987086,57.891059/45.397069.
Reject unconditional center removal. Block128 has TIR-only missing
exposure, no RGB-only/blind exposure. Full hash/metric/past-only/control
checks passed; frozen state is a worker witness, not independent replay.
See docs/SCALE_ONLY_CONTROL_20261031.md for paired intervals and limits.

Template pilot primary protocol now records completed192updates per arm,
selectedepoch6/final5, one retained/four removed each, mean improvement
only0.19%,3of6missing-search cells improve. Formal A already samples
dynamic-template availability; no template omission justifies a live patch.
ViPT source review pins f49f30186ebff5587600a61ff224bb341c0a3243,
documents prompt prior art/availability limits; no code integrated/executed.

At19:39:32 four trainers50004/49999/50013/50008, four final verifiers
128343..128346 and cleanup197828 were actually live. All4power limits250W;
temperatures85/80/81/77C. No power/temperature operation executed by agent.
At19:40:06 fixedU/R12 updates28833/28938; candidateU11/22000,
candidateR13/24065. All recorded task/loss/update means finite.
CandidateRandom12 proxy5.03791772077481 (11/01/10:
4.340420678257942/5.873122118413448/4.900210365653038).
Utility11 validation pending in this snapshot. Final5/common-weight
official results, cleanup and three validated modules remain pending.

19:43 update: Utility11 validation now complete, mean4.984900663296382,
states11/01/10=4.355199903249741/5.744167685508728/4.855334401130676.
Raw Utility11 and Random12 histories are published with hashes in latest
metrics document. This supersedes the pending Utility11 snapshot below.

## 2026-10-31 20:08 UTC+08 - State-template control launched

New isolated state_template_tracker28.py/run_state_template28.py/
launch_state_template28.py, frozen runtime unchanged. Ordinary state-cache
reading, native writes retained, all caches truthful complete initialization.
Native/sham/matched x complete/block32 =48trajectories on sequential
dev68..75,8398frames including6024-frame girlshead; loads6024/2374.
No duration/outcome filtering, fitted gate, extra forward or weight updates.
CPU21calls/condition passed native/sham boxes/confidence, complete matched
boxes, frozen digest;14writes/15differing sources. Witness17e16924...,
tracker8f3c0b48...,runnera4bbf199...,registration7b86360a....
Controller342664 launched20:07:20; at20:08:13 actual alive, workers
342898/342899 each2266MiB beside all4trainers. No full-track receipts yet.
Final analyzer checks48metrics/hashes/past-only truthful banks/native writes/
blind holds/controls and current11 subgroup. Not isolated age causality,
novelty, official gain or independent network replay. ETA20:55-21:40+08,
refine long native receipt;300spolls near completion. Preserve registered
new sources and dependencies until analysis. Goal remains incomplete.

## 2026-10-31 20:22 UTC+08 - Repair trajectory control queued

New run_repair_trajectory28.py / launch_repair_trajectory28.py queued behind
actual state-template controller342664. Retained epoch6 Utility/Random,
fresh dev76..83 (7059frames, long higherthwartbottle_quezhen4221),
native/sham/compensation-off x complete/block32 =96trajectories. Off returns
exact pre-compensation tensor objects while selection/RNG/overhead remains.
CPU7-call/model/arm witness verifies sham/native, off/disabled-module,
RNG restoration and frozen digest. Witness
e9f3bb9d5125e6dcdecde3700a2ef5bd346aa4213d44c5799fb319dc6406dbbb0;
source1ebb46e1f68d66af53d0d4ffccd8d0116a3db5735dc36044827a4b28a1458aa3.
Queue controller366251 registered20:21:15; no GPU workers until prior
48-track report/hash passes, its controller exits, and >=3000MiB free on
GPU0/1. Original epoch6 weights and registered sources remain protected.
This is inference intervention, not matched retraining/module causal proof.

Published8c6d7c7947f813ffef1be4086a7108ce56bc2e92 to
origin/codex/server28-research with exact remote HEAD. Repair control CPU
witness passed and controller366251 is registered behind actual
state-template controller342664; no second GPU job starts early.
At20:22 the preceding state-template worker command lines342898/342899
were verified actual. Preserve both experiment source sets and registered
epoch6 checkpoints until their respective analyses complete.

Correction20:38 UTC+08: first queue PID366251 failed before GPU allocation
because witness used relative `__file__` and launcher absolute path. Failure
log/receipt retained. Corrected absolute-path witness passes, and retry
controller393144 is alive/waiting on342664; no retry GPU worker started.

## 2026-10-31 20:57 UTC+08 - Template control complete

All48 trajectories independently analyzed; report SHA
66b3b776e0152f01bf0115e322e371ca7e15ec0c7b1576e1f8550f11300ad3e9 matches
completion receipt. Native/sham block32 PR20/SR80.072518/65.698845;
state-matched73.127902/59.989876. Matched-native SR-5.708969pp,
95% paired-video interval[-14.650757,-0.689463],1win/1tie/6losses.
Complete all three arms exactly92.123743/74.997601. Reject state-matched
policy; no gate fit or contribution claim.
Queue retry controller393144 verified the predecessor report and started
workers418802/418803 on GPU0/1 at20:52. At20:56:32 all3 were live beside
four trainers. Repair trajectory comparison remains in progress.
Fixed12 task-loss validations are recorded in latest metrics; no tracking
or convergence conclusion follows. Preserve all registered weights.

At21:06:50 controller393144 and workers418802/418803 remained live.
GPU0 completed two required long-video tracks, Utility native on
higherthwartbottle_quezhen: complete PR20/SR42.393365/27.807493 and block32
7.274882/8.803882. GPU1 had34/96 receipts with Utility sham/off partial;
these do not form a complete paired contrast. Four trainers remained alive.
No partial arm or official claim; continue on live process handles.

## 2026-10-31 21:47 UTC+08 - Repair Control Complete

Repair controller393144 completed and exited normally. All96 fresh-
development trajectories were independently checked; report SHA
`73c5e905bc944f27ba5da274284fb3c70808037d64403c324b54a0c56adf37a5`
matches the downloaded report and server completion receipt. Utility
native-minus-off PR20/SR was+0.462/+0.338pp on complete and+0.822/+0.538pp
on block32 outage; all four paired95% intervals cross zero. Random
native-minus-off showed complete-input harm and a high-variance outage
contrast. This is inconclusive Utility-module evidence, not official
benchmark gain or a paper contribution. Detailed results:
`docs/REPAIR_TRAJECTORY_CONTROL_20261031.md`.

At21:56:25+08 all four training command lines remained live. Fixed
Utility/Random were epoch13 updates31525/31825; batch10 Utility was epoch13
update24575 and Random epoch14 update26700. Latest completed candidate
validation was Random epoch13 mean5.001639 over128 cached tuples; this is
not a tracking metric or convergence proof. Final-five selection, official
LasHeR/RGBT234 complete/Miss evaluation, and weight cleanup remain pending.
Preserve registered weights and active source/receipt files.

## 2026-10-31 22:25 UTC+08 - Training progress

At22:25:21+08 the four trainers, four final-validation waiters and original-
weight cleanup waiter were live. Current updates were fixed Utility
epoch13/32225, fixed Random epoch13/32500, batch10 Utility epoch13/25125,
and batch10 Random epoch14/27250. Latest completed cached128-tuple mean
task losses were4.899094(epoch12),4.945365(epoch13),5.004183(epoch12),
and5.001639(epoch13), respectively. These are not PR/SR or convergence
evidence. Random epoch13 improved over its prior cached validation in both
campaigns. Utility log tails had not advanced since22:20, but the processes
remained live at~101% CPU while GPUs0/2 were100% utilized; do not restart
based on log age alone. Official complete/Miss scores and final-five
selected weights remain pending.

GPU0-3 power-limit telemetry was250W; temperatures were83/78/78/76C at
22:25:21. This snapshot does not prove an85C temperature target was set.
Preserve registered weights until their final-five evaluation and cleanup
protocols complete.

## 2026-10-31 22:37 UTC+08 - Training progress

At22:37:03+08 all four trainers, four final-validation waiters, and the
original-weight cleanup waiter remained live. Current updates were Fixed
Utility epoch14/32525, Fixed Random epoch14/32800, Batch10 Utility
epoch13/25350, and Batch10 Random epoch14/27475.

Fixed Utility completed cached epoch13 validation with losses11/01/10
4.259495/5.816765/4.654164, mean4.910141, stale_epochs2, stop=false.
This is0.011047 higher than epoch12 and is not PR/SR. The latest completed
validations for Fixed Random, Batch10 Utility and Batch10 Random remain
epoch13/4.945365, epoch12/5.004183, and epoch13/5.001639, respectively.
These are cached128-tuple mean losses, not convergence evidence. No current-
campaign official complete/Miss scores or final-five selected checkpoint
exist yet.

GPU telemetry at22:37:03 was83/80/78/75C with250W power limits on GPUs0-3;
utilization was100/100/94/100%. This does not verify the85C target setting.
Keep registered weights until the current validation and cleanup gates end.

## 2026-10-31 23:04 UTC+08 - Training progress

At23:04:51+08 all four trainers, four final-validation waiters, and the
original-weight cleanup waiter remained live. Current updates were Fixed
Utility epoch14/33200, Fixed Random epoch14/33475, Batch10 Utility
epoch13/25875, and Batch10 Random epoch15/28025.

Batch10 Random completed epoch14 cached validation over128 tuples:
losses11/01/10=4.316951/5.781932/4.812072, mean4.970318,
stale_epochs0, stop=false. Its mean improved0.031321 versus epoch13.
Latest completed validations for Fixed Utility, Fixed Random and Batch10
Utility remain epoch13/4.910141, epoch13/4.945365 and epoch12/5.004183.
These are task-loss proxies, not PR/SR or convergence proof. No current-
campaign official complete/Miss scores or final-five selected checkpoint
exist yet.

GPU0-3 telemetry was82/79/78/74C with250W power limits and utilization
100/100/100/72%. This does not verify an85C temperature target. Preserve
registered weights until the existing evaluation and cleanup gates finish.

## 2026-10-31 23:14 UTC+08 - Training progress

At23:14:39+08 all four trainers, four final-validation waiters, and the
original-weight cleanup waiter were live. Current updates: Fixed Utility
epoch14/33450, Fixed Random epoch14/33700, Batch10 Utility epoch14/26050,
Batch10 Random epoch15/28200.

Batch10 Utility completed epoch13 cached validation over128 tuples:
losses11/01/10=4.350545/5.756105/4.868650, mean4.991767,
stale_epochs2, stop=false. This improved0.012416 versus epoch12/5.004183
but remains0.006866 above its earlier epoch11 minimum4.984901. Latest
completed validations for Fixed Utility, Fixed Random and Batch10 Random
remain epoch13/4.910141, epoch13/4.945365 and epoch14/4.970318.
These are task-loss proxies, not PR/SR or convergence evidence. Current-run
official complete/Miss scores and final-five checkpoint selection remain
pending.

GPU0-3 temperatures were83/80/78/76C, power-limit readings250W, and
utilization100/100/100/91%. This does not verify an85C target setting.
Preserve registered weights until evaluation and cleanup gates complete.

## 2026-11-01 13:41 CST - RGBT234-Miss gap and official queues

Recorded the target arithmetic and existing-control evidence in
`docs/RGBT234_MISS_GAP_ANALYSIS_20261101.md`. The requested RGBT234-Miss
thresholds exceed original CADTrack E0 by +5.305185 MPR/+4.725565 MSR;
Batch10 Utility epoch13 is +2.108635/+1.082522 over E0 and nearly ties
historical A. This is a persistent unmet target, not a demonstrated regression
from epoch13. Template U1 helps versus U0, while all-BN freezing does not
close the gap; neither identifies a sole cause.

At 13:41 CST four official evaluators remained live: Batch10 Random/Utility
epoch14 complete had 30/479 and 35/479 sequence receipts; Fixed A
Random/Utility epoch16 miss had 48/479 and 42/479. These counts are progress,
not metrics. Preserve weights, do not run val97, and publish no result until
the complete 479-sequence/337,353-frame official receipt and aggregate
verification are present. Final-five test selection must be labeled as
selection-biased.

At 13:52 CST a verified live-process check showed continued receipt progress:
Batch10 Random/Utility epoch14 complete had 50/479 and 57/479; Fixed A
Random/Utility epoch16 miss had 80/479 and 71/479. No new full metrics or
completion receipts were present.

At 13:57 CST all four evaluators were still alive and sequence receipts had
advanced to Batch10 Random/Utility epoch14 complete 63/479 and 72/479, and
Fixed A Random/Utility epoch16 miss 100/479 and 89/479. No new aggregate
metrics or completion receipts were present.

At 14:03 CST all four evaluator processes remained alive. Batch10 Random and
Utility epoch14 complete had 76/479 and 87/479 receipts; Fixed A Random and
Utility epoch16 miss had 125/479 and 111/479. There were no new complete
receipts or aggregate metrics.

## 2026-11-01 21:24 CST - Final-five official queue

Verified on SSH2028 that the four existing evaluator processes remain alive:
Batch10 Random/Utility epoch16 `complete` are at 58/479 and 95/479 receipts;
Fixed A Random/Utility epoch18 `miss` are at 151/479 and 92/479. These are
progress counts, not metrics. No new full Miss aggregate is available.
GPU0-3 are 96-100% utilized, 75-82 C, and 248.66-249.13 W under 250 W caps.
Do not restart the queue, launch training while all four GPUs are occupied,
run val97, or delete candidate weights. Continue final-five official
complete/Miss evaluation and audit each Miss result only after all 479
sequences/337,353 frames and receipt/hash checks pass.

At 21:39 CST, SSH2028 confirmed the same four evaluator PIDs were still alive.
Batch10 Random/Utility epoch16 `complete` had 100/479 and 160/479 receipts;
Fixed A Random/Utility epoch18 `miss` had 271/479 and 172/479. GPU0-3 were
99-100% utilized, 74-82 C, and 249.08-249.40 W under 250 W caps. No new full
Miss aggregate was ready; do not treat these counts as scores.

## 2026-11-01 22:02 CST - Fixed A Random epoch18 Miss audited

The official `miss` evaluation completed with `condition=miss`, `limit=null`,
479 sequence receipts and 337,353 frames. LasHeR-Miss has 245 sequences /
220,703 frames; RGBT234-Miss has 234 / 116,650. `run.json` and
`completed.json` agree. Checkpoint SHA256 is
`929d6538ae350a5dcfb7198620c46c49fa77f752e4c177bd2fd5a2a300da7dc1`;
evaluator SHA256 is
`bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
All 479 individual receipt identities/hashes match `all_receipts.json`, and
all 479 bbox hashes pass.

Metrics: LasHeR PR/NPR/SR `68.162673/63.150544/54.273160`; RGBT234 MPR/MSR
`85.840473/61.418089`. This is now the highest audited RGBT234-Miss MPR/MSR
among completed candidates, but remains 2.259527/2.881911 below the requested
88.1/64.3 thresholds.

At 22:02 CST, the existing queues remained live: Batch10 Random/Utility
epoch16 complete `208/479` and `331/479`; Fixed A Random epoch19 complete
`1/479`; Fixed A Utility epoch18 miss `434/479`. GPU0-3 were 98-100%
utilized, 75-83 C, and 248.44-249.55 W under 250 W caps. Do not restart,
launch training while GPUs are occupied, run val97, or delete weights.

## 2026-11-01 22:11 CST - Fixed A Utility epoch18 Miss audited

This full official Miss evaluation passed: `condition=miss`, `limit=null`,
479 sequences/337,353 frames, including 245/220,703 LasHeR-Miss and
234/116,650 RGBT234-Miss. Run/completed records agree. Checkpoint SHA256:
`1f907ae2238d94a509b4924dcb051699ff97c90d50f75a2998c4d10cd234f492`;
evaluator SHA256:
`bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
The 479 individual receipts exactly match `all_receipts.json`; all 479 bbox
hashes pass.

Metrics: LasHeR PR/NPR/SR `67.730655/62.711185/53.942749`; RGBT234 MPR/MSR
`86.039255/61.454244`. This is the new highest audited RGBT234-Miss result
on both measures, remaining 2.060745/2.845756 below 88.1/64.3.

At 22:11 CST, the active evaluations were Batch10 Random epoch16 complete
`276/479`, Batch10 Utility epoch16 miss `0/479`, Fixed A Random epoch19
complete `4/479`, and Fixed A Utility epoch19 complete `0/479`. GPU0-3 were
95-100% utilized, 75-82 C, and 248.24-249.16 W under 250 W caps. No
restarts or new training; preserve weights and do not run val97.

At 22:21 CST, all four evaluator processes remained live: Batch10 Random
epoch16 complete `388/479`, Batch10 Utility epoch16 miss `1/479`, Fixed A
Random/Utility epoch19 complete `12/479` and `3/479`. GPU0-3 were 100%
utilized, 75-83 C, and 248.91-249.36 W under 250 W caps. No new full Miss
aggregate or training was started; receipt counts are progress only.

## 2026-11-01 22:42 CST - Batch10 Random epoch16 complete audited

The full official `complete` run passed audit: `limit=null`, 479 receipts /
337,353 frames, with 245/220,703 LasHeR and 234/116,650 RGBT234. `run.json`
and `completed.json` agree. Checkpoint SHA256 is
`16dc20c62d8464c884147aecaa732f5acbdbe5529daf1d55867297567a0da932`;
evaluator SHA256 is
`bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
All 479 individual receipt identities match `all_receipts.json`, and all
479 bbox hashes pass.

Metrics: LasHeR PR/NPR/SR `74.541153/71.053097/59.463409`; RGBT234 MPR/MSR
`89.963207/65.482712`. These do not change the current per-metric complete
maxima. The paired epoch16 Miss evaluation has started.

At 22:42 CST, queues were Batch10 Random/Utility epoch16 Miss `4/479` and
`17/479`; Fixed A Random/Utility epoch19 complete `39/479` and `18/479`.
GPU0-3 were 99-100% utilized, 75-83 C, 249.01-249.54 W under 250 W caps.
All weights remain preserved; no val97.

## 2026-11-01 23:00 CST - Official queue progress

The four existing evaluators remain alive. Batch10 Random/Utility epoch16
`miss` have 21/479 and 56/479 sequence receipts; Fixed A Random/Utility
epoch19 `complete` have 83/479 and 48/479. These are progress counts only;
there are no new full-scope results or aggregate metrics.

GPU0-3 are 100%, 100%, 99%, and 100% utilized at 82/80/78/75 C and
249.10/248.88/248.89/249.39 W under the 250 W limits. No training was
started, no evaluator was restarted, all weights remain preserved, and no
val97 evaluation was run. Continue the existing final-five official queues.

## 2026-11-01 23:55 CST - Two official runs audited

Fixed A Random epoch19 `complete` passed full-scope audit:
`condition=complete`, `limit=null`, 479 sequences/337,353 frames, with
245/220,703 LasHeR and 234/116,650 RGBT234. Run/completed records, all 479
individual receipts and bbox hashes, checkpoint/evaluator hashes, and all
110 source-file hashes match. Metrics: LasHeR PR/NPR/SR
`73.398075/69.879029/58.671485`; RGBT234 MPR/MSR `90.159512/65.761226`.

Batch10 Utility epoch16 `miss` passed the same full-scope and hash checks.
Metrics: LasHeR PR/NPR/SR `69.541995/64.221692/55.006010`; RGBT234 MPR/MSR
`84.979303/60.632103`. This clears the LasHeR PR/SR thresholds but does not
reach either RGBT234 target. Neither run changes the current per-metric
maxima; all final-five checkpoints remain preserved.

At 23:59 CST, the four active runs were Batch10 Random epoch16 `miss`
`290/479`, Batch10 Utility epoch17 `complete` `0/479`, Fixed A Random
epoch19 `miss` `0/479`, and Fixed A Utility epoch19 `complete` `346/479`.
GPU0-3 were at 75-82 C and 249.19-249.61 W under 250 W limits. No new
training or val97 evaluation was started.

## 2026-11-02 00:06 CST - Fixed A Utility epoch19 complete audited

The official complete run passed full-scope audit: `condition=complete`,
`limit=null`, 479 sequences/337,353 frames, with 245/220,703 LasHeR and
234/116,650 RGBT234. Run/completed records agree; all 479 individual receipt
records match `all_receipts.json`; all bbox hashes, checkpoint/evaluator
hashes, and 110 runtime source-file hashes match.

Metrics: LasHeR PR/NPR/SR `73.888772/70.183386/59.052663`; RGBT234 MPR/MSR
`90.778065/65.965932`. This does not update the per-metric complete maxima.
Its paired Miss run has started.

At 00:06 CST, the other live queues were Batch10 Random epoch16 Miss
`397/479`, Batch10 Utility epoch17 complete `3/479`, and Fixed A Random
epoch19 Miss `4/479`. Fixed A Utility epoch19 Miss had just started. GPUs
0-3 were 48-82 C, with GPU1 momentarily idle during evaluator startup and
the others near 249 W under 250 W limits. No new training or val97 run.

## 2026-11-02 00:13 CST - Batch10 Random epoch16 Miss audited

The official `miss` run passed full audit: `limit=null`, 479 sequences and
337,353 frames; 245/220,703 LasHeR-Miss and 234/116,650 RGBT234-Miss.
Run/completed records agree. All 479 individual receipts match
`all_receipts.json`; every bbox hash, checkpoint/evaluator hash, and all 110
runtime source-file hashes match.

Metrics: LasHeR PR/NPR/SR `69.243173/63.933594/54.937753`; RGBT234 MPR/MSR
`84.027511/60.083422`. This is not a new per-metric maximum. It clears the
LasHeR-SR threshold but misses LasHeR-PR by 0.056827 and the RGBT234 targets
by 4.072489 MPR and 4.216578 MSR. Paired with the same checkpoint's epoch16
complete run, complete-to-Miss changes are -5.297980 PR, -7.119503 NPR,
-4.525656 SR, -5.935696 MPR, and -5.399290 MSR.

At 00:17 CST, active evaluations were Batch10 Random epoch17 `complete`
`0/479`, Batch10 Utility epoch17 `complete` `8/479`, Fixed A Random epoch19
`miss` `12/479`, and Fixed A Utility epoch19 `miss` `1/479`. GPU0-3 were
74-82 C at 246.70-249.74 W under 250 W limits. No new training or val97.

## 2026-11-02 01:30 CST - Fixed A Random epoch19 Miss audited

Official Miss evaluation passed full audit: `limit=null`, 479 sequences and
337,353 frames; dataset scope 245/220,703 LasHeR-Miss and 234/116,650
RGBT234-Miss. Run/completed records agree. All 479 receipt JSON files match
`all_receipts.json`; all bbox hashes, 110 runtime source-file hashes, and
checkpoint/evaluator hashes match.

Metrics: LasHeR PR/NPR/SR `68.365216/63.171326/54.312404`; RGBT234 MPR/MSR
`86.073187/61.660594`. This does not change the LasHeR-Miss PR/SR maxima;
it raises RGBT234-Miss per-metric maxima by `0.033932/0.206349` points.
Versus original CADTrack E0 Miss `82.794815/59.574435`, the gains are
`3.278372/2.086159`; the `88.1/64.3` targets remain short by
`2.026813/2.639406`.

Checkpoint: `a_random_full20_seed2027/epoch19.pth.tar`, SHA256
`0444578e5d157e8801e537a166c83f4973ed0f56846374b3fd139f7da4ffe8e9`.
This is a final-five candidate, not the selected model; choosing among
official test scores is post-hoc and test-selection-biased. Keep all weights
until the final-five campaign completes. No val97 results are used. Full
audit details: `docs/FIXED_A_EPOCH19_RANDOM_MISS_AUDIT_20261102.md`.

At 01:37 CST, Fixed A Utility epoch19 Miss had 275/479 receipts; Batch10
Random/Utility epoch17 Complete had 145/479 and 271/479; Fixed A Random
epoch20 Complete had just started. These counts are progress only.

## 2026-11-02 01:49 CST - Fixed A Utility epoch19 Miss audited

Official Miss evaluation passed full audit: `limit=null`, 479 sequences and
337,353 frames; per-dataset scope 245/220,703 LasHeR-Miss and 234/116,650
RGBT234-Miss. Run/completed records, all 479 receipt identities, bbox hashes,
110 runtime source-file hashes, and checkpoint/evaluator hashes match.

Metrics: LasHeR PR/NPR/SR `68.095620/63.042714/54.109071`; RGBT234 MPR/MSR
`85.283767/60.972261`. Checkpoint SHA256:
`9a5683045ec8ba7b3560856c20a52ba50e8def85fe12988d25b8dc629825dfbe`.
This candidate does not improve any current per-metric maximum. Preserve it
for final-five comparison; no checkpoint selection or cleanup yet.

## 2026-11-02 01:50 CST - Batch10 Utility epoch17 complete audited

Official complete-input evaluation passed full audit: `limit=null`, 479
sequences/337,353 frames. Benchmark-split receipts are 245/220,703
LasHeR-Miss and 234/116,650 RGBT234-Miss; the `-Miss` suffix is the split
name, while `run.json` condition is `complete`. Run/completed agree; all 479
receipt identities and bbox hashes, 110 runtime source hashes, and
checkpoint/evaluator hashes match.

Metrics: LasHeR PR/NPR/SR `75.751492/71.965679/60.195400`; RGBT234 MPR/MSR
`89.839668/65.314284`. LasHeR PR becomes the complete-input per-metric
maximum, +`0.291668` over the prior maximum and +`0.114989` over original E0.
No other complete-input maximum changes; roughly +2 point complete-input
improvement is not established. This is a final-five candidate, not a
selected checkpoint. Preserve all weights; official test-based selection
has test-selection bias. Full audit:
`docs/BATCH10_UTILITY_EPOCH17_COMPLETE_AUDIT_20261102.md`.

## 2026-11-02 02:13 CST - Batch10 Random epoch17 complete audited

Official complete-input run passed full audit: `limit=null`, 479 sequences
and 337,353 frames; benchmark-split receipts are 245/220,703 LasHeR-Miss
and 234/116,650 RGBT234-Miss. The `-Miss` suffix is the split label; the
run condition is `complete`. Run/completed agree; all receipt identities,
bbox hashes, 110 runtime source hashes, and checkpoint/evaluator hashes
match.

Metrics: LasHeR PR/NPR/SR `75.704048/72.171251/60.419370`; RGBT234 MPR/MSR
`89.946245/65.472600`. LasHeR NPR/SR become new complete-input per-metric
maxima, +`0.122045/0.103244` over prior maxima. LasHeR PR maximum remains
`75.751492`; RGBT234 maxima do not change. Compared with E0, LasHeR NPR/SR
gains are `0.398414/0.293656`; roughly +2 point complete-input improvement
is not established. This remains a final-five candidate, not a selected
checkpoint. Preserve all weights and disclose test-selection bias.
Full audit: `docs/BATCH10_RANDOM_EPOCH17_COMPLETE_AUDIT_20261102.md`.

## 2026-11-02 02:33-02:38 CST - Official queue progress

Reconfirmed the same four evaluator PIDs were live at both ends of a
240-second wait. Receipt counts advanced:

| GPU | Candidate | Condition | Receipts before -> after |
|---|---|---|---:|
| 0 | Fixed A Random epoch20 | Complete | 67 -> 76 / 479 |
| 1 | Fixed A Utility epoch20 | Complete | 28 -> 35 / 479 |
| 2 | Batch10 Random epoch17 | Miss | 7 -> 12 / 479 |
| 3 | Batch10 Utility epoch17 | Miss | 38 -> 48 / 479 |

No run completed and no new aggregate metrics are available. GPU temperatures
were 74-81 C; power draw was 249.34-249.69 W under 250 W limits. These receipt
counts are operational progress only. Do not restart evaluators, launch
training, run val97, or remove candidate checkpoints.

## Research boundary update (2026-11-02)

A read-only audit of the exact active runtime found that availability gates
input tokens and the layer-6 repair, while the shared transformer blocks do
not receive availability as an explicit condition. The tracker retains a
bounded template list but no separate short/long feature-state representation.
This is a testable architectural observation, not a demonstrated cause or
gain. Server source hashes and the experiment gate are recorded in
`docs/MISSING_MODALITY_SOURCE_REVIEW_20261102.md`.

SCDT (CVPR 2026) already uses spatially and temporally conditioned denoising
for missing-modality RGBT tracking. Directly porting or renaming temporal
feature reconstruction/noise-conditioned enhancement would not establish
novelty. Any next module needs a mechanism-specific training-only diagnostic,
matched controls, and complete-input parity; no val97 or RGBT234 development
is allowed. The requested three modules and target metrics remain unproven.

## 2026-11-02 02:52-02:56 CST - Official queue progress

The same four official evaluator PIDs remained live across a 240-second
wait. Receipt counts advanced:

| GPU | Candidate | Condition | Receipts before -> after |
|---|---|---|---:|
| 0 | Fixed A Random epoch20 | Complete | 130 -> 148 / 479 |
| 1 | Fixed A Utility epoch20 | Complete | 67 -> 76 / 479 |
| 2 | Batch10 Utility epoch17 | Miss | 95 -> 112 / 479 |
| 3 | Batch10 Random epoch17 | Miss | 33 -> 42 / 479 |

No full run completed and no aggregate metrics changed. GPU temperatures
were 74-81 C; power draw was 248.93-249.44 W under 250 W limits. Receipt
counts are operational progress only. Preserve all checkpoints and do not
restart the evaluators, launch training, or run val97.

## 2026-11-02 02:59-03:03 CST - Official queue progress

The same four official evaluator PIDs remained live throughout a 240-second
wait. Receipt counts advanced:

| GPU | Candidate | Condition | Receipts before -> after |
|---|---|---|---:|
| 0 | Fixed A Random epoch20 | Complete | 161 -> 184 / 479 |
| 1 | Fixed A Utility epoch20 | Complete | 84 -> 96 / 479 |
| 2 | Batch10 Utility epoch17 | Miss | 125 -> 146 / 479 |
| 3 | Batch10 Random epoch17 | Miss | 48 -> 58 / 479 |

No `completed.json` or `metrics.json` appeared for these runs. GPU
temperatures were 74-81 C and power draw was 248.94-249.71 W, below the
250 W limits. These receipts are progress only; preserve all checkpoints
and do not restart jobs, launch training, or run val97.

## 2026-11-02 03:09-03:13 CST - Official queue progress

The same four official evaluator PIDs remained live across a 240-second
wait. Receipt counts advanced:

| GPU | Candidate | Condition | Receipts before -> after |
|---|---|---|---:|
| 0 | Fixed A Random epoch20 | Complete | 220 -> 252 / 479 |
| 1 | Fixed A Utility epoch20 | Complete | 119 -> 136 / 479 |
| 2 | Batch10 Utility epoch17 | Miss | 182 -> 212 / 479 |
| 3 | Batch10 Random epoch17 | Miss | 76 -> 89 / 479 |

No `completed.json` or `metrics.json` appeared for these runs. GPU
temperatures were 74-81 C and power draw was 249.15-249.48 W, below the
250 W limits. Counts are progress only; preserve all checkpoints and do not
restart the evaluators, launch training, or run val97.

## 2026-11-02 03:31-03:50 CST - Batch10 Utility epoch17 Miss audited

The full official `miss` evaluation passed independent recomputation with
`limit=null`, 479 sequences, and 337,353 frames. Scope matches
245/220,703 LasHeR-Miss and 234/116,650 RGBT234-Miss. The verifier checked
all per-sequence receipts and bbox hashes, source/evaluator/GT/mask hashes,
then recalculated official metrics and curves from saved trajectories.
Checkpoint SHA256:
`fbeaf71a9b4bbfb46e926251b34c7fb1960c4ba41d3da75745342638fd4dc062`.

Metrics: LasHeR-Miss PR/NPR/SR `70.397396/65.127562/55.789882`;
RGBT234-Miss MPR/MSR `85.097378/60.685934`. This is the new audited
per-metric maximum for all three LasHeR-Miss scores. Against E0 Miss, gains
are `+6.924889/+6.545711/+5.206733`; the requested LasHeR PR/SR thresholds
are cleared. RGBT234-Miss is still led by Fixed A Random epoch19 at
`86.073187/61.660594`, short of the requested thresholds by
`2.026813/2.639406`. Audit:
`docs/BATCH10_UTILITY_EPOCH17_MISS_AUDIT_20261102.md`.

At 03:50 CST, the remaining official evaluators were live: Fixed A Utility
epoch20 Complete PID 3596077 at 441/479 receipts, Batch10 Random epoch17
Miss PID 3632441 at 341/479, and Fixed A Random epoch20 Miss PID 3755113
at 6/479. Batch10 Utility's final-five queue completed. GPU3 was idle;
GPUs0-2 were 76-79 C, 248.57-249.19 W with 250 W caps. Do not infer scores
from these receipt counts, restart work, select a checkpoint, delete weights,
or run val97. Allow the three active full evaluations to finish before
considering another training run.

All requested target thresholds and the three experimentally validated
paper modules remain unachieved. Current research boundary and module gate:
`docs/MISSING_MODALITY_SOURCE_REVIEW_20261102.md`.

## 2026-11-02 04:12 CST - Three additional final-five results audited

Fixed A Random epoch20 Complete, Fixed A Utility epoch20 Complete, and
Batch10 Random epoch17 Miss all passed independent full-scope receipt/hash
verification and official-toolkit metric/curve recomputation. Each run is
479/337,353 with `limit=null`; exact values and checkpoint hashes are in
`docs/ADDITIONAL_FINAL5_RESULT_AUDITS_20261102.md`. None changes current
per-metric maxima.

At 04:12 CST, Fixed A Random epoch20 Miss PID 3755113 had 42/479 sequence
receipts and Fixed A Utility epoch20 Miss PID 3788435 had 9/479. Both were
alive. GPUs0/1 were 78-79 C and 248.91-249.13 W under 250 W limits; GPUs2/3
were idle. Counts are progress only. Continue without restarting, do not
launch another GPU job while the paired full Miss evaluations are early in
their long sequences, do not clean checkpoints, and do not run val97.

## 2026-11-02 04:19 CST - Queue ETA and next diagnostic

Fixed A Random epoch20 Miss PID 3755113 and Fixed A Utility epoch20 Miss
PID 3788435 remained alive at 56/479 and 15/479 receipts. At a nearby sample,
their receipts covered about 169,335 and 89,860 frames, respectively, out of
337,353. Frame-throughput estimates suggest roughly 50 minutes and 80 minutes
remain, but sequence mix can move these estimates; recheck near 04:45 CST and
re-estimate. GPUs0/1 were 79 C and about 249.4 W under 250 W; GPUs2/3 idle.

A no-training, train-only diagnostic is specified in
`docs/AVAILABILITY_STATE_BLOCK_DIAGNOSTIC_PLAN_20261102.md`. It tests whether
normal Transformer updates to the position-only missing stream change
tracking loss, with an available-stream sham and exact complete-input parity.
It is not a claimed contribution and has not been launched; wait for the two
official Miss evaluations to finish before allocating a diagnostic GPU.

At 04:29 CST, both same evaluator PIDs remained live. Random had 89/479
receipts and 200,576 frames; Utility had 30/479 and 125,385 frames. Over the
preceding ten-minute interval, frame throughput implies about 45 minutes
remaining for Random and 60 minutes for Utility; check again around 04:50 CST
and update the estimates. GPU state remained 78-79 C and 249.32-249.38 W.

Verified another wait through 04:35 CST: the same PIDs remained live and
receipts advanced to Random 116/479, 222,639 frames and Utility 43/479,
146,284 frames. Over this six-minute interval, estimated remaining time is
about 31 minutes and 55 minutes, respectively; recheck near 04:50 CST.
GPUs0/1 were 78-79 C and 249.10 W, with the 250 W caps unchanged.

Verified another 240-second wait through 04:41 CST. Random advanced to
149/479 receipts and 244,034 frames; Utility to 60/479 and 169,335 frames.
Both evaluator PIDs remained live. Recent frame throughput suggests roughly
26 and 44 minutes remain; re-estimate at the next 240-second check. GPUs0/1
were 78-79 C and 249.26-249.40 W. No full aggregate metrics yet.

Verified 240-second wait through 04:49 CST. Random advanced to 202/479
receipts and 271,978 frames; Utility to 84/479 and 195,666 frames. Both
original evaluator PIDs remained live. Current frame throughput suggests
about 15 and 31 minutes remain, respectively; check near 05:05 CST. GPUs0/1
were 78-79 C and 249.25-249.63 W. No aggregate metrics yet.

## 2026-11-02 05:05 CST - Fixed A Random epoch20 Miss audited

The official Miss run completed and passed independent full-scope
verification: 479 sequences, 337,353 frames, `limit=null`; LasHeR-Miss
245/220,703 and RGBT234-Miss 234/116,650. Receipt identities, bbox/GT/mask
hashes, runtime/evaluator hashes, and recalculated official metrics/curves
all pass. Checkpoint SHA256:
`6f014866e45393edd0c32594a916d1b0e677f1c6bea8f2d898daa7938f4e5e39`.

Metrics: LasHeR-Miss PR/NPR/SR `67.823154/62.731542/53.736184`;
RGBT234-Miss MPR/MSR `86.184864/61.418962`. This is a new RGBT234-Miss
MPR maximum (+0.111677 over epoch19) but does not improve the MSR maximum
61.660594 from epoch19. The current RGBT234 targets remain short by
1.915136 MPR/2.639406 MSR; the LasHeR-Miss best remains Batch10 Utility
epoch17. Audit: `docs/FIXED_A_EPOCH20_RANDOM_MISS_AUDIT_20261102.md`.

Fixed A Utility epoch20 Miss PID 3788435 remained active at 199/479 receipts
and 270,607 frames at 05:12 CST. GPU0 was idle after the Random evaluation;
GPU1 remained active at 79 C/249.15 W, GPUs2/3 idle. Preserve all weights;
do not use receipt progress as a metric or run val97. Recheck the Utility
evaluation after about four minutes and audit only after full completion.

## 2026-11-02 05:33 CST - Fixed A Utility epoch20 Miss audited

The second remaining official Miss run completed and passed independent
full-scope verification: 479 sequences, 337,353 frames, `limit=null`;
245/220,703 LasHeR-Miss and 234/116,650 RGBT234-Miss. Receipts,
bbox/GT/mask hashes, runtime/evaluator hashes, and recomputed official
metrics/curves all pass. Checkpoint SHA256:
`19c008d03f261ebda37c89eff63a0b959bcb6795c0da548e761ef12c91878170`.

Metrics: LasHeR-Miss PR/NPR/SR `68.569074/63.435064/54.347391`;
RGBT234-Miss MPR/MSR `84.703935/60.637694`. It does not update any
per-metric maximum. Audit: `docs/FIXED_A_EPOCH20_UTILITY_MISS_AUDIT_20261102.md`.
All four GPUs were idle after completion. Keep all final-five weights
preserved until queue manifests confirm every registered run has finished;
do not select by official scores, clean weights, or run val97.

Current audited maxima after both epoch20 Miss runs:

| Condition / dataset | Metric | Candidate | Score |
|---|---|---|---:|
| Complete / LasHeR | PR | Batch10 Utility ep17 | 75.751492 |
| Complete / LasHeR | NPR | Batch10 Random ep17 | 72.171251 |
| Complete / LasHeR | SR | Batch10 Random ep17 | 60.419370 |
| Complete / RGBT234 | MPR | Fixed A Random ep16 | 91.043051 |
| Complete / RGBT234 | MSR | CADTrack E0 | 66.845228 |
| Miss / LasHeR | PR / NPR / SR | Batch10 Utility ep17 | 70.397396 / 65.127562 / 55.789882 |
| Miss / RGBT234 | MPR | Fixed A Random ep20 | 86.184864 |
| Miss / RGBT234 | MSR | Fixed A Random ep19 | 61.660594 |

LasHeR-Miss PR/SR targets are met by the current maxima. RGBT234-Miss
remains short by 1.915136 MPR and 2.639406 MSR. Three paper modules and the
approximately +2-point complete-input gains remain unproven.

## 2026-11-02 05:46 CST - All final-five official outputs independently recalculated

On server28, independently recalculated all 40 official conditions using the
isolated `verify_official` path: Fixed A Random/Utility epochs 16-20 and
Batch10 Random/Utility epochs 13-17, each in complete and miss. Every run
passed checkpoint SHA, source/evaluator/GT/mask hashes, all 479 receipt and
bbox checks, 479-sequence/337,353-frame scope with `limit=null`, and toolkit
metric/curve recomputation. No val97 inference or score was used.

All 20 final-five checkpoints were rehashed and match the corresponding
Complete/Miss run manifests. No official-test-based selection or cleanup was
performed. GPUs 0-3 were idle after the audit and remain capped at 250 W.
The per-metric maxima, retained checkpoint hashes, baseline deltas, remaining
targets, and next diagnostic gate are recorded in
`docs/FINAL5_OFFICIAL_RECALC_AUDIT_20261102.md`.

LasHeR-Miss PR/SR exceed their thresholds by 1.097396/1.389882 points.
RGBT234-Miss MPR/MSR remain short by 1.915136/2.639406 points. The best
complete-input gains over E0 are below +0.4 points, so the approximate
+2-point objective and all three paper modules remain unproven.

## 2026-11-02 06:53 CST - Train-only availability-block diagnostic

Completed the registered no-update diagnostic on GPU2 using 128 tuples from
118 LasHeR train882 sequences and frozen E0 epoch20
(`d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`).
The split-list hash matches the audited train882 manifest. Every tuple
records its sequence/frame identities and input hashes. No val97 or official
test data was used; no optimizer update occurred. Complete-input intervention
is bitwise identical to native (maximum absolute output delta0), and native
repeat task-loss noise is0.

Bypass of ordinary block updates to the unavailable stream worsened both
missing-state conditions. Sequence-macro native-minus-intervention task-loss
effects (positive would improve) were -4.847049
([-5.466646, -4.230780]) RGB-only and -12.076711
([-13.086938, -11.087066]) TIR-only. IoU changes were -0.226034
([-0.273621, -0.179301]) and -0.367300
([-0.420182, -0.312548]); center-error improvements were also negative.
Neither direction passes the registered repeat-noise/effect gate.

Decision: reject this mechanism and do not start matched training. It does
not establish a paper module or explain the RGBT234-Miss gap. Full 128-tuple
report: `reports/server28_train882_block_state_diag_20261102/diagnostic.json`
on server28 and local workspace `outputs/server28_train882_block_state_diag_20261102.json`,
SHA256 `fa6e4c98b0d51730b288024d59b369581d717c96cbd878ea335d6105da6fa09f`.
The diagnostic script, Python3.7 fixes, and protocol/result record are on
`codex/server28-research`; no candidate training or checkpoint cleanup was
performed. GPU2 is released.

## 2026-11-02 07:14 UTC+08 - Train-only joint utility-set diagnostic

Previous turn classified as no progress toward the research objective: it
reported the already-audited official table without new experimental evidence.
This turn revalidated clean local/remote commit6be2713, idle GPU0-3 and
the rejected block-bypass result, then completed a new frozen-checkpoint
train882 diagnostic. No val97 or official-test inference was launched.

Fixed A Utility ep16 was fixed as the first retained final-five epoch,
not chosen by official scores. SHA25642ed78bb821a34f2695423a3a5d8db3ea0af0ab7820022b78588378fea803c8d.
GPU2 process4108301 completed128 tuples from118 training sequences, three
search conditions, batch8, no parameter updates. The same pre-repair features
fed no-repair, utility/random/similarity top4 and four selected single-region
interventions. Both missing-state utility-vs-control bootstrap intervals
cross zero, as do joint-minus-single-sum interaction intervals. Registered
training gate=false: do not launch joint-utility training from these results.

Report `reports/server28_joint_utility_sets_20261102/diagnostic.json`,
SHA2561a36cb0f3235fc5d2ad56821efe16d3a6e110a0bcc7ec17e806589b2ac37f0d7.
Separate local analyzer verified384 saved records, sequence membership,
input pairing, means and all sequence-bootstrap calculations. See
`docs/JOINT_UTILITY_SET_DIAGNOSTIC_20261102.md` for results and limitations.
This is a one-checkpoint task-loss diagnostic with complete templates, not
official scores, proof of exact additivity, or an independently reviewed
paper module. All-map parity/repeat noise/frozen-state digests were not
recorded; do not imply otherwise. GPU2 worker exited; preserve all weights.

Next research direction must address useful correction or target localization
with train-only evidence rather than assume subset interactions explain the
gap. Existing negative bypass, unconditional geometry/state-template and
ordinary motion results remain closed; do not restart them. Complete gains
near2 points, RGBT234-Miss targets and three validated modules remain unmet.

## 2026-11-02 07:30 UTC+08 - Fusion adapter paired pilot completed

After the table-only turn (no research progress), checked actual PID4123633:
worker exited normally,197.78seconds, GPUs0-3 idle. Train882-only cached
fitting/development/audit partitions705/88/89;256/128/128 tuples. Frozen E0,
two50,080-parameter matched adapters. Local stopped18epochs/144updates,
global14epochs/112updates; last-five development selection chose15 and11.
No val97, official-test evaluation or baseline update.

Local sequence-macro native-loss gains: TIR-only0.068091,95%CI
[-0.022931,0.160346]; RGB-only0.053419,[-0.013591,0.129761]. Relative
macro improvements3.367973%/3.807889% are positive point estimates, but both
registered absolute-effect CIs cross zero. Local-global gain0.010722,
CI[-0.041733,0.066307]. Registered promotion gate=false. Do not launch
full training or fresh closed-loop promotion from this pilot.

Complete history remains bitwise equal; complete search after missing
templates worsened in both local-arm tuple-mean cells. Baseline state digest
unchanged, gradients absent. Saved-record analysis with server-file hashes
passed source/checkpoint/cache/final-five integrity, stopping/selection and
sequence-bootstrap arithmetic. This does not independently replay inference.
Report SHA2562ca07ff702becc2aa26dc646d05e911810ca18a360b42b8b00e94a5e7a36584f.
Protocol/results:docs/FUSION_ADAPTER_PILOT_20261102.md; small evidence:
reports/server28_fusion_adapter_pilot_20261102/. All weights preserved.

Next evidence should distinguish persistent target localization from
recovery-frame damage with a fresh train-only cohort and explicit controls.
Do not retune this adapter on the audit sample or count an ordinary adapter
as a contribution. Official maxima unchanged; complete approximately+2,
RGBT234-Miss88.1/64.3 and three supported modules remain unmet.

## 2026-11-02 07:40 UTC+08 - Fresh response/geometry diagnostic

Previous goal turn made progress: paired fusion-adapter training completed,
failed its registered promotion gate and was published atbc9d667. This turn
checked actual idle GPUs and completed new train882-only diagnostic PID4143074
on GPU2 in45.296seconds. New seed39101,128 tuples/111 sequences, nine
template/search states,1152 rows; no exact frame-tuple overlap with prior
pilot. Same fitting-name pool, not sequence-independent data.

Complete-template native IoU: full0.768577,TIR-only0.561638,RGB-only0.682737.
GT-peak diagnostic IoU0.797548/0.635005/0.712224. TIR-only oracle gain excess
over complete0.044397,95%CI[0.016508,0.074092]; RGB-only excess CI spans zero.
Peak selection is only part of the observed TIR-only crop deficit. GT oracle
cannot be used at inference or reported as attainable benchmark performance.
All nine mean gradient cosines positive; TIR-only conflict fraction actually
decreases, not evidence for a gradient-conflict method.

Native complete-search/missing-template effects are small with intervals
spanning zero on this cohort. They cannot be equated with the prior adapter's
different-cohort loss increases or temporal recovery effects.

Saved-evidence analyzer passed90 cell summaries/32 paired contrasts and
source/base/cache hashes,1152 row arithmetic, whole frozen-state/gradient
invariants. No independent inference replay. Report SHA256
7521c438e18a80d4ec2a66881bf6d3907b06839439b148b8b7455314e27310e3.
See docs/RESPONSE_GEOMETRY_DIAGNOSTIC_20261102.md and
reports/server28_response_geometry_20261102/.

Reviewed pinned CenterNet target-index regression and OSTrack gt_score_map
API as ordinary supervision controls, not new modules. Next eligible
question is a fresh matched training-objective control comparing native
predicted-peak regression and target-supported supervision while keeping
native inference. No adapter promotion or training started here.
No val97/official evaluation/baseline update/cleanup. Official scores,
approximately+2 complete goal,RGBT234-Miss thresholds and three modules
remain unchanged and incomplete.

## 2026-11-02 07:56 UTC+08 - Paired head supervision failed promotion

Prior turn completed a fresh response/geometry diagnostic (actual progress).
This turn rechecked idle GPUs and ran GPU2 PID4161273,274.387seconds,
paired native versus target-position box-regression supervision. Frozen E0
fusion,6,474,117-parameter copied heads,BN eval,all9 modality states.
New256/128/128 tuples from705/88/89 train882 name partitions; no exact tuple
overlap with two preceding pilots. Audit128 tuples cover70 actual sequences.
No val97 or official tests.

Both arms stopped epoch10/360updates, selected epoch6 from final-five6-10 by
native-head development loss. Target-native missing-cell macro IoU effect
-0.009119,95%CI[-0.017295,-0.000888]. Target-E0 TIR-only-0.000202
CI[-0.017034,0.017395],RGB-only+0.001481CI[-0.015394,0.019496]. Complete
history target-E0-0.006657; recovery10_11-0.017905
CI[-0.033137,-0.005049]. Registered gate=false. Do not promote target
supervision or reinterpret native control as an approved candidate.

Saved-evidence verifier checked source/base/cache/selected/final-five hashes,
partitions and prior tuple exclusion, stopping/selection and all bootstrap
effects. All10 last-five head weights finite with BN buffers equal to E0.
Separately constructed heads directly replayed1152 audit rows for each
E0/native/target:3456 native-inference losses/IoUs exactly match. No GT map
in replay. Baseline full-state digest unchanged,gradients absent.
This replays cached heads, not backbone/images or full tracking.

Report SHA2565e67287881365117c978b89d382911d02d595fd7514ac344f0d00b845f95d4f9.
Full protocol/results:docs/HEAD_SUPERVISION_PILOT_20261102.md; small evidence
reports/server28_head_supervision_20261102/. All weights preserved.
Static crops omit tracker Hann response and dynamic updates. Heads used
less than1GiB allocated; finite pilot does not saturate24GiB, full-model
batch profiling remains pending evidence. No official scores changed.

Next mechanism must address modality-asymmetric geometry with explicit
capacity and complete/recovery controls; do not rerun target-peak or failed
fusion-adapter tuning on these audit samples. Ordinary controls are not
paper modules. Complete+2,RGBT234-Miss targets and three modules remain unmet.

## 2026-11-02 08:12 UTC+08 - Capacity-matched modality heads

Previous turn head-supervision training/replay was progress. New paired
modality-vs-balanced-random two-head training completed GPU2 PID4186101,
233.025seconds. Same12,948,234parameters per arm,1536 fitting rows,
768/head,24updates/epoch. New256/128/128 train882 tuples,705/88/89name
partitions; no exact overlap with previous3pilot samples. Audit73sequences.
Both stopped10epochs/240updates, selected6 from final-five6-10.

Registered macro IoU modality-E0:TIR-only+0.003473CI[-0.006143,0.013821],
RGB-only+0.012225CI[0.004128,0.022354]; modality-random+0.000977
CI[-0.001836,0.003703]. RGB-only single gate passes; full promotion=false.
No clear superiority to capacity-matched random routing or TIR-only gain;
do not expand or count as a module. One random route seed, no robustness claim.

All complete-search states use frozen E0 head; complete/recovery cached
maps exactly equal. This does not establish closed-loop recovery equality
or complete approximately+2 improvement. Baseline state hash unchanged,
gradients absent. All10 last-five two-head checkpoints finite and BN
buffers equal to E0. Separate verifier rederived routes/balance,partitions/
exclusions,source/base/cache/weights,history/stopping/selection/bootstrap.
Separate native-head replay3456records exact loss/IoU and complete maps.
Not image/backbone replay or external scientific review.

Report SHA256ab2f1aca04d047e4d92432e56bc9714687c8890133e0238e104d925e355c3790.
Full protocol:docs/MODALITY_HEAD_PILOT_20261102.md; small evidence
reports/server28_modality_heads_20261102/. All weights preserved,no val97
or official runs,official scores unchanged. Full goal remains incomplete.
Do not keep testing minor head variants on this audit cohort; next work
must test a distinct evidence-backed representation or temporal mechanism.

## 2026-11-02 - Absent-search attention K/V diagnostic

Completed frozen E0 train882 diagnostic on server28 GPU2, PID65200, in
80.864 seconds. The one-batch custom attention sham matched native output
exactly. Corrected full run used128 fresh frame tuples/113 sequences and
1152 rows across nine dynamic-template/search states. Independent analyzer
verified tuple exclusions, source hashes, all 1152 sham parity checks,
384 complete-search parity checks, report digest, and bootstrap arithmetic.
No optimizer, model-state change, or gradients.

Masking current missing-search tokens as attention keys/values reduced
sequence-macro IoU in both missing directions: RGB missing -0.182255
95% CI[-0.221629,-0.143620]; TIR missing -0.062795
95% CI[-0.088324,-0.038894]. Registered gate failed. Close this hypothesis
on this cohort; do not train or official-evaluate it. The first full attempt
failed on CPU target/GPU prediction placement; its log and launch receipts
are retained and the corrected run completed. No val97 or official data.
Full protocol/results: `docs/ABSENT_SEARCH_KV_DIAGNOSTIC_20261102.md`;
report: `reports/server28_absent_search_kv_20261102_v2/`.
The overall objective remains incomplete: RGBT234-Miss targets and the
approximately +2 complete-input gains are unmet; no three paper modules
have been validated.

## 2026-11-02 - Burst-mask closed-loop rollout diagnostic

Compared IID versus temporally clustered modality masks on frozen E0 over
24 fresh contiguous64-frame train882 clips. Each arm used exactly the same
per-clip counts:26 complete and19 in each single-modality state. The
paired rollout produced3024 tracked-frame records; no val97 or official
test data were used. GPU2 PID144335 completed in123.645s.

Clustered-minus-IID sequence-macro IoU was -0.015331 on all missing frames
(95% bootstrap CI[-0.032356,0.001973]), -0.009342 on RGB-only input
(CI[-0.037280,0.018596]), -0.013135 on TIR-only input
(CI[-0.052738,0.026904]), and -0.043505 on first-five-frame recovery
(18 paired sequences; CI[-0.178025,0.056210]). All intervals cross zero;
registered gate=false. Do not launch burst-aware full training or claim a
module from this diagnostic.

Independent saved-evidence verification rechecked all per-frame schedules,
recovery flags, box IoUs, source/checkpoint hashes, and bootstrap intervals.
The model state remained unchanged and gradients absent. Raw report SHA256:
b44dc635c7b6c1807ab08dd6ec9e7d1e3976fce8d4e40ceacb8fd198b420212b.
The producer labeled paired-sequence counts as `frames`; the recomputed
analysis corrects this label to `paired_sequences` without altering the
raw report. Full evidence:docs/BURST_MASK_ROLLOUT_DIAGNOSTIC_20261102.md
and reports/server28_burst_mask_rollout_20261102/.

No training, benchmark evaluation, weight selection or cleanup occurred.
Complete-input gains, RGBT234-Miss thresholds, and three validated modules
remain unmet.
## 2026-11-02 - Fixed box consensus verified and closed

Previous turn published attention evidence (progress). New GPU2PID1131961
completed50.036s,128freshfittingtuples117seq1152states,all priorframes
including attentiontemplates/search excluded. Fixed top8 nativeHann boxes:
scoreweightedmean versus uniformmean versus nativeargmax; nooptimizer.
Weighted-native cropIoU complete-.061691CI[-.091699,-.034760],
RGBmissing-.075441CI[-.104439,-.047579],TIRmissing-.117869
CI[-.152202,-.085691]. Recovery01/10-.078797/-.079187.
Weighted beats uniform but harms native; advancefalse,closeexactestimator,
no tuning candidatecount/temperature/clusters/controlpromotion/module.
Not universalboxfusion rejection or established soleerrorcause.
Server/local verified9216boxes/1152aggregates/bootstrap/gate/exclusions/
provenance;allnative maps checked and144firstbatch normalizedimage
forwardrows bitwiseexact. Sharednetwork,notrawdecoder/fullimagereplay/
externalreview. Source/checkpoint/cache hashes and physicalvisibility pass.
ReportSHA dedaec020a61dea618dd8320bec459fc3ec03956e67631ac811f08eae150806b.
Docs BOX_CONSENSUS_DIAGNOSTIC_20261102.md;largecachesserveronly.
NextfreshsamplersmustexcludeALLthisreportinput_manifestframeIDs.
No val97/official/newweights/cleanup. Officialscores unchanged,goalunmet.
## 2026-11-02 - Matched local dense geometry training verified

Previous goal turn ran/published box-consensus negative=progress. GPU2PID1178458
completed280.274s. train/dev/audit256/128/128 train882 tuples,705/88/89 name
partitions,all old template/search frames excluded. Fixed3x3 GT-heatmap
regression mean vs native Hann-peak regression; common full focal objective,
same six-million-parameter pretrained head and original single-box inference.
Batch2304/LR.00288/one update per epoch,actual peak9440MiB each. Native20
epochs hit max with ongoing dev improvement,selected17 from16-20; dense
stopped14 after patience3,selected13 from10-14. All10 final-five checkpoints
verified finite/BN unchanged. Cached-head only; not full-network convergence.

Audit128/74seq: dense-E0 RGBmissing-.006127CI[-.028993,.015274],
TIRmissing+.022116CI[.001969,.046957]. Dense-native-trained pooled missing
-.010682CI[-.019862,-.001967]. Complete11_11+.011744
CI[-.012965,.044125]; recovery01+.022820CI[-.012885,.065286],
recovery10+.037759CI[.003800,.079068]. Gatefalse: RGBmissing gain absent,
matched control superior to dense, complete parity gate failed. Close exact
3x3 treatment/no neighborhood tuning/no module promotion. Does not reject all
dense regression or explain cross-modal gap.

Local/server analyzers verified all sources/base/cache/final-five weights,
all prior-frame exclusions/visibility/partition/pairing/bootstrap/history,
16 independent objective witness expressions/offset gradients (max error0),
and selected-head3456 audit-row replay via independent Hann decode.
Shared cached features/native taskloss; not image/full-tracker replay or
external review. ReportSHA de33c5b1912be26b4527df5bbdf579f995596680e1459d548ee675b21f308a70.
WorkerSHA b23858e5f3f679af547047873ab4f96d85f8ba3b501833759fedea4f3880f217.
ProtocolSHA8847da6ad48e2219336b0fe1c0554a11714cd4bd5c4f6f0fb04f0acbee686b21.
Docs DENSE_GEOMETRY_TRAINING_20261102.md. Large caches and all20 weights
remain server-only. No val97/official calls/cleanup; official scores unchanged,
goal unmet. Future samples exclude all new manifest frames.
# 2026-11-02 paired full official metrics

New consolidated table: docs/PAIRED_OFFICIAL_METRICS_20261102.md.
It pairs Complete/Miss for E0 epoch20, Batch10 Utility/Random epoch17
and Fixed A Random epoch20, with all five metrics per condition.
These are historical comparisons, not checkpoint selection.
The existing per-metric maxima come from multiple weights.
No candidate in this table meets all four Miss targets.
All40 historical final-five evaluations finished and passed the05:46CST
audit; older individual notes describe their original timestamps.
Current seed2028 training/queue unchanged and has no new official score.

## 2026-11-02 18:05 CST - P implementation and first original epoch

User's pasted P/F/X proposal is adopted in that order. Isolated P code is in
research/curriculum_p/{schedule,check_schedule,train,prepare}.py.
Four20epoch controls:fixed60,fixed50,progressive5/10/5epochs at.2/.5/.8,
and deterministic stage-permuted control. Patch.1,original A Utility,
same-budget controls; no early stopping for P. CPU sampler/native parity,
cross-arm patch/RNG checks PASS in server environment; trainer imports/CLI
and compilation passed. No image smoke/formal P launched. Preparation
correctly stops before campaign creation while original training completion
is absent. Future image smoke/verifier/full-evaluation continuation pending;
check metadata feasibility after freezing original complete manifests.
Details and source HEAD pins:docs/CURRICULUM_P_AND_FEEDBACK_F_20261102.md.

Scope correction:recent40run maxima are not all historical maxima.
Migrated historical A Utility complete RGBT234 MPR91.450712 was directly
read from original metrics artifact; no fresh trajectory recomputation.
Older E4epoch05 Miss MSR61.797040 remains a historical recorded result
requiring original-artifact/provenance checking. Do not silently omit these.

Original PIDs1288280/1288281 live18:05:48CST with epoch1/2000updates done;
internal128dev meanloss4.356546/4.310292,not official PR/SR or convergence.
Firstepoch6389.62/6368.71s;roughly25-35.5h for14-20epochs total.
Queue1355742 live,unchanged. GPU2/3 75/74C,22710/22742MiB,250Wlimits.
No new official score, validated module or checkpoint cleanup.

## 2026-11-02 18:27 CST - P execution and verification tools

Added research/curriculum_p/launch.py,verify.py,official.py,check_contracts.py.
P trainer now records initial BN digests plus actual update gradient norms/LRs.
Verifier requires20epochs/finalfive16-20,all four prepared configs and
actual launch receipts, all data/source/exclusion identities and hashes,
all dev rows, final-five finite weights/actual parameter updates and BN
initialization parity. All six arm pairs compare exact inputs/patch masks;
equal-frequency steps also require modality-mask equality.
Formal preparation requires actual smoke PASS with matching completion and
verifier-source hashes. P launch/evaluation uses GPU0/1,leaving original
GPU2/3 queue unchanged. official.py plans10fullruns/arm,4workers/GPU,
isolated official-only recomputation,no subsets/val97main/weight deletion.

Synthetic contract rejection tests and40unique official job plans PASS;
CUDA0native sampler parity/sharedpatch/RNG checks PASS. All8files compiled
and launcher/verifier/official CLI imports passed in actualPython3.7.
Receipts reports/server28_curriculum_p_contracts_v3_20261102.json and
reports/server28_curriculum_p_sampler_cuda_20261102.json. Not actual P
image/gradient/weight verification or40executed benchmark jobs.
No P campaign prepared/launched:original manifests still growing.
After originalterminal:metadatafeasibility,image4arm smoke,actual verifier,
formal20epochs,formal verifier,full official reports. Do not bypass gates.

At18:27:20CST originaltrainers1288280/1288281 actualcommands live02:08:20,
epoch2/update2400total(400of2000thisepoch),sameobservedinputs/masks,
lossesfinite. Queue1355742 live01:27:34WAITING_FOR_TRAINING.
GPU2/3both75C,22710/22742MiB,250W;GPU0/1idleafterCUDAcheck.
Next second-epoch milestone roughly19:52CST;avoid repeated early polls.
No officialscore/convergence/module claim;fullgoal remains unmet.

## 2026-11-02 19:22 CST - P/F/X source audit

Inspected all four pinned author repositories and actual local insertion
points. Evidence and blob IDs:docs/PFX_SOURCE_AUDIT_20261102.md.
SDSTrack masks complementary RGB/TIR patches (marginals .3/.21), unlike
our independent patches; not a proven cause of the performance gap.
CBM is patch/classification curriculum, not whole-modality frequency.
DetectoRS recomputes backbone/neck; zero sigmoid blend logits give .5,
distinct from its zero feedback projections. DCM uses detached cross-head
targets; classification KL is not a drop-in tracking objective.
No reference code copied/installed/run. No new trained module claim.

F must rerun blocks4-12 INCLUDING block6 repair, preserve query/template
prefix at feedback insertion, test eval-mode zero-residual parity and
random-selection RNG semantics, normalize dual-round losses, and commit
one external tracker update. Active code/runtime/queue unchanged.

At19:20:30CST trainer1288280/1288281 actual command identities live03:01:31;
queue1355742 live02:20:45. GPUs2/3 74/71C,22710/22742MiB,250W limits.
Near19:22 logs Utility epoch2/update3425 and Random epoch2/update3450,
finite losses. Common update3425 input/mask hashes match. No new complete
epoch development result or official score. GPU0/1 idle.
Next epoch2 milestone about19:52CST;P preparation waits terminal originals.

## 2026-11-02 - Isolated F structural implementation

research/feedback_f/model.py implements shared second-pass blocks4-12,
including block6 utility repair, search-only residuals from this view's
shallow/deep features,prediction,availability,plus none/shallow controls.
No active runtime/source or original trainer/queue was changed.
Normalized two-round C/M task-and-KD helper and round2 utility probes work.
Random-region mode intentionally excluded pending its two-round RNG policy.

Actual E0 network synthetic check PASS onGPU0;latest receipt
reports/server28_feedback_f_structure_v3_20261102.json,
SHA6cfa0abeb90f43854aab1152701ab697c982921875126fb7301ba1d790a1086d.
Two128templates/256search;three-search C/M gradient test and oneSGDupdate.
Exact native/round1/round2 parity for11/01/10 with nonzero syntheticrepair;
blockcounts[1,1,1,2,2,2,2,2,2,2,2,2],repair2;prefixunchangedatfeedback;
deepvs shallow dependency controls;physicalmissing/patch invariance;
zero-initialized projectiongradnorm.883728;finitebackbone/feedback/selector
gradients;actualparameterupdate;BN unchanged;fresh-model strictreload exact.
10.206s,peak8002914304bytes(~7.45GiB). GPU0released oncheckerexit.
All five receipt source digests match publication files. Earlier v1/v2
receipts kept with their own earlier checker-source hashes.
No actual image learning,formaltrainer/trackerintegration,latency comparison,
correction/destruction measurement,convergence or official-score claim.
Details:docs/FEEDBACK_F_STRUCTURE_20261102.md.

Tracker integration also PASS on synthetic uint8 frames:
research/feedback_f/tracker.py strict-loads none/shallow/deep and inherits
native initialize/track;receipt
reports/server28_feedback_f_tracker_20261102.json
SHA3b8d6e559b13db3ca43666c45556212340274969ba08e2ad6828c1d2897637f4.
Exactzero-residual parity states11/01/10 across3frames;one network call and
one box/state commit per frame;round2 decode;conditional template append/pop.
No image training/official score/latency efficacy claim.

Published bf580bd, research:F tracker integration check; GitHub branch pushed.
At 19:53:55 CST original trainers 1288280/1288281 and queue 1355742 remain
alive. Both formal trainers entered epoch3 after epoch2 DEVELOPMENT records;
latest observed updates Utility 4025 / Random 4075 with finite losses.
GPU2/3 73/74C, 241/244W under 250W, 22710/22742MiB. No new official score.
P remains blocked on original formal completion receipts and frozen manifests;
do not start P smoke/formal or alter active runtime/queue.

At19:34:37CST originaltrainers1288280/1288281live03:15:37,
queue1355742live02:34:51. Utilityepoch2/update3675,Random3700,finite losses.
GPUs2/3 75/74C,22710/22742MiB,250W;GPU0/1idleatobservation.
Epoch2development milestone stillabout19:52CST. P remains waiting for
originalterminal/frozenmanifests/metadatafeasibility;P->F->X orderunchanged.
No weights pruned,no val97,no newofficialscore;fullgoalunmet.

## F shared training batch interface, 2026-11-02 20:51 CST

Added research/feedback_f/training.py and check_training.py. Actual E0
synthetic four-arm check PASS: shared initialization and post-build CPU RNG
exact; C/M probes each reconstruct objective exactly, backward/AdamW finite,
backbone/selector updated, shallow/deep projection gradients nonzero, BN fixed.
Eight updates, 19.424 seconds, peak 8434151424 bytes; GPU0 released.
Receipt reports/server28_feedback_f_training_20261102.json SHA256
455379b2f90fb0e612acacc0dd1ddf231898e222af57ff9814e4db74886afc62.
No persistent weights or dataset use; not formal training or efficacy.
Formal loop still pending; isolate mask RNG per step because forward RNG
consumption differs between single/double passes. P->F->X order retained.

At 20:47:05 CST original Utility/Random updates5050/5075, epoch3, finite
observed loss5.376097/4.109694; estimated epoch3 end21:37-21:38 CST.
At20:51:05 original PIDs1288280/1288281 and queue1355742 still live;
GPU2/3 75/74C,22710/22742MiB,250W caps. No restart, new official result,
checkpoint selection or cleanup. Full goal remains unmet.

## F per-step mask isolation, 2026-11-02

training.step_masks wraps the unchanged native sampler in a CPU/CUDA RNG fork,
keyed by config mask_seed+microstep. Actual E0 synthetic v2 PASS: native
same-seed parity, exact caller RNG preservation, consecutive-step patch
variation, identical masks across four arms/two steps after their different
forward draws. Those masks drive all8 C/Mprobe AdamW updates; objective
reconstruction exact, gradients finite, backbone update and selector/feedback
gradient checks and frozen BN pass. 19.253s, peak8434151424bytes, GPU0 checker exited.
Receipt reports/server28_feedback_f_training_v2_20261102.json SHA256
0c4052c1c9eb833114c1522c9610be2931abf5562491882b42e30306b3e14d30.
No real-image training/official metric/convergence claim. Formal F loop
must call this helper and verify actual image/input/mask parity across arms.

At20:54:44CST original1288280/1288281/1355742 commands live; epoch3
Utility5175/Random5200 with observed losses6.963657/5.826354 finite.
GPU2/3both74C,22710/22742MiB,250Wcaps. Epoch3 milestone~21:37CST.
No original runtime/training/queue changes, checkpoint pruning or val97.
P still waits original terminal/frozen manifests; full goal remains unmet.

## Isolated F image training entry, 2026-11-02

Added research/feedback_f/train.py: four controls use shared build/batch/
step_masks, inherited train882 partitions/exclusions/loader/records and BN
freeze; logs gradients/LRs/masks/input hashes, saves20weights and selects
development-loss minimum within16-20 with earlier tie. Completes full common
configured P course; no benchmark selection or weight deletion.
Requires actual formal P verification plus prepared/config/source/parent/
manifest hashes before dataset access/output-directory creation.

ActualPython3.7 compile/import/CLI PASS and missing formal P verification
rejected with no F run directory created. Receipt
reports/server28_feedback_f_train_entry_20261102.json SHA256
054f3e7874460bbadfd083ac7cfce28108af19464eebf60a71ed293301920736.
This is entry validation only: no positive-parent check, dataset/epoch loop,
F real-image smoke/formal run, saved-weight verifier or official result.
Batch-memory sizing/matched configs and positive-path checks remain pending.

At21:00:45CST original1288280/1288281/1355742 actualcommands live;
epoch3microsteps10599/10649 (updates5300/5325),observedloss3.276134/6.415179
finite. GPU2/3 75/74C,22710/22742MiB,250Wcaps. Epoch3ETA~21:37CST.
Original runtime/training/queue unchanged. P waits originalterminal;
P->F->X retained and full goal still unmet.

## F parent metadata contracts, 2026-11-02

Added research/feedback_f/check_parent_contracts.py. Actual Python3.7
CPU-only test on temporary synthetic files: positive fixture accepted;
12cases reject missing verification/wrong phase/missing arm/changed verifier,
prepared,config,run-config/incomplete status despite refreshed hashes/changed
manifest/omitted inherited completion or manifest/wrong verification hash.
No F run directory,dataset,GPU or actual image loop. Receipt
reports/server28_feedback_f_parent_contracts_20261102.json SHA256
8db6fd81b56defe6fae71ce38bae2fdd333cadecbc1ef0bcd6f4f891b3569d44.
Real P positive-boundary,image smoke,full-loop and saved-weight checks remain
pending; synthetic acceptance does not establish actual P/F training.

At21:09:19CST actual original PIDs1288280/1288281/1355742 live,
epoch3updates5450/5500,observedloss11.090048/7.013592 finite.
Epoch3 milestone remains about21:37CST; no early repeated polling.
Original training/runtime/queue untouched. No new official metrics.

## F batch profile, 2026-11-02

Added profile_batch.py and measured all four controls with synthetic resident
tensors on RTX3090. Microbatch2/accumulation2/effective4 passed three full
AdamW updates. Single peak allocated8.97GiB; none14.46GiB; shallow14.58GiB;
deep14.57GiB. Deep microbatch3 OOM during backward:20.46GiB allocated,
21MiB free,21.82GiB reserved. Microbatch4 not attempted. Preliminary smoke
choice is microbatch2, accumulation2, effective4, LR5e-5/backbone5e-6;
actual image smoke must recheck memory. Profile docs:
docs/FEEDBACK_F_BATCH_PROFILE_20261102.md and OOM log
reports/server28_feedback_f_deep_b3_oom_20261102.log.
No F image or official run launched.

At21:22:22CST original1288280/1288281/1355742 actualcommands live;
GPU2/3 75/74C,22710/22742MiB,250Wcaps. No original runtime/queue change,
new official metric or checkpoint cleanup.

## Epoch3 completed, 2026-11-02 21:37:44 CST

Actual original PIDs1288280/1288281/1355742 remain live with original commands.
Both trainers completed epoch3/update6000; Random already logged epoch4.
Fixed train882-development mean task losses (epochs1,2,3):
Utility4.356546/4.398573/4.481288; Random4.310292/4.399227/4.472109.
Neither improves over epoch1. This is an early optimization concern, not
official accuracy or proof of convergence; keep the preregistered LR-drop/
stopping policy unchanged. No test checkpoint selection or cleanup.
Raw three-epoch histories copied to reports/server28_partitioned_a_
{utility,random}_epoch03_20261102.json. Next epoch milestone~23:23CST.
P still requires original terminal receipts and frozen manifests.

## F internal round-loss diagnostics, 2026-11-02

Added validation.py/check_validation.py; F train writes per-epoch paired
round1/2 task-loss vectors keyed by development batch/sample index and
summarizes lower/higher/equal counts/fractions and mean deltas for11/01/10.
Original final-round mean-loss aggregation and selection unchanged.
These are per-tuple sums over search frames, not closed-loop tracking
correction/destruction or official metrics. Single controls keep native fields.

Actual E0 synthetic check PASS: final-loss exact native parity for single/
none/deep, zero-feedback ties, preserved CPU/CUDA RNG and selector state,
nonzero-feedback raw losses/counts/deltas recomputed from captured outputs.
Receipt reports/server28_feedback_f_validation_20261102.json SHA256
a32207d49dae468c58a4d9c4f3975774742e9397d58c44d5c74501abea39a665.
Updated Python3.7 train CLI imports; checker exited and GPU0 released.
No real F images/epoch/efficacy/official score; previous source receipts
remain historical. At21:40:32CST original1288280/1288281/1355742 live,
both epoch4, latestupdates6050/6075 finite; no active runtime/queue edit.
Next meaningful training milestone~23:23CST; P waits original completion.

## F full official entry and single-arm loader fix, 2026-11-02

Added research/feedback_f/evaluate.py and verify_official.py. Full evaluator
reuses native prepare/sequence, fixed479/337353, no subset CLI, same rgbt
metrics/curves, checkpoint/runtime/F/native source hashes and all receipts.
Separate F verifier recalculates all scores/curves and checks every receipt,
trajectory,GT/mask/source; original verifier hardcodes original evaluator hash.
No original evaluator/queue edits. Verified F campaign/final-five integration
and actual spawned evaluation/recalculation remain pending.

Fixed tracker.py mode single: native training checkpoint now constructs
MissingTrackerModel rather than rejecting single as a two-round mode.
Actual E0 synthetic tracker v2 PASS, including three-frame single/native
box/template/availability parity and previous round2/template checks.
Receipt server28_feedback_f_tracker_v2_20261102 SHA256
4aab14a1ded19e1d7b71bc176e1c73aed72e3159c563c0cbcaf7b69b4839a468.
ActualPython3.7 new evaluator/verifier compile/import/CLI PASS; --limit rejected
before output creation. Receipt server28_feedback_f_official_entry_20261102 SHA
d4c1df7722c0a378ffa0b8e4a715e2d29c8f2c0cfa0e91f01ee9dc783d1804bb.
No official dataset preparation, F trajectory/metric, image training or
real learned F weights. Earlier source receipts remain historical.
Original training continues; next milestone~23:23CST. Full goal unmet.

## Matched F smoke preparation, 2026-11-02

Added research/feedback_f/prepare.py and check_prepare.py. Preparation
requires the existing verified completed formal P campaign before output
creation. The explicitly supplied --curriculum-arm is shared by single,
none, shallow and deep; no official metric chooses a curriculum here.
Each smoke arm uses microbatch2/accumulation2/effective4, LR5e-5/backbone5e-6,
20 short epochs,8 samples/epoch,40 total updates and8 fixed development
samples. Shared seeds/masks/budgets and inherited frame exclusions match.
This command prepares smoke configs only; formal preparation, saved-run
verification, launch integration and final-five queue remain pending.

Actual server Python3.7 synthetic metadata check passed all four curriculum
options, cross-arm config equality, update/LR arithmetic and missing-parent
rejection before output. Receipt:
reports/server28_feedback_f_prepare_20261102.json.
No real P/F campaign/config directory, image worker, learned weight or
official score created. At21:57:13CST original1288280/1288281/1355742
actual commands remain live; GPU2/3=75/74C,250W,22710/22744MiB.
Next useful original epoch4 inspection remains~23:23CST.

## F saved-run verifier entry, 2026-11-02

Added research/feedback_f/verify.py. Checks prepared/config/source/prior
hashes, inherited P completion boundaries, train705/dev88/audit89 frame
membership/exclusions, complete step/input counts, finite single or nested
round losses, keyed masks, update/LR arithmetic,20epoch history and final5
development-loss selection. Across all4arms requires same common config,
saved initial shared parameter/BN hashes, frame/tensor identities and masks.
Recomputes raw round-loss summaries. Checks all20weight paths and loads
epoch1/final5 for finite state, fixedBN, backbone/head/Utility updates and
shallow/deep feedback updates. It never opens official test images.

Actual Python3.7 CLI imports/help passed. check_evidence.py passed temporary
synthetic helper cases for history/selection/update/mask/nonfinite losses,
LR boundaries, single/nested status and round-summary recomputation.
Receipt reports/server28_feedback_f_evidence_contracts_20261102.json.
No complete saved F run or learned weights exist, so the full verifier and
weight-loading branch have NOT been exercised. Formal F preparation,
launch integration and final5 queue still pending. This is implementation
progress, not training/efficacy/official score evidence.

At22:03:31CST original1288280/1288281/1355742 actualcommands live.
Utility epoch4 microstep12995/update6498 observedloss5.676797;
Random epoch4 microstep13057/update6529 observedloss9.936890.
Different latest steps, no paired-loss comparison. No training edit,
restart, extra image job or checkpoint deletion. Goal remains unmet.

## F smoke-to-formal preparation and launcher

prepare.py now requires --phase smoke|formal. Formal requires a completed
verified4arm F smoke for the same explicit curriculum before output creation,
checks source/prepared/config/run/receipt hashes and adds smoke manifests
to inherited frame exclusions. train.py independently enforces this formal
boundary; verify.py checks it and actual launcher receipt/config/trainer/
prepared hashes. launch.py uses existing GPU readiness checks,GPU0/1 only,
isolated subprocess/log/receipt and frozen runtime cwd; no original job edits.
Formal matched budget: seed2030,mask65403,dev65402,20epochs,
20000samples/epoch,128dev,microbatch2/accum2,LR5e-5/backbone5e-6.
Smoke remains20shortepochs/8samples/40updates. No checkpoint deletion.

ActualPython3.7 prepare_v2 synthetic fixture checks pass all4curricula,
matched smoke/formal configs, inherited exclusions and missing-smoke rejection.
Evidence-helperv2 checks pass; launch.py --help imports pass.
Receipts reports/server28_feedback_f_{prepare_v2,evidence_contracts_v2}_20261102.json.
No real F config/run, positive real-parent verification, spawned image worker,
full saved-run verification or new official metric. Final5 queue pending.
Earlier receipts retain historical source hashes.
22:10:08CST original1288280/1288281/1355742 actual commands live;
GPU2/3=75/74C,250W,22710/22744MiB. Next epoch4 milestone~23:23CST.
Full objective remains active/unmet.

## F final-five official queue entry

Added research/feedback_f/official.py and check_official.py. Entry requires
the full formal4arm saved-run verification, unchanged prepared/config/
completion/manifest/source hashes and inherited P/F smoke boundaries.
For one arm it evaluates epochs16-20 in complete+miss,10 full479/337353
jobs,4workers/GPU0or1,300second child-process polling. Uses F evaluate.py
and independent verify_official.verify for every result. Saves registration,
source hashes, exit records and recomputed metrics. Keeps the train-only
development-selected epoch marker and all checkpoints; no test selection.

ActualPython3.7 CLI/import and synthetic plan checks PASS: all4curricula x
4arms x10jobs have160unique output paths, pairedselected weights, fixed
fullcoverage, no subset option; invalid epochbudget/selection/hash/final5
rejected. Receipt reports/server28_feedback_f_official_plan_20261102.json.
These160 jobs are synthetic plans, NOT registered/executed experiments.
No real F campaign, spawned evaluation, verifier trajectory recalculation
or new scores. Real P/F smoke/formal and efficacy remain pending.
At22:18:56CST original1288280/1288281/1355742 actualcommands stilllive.
Next originalepoch4 milestone~23:23CST; no early repeated statuspoll.
Originalruntime/queue unchanged. Full objective active and unmet.

## Historical fresh-frame feasibility audit

Added read-only CPU metadata audit scripts/audit_partitioned_fresh_frames.py.
It exactly counts native causal anchors for fitting705 sequences using
length>=20, visible>10, anchor range[1,len-3), an earlier visible template
frame and3 later search frames; native gap expands without a fixed cap.
Exclusions are completed historical diagnostics/crops plus completed dense
cache identities only. Active/growing partitioned-A/P manifests were not
read; no images, GPU, workers or training/evaluation launched.

Result296231 possible anchors,704/705 eligible,median229,minimum positive7,
7 sequences below20 anchors. `aboyleft_1202` has71 total/5 visible/66 excluded/
0 anchors. This is historical-boundary upper-bound feasibility, NOT the
post-A final boundary or crop-augmentation acceptance. Do not launch P smoke
yet. Once both original A formal manifests/completion receipts freeze, rerun
with those immutable exclusions before P smoke. Checker pure fixtures PASS.
Raw report `reports/server28_partitioned_fresh_frame_audit_v3_20261102.json`;
analysis `docs/PARTITIONED_FRESH_FRAME_FEASIBILITY_20261102.md`.
Only the dedicated audit report was written; no audit input manifest/run dir.
At22:31CST original1288280/1288281/1355742 remained actual live commands;
GPU0/1 1MiB idle, GPU2/3=75/74C,22710/22744MiB,100%.

Added final-boundary option `--include-completed-partitioned-a`. It requires
formal `verification_v2.json` status=passed, then verifies both completed
and input-manifest hashes before including their identities. Synthetic
receipt tests pass paired acceptance and reject changed manifest/incomplete
status; no live manifest was read. After original A training finishes and
verification_v2 passes, rerun to a NEW report path with this option before
P smoke. Python3.7 checker PASS; no P image smoke/training launched.

## Actual initial/epoch3 same-image diagnosis

Added scripts/diagnose_partitioned_initial.py and executed on GPU0.
All32devbatches/128tuples exactly match original frame identities and
augmented tensor hashes; initialparameterhash matches; epoch3 three losses
reproduce exactly. Initial vs epoch3:
complete3.252417->3.808033(+.555617);
RGBmissing5.764184->5.488136(-.276049);
TIRmissing3.976443->4.147695(+.171252);
mean4.331015->4.481288(+.150273).
This is real-image evidence of uneven early task-loss changes, not official
accuracy/convergence/causality. No optimizer/newframe/weightwrite/selection
change. Process exited0; evaluation37.892s/peak557701120bytes.
Raw reports/server28_partitioned_initial_diagnosis_20261102.json;
analysis docs/PARTITIONED_INITIAL_DIAGNOSIS_20261102.md.
Keep availability-separated analysis for P/F/X and recheck after scheduled
LRdrop without silently intervening in original training.
At22:24:15CST original1288280/1288281/1355742 actualcommands live;
GPU2/3both74C,250W. Still3completedepochs, nextmilestone~23:23CST.
Previous official queue implementation was progress; this turn adds actual
image evidence. No new official score; full objective remains unmet.

## Partitioned A epoch4 diagnostic snapshot

At23:27CST, paired train-only fixed128 development histories through epoch4
were preserved in
reports/server28_partitioned_a_dev_history_epoch04_{utility,random}_20261102.json
and summarized in docs/PARTITIONED_A_EPOCH04_DIAGNOSTIC_20261102.md.
Utility mean losses by epoch1-4:4.356546,4.398573,4.481288,4.492963.
Random:4.310292,4.399227,4.472109,4.386902. Complete and TIR-missing
losses remain above epoch1 in both arms; Utility RGB-missing also increased,
while Random RGB-missing epoch4 remains slightly above epoch1. This is
uneven early optimization evidence, not official accuracy or convergence;
keep preregistered LR/stopping unchanged.

At23:27CST original trainer PIDs1288280/1288281 and official queue1355742
were live. Utility epoch5/update8100; Random epoch5/update8150 (2000 updates
per epoch). GPU2/3 75/74C,250W caps. Estimated epoch5 boundary ~00:59CST;
recheck near that milestone with180-300s polling. Do not read growing
manifests or start P smoke before both trainers terminate and queue
verification_v2 passes. No new official score; no run/config/queue changes
or checkpoint cleanup. Full goal remains active and unmet.

## 2026-11-03 00:38 CST - Current verified state

Read-only SSH confirmed original Utility/Random training parents1288280/
1288281 and official queue v2 PID1355742 are live. PIDs1945771/1945774 and
1942780/1942781 are their dataloader children, not extra training jobs.
Latest log: epoch5, Utility update9475, Random update9450; finite loss is
training telemetry only, not convergence or official accuracy. At the
observed rate, next 2000-update epoch boundary is estimated around
01:05-01:10 CST.

GPU0/1 idle. GPU2 75C/243.38W and GPU3 75C/249.64W, both fully utilized and
within 85C/250W limits. Queue remains waiting for both original trainers and
the formal verifier. No new full official result; no process, queue,
checkpoint, or growing manifest was changed.

Audited Miss per-metric bests remain LasHeR PR/SR70.397396/55.789882
(+1.097396/+1.389882 over targets) and RGBT234 MPR/MSR86.184864/61.660594
(1.915136/2.639406 below targets). The approximate +2-point complete
improvement is also unproven. These maxima are not one jointly selected
checkpoint. Current A is a paired CADTrack-initialization replication, not
one of the three validated paper modules. Preserve the registered last-five
full official evaluation and no-test-selection/no-cleanup boundaries.

## 2026-11-03 00:46 CST - Verified training progress

After a 180-second wait, Utility log microstep advanced from18999 to19149;
Random also advanced. Both original trainer parents and queue remain live,
so prior tail positions did not indicate a hang. GPU2/3 remain fully
utilized at75/74C and247.76/244.44W; GPU0/1 are idle.

Epoch5 is not complete; queue remains WAITING_FOR_TRAINING and no completion
or formal-verification receipt exists, so no official evaluation was
started. At the observed rate, epoch5 boundary is estimated near01:05CST;
recheck at the milestone without intervention. No run/queue edits,
checkpoint cleanup, or new official metric.

## 2026-11-03 02:12 CST - A training remains active

After continued 300-second polling, both original A trainers remain live
and are still writing finite-loss records in epoch6: Utility update11200
and Random update11250. The official queue is still WAITING_FOR_TRAINING;
no completion receipt or verification_v2 receipt exists, so no downstream
formal training or partial evaluation was started.

GPU2/3 are fully utilized at75/74C and244.31/249.20W with a 250W cap.
GPU0/1 remain idle because they are reserved for the registered official
full-dataset evaluation queue after the A formal gate. Preserve the
last-five-epoch Complete+Miss evaluation and do not use val97 or test
metrics for selection.

## 2026-11-04 01:24 CST - OHCAS residual contract and queue state

Added an isolated `HistoryResidualAdapter` and CPU contract checker under
`research/history_stream/`. The adapter returns
`native + (1 - availability) * delta(native, history, log1p(age))`; its final
projection is zero-initialized. The check passed in Server 28's Python
3.7/PyTorch 1.12 environment with CUDA hidden: exact initialization parity,
available-stream invariance after making the residual nonzero, history/age
sensitivity on the missing stream, and finite nonzero output-projection
gradient. This was an in-memory execution of the local source; no new source
was copied into the active project or frozen snapshots. It is an interface
contract only, not model integration or efficacy evidence.

At 01:24:28 CST, A Utility/Random completion receipts were present at
`runs/server28_partitioned_a_20261102_formal/{utility,random}/completed.json`.
The receipt counts are 17 epochs/34,000 updates and 18 epochs/36,000 updates;
both retain all checkpoints and exclude official-test-based selection.
`reports/server28_partitioned_a_20261102_formal/verification_v2.json` is
`passed`, with scope limited to saved-evidence verification.

The A official queue PID1355742 remains `OFFICIAL_RUNNING`. Its Utility epoch13
Complete job (PID1601993/GPU2) and Random epoch14 Complete job (PID1601948/GPU3)
each require 479 sequences / 337,353 frames. The queue's final formal
verification has not passed. R1 PID3388846 and R2 PID3395457 remain alive on
GPU0/1; P queue PID3539938 remains `WAITING_FOR_R` for the four registered R
training/official receipts. No GPU is free and no next-stage task was
dispatched.

GPU0-3 at 01:24:28 CST: temperatures 81/78/76/73 C, power draw
245.75/246.61/247.09/244.74 W under 250 W limits, memory
19160/19160/9106/9138 MiB, utilization 100/100/94/90%. No new official score,
checkpoint cleanup, restart, or active protocol-source change. Overall
targets and three validated modules remain unproven.

## 2026-11-04 02:14 CST: F response-control amendment and CPU checks

- Added the `deep-no-response` F arm to separate the deep-feature contrast from first-pass response-map conditioning. It has the same feedback shape and initialization as `deep`, with only the response-map input zeroed. Updated protocol training, validation, smoke gates, saved-evidence pairing, official reporting and tests from four to five arms. The older `training.py` / `prepare.py` path remains the retired four-arm P path.
- Server 28 Python 3.7 / PyTorch 1.12 CPU checks passed with CUDA hidden: compileall; 90 training contract cases and five OSTrack CPU initializations; 30 runner cases / 720 microsteps / 360 updates; five-mode synthetic validation across three availability states with actual OSTrack CPU forwards; 30 saved-evidence cases / 720 synthetic steps; 2 smoke tests; 2 official-plan tests; 8 mocked official-controller/report tests; and synthetic parent-gate checks. Raw JSON receipts are saved in `reports/server28_feedback_five_arm_checks_20261103/`.
- The plan fixture's 300 destinations enumerate six possible curricula; a single selected curriculum requires 50 full evaluations across five arms, five retained checkpoints and Complete/Miss conditions. Each evaluation must cover 479 sequences / 337,353 frames (LasHeR 245 / 220,703; RGBT234 234 / 116,650). All weights remain retained.
- This is protocol/interface evidence only: no image smoke, F training, learned weights, full GPU tracker integration, official metrics or efficacy claim. All four GPUs remained occupied, so `check_tracker.py` was not run. No active source snapshot, config, queue, checkpoint or growing manifest was changed.
- At 02:14:15 CST, A training receipts and saved-evidence verification remained complete. A official queue PID1355742 remained `OFFICIAL_RUNNING`; its Utility epoch13 Complete PID1601993/GPU2 and Random epoch14 Complete PID1601948/GPU3 were still live. R1/R2 PIDs3388846/3395457 were live and P queue3539938 remained `WAITING_FOR_R`. Temperatures were81/78/76/73 C, power246.96-249.44 W, and all four GPUs were occupied. No official campaign completion or next-stage authorization was present.
- One isolated test attempt initially lacked copied dependencies required for source-hash resolution; after adding the read-only source files to the `/tmp` overlay, the saved-evidence and official tests passed. No product-code change was needed for that harness setup issue.

## 2026-11-03 18:34 UTC - A full-official queue progress

Read-only Server 28 checks confirmed A Utility/Random receipts complete at
17/34,000 and 18/36,000 epochs/updates; both checkpoint sets are retained.
The training-scope `verification_v2.json` is `passed`. This is distinct from
the still-running official-evaluation queue's final formal verification.

Utility epoch13 Complete and Random epoch14 Complete each completed the full
479-sequence / 337,353-frame evaluation and passed independent
`verify_official` recomputation. Utility metrics (LasHeR PR/NPR/SR,
RGBT234 MPR/MSR): 73.891323/70.522389/59.025243,
87.456205/64.369160. Random: 74.008829/70.746387/59.087931,
87.647063/64.243282. These are two of twenty post-hoc final-five
evaluations, on Complete input; they do not answer the LasHeR-Miss target
and do not select checkpoints.

At server-reported 2026-11-03 18:34 UTC, Utility epoch13 Miss and Random
epoch14 Miss were running on GPU2/3. R1/R2 trainers3388846/3395457 remained
live on GPU0/1, around epoch11 (latest logged updates26950/13200). The A
queue was still `OFFICIAL_RUNNING`; no queue-level completed receipt or
final formal verification existed. The RPPM queue3539938 remained
`WAITING_FOR_R`, missing r1/r2 training-verification and full-official
completion receipts. Therefore no follow-on campaign was authorized.

GPU0-3 were 77/78/75/73 C, drawing246.60/248.99/249.32/249.29 W under
250 W caps, with GPU0/1/2/3 utilization86/100/100/100%. No training,
configuration, queue or checkpoint was changed.

## 2026-11-03 18:54 UTC - R1 crossed epoch12

Read-only Server 28 polling confirmed R1 PID3388846 advanced to epoch12,
update27550; R2 PID3395457 remained at epoch11, update13500. These are
training-log telemetry, not convergence or official metric evidence.
The two A Miss evaluations (Utility epoch13 and Random epoch14) remained
live on GPU2/3, while A queue PID1355742 was still `OFFICIAL_RUNNING`.
RPPM queue3539938 remained `WAITING_FOR_R` on the same four R training and
full-official receipts. No new result or dispatch authorization was present.

GPU0-3 were 81/78/76/73 C and drew247.24/246.28/249.40/249.12 W, all under
250 W caps. All four cards remained allocated to their registered work.

## 2026-11-03 19:05 UTC - Official A evaluation and P preflight still gated

Read-only Server 28 polling at 19:05:59 UTC confirmed the A official queue
PID1355742 remains `OFFICIAL_RUNNING`. Utility epoch13 Miss PID1731939 and
Random epoch14 Miss PID1731824 are still active on GPU2/3; neither completed
receipt nor queue-level formal verification exists. Both jobs target the full
479-sequence / 337,353-frame scope (LasHeR 245 / 220,703 and RGBT234
234 / 116,650).

The R1/R2 trainers PID3388846/3395457 and curriculum queue PID3539938 remain
alive. The autoscheduler log records `PREPARED_NOT_LAUNCHED` for the P smoke:
its GPU preflight failed at 19,160 MiB allocated versus the required
less-than-500-MiB free-card threshold (temperature 79 C, 250 W cap). No P
training was launched. The four GPUs remain occupied, with 9,106-19,160 MiB
used, temperatures 73-81 C, power 243.84-249.42 W, and utilization 60-100%.
No intervention, duplicate dispatch, configuration change, or checkpoint
cleanup was performed. A training-evidence `verification_v2.json` remains
`passed`, but it is not the official queue's final formal verification.

## 2026-11-03 19:22 UTC - A Miss evaluation progress

At 19:22:28 UTC, both full Miss evaluations remained live. Utility epoch13
had logged 56/479 sequences and 164,286/337,353 frames; Random epoch14 had
logged 58/479 sequences and 166,833/337,353 frames. The frame-based linear
ETA is roughly 52-54 minutes from this observation and is only an estimate.
Neither completion receipt nor queue-level formal verification exists yet;
the A queue PID1355742 remains `OFFICIAL_RUNNING`.

R1/R2 trainers PID3388846/3395457 and curriculum queue PID3539938 remain
alive. GPU0-3 were at 81/78/77/73 C, drawing247.23/246.81/249.16/248.09 W,
with 19,160/19,160/9,106/9,138 MiB allocated and 95-100% utilization.
The P smoke remains `PREPARED_NOT_LAUNCHED` after the logged free-memory
preflight failure; no new training was dispatched and no process/config was
changed.

## 2026-11-03 19:27 UTC - A Miss evaluations pass halfway

Read-only polling at 19:27:36 UTC found both full Miss evaluators still alive.
Utility epoch13 had completed 71/479 sequences and 182,068/337,353 frames;
Random epoch14 had completed 72/479 sequences and 183,170/337,353 frames.
Frame-rate extrapolation gives approximately 47-48 minutes remaining, subject
to sequence-length variation. Completion receipts and queue-level formal
verification are still absent; A queue PID1355742 remains
`OFFICIAL_RUNNING`.

R1/R2 PIDs3388846/3395457 and curriculum queue PID3539938 remain alive.
All four GPUs remain allocated: temperatures 81/77/77/74 C, power
247.04/244.82/249.12/249.58 W, memory 19,160/19,160/9,106/9,138 MiB, and
utilization 86-100%. P remains not launched because its preflight requires a
free card below 500 MiB allocated memory. No intervention was made.

## 2026-11-03 19:33 UTC - A Miss evaluations near 60 percent

At 19:33:24 UTC, Utility epoch13 Miss had logged 89/479 sequences and
200,576/337,353 frames; Random epoch14 Miss had logged 91/479 sequences and
202,467/337,353 frames. Both evaluator PIDs 1731939/1731824 and A queue
PID 1355742 remain alive. Linear frame-rate estimates are about 41 minutes
remaining, with sequence-duration variability. No completion receipt or
queue-level formal verification exists yet.

R1/R2 PIDs 3388846/3395457 and curriculum queue PID 3539938 remain live.
GPU0-3 remain allocated at 81/78/77/74 C and 246.68/246.29/249.10/249.34 W;
memory is 19,160/19,160/9,106/9,138 MiB. P is still not launched because the
free-card memory preflight failed. No job, configuration, or checkpoint was
changed.

## 2026-11-03 19:41 UTC - A Miss evaluations continue

At 19:41:58 UTC, Utility epoch13 Miss had logged 127/479 sequences and
230,103/337,353 frames; Random epoch14 Miss had logged 131/479 sequences and
232,750/337,353 frames. Frame-rate extrapolation estimates about 32 minutes
remaining. Both evaluator processes and A official queue PID 1355742 remain
alive; neither full completion receipt nor queue-level formal verification
exists yet.

R1/R2 trainers PID 3388846/3395457 and curriculum queue PID 3539938 also
remain alive. GPU0-3 were at 81/78/77/74 C, drawing
246.94/245.78/249.32/249.60 W, with 19,160/19,160/9,106/9,138 MiB allocated.
No task was stopped, restarted, or newly dispatched; no configuration or
checkpoint was changed.

## 2026-11-03 19:52 UTC - A Miss evaluations near 80 percent

At 19:52:32 UTC, Utility epoch13 Miss had logged 188/479 sequences and
265,329/337,353 frames; Random epoch14 Miss had logged 195/479 sequences and
268,729/337,353 frames. These correspond to about 78.7% and 79.7% of expected
frames. Frame-rate extrapolation estimates roughly 21 minutes remaining.
Both evaluators and A official queue PID 1355742 remain alive; completion
receipts and queue-level formal verification are still absent.

R1/R2 PIDs 3388846/3395457 and curriculum queue PID 3539938 remain live.
GPU0-3 were at 81/78/77/74 C, drawing 242.59/246.67/249.83/249.28 W, with
19,160/19,160/9,106/9,138 MiB allocated. No job or configuration was
changed and no new task was dispatched.

## 2026-11-03 19:58 UTC - A Miss evaluations approach completion

At 19:58:29 UTC, Utility epoch13 Miss had logged 235/479 sequences and
286,040/337,353 frames; Random epoch14 Miss had logged 244/479 sequences and
289,526/337,353 frames, about 84.8% and 85.8% frame coverage. Linear estimates
put remaining runtime at roughly 15-16 minutes, subject to sequence-length
variation. Both evaluators and A queue PID 1355742 remain alive. The two
completion receipts and queue-level formal verification are still absent.

R1/R2 PIDs 3388846/3395457 and curriculum queue PID 3539938 remain alive.
GPU0-3 are at 81/78/77/74 C, drawing 248.17/246.87/248.87/249.39 W, with
19,160/19,160/9,106/9,138 MiB allocated. No intervention or new dispatch.

## 2026-11-03 20:05 UTC - A Miss evaluations above 90 percent

At 20:05:39 UTC, Utility epoch13 Miss had logged 308/479 sequences and
310,007/337,353 frames; Random epoch14 Miss had logged 322/479 sequences and
313,656/337,353 frames, about 91.9% and 93.0% frame coverage. Remaining-frame
extrapolation is roughly 8-10 minutes but varies with sequence duration.
Both evaluators (PIDs1731939/1731824) and A queue PID1355742 remain alive.
Neither evaluation completion receipt nor queue-level formal verification
exists yet.

R1/R2 trainers PID3388846/3395457 and curriculum queue PID3539938 remain
alive. GPU0-3 were at81/78/76/74 C and246.88/246.43/248.95/249.18 W, with
19,160/19,160/9,106/9,138 MiB allocated. No task was stopped/restarted and
no new task was dispatched.

## 2026-11-03 20:15 UTC - A final-five Miss evaluations finished

At 20:15:23 UTC, Utility epoch13 Miss and Random epoch14 Miss each had
completed the full scope: 479 sequences / 337,353 frames, comprising LasHeR
245 / 220,703 and RGBT234 234 / 116,650. Their `completed.json` and
`metrics.json` outputs exist. Recorded metrics:

- Utility epoch13: LasHeR-Miss PR/NPR/SR 67.791326/62.320427/53.763340;
  RGBT234-Miss MPR/MSR 84.418707/60.717600.
- Random epoch14: LasHeR-Miss PR/NPR/SR 67.890157/63.093999/53.846060;
  RGBT234-Miss MPR/MSR 84.703949/60.568584.

Neither candidate reaches the stated targets (LasHeR-Miss PR/SR 69.3/54.4;
RGBT234-Miss MPR/MSR 88.1/64.3). These are two of twenty post-hoc
final-five results; do not use them to select a checkpoint. Independent
`verify_official` recomputation is still running in the arm controllers
PID1601932/1601933; queue PID1355742 remains `OFFICIAL_RUNNING`, and the
queue-level formal-verification receipt is absent. Thus these metrics are
provisional pending the registered integrity checks.

Evaluation GPUs 2/3 were released (1/33 MiB used, 0% utilization); GPUs 0/1
remain occupied by R1/R2. No next stage is authorized before A queue formal
verification passes. No running job or checkpoint was changed.

Update at 20:22:07 UTC: both Miss results have since passed independent
`verify_official` recomputation; `utility_verified_metrics.json` and
`random_verified_metrics.json` each contain the expected Complete and Miss
epoch records. The two Miss metrics above are now independently verified,
not provisional. A queue-level formal verification still awaits all twenty
final-five evaluations. The next jobs, Utility epoch14 Complete (PID1794995)
and Random epoch15 Complete (PID1794990), are running on GPUs2/3. R1/R2
remain on GPUs0/1; all four GPUs are occupied again.

## 2026-11-03 20:57 UTC - A training receipts complete; official queue remains active

Read-only check confirmed both A training processes have exited and their
stable `completed.json` receipts report `complete`: Utility finished 17
epochs/34,000 updates and selected epoch14; Random finished 18 epochs/36,000
updates and selected epoch15. Both receipts state all checkpoints retained
and no official-test selection. The formal training
`reports/server28_partitioned_a_20261102_formal/verification_v2.json` is
`passed` for saved-evidence checks; its scope explicitly is not independent
image/gradient replay.

This is not the official queue's final formal verification. Queue PID1355742
and both arm controllers 1601932/1601933 remain alive in `OFFICIAL_RUNNING`;
the queue-level formal receipt is absent. Utility epoch14 Complete (PID1794995)
and Random epoch15 Complete (PID1794990) are still evaluating full 479-sequence
runs. GPU0-3 are all allocated (19,160/19,160/9,106/9,138 MiB) at 81/78/77/74 C
and 246.56/246.68/249.05/249.57 W, below their 250 W limits. No next-stage
dispatch is authorized until the official queue's final formal verification
passes; no process, config, or checkpoint was changed.

## 2026-11-03 22:13 UTC - Final-five Complete evaluations near full coverage

Read-only Server28 poll at22:13:22Z confirmed Utility epoch14 Complete
evaluator1794995 and Random epoch15 Complete evaluator1794990 remain alive;
their queue controllers1601932/1601933 and parent queue1355742 also remain
alive in `OFFICIAL_RUNNING`. Log-derived progress is Utility368/479 sequences,
323,761/337,353 frames and Random393/479 sequences,328,048/337,353 frames.
No completion receipts or queue-level formal-verification receipt exist yet.

Both A training `completed.json` receipts remain `complete` (17/34,000 and
18/36,000 epochs/updates), retain all checkpoints, and state no official-test
selection. Training-scope `verification_v2.json` remains `passed`; this is
separate from the pending official queue verification. R1/R2 trainers3388846/
3395457 and RPPM queue3539938 remain alive; RPPM is still `WAITING_FOR_R` on
the same four R training-verification/full-official receipts. GPU0-3 are all
occupied at80/78/76/73 C,246.58/248.36/249.41/248.77W and100% utilization.
No task was stopped, restarted or dispatched; no config, manifest or
checkpoint was changed.

## 2026-11-03 22:24 UTC - Two more full Complete rows verified; Miss rows running

At22:24:33Z the A official controllers independently recomputed and appended
Utility epoch14 Complete and Random epoch15 Complete, both at479 sequences /
337,353 frames. Utility metrics (LasHeR PR/NPR/SR; RGBT234 MPR/MSR) are
74.158725/70.542595/59.154405 and87.473682/63.972659. Random metrics are
73.602038/70.002574/58.640863 and88.414559/64.548808. The Random Complete
RGBT234 values exceed the numeric Miss thresholds but are Complete-condition
results and do not establish the requested Miss targets.

The queue remains `OFFICIAL_RUNNING`, with no queue-level completion or final
formal-verification receipt. Utility epoch14 Miss PID1873594 is now running
on GPU2 and Random epoch15 Miss PID1870217 on GPU3; both require the same
479/337,353 full scope. R1/R2 trainers3388846/3395457 remain alive and RPPM
queue3539938 remains `WAITING_FOR_R` on four missing R receipts. GPU0-3:
80/77/74/73 C,246.43/247.12/249.14/247.98W, utilization100/100/100/91%.
No additional task was dispatched; configs, manifests and checkpoints were
left untouched.

## 2026-11-03 22:31 UTC - Miss evaluations active; early sequences are long

Read-only poll at22:31:43Z confirmed both full Miss evaluators remain alive:
Utility epoch14 PID1873594 on GPU2 and Random epoch15 PID1870217 on GPU3.
Random has completed four LasHeR-Miss sequences totaling39,393 frames:
"blkboy`shead" (12,862; 695.68s), `broom` (9,538; 742.95s),
`boyshead9684` (9,684; 755.96s), and `midredboy` (7,309; 774.77s).
This is 11.7% of its registered337,353-frame scope, not a partial result.
Utility has no completed sequence record yet; its process and workers remain
live with GPU2 at100% utilization, so no stall or failure is established.
The long initial sequences explain the earlier lack of per-sequence log rows;
any ETA remains provisional.

A queue1355742 and arm controllers1601932/1601933 remain
`OFFICIAL_RUNNING`; no queue-level completion or final formal verification
exists. A training receipts and saved-evidence `verification_v2.json` remain
complete/passed. R1/R2 trainers3388846/3395457 remain live and RPPM3539938
still waits for four R receipts. GPU0-3:81/78/77/73 C,
246.45/246.75/249.43/249.02W, utilization100/60/100/100%. No new dispatch
or intervention; configurations, manifests and checkpoints are unchanged.

## 2026-11-03 22:36 UTC - Both full Miss evaluations continue

Read-only poll at22:36:06Z confirmed Utility epoch14 Miss PID1873594 and
Random epoch15 Miss PID1870217 are alive on GPU2/3. Utility has completed
2/479 sequences and20,171/337,353 frames; Random has completed5/479 and
43,779/337,353 frames. No Miss metrics or completion receipts are available.
The A queue and controllers remain `OFFICIAL_RUNNING`; training receipts and
training-evidence `verification_v2.json` remain complete/passed, but no
queue-level final verification exists. R1/R2 remain alive and RPPM still
waits on four R receipts. GPU0-3 are81/78/77/73 C,
246.50/247.06/249.29/248.98W, all at100% utilization. No new task or
intervention; all configs, manifests and checkpoints remain untouched.

## 2026-11-03 22:39 UTC - Full Miss evaluation frame counts advanced

At22:39:03Z both official Miss evaluators remained alive. Utility epoch14
PID1873594 logged4/479 completed sequences and39,393/337,353 frames;
Random epoch15 PID1870217 logged7/479 and53,017/337,353 frames. No full
completion receipt or Miss metric is present yet. Queue1355742 and both arm
controllers1601932/1601933 remain `OFFICIAL_RUNNING`; A training receipts
and its saved-evidence verification remain complete/passed. R1/R2 trainers
3388846/3395457 remain live, while RPPM3539938 still waits for four R
receipts. GPU0-3 are81/78/76/73 C,246.68/246.86/249.13/249.19W,
utilization100/100/100/100%. No intervention or dispatch; configs,
manifests and checkpoints remain unchanged.

## 2026-11-03 22:41 UTC - Miss evaluation progress; brief power readings above cap

At22:41:42Z the two full Miss evaluators remained live. Utility epoch14
PID1873594 logged6/479 sequences and51,037/337,353 frames; Random epoch15
PID1870217 logged10/479 and71,398/337,353 frames. No completion receipts or
Miss metrics are available. A queue1355742/controllers1601932/1601933 remain
`OFFICIAL_RUNNING`; A training receipts and saved-evidence verification
remain complete/passed, while final queue verification is absent. R1/R2
trainers3388846/3395457 remain live; RPPM3539938 still waits for four R
receipts.

GPU0-3 temperatures were80/78/77/73 C; power draws were246.42/251.22/
250.04/249.26W against250W limits; utilization was100% on all cards. This
single sample shows small GPU1/2 power-limit overruns and does not establish
a sustained condition. No task, config, manifest or checkpoint was changed.

## 2026-11-03 22:44 UTC - Miss evaluation progress continues

At22:44:45Z Utility epoch14 Miss PID1873594 logged7/479 sequences and
55,958/337,353 frames; Random epoch15 Miss PID1870217 logged13/479 and
82,833/337,353 frames. Both evaluators, A queue1355742 and controllers
1601932/1601933 remain live; no complete/Miss receipts or queue-level formal
verification exist. A training receipts and training-evidence verification
remain complete/passed. R1/R2 trainers3388846/3395457 remain live; RPPM
3539938 still waits for four R receipts.

GPU0-3 were81/78/77/73 C, power245.79/247.28/249.79/249.44W under250W
limits, utilization100% each. No new task, intervention, config/manifest
change or checkpoint cleanup.

## 2026-11-03 22:52 UTC - Full Miss evaluations continue

At22:52:55Z, Utility epoch14 Miss PID1873594 had logged15/479 sequences
and89,860/337,353 frames; Random epoch15 Miss PID1870217 had logged23/479
and112,030/337,353. Both evaluators and A queue1355742/controllers1601932/
1601933 remain live in `OFFICIAL_RUNNING`; their full `completed.json`
receipts and queue-level final formal verification are still absent.

The A training receipts remain complete (Utility17 epochs/34,000 updates,
Random18/36,000); all checkpoints are retained and official-test selection
was not used. Training `verification_v2.json` remains `passed` for saved
evidence only. R1/R2 trainers3388846/3395457 remain live, and RPPM queue
3539938 remains `WAITING_FOR_R` on four missing training/full-official
receipts. GPU0-3 were80/77/76/73 C, drawing246.06/247.21/248.64/249.08W
under250W; utilization100% and memory19160/19160/9106/9138MiB. No task was
dispatched or changed; manifests and checkpoints were left untouched.

## Follow-up poll - R replay complete; official gates remain closed

Server 28's CPU-only protocol-R reconstruction report now records
`status=complete`, reconstructed epochs1-20 and 380,000 replayed tuples for
epochs2-20. CUDA remained uninitialized. Report SHA256:
`5ac52cde8a07af99e344ed6a724a2ef530ded77d7d4cadcd286f342fce96437d`.
This is candidate loader reconstruction, not proof of the actual inputs seen
by R1/R2. The report keeps original-input certification, historical
exclusion verification, and the OHCAS data gate false.

The live poll still found A queue1355742/controllers1601932/1601933,
Utility Miss evaluator1873594, Random Miss evaluator1870217, R1/R2 trainers
3388846/3395457, and RPPM queue3539938 alive. The queue remains
`WAITING_FOR_R` on four R training-verification/full-official receipts.
Utility and Random A training `completed.json` receipts were rechecked
(17/34,000 and18/36,000); saved-evidence `verification_v2.json` is
`passed`, but is not queue-level official verification. GPU0-3 measured
81/78/76/73 C and246.52/247.44/248.97/249.64 W under250 W caps, at
82/97/100/100% utilization. All four cards remain occupied. No task was
dispatched, and no process, config, manifest, or checkpoint was modified.

## 2026-11-03 23:19 UTC - R1/R2 training advanced

Read-only inspection of the live training logs showed R1 at epoch14,
update34450, and R2 at epoch14, update16875. Both trainer PIDs remain alive;
GPU process monitoring showed active compute on GPUs0/1. The A queue and its
two full Miss evaluators remain alive without Miss completion receipts or
queue-level formal verification. RPPM remains `WAITING_FOR_R` on the same
four receipts. GPU0-3 were80/78/77/73 C and249.52/247.66/249.72/249.16W,
all below250W. No run was dispatched or altered.

## 2026-11-03 23:27 UTC - R2 log gap resolved

A follow-up poll several minutes after R2's log last advanced found both R
trainers alive with fresh log records: R1 reached epoch14/update34625 and
R2 epoch14/update16950. GPUs0/1 both reported100% utilization; temperatures
were81/78 C and power246.61/248.11W. The earlier R2 logging gap did not
persist, so no stall is established and no intervention was made. A queue
and both Miss evaluators remain live without full Miss completion receipts;
RPPM remains `WAITING_FOR_R` on the same four receipts.

## 2026-11-03 23:42 UTC - R1 crossed into epoch 15

The live R training logs show R1 at epoch15/update35025 and R2 at
epoch14/update17150; both trainer PIDs remain alive. This epoch boundary is
training progress only, not convergence or official-metric evidence. A
queue1355742, its controllers, and both full Miss evaluators remain alive;
no Miss `completed.json` receipts or queue-level formal verification are
present. RPPM remains `WAITING_FOR_R` on four receipts. GPU0-3 were
78/77/76/73 C,249.48/247.02/249.17/248.81W under250W, at
78/100/100/100% utilization. No new task was dispatched or modified.

## 2026-11-03 23:59 UTC - Random epoch15 full Miss evaluation finished

Random epoch15 Miss now has a stable `completed.json` receipt covering
479/479 sequences and337,353/337,353 frames. Its saved `metrics.json`
covers LasHeR245/220,703 and RGBT234234/116,650:
LasHeR-Miss PR/NPR/SR=67.890547/62.761895/53.760802;
RGBT234-Miss MPR/MSR=83.890983/59.846600. Relative to the requested strict
thresholds, this candidate is short by1.409453 PR,0.639198 SR,4.209017 MPR,
and4.453400 MSR points. These are full-scope raw candidate metrics, not yet
queue-level formal verification or promotion evidence.

Utility epoch14 Miss PID1873594 and A queue1355742/controllers1601932/1601933
remain active, with no Utility completion receipt and no queue-level
verification artifact. Random evaluator1870217 has exited and GPU3 is
released, but the protocol requires both A Miss receipts and official queue
formal verification `passed` before scheduling the next phase. RPPM3539938
also remains `WAITING_FOR_R` on four R training/official receipts. No task
was started on GPU3 and no checkpoint was changed or removed.

## 2026-11-04 00:11 UTC - Utility epoch14 Miss complete; official queue continues

Read-only Server28 poll at00:11:50Z verified Utility epoch14 Miss
`completed.json`:479/479 sequences and337,353/337,353 frames. Full metrics:
LasHeR-Miss PR/NPR/SR=67.174178/61.863332/53.313374;
RGBT234-Miss MPR/MSR=84.735751/60.410309. Receipt SHA256
f5a2a5305c3027e48e409ab3417013833f919e5efa3c25738cd7be6e38a41af4;
metrics SHA256 6d00efb6b36aa7be3ceb40c49a493a0b45ec6c644aee87da85edf3ee0e7ba2d7.
This remains a full-scope candidate result, below the requested strict
thresholds; it is not queue-level completion/promotion evidence.

The registered queue PID1355742 and controllers1601932/1601933 remain live.
They automatically advanced Utility to epoch15 Complete PID1938748/GPU2
(started00:08:34Z) and Random to epoch16 Complete PID1934927/GPU3
(started00:03:33Z). R1/R2 trainers3388846/3395457 and RPPM queue3539938
remain live. The A training receipts remain complete (17/34,000 and
18/36,000), and the training-scope `verification_v2.json` is `passed`;
however, the official queue remains `OFFICIAL_RUNNING`, with no queue
`completed.json` or final queue-level verification. Do not schedule the
next stage until that gate passes. RPPM remains gated on its four R receipts.

GPU0-3 were80/77/76/73 C, drawing247.11/245.87/249.22/248.87W under250W,
at96/100/100/99% utilization. All four GPUs are occupied. No new job,
configuration, manifest, or checkpoint was changed.

## 2026-11-04 00:31 UTC - A queue and RPPM remain gated

Read-only Server28 poll: A official queue1355742 and controllers1601932/
1601933 remain live in `OFFICIAL_RUNNING`. Utility epoch15 Complete PID
1938748/GPU2 and Random epoch16 Complete PID1934927/GPU3 are still running;
there is no queue-level `completed.json` or final formal-verification
receipt. The A training receipts and saved-evidence `verification_v2.json`
remain complete/passed, which does not satisfy the queue-level gate.

R1/R2 trainers3388846/3395457 remain live. Latest log rows are epoch15,
updates36300/17775; this is progress, not training completion or score
evidence. RPPM queue3539938 remains `WAITING_FOR_R` on the four R training
verification and full-official-evaluation receipts. Its actual-input frame
provenance and OHCAS exclusion gate remain unverified.

The previously completed full-scope Miss candidates remain below the strict
targets: Utility epoch14 LasHeR PR/SR67.174/53.313 and RGBT234 MPR/MSR
84.736/60.410; Random epoch15 67.891/53.761 and83.891/59.847.

GPU0-3 were80/77/76/73 C, drawing246.66/246.86/248.98/249.37W under250W,
at100/100/100/99% utilization. All four GPUs are occupied.

Local source review confirmed the OHCAS adapter remains an isolated,
zero-initialized residual prototype with CPU contract tests only; it is not
integrated into `MissingTrackerModel` and has no tracking-efficacy evidence.
The fresh-frame metadata report still marks historical-frame exclusions
unverified, including R1/R2, so the OHCAS data gate is closed. No process,
runtime source, config, manifest, or checkpoint was changed; no next-stage
job was dispatched.

## 2026-11-04 00:40 UTC - OHCAS online interface gap confirmed

Read-only source review found that the proposed adapter consumes one
`[B,N,D]` native/history stream pair, age, and modality-level availability.
`MissingTrackerModel` operates on paired RGB/TIR block outputs containing
class, template, and search tokens, with the search tokens at the tail and a
per-token mask. `MissingTracker` currently stores templates/availability
only; it has no online history crop/feature cache or age. The existing
history diagnostic uses separately cached crops and hooks, so it does not
establish online tracking behavior.

This is a concrete integration contract gap, not evidence for a new
defensive branch or an efficacy claim. OHCAS implementation remains gated
on the active A/R protocols and verified fresh-frame exclusions. Runtime and
active jobs were not changed.

## 2026-11-04 00:47 UTC - Full evaluation queues continue

Read-only Server28 poll at00:47:30Z found A official queue1355742 and both
arm controllers still live in `OFFICIAL_RUNNING`. Utility epoch15 Complete
PID1938748/GPU2 has logged19/479 sequence completions; Random epoch16
Complete PID1934927/GPU3 has logged27/479. Each run is registered for the
full 337,353-frame evaluation. No queue-level completion or verification
file exists yet. A training receipts remain complete and training-scope
`verification_v2.json` remains passed; this does not release the queue gate.

R1/R2 trainers3388846/3395457 remain live at epoch15, latest updates
36675/17950. RPPM queue3539938 remains `WAITING_FOR_R` on both training
verification and full-official receipts for each arm.

GPU0-3 were80/76/76/73 C, drawing247.46/247.91/249.58/249.32W under250W,
at100/100/100/99% utilization. All cards remain occupied. No dispatch or
runtime/config/manifest/checkpoint change was made.

## 2026-11-04 00:50 UTC - A full evaluations and R training advance

Read-only Server28 poll: A queue1355742 and controllers1601932/1601933
remain live in `OFFICIAL_RUNNING`. Utility epoch15 Complete PID1938748 has
logged24/479 sequences; Random epoch16 Complete PID1934927 has logged
32/479. Both are full 337,353-frame evaluations. Queue-level completion
and formal verification files are still absent.

R1/R2 trainers3388846/3395457 remain live at epoch15, latest updates
36825/18025. RPPM queue3539938 remains `WAITING_FOR_R` on the four R
training-verification and full-official receipts.

GPU0-3 were80/77/75/72 C, drawing246.00/247.85/249.03/249.11W under250W,
at100/100/100/100% utilization. All cards remain occupied; no new job or
runtime/config/manifest/checkpoint change was made.

## 2026-11-04 00:55 UTC - Evaluation progress confirmed; no R1 stall

Two read-only observations confirmed the full A evaluations are advancing:
Utility epoch15 Complete PID1938748 moved from24 to31/479 completed
sequences, and Random epoch16 Complete PID1934927 moved from32 to39/479.
The corresponding logs include completed Miss sequences; both jobs remain
the registered full evaluations. No completion receipts or queue-level
formal verification are present.

R1/R2 remain live at epoch15, latest updates36925/18075. R1 briefly showed
state `D` in a single process snapshot, but the next snapshot showed it
runnable and its log advanced; this does not establish a persistent stall.
RPPM remains `WAITING_FOR_R` on the four expected R receipts.

All four GPUs have active assigned jobs. At this sample GPU0-3 were
80/77/76/72 C, drawing244.30/247.08/249.30/249.17W under250W; utilization
was90/100/100/100%. No process, config, manifest, checkpoint, or dispatch
was changed.

## 2026-11-04 00:57 UTC - A and R progress; utilization fluctuates

The two full A evaluations remain live and advanced: Utility epoch15
Complete is35/479 sequences and Random epoch16 Complete is44/479.
R1/R2 remain live at epoch15, latest updates37000/18100. R1's earlier
single-snapshot `D` state was transient; subsequent process and log checks
showed it runnable with continued updates. No persistent stall is evidenced.

The official A queue remains `OFFICIAL_RUNNING` without queue completion or
formal verification. RPPM remains `WAITING_FOR_R` on the four expected
training-verification/full-official receipts.

GPU0-3 still have their assigned R1/R2/A evaluator processes and associated
memory allocations, but utilization at this sample was48/73/98/100%, not
four-card saturation. Temperatures were80/75/75/72 C and power
242.18/247.02/248.75/249.52W under250W. No card was released and no new job
was dispatched; active processes, configs, manifests, and checkpoints were
left unchanged.

## 2026-11-04 01:00 UTC - A full evaluations continue

Read-only poll found Utility epoch15 Complete at39/479 sequences and
Random epoch16 Complete at50/479. Both evaluator PIDs and A queue1355742
remain live; no queue completion or formal-verification receipt exists.
R1/R2 remain live at epoch15, latest updates37050/18125. RPPM queue3539938
still waits for both R training-verification and full-official receipts.

GPU0-3 were80/76/76/72 C, drawing246.33/246.52/249.33/248.98W under250W,
at100% utilization for this sample. All cards have active assignments.
No new dispatch or runtime/config/manifest/checkpoint change was made.

## 2026-11-04 01:02 UTC - A and R jobs continue

Utility epoch15 Complete advanced to43/479 sequences and Random epoch16
Complete to55/479. Both remain live full evaluations under A queue1355742;
there is still no queue-level completion or formal-verification receipt.
R1/R2 remain live at epoch15, latest updates37100/18150. RPPM queue3539938
remains `WAITING_FOR_R` on the four training-verification and full-official
receipts.

GPU0-3 were80/77/76/72 C, drawing246.50/247.13/249.21/249.24W under250W,
at100% utilization. All four cards have active jobs. No process, config,
manifest, checkpoint, or dispatch was changed.

## 2026-11-04 01:04 UTC - A progress and transient power sample

Utility epoch15 Complete advanced to48/479 sequences and Random epoch16
Complete to59/479. A queue1355742 and both full evaluation workers remain
live without queue completion/formal-verification receipts. R1/R2 remain
live at epoch15, latest updates37150/18175; RPPM remains gated on the four
R training-verification and full-official receipts.

GPU0-3 measured79/76/76/72 C, 247.76/247.81/249.12/249.40W against250W
limits, with100% utilization. An immediately preceding sample briefly read
GPU1 at251.06W; this follow-up read returned247.81W, so sustained power
exceedance was not observed. All four GPUs have active assignments. No
intervention or runtime/config/manifest/checkpoint/dispatch change was made.

## 2026-11-04 01:06 UTC - A and R progress continues

Utility epoch15 Complete advanced to52/479 sequences and Random epoch16
Complete to64/479. A queue1355742 and both evaluators remain live, with no
queue-level completion or formal-verification receipt. R1/R2 remain live at
epoch15, latest updates37225/18225. RPPM queue3539938 still waits for the
four R training-verification/full-official receipts.

GPU0-3 were80/77/75/72 C and246.32/246.15/248.81/248.99W under250W,
at82/100/100/100% utilization. All four cards remain assigned to active
jobs; utilization is not uniformly saturated. No run, configuration,
manifest, or checkpoint was changed, and no additional job was dispatched.

## 2026-11-04 01:22 UTC - Research decision and gate recheck

The attached research review was cross-checked against the paired full-scope
metrics and current RPPM source. Batch10 Utility epoch17 clears the LasHeR-Miss
PR/SR targets but remains below the RGBT234-Miss MPR/MSR targets by
3.002622/3.614066 points; its Complete RGBT234 MPR/MSR also remain below the
SCDT table values. See `docs/RESEARCH_DECISION_20261104.md` for the same-
checkpoint comparison, caveats, and module order.

The active RPPM `paced` arm assigns lower missing probability to samples
classified hard by complete-view loss EMA. Its `shuffled` arm only permutes
the epoch-level 0.2/0.5/0.8 schedule, so it is not a matched per-sample
assignment control. This limits attribution of any paced gain; the frozen
campaign was not changed. Any matched control must be separately preregistered
after existing queue gates.

At01:22UTC A queue1355742/controllers1601932/1601933 and the epoch15/16
Complete evaluators remained live. A training receipts and saved-evidence
`verification_v2.json` were complete/passed, but A queue terminal receipts
and queue-level formal verification were absent. R1/R2 remained live at
epoch16/update37625 and epoch15/update18425; RPPM queue3539938 remained
`WAITING_FOR_R` on four receipts.

GPU0-3 were77/77/76/72 C, drawing247.43/246.01/249.49/248.97W under250W,
at91/100/100/100% utilization. All cards remained assigned. No active
process, config, manifest, checkpoint, or queue was changed; no task was
dispatched. All checkpoints remain retained.

## 2026-11-03 - SCDT scorer provenance follow-up

The arXiv paper's code link resolves to the SCDT-CVPR2026 GitHub repository,
which currently displays as empty; open issue #1 asks when code will be
released, and `git ls-remote` returned no refs during this audit. This means
exact SCDT scorer parity remains unverified; it does not prove that no private
or separately hosted scorer exists. The pinned dataset/benchmark resource
repository is not proof of which archive SCDT executed.

See `docs/SCORING_PROVENANCE_FOLLOWUP_20261103.md`. Continue labeling SCDT
numbers as published comparison targets, not controlled head-to-head
results. No local score, active job, or checkpoint was changed. The latest
Server28 poll still showed A's full evaluations and R1/R2 training active,
with all GPUs assigned; the queue-level formal gate remains pending, so no
next-stage job is authorized.

## 2026-11-04 01:52 UTC - R training enters epoch 16

Read-only logs show R1 PID3388846 at epoch16/update38425 and R2 PID3395457
at epoch16/update18800. Their fixed 128-tuple development mean losses from
epochs10-15 moved from4.8844 to4.6684 for R1, monotonically down, and from
4.8249 to4.7128 for R2 with an epoch14 rebound. This internal trend is not
an official score; neither run has a completed 20-epoch history, so it is
not enough to declare convergence or stop early. Continue the registered
20-epoch runs and final-five development selection.

## 2026-11-04 02:05 UTC - R2 epoch16 Complete evaluation audited

The registered A queue's Random epoch16 Complete run has a stable receipt:
479/479 sequences and337,353/337,353 frames, including LasHeR245/220,703
and RGBT234234/116,650. The queue controller appended the result with
`independently_recomputed=true`.

| Dataset | Metric | R2 epoch16 | E0 Complete | Delta |
|---|---|---:|---:|---:|
| LasHeR | PR | 73.154501 | 75.636503 | -2.482002 |
| LasHeR | NPR | 69.602719 | 71.772837 | -2.170118 |
| LasHeR | SR | 58.395139 | 60.125714 | -1.730575 |
| RGBT234 | MPR | 89.198793 | 90.677134 | -1.478341 |
| RGBT234 | MSR | 65.126610 | 66.845228 | -1.718618 |

This is an interim final-five candidate, not the development-selected
checkpoint: Random selected epoch15. It is Complete-only evidence; Random
epoch16 Miss is still running, so this is not a same-checkpoint paired
conclusion. Do not select or discard weights from these official-test
numbers. Keep all checkpoints and wait for the complete registered queue
and its formal verification.

At02:05UTC, A queue1355742/controllers1601932/1601933 remained live.
Utility epoch15 Complete PID1938748 and Random epoch16 Miss PID2009135 were
active on GPUs2/3. R1/R2 trainers3388846/3395457 and RPPM queue3539938
remained live/gated. No queue-level A completion/formal verification or R
receipts were present; no independent task was dispatched.

## 2026-11-04 02:53 UTC - A Miss evaluation and R training progress

Read-only poll at02:53:54Z confirmed A queue1355742/controllers1601932/
1601933, Utility epoch15 Miss PID2016182, Random epoch16 Miss PID2009135,
R1 PID3388846, R2 PID3395457, and RPPM queue3539938 live. Both Miss
evaluations continue on the full registered479-sequence/337,353-frame
scope; their `completed.json` receipts, A queue `completed.json`, and
queue-level `verification_v2.json` are absent. Latest worker log rows
advanced, but no final metrics are available.

R1 reached epoch16/update40000 with fixed128-tuple development mean loss
4.692165, slightly above epoch15's 4.668373. R2 was at epoch16/update19575;
its latest completed development mean is epoch15's 4.712759. The registered
configs retain20epochs and select by development mean across epochs16-20.
These internal losses do not establish convergence or official benchmark
performance; continue both runs and preserve all checkpoints.

GPU0-3 were79/74/76/73C,243.61/243.11/249.07/249.30W, and
81/53/100/100% utilization. All cards remain assigned. No training,
configuration, manifest, checkpoint, or queue was changed; no additional
job was dispatched because A's formal queue gate remains pending.

## 2026-11-04 03:02 UTC - A evaluation advances; R1 enters epoch 17

Read-only poll at03:02:16Z confirmed A queue1355742/controllers1601932/
1601933, Utility epoch15 Miss PID2016182, Random epoch16 Miss PID2009135,
R1 PID3388846, R2 PID3395457, and RPPM queue3539938 live. Both full-scope
Miss evaluations appended more completed sequence rows, but neither has a
479/479 completion receipt. The A queue remains `OFFICIAL_RUNNING`; its
queue-level completion and formal-verification receipts remain absent.

R1 advanced to epoch17/update40225. R2 remains in epoch16/update19700.
The final-five development selection is still pending, and current batch
losses do not establish convergence or benchmark performance.

GPU0-3 were80/77/76/72C,246.74/246.39/248.94/248.83W, all at100%
utilization. All cards remain assigned. No run, configuration, manifest,
checkpoint, or queue was changed; no additional job was dispatched.

## 2026-11-04 03:08 UTC - Active evaluation progress and power recheck

The 03:07:48Z poll found the same A official queue/controllers, Utility
epoch15 Miss and Random epoch16 Miss evaluators, R1/R2 trainers, and RPPM
queue alive. Both Miss logs appended completed-sequence rows; neither
evaluation nor the A queue has a terminal completion receipt or formal
verification. R1 was at epoch17/update40375 and R2 at epoch16/update19750.

One GPU1 sample read253.00W at77C; a follow-up at03:08:55Z read247.01W.
GPU0-3 then measured246.44/247.01/248.86/249.10W at80/77/75/72C, all
100% utilized. The single over-limit sample was not sustained. No power
setting, job, configuration, manifest, or checkpoint was changed.

## 2026-11-04 03:21 UTC - A Miss rows and R epoch progress

At03:21:39Z the official A queue1355742/controllers1601932/1601933,
Utility epoch15 Miss PID2016182, Random epoch16 Miss PID2009135, R1
PID3388846, R2 PID3395457, and RPPM queue3539938 were all live. Both
Miss evaluators appended sequence rows; both remain on the complete
479-sequence/337,353-frame plans and still lack `completed.json`.
The official queue remains `OFFICIAL_RUNNING` without queue-level
completion or `verification_v2.json`.

R1 advanced to epoch17/update40725; R2 advanced within epoch16 to
update19950. No final-five selection or convergence result is available.
GPU0-3 were80/76/75/72C, drawing246.30/247.38/249.44/249.34W at100%
utilization. All cards remain assigned; no additional work was dispatched
and no training/configuration/manifest/checkpoint state was altered.

## 2026-11-04 03:26 UTC - A evaluations remain active; R development update

At03:26:57Z A queue1355742/controllers1601932/1601933, Utility epoch15
Miss PID2016182, Random epoch16 Miss PID2009135, R1 PID3388846, R2
PID3395457, and RPPM queue3539938 remained live. Both Miss logs appended
sequence rows, but neither full479-sequence evaluation has a completion
receipt. The A official queue remains `OFFICIAL_RUNNING` without queue-level
completion or formal verification.

R1 was at epoch17/update40875. R2 reached epoch16/update20000 and recorded
development mean task loss4.725430 over128 fixed tuples. This is the
registered internal development signal, not an official metric or evidence
of convergence. Final-five selection remains pending.

GPU0-3 were79/77/76/73C,243.53/245.29/249.39/249.24W; GPU0 utilization
was48%, with GPUs1-3 at100%. All cards remain assigned. No additional job
or experiment intervention was made.

## 2026-11-04 03:37 UTC - A/R live status and GPU0 power recheck

At03:36:03Z the official A queue/controllers and both full Miss evaluators
remained live, as did R1/R2 training and the gated RPPM queue. Utility
epoch15 Miss and Random epoch16 Miss appended sequence rows but still have
no full-scope completion receipts; A queue completion and formal
verification are also absent. R1 was epoch17/update41100 and R2
epoch17/update20125.

GPU0 briefly read250.55W at80C. A follow-up at03:37:07Z read246.06W;
GPU0-3 then measured246.06/247.94/249.24/249.37W at80/76/76/72C.
The exceedance was not sustained. No power limit, active run, configuration,
manifest, or checkpoint was changed; no task was dispatched.

## 2026-11-04 03:42 UTC - A Miss workers and R training still active

At03:42:23Z A queue1355742/controllers1601932/1601933, Utility epoch15
Miss PID2016182, Random epoch16 Miss PID2009135, R1 PID3388846, R2
PID3395457, and RPPM queue3539938 remained live. Both Miss evaluation logs
appended sequence rows but lack full completion receipts. The queue remains
`OFFICIAL_RUNNING` with no queue-level completion or formal verification.

R1 was at epoch17/update41275; R2 at epoch17/update20200. No final-five
development selection is available yet. GPU0-3 were80/74/76/72C,
246.93/246.20/249.10/249.19W, all at100% utilization. All cards remain
assigned. No new task or intervention was made.

## 2026-11-04 03:47 UTC - Random epoch16 Miss completed; queue gate remains closed

At03:45-03:47UTC, Random epoch16 Miss PID2009135 exited and its stable
`completed.json` recorded479/479 sequences and337,353/337,353 frames. Its
dataset counts are LasHeR245/220,703 and RGBT234234/116,650. The generated
metrics are LasHeR-Miss PR/NPR/SR67.805983/62.657725/53.763168 and
RGBT234-Miss MPR/MSR84.363668/60.414559. These numbers are provisional:
the official queue's independently verified metrics do not yet include this
run, and queue-level `verification_v2.json`/`completed.json` remain absent.

Utility epoch15 Miss PID2016182 remains alive with no completion receipt.
Queue1355742 and controllers1601932/1601933 remain live in
`OFFICIAL_RUNNING`. GPU3 is released; at03:47:28UTC GPU0-3 were
77/78/75/39C and247.23/247.57/249.27/25.21W, with utilization
55/100/100/0%. R1/R2 trainers3388846/3395457 and RPPM queue3539938 remain
live. Do not dispatch onto GPU3: the paired A Miss receipts and queue-level
formal verification=`passed` are still required. No runtime, config,
checkpoint, or manifest was changed.

## 2026-11-04 03:51 UTC - Random epoch16 independently verified; epoch17 evaluation started

The A queue's independent recomputation now includes Random epoch16 Complete
and Miss, both on the full479-sequence/337,353-frame scope. Random16 Miss
metrics are LasHeR PR/NPR/SR67.805983/62.657725/53.763168 and RGBT234
MPR/MSR84.363668/60.414559; all four remain below the requested
69.3/54.4/88.1/64.3 thresholds. Random16 Complete is
73.154501/69.602719/58.395139 and89.198793/65.126610. Its Miss metrics
are verified, but the A queue's overall `completed.json` and formal
`verification_v2.json` are still absent.

The queue launched Random epoch17 Complete PID2070557 on GPU3. Utility
epoch15 Miss PID2016182 remains alive; its log has377 completed-sequence
rows, with no full completion receipt. R1/R2 remain live at epoch17 updates
41550/20325; individual training losses fluctuate and are not convergence
or benchmark evidence. RPPM queue3539938 remains gated. GPU0-3 after a
repeat power reading were79/77/75/72C,246.87/246.21/249.20/249.13W,
all100% utilized. GPU1 had one251.21W sample and then returned to246.21W;
no sustained over-limit condition or intervention. All four GPUs are
assigned to authorized runs; no new task was dispatched.

## 2026-11-04 04:00 UTC - Utility epoch15 Miss independently verified

Utility epoch15 Miss PID2016182 exited normally; the authoritative receipt
records `ended_utc=2026-11-04T03:59:14Z`. Its stable
completion receipt covers479/479 sequences and337,353/337,353 frames
(LasHeR245/220,703; RGBT234234/116,650). The queue independently recomputed
the same full run. Metrics are LasHeR-Miss PR/NPR/SR
67.511848/62.192528/53.572766 and RGBT234-Miss MPR/MSR
84.522621/60.189169. They remain below the requested thresholds by
1.788152/0.827234/3.577379/4.110831 points.

The queue immediately reused GPU2 for Utility epoch16 Complete PID2077250;
Random epoch17 Complete PID2070557 remains live on GPU3. Queue1355742 and
both arm controllers remain alive. Despite the two full Miss receipts and
their independently recomputed metrics, the queue-level `completed.json`
and formal `verification_v2.json` have not appeared, so no next-stage
dispatch is authorized. R1/R2 remain in epoch17 at updates41750/20425;
their fluctuating training losses are not convergence evidence. RPPM queue
3539938 remains alive and gated.

At04:00:28UTC GPU0-3 were80/77/64/73C and247.10/246.55/249.10/249.38W,
all100% utilized. All GPUs are assigned to registered jobs. No active run,
configuration, manifest, or checkpoint was changed or removed.

## Verified A Miss candidate gap analysis

Among the six independently recomputed full-scope Miss runs currently
available (Utility epochs13-15; Random epochs14-16), none reaches all four
requested thresholds. Per-metric maxima are split across checkpoints:
LasHeR-Miss PR67.890547 (Random15;1.409453 below69.3), SR53.846060
(Random14;0.553940 below54.4), RGBT234-Miss MPR84.735751
(Utility14;3.364249 below88.1), and MSR60.717600 (Utility13;3.582400
below64.3). These are descriptive maxima across tested checkpoints, not a
single deployable model or a basis for test-set selection. The A queue is
still evaluating later registered checkpoints.

Within this verified subset, RGBT234-Miss has the largest remaining
threshold gaps; that observation does not establish a cause. It supports
measuring complete-condition localization and failure modes alongside
missing-modality behavior, rather than assuming stronger masking alone is
the remedy. Queue-level formal verification remains pending.

## 2026-11-04 04:09 UTC - R-to-P test-conditioned gate limitation recorded

Static review of the frozen RPPM protocol found that its preparation gate
uses the R1-selected checkpoint's five Complete official benchmark metrics
against E0-0.1pp to decide whether P proceeds. This makes official-test
outcomes affect the next research branch; preregistration does not turn
those test sets into development data. The active protocol remains frozen:
do not alter the gate or replace its outcome. The caveat and the future
rule (base new method/continuation decisions on a predeclared development
split, reserving official test results for final reporting) are recorded in
`docs/RESEARCH_DECISION_20261104.md`.

At04:09:28UTC A queue1355742/controllers1601932/1601933, Random epoch17
Complete PID2070557, Utility epoch16 Complete PID2077250, R1/R2 trainers
3388846/3395457, and RPPM queue3539938 remained alive. Queue-level
`completed.json` and `verification_v2.json` were still absent. GPU0-3 were
79/77/75/73C and246.81/247.16/249.04/249.00W, all fully assigned. No
runtime, config, training process, manifest, or checkpoint was changed.

## 2026-11-04 04:45 UTC - A Complete evaluators and R training progress

At04:45:35UTC the A queue1355742, controllers1601932/1601933, Utility
epoch16 Complete PID2077250, Random epoch17 Complete PID2070557, R1/R2
trainers3388846/3395457, and RPPM queue3539938 remained alive. The two
full479-sequence/337,353-frame Complete evaluations had30 and50 `DONE`
sequence records respectively; neither has a `completed.json` receipt.
Queue-level `completed.json` and formal `verification_v2.json` remain
absent, so downstream dispatch remains unauthorized.

R1 advanced within epoch18 to update42925; R2 remains in epoch17 at
update21000. Logged minibatch losses fluctuate and do not establish
convergence or benchmark performance. GPU0-3 were80/76/76/72C,
247.26/246.95/249.54/249.32W, all assigned to registered jobs. No
additional job was dispatched; no active process, configuration,
manifest, or checkpoint was changed.

## 2026-11-04 04:53 UTC - A evaluation progress and transient GPU1 power sample

At04:53:18UTC the A official queue/controllers, Utility epoch16 Complete
PID2077250, Random epoch17 Complete PID2070557, R1/R2 trainers, and RPPM
queue3539938 remained alive. Their full-scope evaluation logs reached43
and67 completed-sequence records; neither evaluator has a completion
receipt, and the queue-level completion/formal verification files remain
absent.

R1/R2 are at epoch18/update43125 and epoch17/update21100. GPU1's earlier
253.28W sample was not sustained: the 180-second recheck measured
245.86W. GPU0-3 were80/77/76/72C and248.24/245.86/249.14/249.16W,
all assigned to the registered runs. No power limit or workload was
changed; no next-stage dispatch is authorized.

## 2026-11-04 05:00 UTC - A evaluation progress and second GPU1 power recheck

At05:00:15UTC the A official queue/controllers, Utility epoch16 Complete
PID2077250, Random epoch17 Complete PID2070557, R1/R2 trainers, and RPPM
queue3539938 remained live. The two full-scope A evaluations logged58
and87 `DONE` sequence records. Neither evaluator completion receipt nor
the queue-level `completed.json`/formal `verification_v2.json` exists.

R1 remains epoch18 at update43300; R2 remains epoch17 at update21200.
GPU1's253.39W sample at04:56:37UTC was not sustained: the05:00:15UTC
reading was241.83W. GPU0-3 were80/75/75/72C and245.81/241.83/249.14/
248.98W. All cards remain assigned to registered workloads. No power
setting or running job was changed; no next-stage work is authorized.

## 2026-11-04 04:33 UTC - R1 entered epoch18; A full evaluations continue

At04:33:25UTC, A official queue1355742/controllers1601932/1601933,
Utility epoch16 Complete PID2077250, Random epoch17 Complete PID2070557,
R1/R2 trainers3388846/3395457, and RPPM queue3539938 were alive.
The two A Complete evaluators remain on their full479-sequence/
337,353-frame scopes; their logs contain14 and27 completed-sequence
records respectively, with neither `completed.json` present. Queue-level
`completed.json` and formal `verification_v2.json` are still absent.

R1 advanced to epoch18/update42600. R2 remains epoch17/update20850.
Per-step losses fluctuate and are not convergence or benchmark evidence.
RPPM remains gated on R receipts. GPU0-3 were80/76/75/73C,
248.62/247.21/248.93/249.16W; all four cards were assigned to the
registered workloads. No new job was dispatched and no active run,
configuration, manifest, or checkpoint was changed.

## 2026-11-04 05:11 UTC - A evaluations continue; R2 enters epoch18

At05:11:03UTC A queue1355742/controllers1601932/1601933, Utility epoch16
Complete evaluator2077250, Random epoch17 Complete evaluator2070557,
R1/R2 trainers3388846/3395457, and RPPM queue3539938 remained live.
Neither evaluator has a `completed.json`; queue-level completion and formal
`verification_v2.json` remain absent. The A training receipts and
saved-evidence training verification remain complete/passed, respectively.

R1 was at epoch18/update43575. R2 advanced to epoch18/update21325; its
latest fixed development entry at epoch17/update21250 reported mean task
loss4.705640 over128 tuples. This single internal-loss observation is not
convergence or official benchmark evidence. RPPM remains gated on R training
verification and full-official receipts.

GPU0-3 were77/76/75/72C, with242.59/247.00/249.41/249.10W draw and
52/100/100/100% utilization. All cards remain assigned to registered
workloads. No new task was dispatched and no process, configuration,
manifest, checkpoint, or power setting was changed.

## 2026-11-04 05:17 UTC - A Complete evaluation progress

At05:16:54UTC the A official queue1355742/controllers1601932/1601933,
Utility epoch16 Complete evaluator2077250, Random epoch17 Complete
evaluator2070557, R1/R2 trainers3388846/3395457, and RPPM queue3539938
were alive. Their evaluator logs advanced to107/479 and155/479 `DONE`
sequence records. Neither evaluator has a completion receipt; the queue
remains `OFFICIAL_RUNNING` without queue-level completion or formal
`verification_v2.json`.

R1 reached epoch18/update43725; R2 reached epoch18/update21400. The latest
R2 fixed-development loss remains the epoch17/update21250 result4.705640
over128 tuples; the new step losses are not convergence or benchmark
evidence. RPPM still requires completed R training verification and all
full-official receipts.

GPU0-3 were80/76/75/72C and246.43/246.56/248.96/248.69W at
05:16:54UTC, with89/100/100/100% utilization. At05:17:18UTC temperatures
and power were79/76/75/72C and247.03/246.74/249.09/249.25W. All cards
remain assigned. No task or training state was changed.

## 2026-11-04 05:22 UTC - A full evaluations advance

At05:22:36UTC A official queue1355742/controllers1601932/1601933, Utility
epoch16 Complete evaluator2077250, Random epoch17 Complete evaluator
2070557, R1/R2 trainers3388846/3395457, and RPPM queue3539938 remained
alive. The two Complete evaluations advanced to128/479 and184/479 `DONE`
sequence records. Neither has a completion receipt; queue-level completion
and formal `verification_v2.json` remain absent.

R1 was at epoch18/update43875 and R2 at epoch18/update21475. Their
minibatch losses remain telemetry only. RPPM is still gated on R training
verification and full-official receipts.

GPU0-3 were80/75/75/72C, using246.79/246.25/248.92/248.91W, all within
their250W limits and at100/100/100/99% utilization. All cards remain
assigned to the registered jobs. No new task or intervention occurred.

## 2026-11-04 05:26 UTC - A full evaluations continue

At05:26:07UTC A official queue1355742/controllers1601932/1601933, Utility
epoch16 Complete evaluator2077250, Random epoch17 Complete evaluator
2070557, R1/R2 trainers3388846/3395457, and RPPM queue3539938 remained
alive. Complete evaluation logs advanced to144/479 and204/479 `DONE`
records. Neither evaluator completion receipt nor queue-level
`completed.json`/formal `verification_v2.json` is present.

R1 reached epoch18/update43975; R2 reached epoch18/update21525. These
minibatch records do not establish convergence or official performance.
RPPM remains gated on R training verification and full-official receipts.

GPU0-3 were77/76/75/72C and245.07/246.83/248.56/249.08W, with
67/100/100/100% utilization. All cards remain assigned to registered
jobs. No additional task or intervention occurred.

## 2026-11-04 05:31 UTC - A full evaluations continue

At05:31:21UTC A official queue1355742/controllers1601932/1601933,
Utility epoch16 Complete evaluator2077250, Random epoch17 Complete
evaluator2070557, R1/R2 trainers3388846/3395457, and RPPM queue3539938
remained alive. The Complete evaluation logs advanced to168/479 and
240/479 `DONE` records. Both evaluator receipts and the A queue-level
`completed.json`/formal `verification_v2.json` remain absent.

R1 was at epoch18/update44075; R2 was at epoch18/update21575. GPU0-3 were
80/77/75/72C, drawing246.22/246.82/249.02/249.00W at100% utilization.
All cards remain assigned to registered workloads. The A verification
gate remains closed, and RPPM still waits for R receipts. No process,
configuration, manifest, checkpoint, or power setting was changed.

## 2026-11-04 05:42 UTC - A full evaluations advance

At05:42:34UTC A official queue1355742/controllers1601932/1601933, Utility
epoch16 Complete evaluator2077250, Random epoch17 Complete evaluator
2070557, R1/R2 trainers3388846/3395457, and RPPM queue3539938 remained
alive. Complete evaluation logs advanced to240/479 and352/479 `DONE`
records. Both evaluator receipts and queue-level `completed.json`/formal
`verification_v2.json` remain absent.

R1 was at epoch18/update44400 and R2 at epoch18/update21725. No completed
training verification or R official evaluation receipts are present for
the active R arms; RPPM remains gated.

GPU0-3 were80/76/75/72C, drawing246.83/248.27/248.89/248.79W at
100/100/100/99% utilization. All cards remain assigned to registered
workloads. No process, configuration, manifest, checkpoint, or power
setting was changed; no new task was dispatched.

## 2026-11-04 05:47 UTC - Random epoch17 Complete near full coverage

At05:47:43UTC A queue1355742/controllers1601932/1601933, Utility epoch16
Complete evaluator2077250, Random epoch17 Complete evaluator2070557,
R1/R2 trainers3388846/3395457, and RPPM queue3539938 remained alive.
Random's full-scope log reached456/479 sequences and Utility reached
281/479. Neither evaluation has a completion receipt; queue-level
`completed.json` and formal `verification_v2.json` remain absent.

R1/R2 remain in epoch18 (latest observed updates44550/21800). RPPM's
05:45:47UTC state still lists all four R training-verification and
full-official receipts as missing.

GPU0-3 were80/76/76/72C, at247.16/243.14/248.56/248.77W with
95/52/100/100% utilization. All cards remain assigned to registered
workloads. No process, configuration, manifest, checkpoint, or power
setting was changed; downstream dispatch remains unauthorized.

## 2026-11-04 05:52 UTC - Random epoch17 Complete receipt; Miss starts

Random epoch17 Complete evaluator2070557 exited after a full-scope run.
Its `completed.json` records479 sequences/337,353 frames; `metrics.json`
records LasHeR245/220,703 and RGBT234234/116,650:
LasHeR PR/NPR/SR73.540330/70.003071/58.610920 and RGBT234
MPR/MSR88.267873/63.729881. This Complete-condition result exceeds
RGBT234 MPR88.1 but is0.570119 below MSR64.3; it is not evidence that the
requested Miss thresholds are met and is not a basis for test-set checkpoint
selection. Queue-level independent verification remains pending.

The A queue automatically started Random epoch17 Miss evaluator2138910 on
GPU3. Utility epoch16 Complete evaluator2077250 remains live on GPU2 at
336/479 sequence rows. Queue1355742/controllers1601932/1601933 remain
alive; queue-level completion and formal `verification_v2.json` are absent.
R1/R2 and RPPM remain live/gated. At05:53:30UTC GPUs0-3 were79/76/75/73C
and246.20/246.22/249.16/249.31W, all cards assigned. No manual dispatch,
process/configuration change, or checkpoint cleanup occurred.

## 2026-11-04 05:58 UTC - Utility epoch16 Complete near full coverage

At05:58:40UTC A queue1355742/controllers1601932/1601933, Utility epoch16
Complete evaluator2077250, Random epoch17 Miss evaluator2138910, R1/R2
trainers3388846/3395457, and RPPM queue3539938 remained alive. Utility
epoch16 Complete reached436/479 `DONE` sequence rows without its completion
receipt. Random epoch17 Complete has a full-scope completion receipt and
its Complete metrics recorded above; the A queue has not yet produced its
overall completion or formal `verification_v2.json`.

Random epoch17 Miss began on GPU3 and had no completed sequence row at
this observation. Treat this as early-run progress only, not a stall or
failure. R1/R2 remained in epoch18 at updates44850/21950. RPPM still lists
all four R training-verification/full-official receipts as missing.

GPU0-3 were80/75/75/72C and246.40/246.35/248.89/249.03W, with all cards
at100% utilization. No active job or configuration was changed; no new
task was dispatched.

## 2026-11-04 06:04 UTC - Utility epoch16 Complete receipt; Random Miss runs

Utility epoch16 Complete evaluator2077250 completed the full479-sequence/
337,353-frame scope. Its stable receipt records LasHeR245/220,703 and
RGBT234234/116,650. Metrics are LasHeR PR/NPR/SR
73.492540/70.024632/58.725722 and RGBT234 MPR/MSR89.013478/64.695470.
The RGBT234 Complete values exceed88.1/64.3, but remain below E0's
90.677/66.845 by1.663044/2.149530 points; LasHeR Complete PR/SR also trail
E0's75.637/60.126. These results do not establish the requested gains or
any Miss-condition target. The A official controller's independent
recomputation and queue-level formal verification are still pending.

Random epoch17 Miss evaluator2138910 remained live with four completed
sequence rows. The A queue/controller processes remain alive. GPU2 is
temporarily idle while its Utility controller verifies/advances the
registered queue; no manual work was placed on it. At06:04:52UTC GPU0-3
were80/76/41/73C and248.93/247.14/23.53/249.18W; utilization99/100/0/100%.
No runtime/configuration/checkpoint changes or cleanup occurred.
