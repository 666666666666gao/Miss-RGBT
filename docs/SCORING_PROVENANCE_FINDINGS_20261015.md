# Python rgbt 评分来源核查（2026-10-15）

**结论：当前归档评分器可确定对应 PyPI `rgbt==1.0.1` 的包内字节；其评分规则可追溯到 `opacity-black/RGBT_toolkit`。这仍不足以称为“数据集作者官方工具箱”，也不足以宣称与 SCDT 的实际评分程序严格等价。** 本次只读核查本地文本、公开发行包和上游源码；未连接训练服务器、运行神经网络、替换评分器或改动既有分数。逐文件机器比对明细保留于文末指定的私有缓存。

## 来源与版本

PyPI 发布元数据列作者为 `zjc`，主页指向 `opacity-black/RGBT_toolkit`；Git 提交记录使用 `opacity-black` 和相同联系邮箱。这建立了发行项目与仓库的关联，没有据此认定维护者就是数据集作者。[PyPI 1.0.1](https://pypi.org/project/rgbt/1.0.1/)、[项目元数据](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/pyproject.toml)

| 核查对象 | 实际建立的关系 |
|---|---|
| PyPI wheel，2023-06-19 发布 | 下载 SHA256 为 `00209d49c5f0a93d6fc931df056162a9c852adaad436ae6d80189b6e143060be`，与发布页一致；仅下载，未安装。 |
| 本地 `first5_metric_package/rgbt` | **1967/1967** 文件与 wheel 对应成员逐字节相同，包括已用 713 个 GT 文件、源码和序列/属性元数据。 |
| 评测后 `collection.json` 的安装包清单 | **3556/3556** 路径及 SHA256 与 wheel 一致。原先未持有的 1589 个安装副本仍未读取；本次取得的是哈希相同的公开发行字节。它不能将评测后的观察追认成启动前封存，也不证明安装方式或当时的 dist-info。 |
| `Alexadlu/RGBT_toolkit_python` | GitHub 标为上述仓库的 fork；HEAD 为 `be78ddf30d8d5b389bb5bcc975a4477daa0f9370`，2023-06-19。全部 3556 包文件内容对应 wheel，其中 3416 字节相同，140 仅 CRLF/LF 不同；无内容差异。 |
| 包主页仓库当前 HEAD | `33da1eb7bbe53d0bff67ca77b342013365749d39`，2024-03-09；上一提交即 `be78ddf…`。`metrics/metrics.py` 与本地逐字节相同；`utils.py` 全部 16 个函数及两个 GT/结果加载函数在去除类型标注后 AST 相同。当前 tree 另有绘图/API 变化，不能把整个 tree 写成字节相同。 |

版本依据：[wheel 及发布哈希](https://pypi.org/project/rgbt/1.0.1/#files)、[fork 关系](https://github.com/Alexadlu/RGBT_toolkit_python)、[共享提交](https://github.com/opacity-black/RGBT_toolkit/commit/be78ddf30d8d5b389bb5bcc975a4477daa0f9370)、[当前提交](https://github.com/opacity-black/RGBT_toolkit/commit/33da1eb7bbe53d0bff67ca77b342013365749d39)。比较读取 Git 对象原始字节，避免 Windows checkout 换行转换干扰。

包 README 自报与原始工具箱有时约 `0.1%` 差异，并称做过论文结果校准；这是维护者的陈述，既不是全输入误差上界，也不能证明当前模型排序不变。[固定版 README](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/README.md)

## 实际分数含义

下列规则来自本地执行代码，且与上述发行版本对应；既有[独立审计](../reports/e0_availability_20261015_v1/EXPERIMENT_AUDIT.md)已对真实轨迹重算，本次没有另造一组“修正成绩”。

| 项目 | 当前实际规则 |
|---|---|
| dtype / 取整 | 文本默认读为 float32；预测框在转换前用 NumPy `.round(0)`，半整数取偶。LasHeR GT 走同一加载函数，也取整；RGBT234 的双模态 GT 直接读，不取整。 |
| LasHeR 首帧与异常值 | 首预测框替换为 GT，**首帧仍在分母内**；注释的“ignore”不是删除首帧。截断多余结果到 GT 长度；从第二帧起，预测宽/高非正则用前一框。任一 GT 坐标/宽高 `<=0` 时把误差或 IoU 置 `-1`：PR/NPR 因 `<=阈值` 而算成功，SR 因 `>阈值` 而算失败，均不删分母。既有审计在已用 LasHeR GT 中计到 1260 帧，故它是实际相关规则。 |
| PR / NPR | 每序列先求帧比例，再对序列等权。PR 是中心误差 `<=20`；NPR 是按 **GT 宽高+1e-8** 归一化中心后，误差 `<=0.20` 的代表点。两条曲线均 51 个阈值；NPR 标量不是整条曲线平均或 AUC。 |
| SR / MSR | IoU 使用含端点像素的角点几何；21 个阈值 `0:0.05:1`，严格 `IoU>阈值`。最终对全部“序列×阈值”单元等权求均值，非按帧长加权，非梯形积分。 |
| RGBT234 MPR / MSR | 每帧分别与 visible/infrared GT 比较：MPR 取较小中心误差，MSR 取较大 IoU；随后按上行规则汇总。MPR/MSR 本身不执行 LasHeR 的首帧覆盖和无效 GT 分支。这里的 max/min 是外部双模态标注选择，不是模型输出自归一化。 |

直接源码：[结果与 GT 加载](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/src/rgbt/dataset/basedataset.py#L9-L24)、[几何与归一化](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/src/rgbt/utils.py#L117-L185)、[MPR/MSR](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/src/rgbt/metrics/metrics.py#L8-L85)、[LasHeR 三指标](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/src/rgbt/metrics/metrics.py#L240-L374)。因此既有 SR/MSR 可注明“21 点平均的 success AUC 实现”，不能声称数学上等同任意 AUC 算法；PR/NPR 也不能用整曲线均值替代代表点。

LasHeR 默认名单有 245 条，与数据集作者 `List_for_Protocol2/testingsetList.txt` **集合相同、顺序不同**；RGBT234 默认名单有 234 条。作者名单固定到 `BUGPLEASEOUT/LasHeR` 的 `c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399`（2022-07-09）。LasHeR 读取每序列 19 项属性，RGBT234 读取 12 份属性向量；属性筛选不用于默认全量分数。名单对应不能认证 GT 原始来源。[LasHeR 加载](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/src/rgbt/dataset/lasher_dataset.py#L11-L60)、[RGBT234 加载](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/src/rgbt/dataset/rgbt234_dataset.py#L13-L51)、[作者测试名单](https://github.com/BUGPLEASEOUT/LasHeR/blob/c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399/List_for_Protocol2/testingsetList.txt)

## 与作者推荐及 SCDT 的关系

CADTrack 的固定版 README 将这四个数据集的评测导向 Chenglong Li 的数据集页面。LasHeR 论文 v2 的 arXiv 摘要页直接链接 `BUGPLEASEOUT/LasHeR`，论文正文摘要另指 `mmic-lcl/Datasets-and-benchmark-code`，由此建立数据集作者来源，而非仅接受仓库自称“official”。后者固定版本 `16db98a85c0fc6b7661550a4970f10f33d6e048a` 的 Tool-Kit 栏分别指向 LasHeR 和 RGBT234 的百度网盘包，**没有将 `pip rgbt` 指定为这两个工具箱**。[CADTrack 评测说明](https://github.com/IdolLab/CADTrack/blob/93d0a67d3eb328617751b18510bdab072705c1a0/README.md#L131-L133)、[论文摘要页](https://arxiv.org/abs/2104.13202v2)、[论文正文](https://arxiv.org/html/2104.13202v2#abstract)、[作者资源表](https://github.com/mmic-lcl/Datasets-and-benchmark-code/blob/16db98a85c0fc6b7661550a4970f10f33d6e048a/index.html)

本次下载作者站和两个网盘入口后，入口返回提取码页面，**未取得作者工具箱压缩包或其执行结果**。所以尚不能逐行核实其 dtype、半整数舍入、首帧、无效 GT、阈值端点和聚合实现，也不能断言这些细节与 Python 包相同或不同。包内 GT 与公开发行包相同，同样不等于已与作者数据下载独立校验。[LasHeR 工具箱入口](https://pan.baidu.com/s/1SDohdsXcEkubF_pS_o43jw)、[RGBT234 工具箱入口](https://pan.baidu.com/s/1UksOGtD2yl6k8mtB-Wr39A)

已经能核实的是：LasHeR 论文规定 PR 代表阈值 20 像素、SR 用 AUC；RGBT234 论文明确 MPR 每帧取两模态较小距离、阈值 20，MSR 使用 IoU success 曲线 AUC。这些高层定义与本地主要规则相符，但论文未充分固定上述全部实现细节。[LasHeR §III-G](https://arxiv.org/html/2104.13202v2#S3.SS7)、[RGBT234 §III-C](https://arxiv.org/html/1805.08982v1#S3.SS3)

NPR 需要特别区分：LasHeR 论文将详细定义引到 TrackingNet；TrackingNet 作者 devkit 的 `36b939410e5f69c1f00b90606812811ff415474a` 对归一化精度曲线作梯形汇总，而本地取 0.20 单点，两者标量算法确实不同。**这不能替代尚未取得的 LasHeR 自身工具箱，也不能据此直接判定当前 LasHeR NPR 错误。** [TrackingNet 作者实现](https://github.com/SilvioGiancola/TrackingNet-devkit/blob/36b939410e5f69c1f00b90606812811ff415474a/metrics.py#L128-L154)

SCDT `2607.24701v1` §4.2 规定 OPE、LasHeR-Miss 245 条和 RGBT234-Miss 234 条，表 1 使用 PR/SR 与 MPR/MSR；它没有给出评分器版本或上述舍入/首帧/无效 GT 细节。论文所链仓库本次仍为空，`git ls-remote ... HEAD` 无返回引用，无法固定其评分源码提交。`Alexadlu` 持有 Python 包 fork，不证明 SCDT 使用了该 fork，更不证明其发表数字可由当前轨迹/GT/缺失协议直接复算。[SCDT §4.2、表1](https://arxiv.org/html/2607.24701v1#S4.SS2)、[论文所链仓库](https://github.com/Multi-Modality-Tracking/SCDT-CVPR2026)

因此可称“使用已固定的公开 Python rgbt 1.0.1 实现，对完整名单按已披露规则评分”；可以保留既有同协议内部比较。不能称“已认证数据集作者官方评测”“与作者工具箱/SCDT 严格等价”，也不能把本次来源核查写成模型提升或新模块效果。

## 最小后续动作与许可

1. 取得作者资源表所链的两个原始工具箱，保存下载 URL、压缩包 SHA256、版本和许可；在独立目录用**同一份已保存全量轨迹**作 CPU 双评分，逐序列比较曲线及差异原因。保留现有分数并并列标注口径；该工作不需重新推理或训练。未完成前，对 SCDT 的跨论文数值比较保留协议未完全认证的边界。
2. 在参数选择前固定开发集名单/哈希及 scorer。作者原始 LasHeR Protocol2 名单为 train979/test245，并无该仓库发布的 val97；项目若用 train882/val97，须记录划分及原 baseline 是否曾见过这 97 条。此处未认证历史独立性。RGBT234 的模块参数选择也须先有独立开发协议；不能靠测试集反复选参数后再称独立测试。Complete/Miss 采用预先固定的同一数据集配置。[作者训练名单](https://github.com/BUGPLEASEOUT/LasHeR/blob/c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399/List_for_Protocol2/trainingsetList.txt)

PyPI wheel 与包源仓库均含 MIT 许可原文（版权行保留为 `2018 The Python Packaging Authority`）；将来复用代码须保留该声明与许可。不能仅凭包的 MIT 就断言所有附带 GT、数据集或尚未取得的作者工具箱也获同样授权。本次没有移植代码。[固定源 LICENSE](https://github.com/opacity-black/RGBT_toolkit/blob/be78ddf30d8d5b389bb5bcc975a4477daa0f9370/LICENSE)

原始下载、Git 对象及比较明细仅缓存于 `C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/scoring_upstream_20261015/`。GitHub REST 初次访问被 rate limit，空 API 缓存不作版本证据；上表提交来自实际 Git 获取。所有来源判断均为本次核查时点的结论。
