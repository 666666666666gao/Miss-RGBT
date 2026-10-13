# 近邻与来源

这些机制是实现和对照的起点，不能直接写成独有贡献。

- [SCDT](https://arxiv.org/html/2607.24701v1)：核对缺失 benchmark、OPE 和目标指标；公开代码仓库在本次核查为空，不能声称复现其具体实现。
- [IPL / Modality-missing-RGBT-Tracking](https://github.com/Alexadlu/Modality-missing-RGBT-Tracking)：官方固定缺失 masks 及公共入口的第一帧完整、单缺失复制、双缺失保持逻辑。
- [SDSTrack](https://arxiv.org/html/2403.16002v1)：跟踪中完整/masked 共享模型与中间特征自蒸馏的直接近邻。
- [SMML](https://arxiv.org/html/2507.07592v1)：利用训练标签下预测质量选择蒸馏方向，说明双向互教和质量选择本身已有先例。
- [DCM](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600290.pdf)：不同深度通过辅助头进行双向蒸馏；跨层连线本身已有先例。
- [I2M2Net](https://link.springer.com/chapter/10.1007/978-3-031-77610-6_1)：完整与缺失条件的双路一致性近邻，具体开源实现尚未核查。
- [RobustSeg，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.html)：正式论文、CVF附带源码及MIT许可证已核实。旧仓库地址404，当前可读仓库为[RobustSeg/RMMSS](https://github.com/RobustSeg/RMMSS)。固定版本反馈脚本optimizer仍绑定student，与论文的选择性teacher更新不一致；同时存在缺失导入，不能称已运行复现。详见[逐文件核查](OPEN_SOURCE_FINDINGS_20261013.md)。

复用代码时需另外保存仓库 commit、具体文件、许可证及修改说明。当前新代码尚未复制这些论文仓库的实现。

2026-10-13新增源码核查：SDSTrack使用双端可导的中间特征MSE；SMML论文的标签质量互教可核实，但仓库缺关键loss实现、无已核实复用许可证。三项研究的具体代码、版本差异及本项目梯度约束见[调研文档](OPEN_SOURCE_FINDINGS_20261013.md)。本次仅研究，没有复制代码进冻结模型。
