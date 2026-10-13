# 近邻与来源

这些机制是实现和对照的起点，不能直接写成独有贡献。

- [SCDT](https://arxiv.org/html/2607.24701v1)：核对缺失 benchmark、OPE 和目标指标；公开代码仓库在本次核查为空，不能声称复现其具体实现。
- [IPL / Modality-missing-RGBT-Tracking](https://github.com/Alexadlu/Modality-missing-RGBT-Tracking)：官方固定缺失 masks 及公共入口的第一帧完整、单缺失复制、双缺失保持逻辑。
- [SDSTrack](https://arxiv.org/html/2403.16002v1)：跟踪中完整/masked 共享模型与中间特征自蒸馏的直接近邻。
- [SMML](https://arxiv.org/html/2507.07592v1)：利用训练标签下预测质量选择蒸馏方向，说明双向互教和质量选择本身已有先例。
- [DCM](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600290.pdf)：不同深度通过辅助头进行双向蒸馏；跨层连线本身已有先例。
- [I2M2Net](https://link.springer.com/chapter/10.1007/978-3-031-77610-6_1)：完整与缺失条件的双路一致性近邻，具体开源实现尚未核查。
- RobustSeg：用户提供的 CVF 页面本次无法读取，完整论文与代码仍待核查，不将未核实细节写成实施依据。

复用代码时需另外保存仓库 commit、具体文件、许可证及修改说明。当前新代码尚未复制这些论文仓库的实现。
