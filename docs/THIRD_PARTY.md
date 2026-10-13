# 代码来源

实际 baseline 来自用户服务器 `/data2/zhouy/CADTrack-main` 的修改版，与官方代码存在 forward 差异；87个源码/配置的实际字节记录在 `reports/e0/source_sha256.json`。

原 README 指向 [IdolLab/CADTrack](https://github.com/IdolLab/CADTrack)。官方 [LICENSE](https://github.com/IdolLab/CADTrack/blob/main/LICENSE) 为 MIT，版权 `2026 Hao Li`；原样保留于 `docs/CADTrack_LICENSE`，SHA256 `3dc829f4d57404dfb2b71b5c7f36968f2f10809562ce1af27ade0dc160949f2e`。原代码中已有上游版权注释保留。

新方法代码位于 `miss_rgbt/`。没有直接复制 SDSTrack/SMML/DCM/RobustSeg 的实现。公共 masks 和 protocol 来自 IPL 仓库；数据与权重不在 Git 中分发。
