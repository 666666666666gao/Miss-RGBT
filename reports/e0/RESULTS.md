# CADTrack 第20轮：缺失模态测试结果

在25服务器物理GPU2、3完成LasHeR-Miss全部245条序列（220,703帧）和RGBT234-Miss全部234条序列（116,650帧）的OPE测试。

| 方法及来源 | 权重轮次 | LasHeR-Miss PR | LasHeR-Miss SR | RGBT234-Miss MPR | RGBT234-Miss MSR |
|---|---:|---:|---:|---:|---:|
| CADTrack，本次实测 | 20 | 63.47 | 50.58 | 82.79 | 59.57 |
| SCDT，论文报告 | — | 69.30 | 54.40 | 88.10 | 64.30 |

指标单位为百分比。LasHeR-Miss NPR为 **58.58**。SCDT数值来自[论文Table 1](https://arxiv.org/html/2607.24701v1#S4.T1)，本次未重新运行SCDT。

| 数据集 | 用户提供的完整模态指标 | 本次缺失模态指标 | 下降百分点 |
|---|---:|---:|---:|
| LasHeR PR | 75.64 | 63.47 | 12.17 |
| LasHeR NPR | 71.77 | 58.58 | 13.19 |
| LasHeR SR | 60.13 | 50.58 | 9.55 |
| RGBT234 MPR | 90.68 | 82.79 | 7.89 |
| RGBT234 MSR | 66.86 | 59.57 | 7.29 |

完整模态列引用用户给定表格；本次只重新推理缺失模态数据。现存两个常规LasHeR结果目录的CPU重算值与给定表不同，详情见README，不能把这些来源当作同一次实测。

使用原权重`/data2/zhouy/CADTrack-main/baseline/checkpoints/train/cadtrack/cadtrack/CADTrack_ep0020.pth.tar`，模型严格加载，模型源码及测试YAML保持原样。

采用已有官方IPL逐帧缺失标注：首帧完整初始化；单模态缺失时复制可用模态；双模态缺失时保持上一帧预测和跟踪状态。全部缺失帧均参与最终指标，未使用后续真值框修正预测。指标由已有rgbt工具包计算，保留其原生取整和阈值规则。

该测试对应SCDT使用的相同基准及OPE指标，输入缺失处理来自公开IPL评测入口。SCDT链接的代码仓库当时为空，因此不声称复现其未公开推理实现。标注统一使用rgbt工具包完整标注；原LasHeR目录中存在15行框标注对应1,411帧图像的情况。

远端原始结果、日志、指标曲线及校验记录保存在`/home/gaob/CADTrack_miss_ep20_20261013/`；本地归档为`C:\Users\gb\.codex_tmp\cadtrack_miss_ep20_20261013\baseline_raw_results.tar.gz`，SHA256为`85b302508ce56b7e688c0f8c21f58a6dd07fd883ea8f57db5bf99a47bae3dc0a`。

独立审查发现原生 LasHeR toolkit 将 GT 任一坐标<=0 的1260帧误差设为-1，PR/NPR把这些帧计为成功，而SR计失败。相对同分母把这些帧计为失败，PR约含0.556805个百分点的该规则贡献。本次保留原生协议，所有后续模型使用同一GT与规则；不能把这部分算作模型改进。正式指标与按帧连续坐标的描述诊断分别记录。
