# Miss-RGBT 交接文档

更新日期：2026-10-13。当前状态：epoch20 两项缺失评估完成；项目与独立 conda 环境正在建立；新方法尚无完整训练结果，研究目标未达成。

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
