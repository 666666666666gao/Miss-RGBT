# 模板更新的隔离对照：已准备，未运行神经网络或正式评估

## 已确认的现有行为

E0/E3/E4已有动态模板更新，不能把原生更新重新作为新增模块。当前两个模板槽保留首帧参考，动态槽每5次有效track调用且max(Hann×score_map)>0.65时写入预测框裁剪；MissingTracker同时写入availability。位置/尺度先提交，再决定是否写模板。双缺失由评估循环保持上一框并跳过track，不推进更新调用计数。

本分支从a32e05d隔离，仅改`miss_rgbt/tracker.py`和`scripts/evaluate_model.py`的推理控制及可选记录，不改模型、权重、训练、模板数量、裁剪/预测或评分。没有合入活动main、部署活动项目或占用GPU。用户提供的外部sandbox试验包没有在本地找到；以下是本仓库独立实现及实际检查，不沿用该包的14项测试说法。

| 对照 | CLI策略 | 固定条件 | 唯一变化 |
|---|---|---|---|
| U0 | off | 两模板槽、同权重/输入/预测 | 禁止动态写入，位置仍提交 |
| U1 | current | 间隔5、阈值0.65 | 现有参照规则 |
| U2 | complete_only | U1全部条件 | 当前RGB/TIR均可用才写入 |
| U3 | current | 阈值0.65 | 间隔5与10 |
| U4 | current | 间隔5 | 阈值0.60/0.65/0.70 |

U2在全程完整输入下与U1规则相同，不能据此声称完整输入增强。修复后的特征没有写回图像模板；模板仍来自当前预测位置的真实可用图像裁剪和可用标记。

## 入口和后续运行条件

新增可选参数`--template-policy`、`--update-interval`、`--update-threshold`、`--update-audit`；不传参数时策略current，间隔/门限仍读取冻结YAML。保留每次从首帧执行的479序列/337353帧单条件入口。对每个模型、每种策略完整跑complete和miss，即958次序列运行/674706帧及10主指标，不能只重算旧bbox。

以下命令是未来运行说明，不代表已执行。先等当前E0-FT/E2及全部全评估真正完成，再完成BN归因。正式使用前，在独立部署目录核验源码、固定权重SHA、已安装评分包/GT/masks和两卡准入，并用真实训练序列检查U1与冻结入口的神经网络输出/轨迹等价；当前9项stdlib测试没有提供这项原生证据。不修改活动home中的科学源码，不启动自动后台模板队列。

```bash
# 在本分支的独立部署目录；复用现有conda，不新建环境。
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=2,3 /home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python scripts/evaluate_model.py \
  --checkpoint /home/gaob/Miss-RGBT/checkpoints/e4_full20_seed2027/epoch20.pth.tar \
  --output /data2/gaob/Miss-RGBT/results/template_e4_u0_complete_v1 \
  --condition complete --gpus 2,3 --template-policy off --update-audit
```

对同一固定E3/E4权重，另开输出路径切换condition与策略。E3权重SHA为01fa99822ab8936c02cd1f6223d0732ab450f2c942b199928b6de65903d87d5d；E4为c1b1ba0ff60b854e07f970c171a44fc543b4f4349210295632309519317aa99a。只选择这两份固定末轮，不拼接历史epoch05或按Test选轮。后续FT/E2正迁移比较沿用相同U1，不能把推理策略差异算为缺失训练收益。

## 实际记录和限制

开启audit后，每个非初始化视频帧记录video_frame（从0计）、track_call（不含双缺失保持帧）、可用状态、是否写入、Hann加权置信度、写入前模板年龄（分别按视频帧与有效调用计）和前一动态模板的可用状态；双缺失帧confidence=null。run.json保存实际策略、间隔/门限、audit开关和冻结YAML哈希，每序列回执保存trace哈希及写入数。未写入时年龄继续增长，换序列时重置。初始帧不记录为动态写入。

这些事件和bbox可用于后续离线配对GT，统计写入前后定位与错误写入代理；本次没有测得错误写入率、未来收益或身份可靠性。GT不进入写入判据。诊断开启有额外记录与I/O成本；本入口的多worker、逐序列时间不等于串行部署FPS/P95，未做延迟收益声明。

本地stdlib控制流检查9项通过：含3240组原规则边界组合（9720次策略断言）、两模板槽保留/首帧固定、部分缺失标记随写入、U2完整流等价、审计不改变stub轨迹、严格阈值、间隔/重置，以及实际sequence函数中的双缺失保持/有效调用计数。使用stub网络输出，通过AST加载实际方法/循环，不导入Torch，不是原生神经网络或完整指标。自审还去除了本地Python3.12可用但目标Python3.7缺少的math.nextafter调用，直接使用固定边界数值；没有增加兼容分支。

后续先在开发协议下单轴调整，不能以公开Test反复挑阈值。LasHeR当前882/97是项目开发划分，原E0是否暴露val97尚未认证；RGBT234独立开发协议待确定。现有Test模板干预若执行，作为预先声明的固定对照报告，不作为隐性选参集。
