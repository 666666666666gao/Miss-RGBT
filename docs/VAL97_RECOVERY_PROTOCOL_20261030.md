# val97 恢复诊断准备

2026-10-30，服务器28。当前正式训练与评估队列的源码、配置、权重选择规则未修改。

## 已落地内容

开发数据根目录：`/data/wangwj/dataset/LasHeR/trainingset`。
协议及mask：`/home/gaob/Miss-RGBT/development/recovery_val97_20261030`。

97条序列、54646帧，固定32视频帧为一个状态块：

| 条件 | 循环状态 | 00帧 | 01帧 | 10帧 | 11帧 |
|---|---|---:|---:|---:|---:|
| complete | 11 | 0 | 0 | 0 | 54646 |
| rgb_outage | 11→01 | 0 | 26580 | 0 | 28066 |
| tir_outage | 11→10 | 0 | 0 | 26580 | 28066 |
| alternating_outage | 11→01→11→10→11→00 | 7546 | 10134 | 8900 | 28066 |

状态依次表示RGB/TIR是否可用。首帧始终完整初始化；单缺失复制当前可用图像，真实可用标识仍传给模型；双缺失保持上一框、不调用tracker。32帧是事先固定的受控诊断周期，不是根据公开测试mask或模型得分拟合的参数。

每份mask、两份训练/验证划分、97份init.txt及协议生成脚本均记录SHA256。当前train882与val97名称无交集；原E0是否历史见过val97仍未认证。

## 诊断入口

`/home/gaob/Miss-RGBT/evaluate_val97_recovery28.py` 复用冻结运行副本中的模型和tracker，输出自身完整闭环轨迹，不向模型传入首帧之外的GT。记录：

- 当时的真实可用状态、有效track调用计数及双缺失保持。
- 原生Hann加权响应；不把它称为已校准存在概率。
- 真实动态模板对象是否被替换、动态模板的源视频帧。
- 相对上一预测框的原生搜索crop是否包含当前GT中心、是否与GT框相交。
- 当前预测框与GT的IoU、是否刚从缺失恢复完整输入。

GT仅用于离线标签，不能进入门控、扩域接受或模板写入判据。几何相交不保证像素中有可见目标，IoU也不能独立证明身份识别。双缺失没有实际新crop，相关crop标签为null。

260帧GT宽高非正；仍保留每帧轨迹记录，但不计算这些帧的开发IoU/几何标签。不能将非正x/y直接视为目标消失。本入口不输出官方PR/SR，也不替代LasHeR-Miss或RGBT234-Miss的正式评估。

以下是准备好的命令示例，**尚未执行神经网络诊断**。不要在当前GPU0/1正式训练运行时启动：

```bash
source /home/gaob/Miss-RGBT/server28-env.sh
python /home/gaob/Miss-RGBT/evaluate_val97_recovery28.py \
  --checkpoint /home/gaob/Miss-RGBT/runs/server28/a_utility_full20_seed2027/epoch20.pth.tar \
  --condition alternating_outage --gpu 0 \
  --output /home/gaob/Miss-RGBT/development/recovery_val97_20261030/utility_alternating_trace_v1
```

固定同一checkpoint分别运行四种条件，才能比较缺失、恢复、搜索支持丢失和模板更新；单次条件不能建立模块增益。该输入准备没有自动插入当前正式评估队列。

## 实际验证范围

Python3.7目标环境语法编译和CLI检查通过。CPU上用100组固定随机框/图像，将crop相交结果与实际原生`sample_target`裁剪后的目标像素比较；另100组IoU与原生`box_iou`比较，均通过；非正尺寸GT排除检查通过。

首次验证脚本把原生逐样本一维IoU误按二维矩阵索引，出现IndexError；按真实函数返回契约修正后通过。没有加入fallback或异常吞噬。

检查只证明几何辅助函数和准备接口，不证明神经网络诊断端到端已通过。当前两卡仍用于正式训练，后续需真实数据闭环运行后再判断B模块的具体干预。

证据：`reports/server28-val97-recovery-preparation.json`、`reports/server28-recovery-geometry-validation.json`。
