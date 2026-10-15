# 融合BN冻结对照：隔离准备版

此分支基于`cc239d8`，在隔离worktree准备。当前主项目与归因队列的源码/配置未改；本对照没有正式启动，也没有性能指标。须先完成E0-availability、E0-FT和E2的完整训练/评估与归因。

## 唯一方法变化

现有`fix_bn(model)`只对`baseline.box_head`中的12个BatchNorm2d设eval，融合卷积中另外1个BN继续更新。新增`freeze_all_bn=true`时，对模型中全部13个BatchNorm2d设eval。只固定running mean/variance和批次计数的更新行为，weight/bias仍可训练；其他模块保持train，DropPath与任务路径不变。不增加参数或改变网络连接。

默认仍为原box-head范围，因为现有配置、`validate_method.py`和反向CPU探针实际使用旧调用。开关的默认值用于保留这些已有实验的行为，不增加其他输入处理或兼容路径。

E3/E4的两个allbn配置与原full20配置仅差冻结开关、独立输出目录和执行版本记录路径。仍从同一原E0 epoch20重新初始化，各20轮×20000元组、有效batch8、50000更新、seed2027；固定最终epoch20在完整/Miss条件下评估，不按Test选轮次。修复位置仍为第6块、宽128，不加入A/B/C或按数据集调参。

## 已实际完成的CPU机械检查

`scripts/validate_bn_freeze.py`在服务器独立准备目录用原环境和原E0权重构建真实E3/E4模型，CUDA_VISIBLE_DEVICES为空，PyTorch仅1线程。17.408秒实际退出0，原始结果见`reports/cpu_bn_witness_20261015_v1/completed.json`。

两模型都实际有13个BN；E3为103138437参数/250状态键，E4为103536903参数/258状态键。每种冻结模式直接给各BN一个2×C×2×2的CPU张量，并对BN affine做一次小型AdamW更新：原默认模式只有融合BN计数+1；allbn模式39个运行缓冲全部不变；所有BN affine有有限非零梯度并实际改变，其他模块仍为train。两种模式的参数结构不变。

这是BN子模块检查，没有图像、完整任务前向、GPU或正式训练。不能据此判断完整性能退化原因，也不能声称BN冻结会涨点。准备目录`/data2/gaob/Miss-RGBT/prepared/bn_freeze_20261015_v1`不参与当前训练。

## 队列空闲后的执行

部署并记录实际科学提交SHA到`reports/BN_FREEZE_SOURCE_VERSION.txt`后，执行`scripts/run_bn_controls.py`。入口要求阶段1为CAMPAIGN_COMPLETE且有结束记录，再只检查GPU2、3的显存准入，不查询温度、功率或其他卡，不抢占任何任务。

队列先在GPU2/3分别运行E3/E4各2次真实任务更新，验证全部39个BN缓冲不变、融合affine实际更新、有限状态和严格重载。短检查使用独立路径，正式训练重新加载原E0，不使用短检查的权重或优化器。该GPU短检查尚未执行。

随后两卡各一项20轮训练；保存到data2独立目录。完成后实际CPU读取固定epoch20，核对50000更新/config、严格加载、全部39个BN缓冲等于原E0，再依次完成E3/E4的完整/Miss四套全量评估。每模型958次序列运行/674706帧；主指标及逐序列轨迹必须全部保留，不能用短检查代替。

启动清单额外封存实际data_specs文本、882份训练init.txt、当前安装评分包/GT和两份公开mask的哈希。该见证证明启动时的字节，不认证上游官方来源，也不等于全部图像或完整采样轨迹已经逐字节审计。

正式运行仍须根据实际启动吞吐估计必要诊断/结束节点；不在当前队列上追加评估或改并发。GPU、端到端任务路径和完整性能验证都仍待执行。
