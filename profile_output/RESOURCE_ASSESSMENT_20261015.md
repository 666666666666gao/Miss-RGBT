# GPU2/3并行资源评估

建议保持每张GPU一个训练任务，总计E3/E4两项。另开同配置训练所需显存大于余量；没有进行并发神经任务压测，不声称多任务減速已被实测。

实际采样：2026-10-14 23:57:26～23:58:26北京时间，一次SSH、6个GPU快照、12秒间隔。GPU利用率是样本均值，不能等同于SM占用率或并发吞吐；CPU是5个区间的/proc/stat差分。

| GPU | 总显存GiB | 已用GiB | 剩余GiB | GPU利用率均值 | 范围 |
|---|---:|---:|---:|---:|---:|
| 2 | 24 | 18.67 | 5.02 | 95.5% | 77～100% |
| 3 | 24 | 18.88 | 4.81 | 88.8% | 71～100% |

64逻辑核，CPU区间均值43.97%，I/O等待均值0.228%。末次load average [32.52,33.21,34.54]，可用RAM 224.35GiB。负载均值不能按百分数解释；这也不是某个训练进程的CPU占用。

17:00原训练记录的PyTorch分配峰值：E3 17285506048字节（约16.10GiB），E4 17476413440字节（约16.28GiB）。nvidia-smi已用显存还含缓存和运行时等，不能把两者混为同一量。仅剩约5GiB不能再放一份当前microbatch4/accumulation2训练。CPU有余量可用于离线统计；不缩小冻结batch预算以强行并行。

只读取GPU2、3显存、GPU/memory utilization和整机CPU/RAM；没有温度、功耗、其他GPU或项目进程、训练步数查询。原11:00最终观察器未动。

## 采样文件记录

| 文件 | 类型 | 内容 |
|---|---|---|
| C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/profile_server_resources.py | 新建独立包装脚本 | 标准库远程采样，不插入训练源码 |
| C:/Users/gb/.codex_tmp/miss_rgbt_campaign_20261013/resource_profile_transport_20261015.json | 新建私有记录 | SSH实际returncode/stdout/stderr |
| profile_output/resources_20261015_0000.json | 新建 | 6个原始样本 |
| profile_output/RESOURCE_ASSESSMENT_20261015.md | 新建 | 分析与建议 |

没有训练内插桩需要撤销；独立采样脚本后续可移除，原始记录保留。
