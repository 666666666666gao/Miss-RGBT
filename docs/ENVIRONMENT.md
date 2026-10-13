# 独立 conda 环境

2025 服务器，用户 gaob，目录 `/home/gaob/Miss-RGBT`。使用已有完整的 conda-pack 归档恢复到本项目新前缀，再执行 conda-unpack；不会修改 Dual-RGBT 的环境。

恢复入口：`bash scripts/restore_environment.sh`。只执行一次，失败保留日志。

记录在 `env-spec.json`；恢复方式和验证级别在项目私有 `.aris/compute/2025.md`。环境不写入 Git。

独立验证者必须原样执行：

```powershell
ssh -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=20 2025 'cd /home/gaob/Miss-RGBT && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python scripts/environment_witness.py'
```

该命令应退出 0，并打印 `WITNESS`、实际软件版本、真实 CUDA 矩阵乘积的形状和有限值。本验证只证明环境可执行，不证明模型、训练或完整评估通过。
