# 环境独立验证记录

结论：**PASS**。文档中的 PowerShell SSH 命令原样执行一次，实际退出码为 **0**，WITNESS 与文档及环境台账一致；未发现执行结果与被检验声明之间的分歧。

- 日期：2026-10-13
- 验证者：/root/validate_miss_env（fresh agent-follows-doc）
- model：gpt-6-astra
- effort：max
- review_independence：same-family
- acceptance_status：provisional
- 验证范围：环境可执行性；不代表模型、训练或完整评估通过。
- GPU 调用：仅执行下列文档命令一次；未修复环境或修改命令。

原样执行的工具输入：

```powershell
ssh -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=20 2025 'cd /home/gaob/Miss-RGBT && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python scripts/environment_witness.py'
```

工具参数：workdir=C:/Users/gb，max_output_tokens=16000，yield_time_ms=10000。

实际完整工具结果：

```json
{
  "chunk_id": "776210",
  "wall_time_seconds": 7.9092614,
  "exit_code": 0,
  "original_token_count": 70,
  "output": "Environment configured. Type 'openclaw' to start.\r\nWITNESS {\"spec_hash\": \"cc6cd075\", \"torch\": \"1.12.1+cu113\", \"numpy\": \"1.21.5\", \"timm\": \"0.9.12\", \"cv2\": \"4.8.1\", \"yaml\": \"6.0.1\", \"device\": \"NVIDIA GeForce RTX 3090\", \"shape\": [32, 32], \"value\": 31.01445198059082, \"finite\": true}\n"
}
```

| 核验项 | 实际结果 | 判定 |
|---|---|---|
| 退出码 | 0 | PASS |
| WITNESS 标记 | 已输出 | PASS |
| 台账 spec hash | cc6cd075 | PASS |
| CUDA 设备名称 | NVIDIA GeForce RTX 3090 | PASS |
| 矩阵乘积形状 | [32, 32] | PASS |
| 有限值 | finite=true；value=31.01445198059082，与台账完全一致 | PASS |
| 软件版本 | torch 1.12.1+cu113；numpy 1.21.5；timm 0.9.12；cv2 4.8.1；yaml 6.0.1 | PASS |

真实分歧：无。台账在验证前记录 tier3 PENDING，本验证者未修改台账。

终端直接读取中文说明时出现显示编码问题；通过读取同一文件的原始字节并按 UTF-8 解码，已确认文档内容与命令。此问题没有改变 SSH 命令，也没有影响实际 WITNESS 输出。

完整读取与执行工具输入、原始工具输出、核验项及结论保存在 [.aris/traces/environment_validation_20261013.json](../.aris/traces/environment_validation_20261013.json)。本次仅读取任务限定的三个源文件；未独立重算 env-spec 哈希，未读取或验证归档及权重文件内容。

