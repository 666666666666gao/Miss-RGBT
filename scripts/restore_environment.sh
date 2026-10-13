#!/usr/bin/env bash
set -euo pipefail
cd /home/gaob/Miss-RGBT
prefix=/home/gaob/Miss-RGBT/conda-envs/miss_rgbt
mkdir "$prefix"
tar -xzf /home/gaob/Dual-RGBT/pytracking_relocatable.tar.gz -C "$prefix"
"$prefix/bin/python" "$prefix/bin/conda-unpack"
cp /data2/zhouy/CADTrack-main/baseline/checkpoints/train/cadtrack/cadtrack/CADTrack_ep0020.pth.tar checkpoints/pretrained/
sha256sum checkpoints/pretrained/CADTrack_ep0020.pth.tar
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$prefix/bin/python" scripts/environment_witness.py
