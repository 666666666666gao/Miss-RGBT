#!/usr/bin/env bash
cd /home/gaob/CADTrack_miss_ep20_20261013 || exit 1
export CUDA_VISIBLE_DEVICES=2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/home/zhouy/miniconda3/envs/pytracking/bin/python -u evaluate_miss.py run
code=$?
printf '%s\n' "$code" > exit_code.txt
exit "$code"
