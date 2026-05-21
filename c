#!/usr/bin/env bash
set -euo pipefail

M="${M:-24576}"
N="${N:-24576}"
K="${K:-24576}"
CUTLASS_DIR="${CUTLASS_DIR:-/tmp/cutlass}"
BUILD_DIR="${BUILD_DIR:-$CUTLASS_DIR/build}"
PROFILER="$BUILD_DIR/tools/profiler/cutlass_profiler"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Use a CUDA devel image, for example pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y git cmake ninja-build build-essential
  else
    echo "ERROR: git/cmake/ninja missing and apt-get is unavailable." >&2
    exit 1
  fi
fi

if [ ! -d "$CUTLASS_DIR/.git" ]; then
  git clone --depth=1 https://github.com/NVIDIA/cutlass "$CUTLASS_DIR"
fi

if [ ! -x "$PROFILER" ]; then
  cmake -S "$CUTLASS_DIR" -B "$BUILD_DIR" -GNinja \
    -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_ENABLE_PROFILER=ON \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_UNITY_BUILD_ENABLED=ON
  cmake --build "$BUILD_DIR" --target cutlass_profiler -j"$(nproc)"
fi

run_case() {
  local name="$1"
  shift
  echo
  echo "=== $name GEMM m=$M n=$N k=$K ==="
  "$PROFILER" "$@" 2>&1 | tee "/tmp/cutlass_${name,,}_gemm.log" | tail -80
  echo
  echo "--- $name summary candidates ---"
  grep -E "GFLOP/s|gflops|Runtime|Math|cutlass" "/tmp/cutlass_${name,,}_gemm.log" | tail -30 || true
}

run_case FP16 \
  --operation=Gemm \
  --m="$M" --n="$N" --k="$K" \
  --A=f16:row --B=f16:column --C=f16:row \
  --accumulator-type=f32 \
  --op_class=tensorop \
  --enable-best-kernel-for-fixed-shape \
  --sort-results-flops-per-sec \
  --verification-enabled=false

run_case FP8 \
  --operation=Gemm \
  --m="$M" --n="$N" --k="$K" \
  --A=f8:row --B=f8:column --C=f16:row \
  --accumulator-type=f32 \
  --op_class=tensorop \
  --runtime_input_datatype_a=e4m3 \
  --runtime_input_datatype_b=e4m3 \
  --enable-best-kernel-for-fixed-shape \
  --sort-results-flops-per-sec \
  --verification-enabled=false

echo
echo "Logs:"
echo "  /tmp/cutlass_fp16_gemm.log"
echo "  /tmp/cutlass_fp8_gemm.log"
echo
echo "Convert GFLOP/s to TFLOPS by dividing by 1000."
