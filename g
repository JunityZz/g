#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-$(pwd)}"
mkdir -p "$OUT_DIR"

PY_BENCH="$OUT_DIR/gpu_fp8_fp16_bandwidth_bench.py"
cat > "$PY_BENCH" <<'PY'
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

out_dir = sys.argv[1]

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"FAILED: {exc}"


def cuda_event_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "iters": iters,
    }


def matmul_tflops(n, ms):
    return 2.0 * n * n * n / (ms / 1000.0) / 1e12


def try_fp8_scaled_mm(a, b, scale_a, scale_b):
    fn = torch._scaled_mm
    attempts = [
        lambda: fn(a, b, scale_a, scale_b, out_dtype=torch.float16, use_fast_accum=True),
        lambda: fn(a, b, scale_a, scale_b, out_dtype=torch.float16),
        lambda: fn(a, b, scale_a, scale_b, None, None, torch.float16, True),
        lambda: fn(a, b, scale_a, scale_b, None, None, torch.float16),
    ]
    last = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last = exc
    raise last


def alloc_fp16_problem(n):
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    # Column-major B is the layout cublasLt commonly wants for peak GEMM.
    b = torch.randn((n, n), device="cuda", dtype=torch.float16).t()
    return a, b


def alloc_fp8_problem(n):
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable in this PyTorch build")
    a = torch.randn((n, n), device="cuda", dtype=torch.float16).to(fp8_dtype)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16).to(fp8_dtype).t()
    scale_a = torch.ones((), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((), device="cuda", dtype=torch.float32)
    return a, b, scale_a, scale_b


def choose_sizes(total_mem_gib):
    sizes = [8192, 16384]
    if total_mem_gib >= 20:
        sizes.append(24576)
    if total_mem_gib >= 32:
        sizes.append(32768)
    return sizes


def run_fp16(sizes):
    rows = []
    best = None
    for n in sizes:
        torch.cuda.empty_cache()
        try:
            a, b = alloc_fp16_problem(n)
            timing = cuda_event_ms(lambda: torch.mm(a, b), iters=12 if n >= 24576 else 20)
            row = {
                "n": n,
                **timing,
                "tflops": matmul_tflops(n, timing["median_ms"]),
            }
            rows.append(row)
            if best is None or row["tflops"] > best["tflops"]:
                best = row
            del a, b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as exc:
            rows.append({"n": n, "error": f"OOM: {exc}"})
            torch.cuda.empty_cache()
        except Exception as exc:
            rows.append({"n": n, "error": str(exc)})
            torch.cuda.empty_cache()
    return {"best": best, "runs": rows}


def run_fp8(sizes):
    if not hasattr(torch, "_scaled_mm"):
        return {"error": "torch._scaled_mm is unavailable in this PyTorch build", "best": None, "runs": []}

    rows = []
    best = None
    for n in sizes:
        torch.cuda.empty_cache()
        try:
            a, b, scale_a, scale_b = alloc_fp8_problem(n)
            try_fp8_scaled_mm(a, b, scale_a, scale_b)
            timing = cuda_event_ms(lambda: try_fp8_scaled_mm(a, b, scale_a, scale_b), iters=12 if n >= 24576 else 20)
            row = {
                "n": n,
                **timing,
                "tflops": matmul_tflops(n, timing["median_ms"]),
                "dtype": "e4m3/e4m3 -> fp16",
            }
            rows.append(row)
            if best is None or row["tflops"] > best["tflops"]:
                best = row
            del a, b, scale_a, scale_b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as exc:
            rows.append({"n": n, "error": f"OOM: {exc}"})
            torch.cuda.empty_cache()
        except Exception as exc:
            rows.append({"n": n, "error": str(exc)})
            torch.cuda.empty_cache()
    return {"best": best, "runs": rows}


def run_torch_copy_bandwidth(total_mem_gib):
    torch.cuda.empty_cache()
    gib = min(16, max(2, int(total_mem_gib * 0.12)))
    num_bytes = gib * 1024**3
    x = torch.empty((num_bytes,), device="cuda", dtype=torch.uint8)
    y = torch.empty_like(x)
    timing = cuda_event_ms(lambda: y.copy_(x), warmup=8, iters=30)
    sec = timing["median_ms"] / 1000.0
    payload_gbps = num_bytes / sec / 1e9
    hbm_traffic_gbps = (2 * num_bytes) / sec / 1e9
    return {
        "buffer_gib": gib,
        **timing,
        "payload_GBps": payload_gbps,
        "estimated_hbm_traffic_GBps": hbm_traffic_gbps,
        "note": "payload counts bytes copied; estimated_hbm_traffic counts read+write and is the number to compare with HBM peak bandwidth",
    }


def run_triton_copy_bandwidth(total_mem_gib):
    try:
        import triton
        import triton.language as tl
    except Exception as exc:
        return {"error": f"triton unavailable: {exc}"}

    try:
        @triton.jit
        def copy_kernel(x, y, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            vals = tl.load(x + offs, mask=mask)
            tl.store(y + offs, vals, mask=mask)
    except Exception as exc:
        return {"error": f"triton jit unavailable in this launch mode: {exc}"}

    try:
        torch.cuda.empty_cache()
        gib = min(16, max(2, int(total_mem_gib * 0.12)))
        num_bytes = gib * 1024**3
        n = num_bytes // 4
        x = torch.empty((n,), device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        block = 1024
        grid = (triton.cdiv(n, block),)
        timing = cuda_event_ms(lambda: copy_kernel[grid](x, y, n, BLOCK=block), warmup=8, iters=30)
        sec = timing["median_ms"] / 1000.0
        payload_gbps = num_bytes / sec / 1e9
        hbm_traffic_gbps = (2 * num_bytes) / sec / 1e9
        return {
            "buffer_gib": gib,
            **timing,
            "payload_GBps": payload_gbps,
            "estimated_hbm_traffic_GBps": hbm_traffic_gbps,
            "note": "custom kernel copy; estimated_hbm_traffic counts read+write",
        }
    except Exception as exc:
        torch.cuda.empty_cache()
        return {"error": f"triton copy benchmark failed: {exc}"}


def pct(value, denom):
    if value is None or denom == 0:
        return None
    return 100.0 * value / denom


if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
torch.cuda.set_device(0)

props = torch.cuda.get_device_properties(0)
total_mem_gib = props.total_memory / 1024**3
sizes = choose_sizes(total_mem_gib)

result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "host": platform.node(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_runtime_from_torch": torch.version.cuda,
    "gpu": {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gib": total_mem_gib,
        "multi_processor_count": props.multi_processor_count,
    },
    "nvidia_smi": sh(
        "nvidia-smi --query-gpu=name,pstate,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,mig.mode.current --format=csv,noheader,nounits"
    ),
    "nvidia_smi_full": sh("nvidia-smi"),
    "test_sizes": sizes,
}

fp16 = run_fp16(sizes)
fp8 = run_fp8(sizes)
torch_copy = run_torch_copy_bandwidth(total_mem_gib)
if os.environ.get("RUN_TRITON_COPY") == "1":
    triton_copy = run_triton_copy_bandwidth(total_mem_gib)
else:
    triton_copy = {"error": "disabled by default; set RUN_TRITON_COPY=1 to run the experimental Triton copy kernel"}

result["fp16_dense_gemm"] = fp16
result["fp8_dense_gemm"] = fp8
result["bandwidth"] = {
    "torch_copy": torch_copy,
    "triton_copy": triton_copy,
}

h200_dense_targets = {
    "fp16_dense_tflops": 989.5,
    "fp8_dense_tflops": 1979.0,
    "hbm_traffic_GBps": 4800.0,
}
result["h200_sxm_dense_reference"] = h200_dense_targets
result["h200_sxm_dense_reference_percent"] = {
    "fp16_dense": pct((fp16.get("best") or {}).get("tflops"), h200_dense_targets["fp16_dense_tflops"]),
    "fp8_dense": pct((fp8.get("best") or {}).get("tflops"), h200_dense_targets["fp8_dense_tflops"]),
    "torch_copy_hbm_traffic": pct(torch_copy.get("estimated_hbm_traffic_GBps"), h200_dense_targets["hbm_traffic_GBps"]),
    "triton_copy_hbm_traffic": pct(triton_copy.get("estimated_hbm_traffic_GBps"), h200_dense_targets["hbm_traffic_GBps"]) if "estimated_hbm_traffic_GBps" in triton_copy else None,
}

json_path = os.path.join(out_dir, "gpu_bench_result.json")
txt_path = os.path.join(out_dir, "gpu_bench_summary.txt")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, sort_keys=True)

def fmt_best(section):
    best = section.get("best")
    if not best:
        return "unavailable"
    return f"{best['tflops']:.1f} TFLOPS @ n={best['n']} median={best['median_ms']:.3f} ms"

summary = []
summary.append(f"GPU: {props.name} ({total_mem_gib:.1f} GiB), SM {props.major}.{props.minor}, torch {torch.__version__}, CUDA {torch.version.cuda}")
summary.append(f"nvidia-smi: {result['nvidia_smi']}")
summary.append(f"FP16 dense GEMM best: {fmt_best(fp16)}")
summary.append(f"FP8 dense GEMM best: {fmt_best(fp8)}")
summary.append(
    f"Torch copy bandwidth: payload={torch_copy['payload_GBps']:.1f} GB/s, estimated HBM traffic={torch_copy['estimated_hbm_traffic_GBps']:.1f} GB/s"
)
if "estimated_hbm_traffic_GBps" in triton_copy:
    summary.append(
        f"Triton copy bandwidth: payload={triton_copy['payload_GBps']:.1f} GB/s, estimated HBM traffic={triton_copy['estimated_hbm_traffic_GBps']:.1f} GB/s"
    )
else:
    summary.append(f"Triton copy bandwidth: {triton_copy.get('error', 'unavailable')}")
summary.append("H200 SXM dense reference used here: FP16 989.5 TFLOPS, FP8 1979 TFLOPS, HBM traffic 4800 GB/s.")
summary.append("NVIDIA's headline FP16 1979 / FP8 3958 TFLOPS numbers are with sparsity; this script measures dense GEMM.")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary) + "\n")

print("\n".join(summary))
print(f"\nWrote: {json_path}")
print(f"Wrote: {txt_path}")
PY

python3 "$PY_BENCH" "$OUT_DIR"
