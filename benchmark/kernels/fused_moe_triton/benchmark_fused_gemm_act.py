"""
Benchmark: Fused Activation+GEMM2 vs Separate Activation + GEMM2

This benchmark compares the performance of the fused Activation+GEMM2 kernel
(which eliminates the intermediate_cache2 buffer and separate activation kernel
by fusing activation into GEMM2's A-tile loading) against the original separate
Activation + GEMM2 approach.

Usage:
    python benchmark/kernels/fused_moe_triton/benchmark_fused_gemm_act.py

Environment variables:
    SGLANG_FUSED_MOE_ACT_GEMM2=1  Enable fused path (default)
    SGLANG_FUSED_MOE_ACT_GEMM2=0  Disable fused path (original)
"""

import os
import time

import torch
import triton

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_experts_impl,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

# Model configurations matching common MoE architectures
MODEL_CONFIGS = {
    "mixtral-8x7b": {
        "num_experts": 8,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "topk": 2,
    },
    "deepseek-v2-lite": {
        "num_experts": 64,
        "hidden_size": 2048,
        "intermediate_size": 1408,
        "topk": 6,
    },
    "qwen2-57b-moe": {
        "num_experts": 64,
        "hidden_size": 3584,
        "intermediate_size": 2560,
        "topk": 8,
    },
}

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]


def benchmark_fused_moe(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    use_fused: bool = True,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Run benchmark for fused MoE with or without Act+GEMM2 fusion."""
    # Set the environment variable to control fusion
    os.environ["SGLANG_FUSED_MOE_ACT_GEMM2"] = "1" if use_fused else "0"

    # Need to reimport to pick up the env var change
    import importlib

    import sglang.srt.layers.moe.fused_moe_triton.fused_moe as fused_moe_module

    importlib.reload(fused_moe_module)
    fused_experts_impl_fn = fused_moe_module.fused_experts_impl

    torch.cuda.manual_seed(42)
    device = "cuda"

    # Create input tensors
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    # w1 shape: [E, 2*intermediate_size, hidden_size] (gate + up stacked)
    w1 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device
    )
    # w2 shape: [E, hidden_size, intermediate_size]
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )

    # Generate routing
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    topk_weights = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk)
    topk_weights = topk_weights.to(dtype=torch.float32)

    # Warmup
    for _ in range(num_warmup):
        _ = fused_experts_impl_fn(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
            is_gated=True,
        )
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        _ = fused_experts_impl_fn(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
            is_gated=True,
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


def main():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    print("=" * 80)
    print("Fused Activation+GEMM2 Benchmark")
    print("=" * 80)
    print()
    print("Approach: Fuse activation(gate)*up into GEMM2's A-tile loading.")
    print("Benefits: Eliminates intermediate_cache2 + activation kernel launch.")
    print("          Only 1 accumulator (same as original GEMM2). Works for all batch sizes.")
    print()

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\nModel: {model_name}")
        print(f"  Experts={config['num_experts']}, "
              f"Hidden={config['hidden_size']}, "
              f"Intermediate={config['intermediate_size']}, "
              f"TopK={config['topk']}")
        print(f"{'Batch':>8} {'Original (ms)':>14} {'Fused (ms)':>12} {'Speedup':>10}")
        print("-" * 50)

        for batch_size in BATCH_SIZES:
            try:
                time_original = benchmark_fused_moe(
                    num_tokens=batch_size,
                    use_fused=False,
                    **config,
                )
                time_fused = benchmark_fused_moe(
                    num_tokens=batch_size,
                    use_fused=True,
                    **config,
                )
                speedup = time_original / time_fused if time_fused > 0 else float("inf")
                marker = " <--" if speedup > 1.05 else (" !!!" if speedup < 0.95 else "")
                print(
                    f"{batch_size:>8} {time_original:>14.3f} {time_fused:>12.3f} {speedup:>9.2f}x{marker}"
                )
            except Exception as e:
                print(f"{batch_size:>8} ERROR: {e}")

        print()


if __name__ == "__main__":
    main()
