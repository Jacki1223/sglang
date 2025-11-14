#!/usr/bin/env python3
"""Kernel Fusion Demo - Simple demonstration of kernel fusion benefits.

This script demonstrates the performance benefits of kernel fusion
by comparing fused vs unfused operations.

Usage:
    python examples/kernel_fusion_demo.py
"""

import argparse
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import fused layers
try:
    from sglang.srt.layers.fused_layers import (
        FusedLinearSiLU,
        FusedLinearGELU,
        FusedRMSNormLinear,
        configure_kernel_fusion_compile,
    )

    FUSED_LAYERS_AVAILABLE = True
except ImportError:
    FUSED_LAYERS_AVAILABLE = False
    print("⚠️  Fused layers not available. Please ensure sglang is installed.")


def benchmark_operation(
    fn: Callable,
    *args,
    warmup: int = 10,
    iterations: int = 100,
    name: str = "Operation",
    **kwargs,
) -> float:
    """Benchmark an operation.

    Args:
        fn: Function to benchmark
        args: Positional arguments to fn
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        name: Name for logging
        kwargs: Keyword arguments to fn

    Returns:
        Average latency in milliseconds
    """
    device = args[0].device if torch.is_tensor(args[0]) else torch.device("cpu")

    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fn(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_latency_ms = (end - start) / iterations * 1000
    print(f"{name:40s}: {avg_latency_ms:8.4f} ms")
    return avg_latency_ms


def demo_linear_silu_fusion(device: torch.device, dtype: torch.dtype):
    """Demonstrate Linear + SiLU fusion."""
    print("\n" + "=" * 80)
    print("Demo 1: Linear + SiLU Fusion")
    print("=" * 80)

    # Configuration
    batch_size = 32
    seq_len = 2048
    hidden_size = 4096
    intermediate_size = 11008

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden size: {hidden_size} → {intermediate_size}")
    print(f"Total tokens: {batch_size * seq_len:,}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("-" * 80)

    # Input
    x = torch.randn(
        batch_size * seq_len, hidden_size, device=device, dtype=dtype
    )

    # ========================================================================
    # Unfused baseline
    # ========================================================================
    linear = nn.Linear(
        hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
    )

    def unfused_forward(x):
        y = linear(x)
        z = F.silu(y)
        return z

    # ========================================================================
    # Fused version
    # ========================================================================
    if FUSED_LAYERS_AVAILABLE:
        fused_linear_silu = FusedLinearSiLU(
            hidden_size, intermediate_size, bias=False
        ).to(device).to(dtype)

        # Copy weights for fair comparison
        fused_linear_silu.weight.data.copy_(linear.weight.data)

    # ========================================================================
    # Benchmark
    # ========================================================================
    print("\nBenchmarking...")
    unfused_latency = benchmark_operation(
        unfused_forward, x, name="Unfused (Linear → SiLU)"
    )

    if FUSED_LAYERS_AVAILABLE:
        fused_latency = benchmark_operation(
            fused_linear_silu, x, name="Fused (Linear+SiLU)"
        )

        speedup = unfused_latency / fused_latency
        improvement = (speedup - 1) * 100

        print("-" * 80)
        print(f"Speedup: {speedup:.3f}x ({improvement:+.1f}%)")
        print("=" * 80)

        return speedup
    else:
        print("⚠️  Fused layers not available. Skipping fused benchmark.")
        return 1.0


def demo_rmsnorm_linear_fusion(device: torch.device, dtype: torch.dtype):
    """Demonstrate RMSNorm + Linear fusion."""
    print("\n" + "=" * 80)
    print("Demo 2: RMSNorm + Linear Fusion")
    print("=" * 80)

    # Configuration
    batch_size = 16
    seq_len = 2048
    hidden_size = 4096
    out_features = hidden_size * 3  # QKV projection

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden size: {hidden_size}")
    print(f"Output features: {out_features} (QKV)")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("-" * 80)

    # Input
    x = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)

    # ========================================================================
    # Unfused baseline
    # ========================================================================
    class SimpleRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x

    rmsnorm = SimpleRMSNorm(hidden_size).to(device).to(dtype)
    linear = nn.Linear(
        hidden_size, out_features, bias=False, device=device, dtype=dtype
    )

    def unfused_forward(x):
        y = rmsnorm(x)
        z = linear(y)
        return z

    # ========================================================================
    # Fused version
    # ========================================================================
    if FUSED_LAYERS_AVAILABLE:
        fused = FusedRMSNormLinear(
            hidden_size, out_features, bias=False
        ).to(device).to(dtype)

        # Copy weights
        fused.norm_weight.data.copy_(rmsnorm.weight.data)
        fused.linear_weight.data.copy_(linear.weight.data)

    # ========================================================================
    # Benchmark
    # ========================================================================
    print("\nBenchmarking...")
    unfused_latency = benchmark_operation(
        unfused_forward, x, name="Unfused (RMSNorm → Linear)"
    )

    if FUSED_LAYERS_AVAILABLE:
        fused_latency = benchmark_operation(
            fused, x, name="Fused (RMSNorm+Linear)"
        )

        speedup = unfused_latency / fused_latency
        improvement = (speedup - 1) * 100

        print("-" * 80)
        print(f"Speedup: {speedup:.3f}x ({improvement:+.1f}%)")
        print("=" * 80)

        return speedup
    else:
        print("⚠️  Fused layers not available. Skipping fused benchmark.")
        return 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Kernel Fusion Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on CUDA with FP16
  python examples/kernel_fusion_demo.py --device cuda --dtype fp16

  # Run on CPU with FP32
  python examples/kernel_fusion_demo.py --device cpu --dtype fp32
        """,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type (default: fp16)",
    )

    args = parser.parse_args()

    # Device setup
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Dtype setup
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Check dtype availability
    if dtype == torch.float16 and device.type == "cpu":
        print("⚠️  FP16 not well supported on CPU, using FP32")
        dtype = torch.float32
    if dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("⚠️  BF16 not supported on this GPU, using FP16")
            dtype = torch.float16

    # Configure torch.compile if fused layers available
    if FUSED_LAYERS_AVAILABLE:
        configure_kernel_fusion_compile()

    print("\n" + "=" * 80)
    print("Kernel Fusion Performance Demo")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("=" * 80)

    # Run demos
    speedup1 = demo_linear_silu_fusion(device, dtype)
    speedup2 = demo_rmsnorm_linear_fusion(device, dtype)

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Linear + SiLU fusion:    {speedup1:.3f}x speedup")
    print(f"RMSNorm + Linear fusion: {speedup2:.3f}x speedup")
    print("=" * 80)

    if FUSED_LAYERS_AVAILABLE:
        print("\n✅ Kernel fusion demo completed successfully!")
        print("\nNext steps:")
        print("1. Run benchmarks on larger models")
        print("2. Profile with torch.profiler to see fusion in action")
        print("3. Integrate fused layers into model implementations")
    else:
        print("\n⚠️  Fused layers not available.")
        print("Please install sglang with kernel fusion support.")


if __name__ == "__main__":
    main()
