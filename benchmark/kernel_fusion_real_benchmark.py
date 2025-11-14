#!/usr/bin/env python3
"""Real-world kernel fusion benchmark using actual SGLang models.

This benchmark uses the actual LlamaMLP implementation to measure
kernel fusion performance improvements.

Usage:
    # Basic benchmark
    python benchmark/kernel_fusion_real_benchmark.py

    # With specific model config
    python benchmark/kernel_fusion_real_benchmark.py \
        --hidden-size 4096 \
        --intermediate-size 11008 \
        --batch-size 32 \
        --seq-len 2048

    # With quantization
    python benchmark/kernel_fusion_real_benchmark.py --quant fp8
"""

import argparse
import time
from typing import Optional

import torch
import torch.nn as nn

# Import actual SGLang components
try:
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.linear import MergedColumnParallelLinear
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.quantization.fused_quant import (
        FusionConfig,
        wrap_with_silu_mul_fusion,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.llama import LlamaMLP

    # Try to import fused version
    try:
        from sglang.srt.models.llama_fused import (
            LlamaMLP as LlamaMLPFused,
            enable_kernel_fusion,
        )

        FUSION_AVAILABLE = True
    except ImportError:
        FUSION_AVAILABLE = False
        LlamaMLPFused = None
        print(
            "⚠️  Fused MLP not available. Using wrapper for demonstration."
        )

    SGLANG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SGLang not available: {e}")
    print("This benchmark requires SGLang to be installed.")
    SGLANG_AVAILABLE = False
    exit(1)


def benchmark_operation(
    fn,
    *args,
    warmup: int = 10,
    iterations: int = 100,
    name: str = "Operation",
    **kwargs,
) -> float:
    """Benchmark an operation."""
    device = args[0].device if torch.is_tensor(args[0]) else torch.device("cpu")

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = fn(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            _ = fn(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_latency_ms = (end - start) / iterations * 1000
    print(f"{name:50s}: {avg_latency_ms:8.4f} ms")
    return avg_latency_ms


def create_llama_mlp_unfused(
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    dtype: torch.dtype,
    quant_config: Optional[QuantizationConfig] = None,
):
    """Create unfused LlamaMLP (baseline)."""
    mlp = LlamaMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        quant_config=quant_config,
        prefix="mlp",
        reduce_results=False,  # Disable for benchmarking
        enable_fusion=False,  # Explicitly disable fusion
    )
    mlp = mlp.to(device).to(dtype)
    mlp.eval()
    return mlp


def create_llama_mlp_fused(
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    dtype: torch.dtype,
    quant_config: Optional[QuantizationConfig] = None,
):
    """Create fused LlamaMLP."""
    if FUSION_AVAILABLE:
        # Use actual fused implementation
        mlp = LlamaMLPFused(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            quant_config=quant_config,
            prefix="mlp_fused",
            reduce_results=False,
            enable_fusion=True,  # Enable fusion
        )
    else:
        # Fallback: manually enable fusion
        FusionConfig.enable_silu_mul_fusion = True
        FusionConfig.configure_torch_compile()
        mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            quant_config=quant_config,
            prefix="mlp_fused",
            reduce_results=False,
            enable_fusion=True,
        )

    mlp = mlp.to(device).to(dtype)
    mlp.eval()
    return mlp


def benchmark_llama_mlp(
    hidden_size: int,
    intermediate_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    quant_config: Optional[QuantizationConfig] = None,
):
    """Benchmark Llama MLP with/without fusion."""
    print("\n" + "=" * 80)
    print("LlamaMLP Kernel Fusion Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Total tokens: {batch_size * seq_len:,}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  Quantization: {quant_config.get_name() if quant_config else 'None'}")
    print("-" * 80)

    # Create input
    x = torch.randn(
        batch_size * seq_len, hidden_size, device=device, dtype=dtype
    )

    # Create unfused MLP (baseline)
    print("\nCreating unfused MLP (baseline)...")
    mlp_unfused = create_llama_mlp_unfused(
        hidden_size, intermediate_size, device, dtype, quant_config
    )

    # Create fused MLP
    print("Creating fused MLP...")
    mlp_fused = create_llama_mlp_fused(
        hidden_size, intermediate_size, device, dtype, quant_config
    )

    # Verify outputs match (correctness check)
    print("\nVerifying correctness...")
    with torch.no_grad():
        out_unfused = mlp_unfused(x)
        out_fused = mlp_fused(x)

    # Check shapes match
    assert out_unfused.shape == out_fused.shape, (
        f"Shape mismatch: {out_unfused.shape} vs {out_fused.shape}"
    )

    # Check values are close
    max_diff = (out_unfused - out_fused).abs().max().item()
    mean_diff = (out_unfused - out_fused).abs().mean().item()
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff > 1e-2:
        print(f"  ⚠️  WARNING: Large difference detected ({max_diff:.6f})")
    else:
        print("  ✓ Outputs match (within tolerance)")

    # Benchmark
    print("\n" + "-" * 80)
    print("Benchmarking...")
    print("-" * 80)

    unfused_latency = benchmark_operation(
        mlp_unfused, x, name="Unfused MLP (baseline)"
    )

    fused_latency = benchmark_operation(mlp_fused, x, name="Fused MLP")

    # Calculate improvement
    speedup = unfused_latency / fused_latency
    improvement = (speedup - 1) * 100

    print("-" * 80)
    print(f"Speedup: {speedup:.3f}x ({improvement:+.1f}%)")
    print("=" * 80)

    return speedup


def benchmark_component_breakdown(
    hidden_size: int,
    intermediate_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Benchmark individual components to understand fusion impact."""
    print("\n" + "=" * 80)
    print("Component-Level Breakdown")
    print("=" * 80)

    num_tokens = batch_size * seq_len
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create components
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    quant_method = UnquantizedLinearMethod()

    # gate_up_proj (merged linear)
    gate_up_proj = MergedColumnParallelLinear(
        hidden_size,
        [intermediate_size] * 2,
        bias=False,
        quant_config=None,
        prefix="test",
    )
    gate_up_proj = gate_up_proj.to(device).to(dtype)
    gate_up_proj.eval()

    # SiluAndMul
    silu_and_mul = SiluAndMul()

    print("\nBenchmarking individual operations:")
    print("-" * 80)

    # Benchmark gate_up_proj alone
    def gate_up_forward(x):
        y, _ = gate_up_proj(x)
        return y

    gate_up_latency = benchmark_operation(
        gate_up_forward, x, name="1. gate_up_proj (Linear)"
    )

    # Benchmark gate_up_proj + SiluAndMul (unfused)
    def unfused_forward(x):
        gate_up, _ = gate_up_proj(x)
        out = silu_and_mul(gate_up)
        return out

    unfused_latency = benchmark_operation(
        unfused_forward, x, name="2. gate_up_proj + SiluAndMul (unfused)"
    )

    # Benchmark with torch.compile fusion
    @torch.compile(mode="max-autotune", fullgraph=True)
    def fused_forward(x):
        gate_up, _ = gate_up_proj(x)
        out = silu_and_mul(gate_up)
        return out

    fused_latency = benchmark_operation(
        fused_forward, x, name="3. gate_up_proj + SiluAndMul (fused)"
    )

    print("-" * 80)
    print(f"SiluAndMul overhead: {unfused_latency - gate_up_latency:.4f} ms")
    print(
        f"Fusion benefit: {unfused_latency - fused_latency:.4f} ms "
        f"({(unfused_latency / fused_latency - 1) * 100:.1f}% faster)"
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Kernel Fusion Benchmark (Real SGLang Models)"
    )

    # Model config
    parser.add_argument(
        "--hidden-size", type=int, default=4096, help="Hidden dimension"
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=11008,
        help="Intermediate dimension",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048, help="Sequence length"
    )

    # Device config
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type",
    )

    # Quantization
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        choices=["fp8", "int8", None],
        help="Quantization method",
    )

    # Benchmark options
    parser.add_argument(
        "--component-breakdown",
        action="store_true",
        help="Run component-level breakdown",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Setup dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Setup quantization
    quant_config = None
    if args.quant == "fp8":
        try:
            from sglang.srt.layers.quantization.fp8 import Fp8Config

            quant_config = Fp8Config()
            print("Using FP8 quantization")
        except ImportError:
            print(
                "⚠️  FP8 quantization not available, using unquantized"
            )

    # Print header
    print("=" * 80)
    print("SGLang Kernel Fusion Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("=" * 80)

    # Run main benchmark
    speedup = benchmark_llama_mlp(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        dtype=dtype,
        quant_config=quant_config,
    )

    # Component breakdown
    if args.component_breakdown:
        benchmark_component_breakdown(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            dtype=dtype,
        )

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Configuration: Llama-{args.hidden_size // 1024}B-like")
    print(f"Kernel fusion speedup: {speedup:.3f}x")
    print(f"Expected end-to-end improvement: {(speedup - 1) * 50:.1f}%")
    print("  (assuming MLP is ~50% of computation)")
    print("=" * 80)

    print("\n✅ Benchmark completed!")
    print("\nNext steps:")
    print("1. Enable fusion in production:")
    print("   export SGLANG_ENABLE_KERNEL_FUSION=1")
    print("2. Run end-to-end benchmarks on real workloads")
    print("3. Profile with torch.profiler to verify fusion")


if __name__ == "__main__":
    main()
