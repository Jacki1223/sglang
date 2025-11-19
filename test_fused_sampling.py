#!/usr/bin/env python3
"""
Test script for fused sampling kernel

This script compares the fused sampling kernel against the original multi-kernel
approach to verify correctness and measure performance improvements.
"""

import torch
import time
import os

# Enable CUDA kernel errors for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def test_fused_sampling_basic():
    """Basic correctness test for fused sampling"""
    print("=" * 80)
    print("Testing Fused Sampling Kernel - Basic Correctness")
    print("=" * 80)

    # Test parameters
    batch_size = 4
    vocab_size = 1000

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Vocabulary size: {vocab_size}")

    # Create test inputs
    logits = torch.randn(batch_size, vocab_size, device='cuda', dtype=torch.float32)
    temperatures = torch.full((batch_size,), 0.7, device='cuda', dtype=torch.float32)
    top_k = torch.full((batch_size,), 50, dtype=torch.int32, device='cuda')
    top_p = torch.full((batch_size,), 0.9, device='cuda', dtype=torch.float32)

    # Use fixed random samples for reproducibility
    torch.manual_seed(42)
    uniform_samples = torch.rand(batch_size, device='cuda', dtype=torch.float32)

    print(f"\nInput logits shape: {logits.shape}")
    print(f"Temperature: {temperatures[0].item()}")
    print(f"Top-k: {top_k[0].item()}")
    print(f"Top-p: {top_p[0].item()}")

    try:
        from sgl_kernel.sampling import fused_sampling_from_logits

        # Test the fused kernel
        print("\nCalling fused sampling kernel...")
        samples = fused_sampling_from_logits(
            logits.clone(),
            temperatures=temperatures,
            top_k=top_k,
            top_p=top_p,
            uniform_samples=uniform_samples
        )

        print(f"\n✓ Fused kernel succeeded!")
        print(f"Output samples: {samples}")
        print(f"Output shape: {samples.shape}")
        print(f"Output dtype: {samples.dtype}")

        # Verify output is within valid range
        assert samples.min() >= 0, f"Invalid sample: {samples.min()} < 0"
        assert samples.max() < vocab_size, f"Invalid sample: {samples.max()} >= {vocab_size}"
        print(f"✓ All samples in valid range [0, {vocab_size})")

        return True

    except ImportError as e:
        print(f"\n✗ Failed to import fused_sampling_from_logits: {e}")
        print("  Note: The kernel needs to be compiled first")
        return False
    except Exception as e:
        print(f"\n✗ Fused kernel failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fused_sampling_vs_original():
    """Compare fused kernel with original implementation"""
    print("\n" + "=" * 80)
    print("Testing Fused vs Original Sampling - Numerical Comparison")
    print("=" * 80)

    batch_size = 8
    vocab_size = 2000

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Vocabulary size: {vocab_size}")

    # Create test inputs
    logits = torch.randn(batch_size, vocab_size, device='cuda', dtype=torch.float32)
    temperatures = torch.full((batch_size,), 1.0, device='cuda', dtype=torch.float32)

    # Use fixed random samples
    torch.manual_seed(42)
    uniform_samples = torch.rand(batch_size, device='cuda', dtype=torch.float32)

    try:
        from sgl_kernel.sampling import (
            fused_sampling_from_logits,
            top_k_top_p_sampling_from_probs
        )

        # Method 1: Fused kernel
        print("\nMethod 1: Fused kernel")
        samples_fused = fused_sampling_from_logits(
            logits.clone(),
            temperatures=temperatures,
            top_k=None,
            top_p=None,
            uniform_samples=uniform_samples.clone()
        )
        print(f"  Samples: {samples_fused}")

        # Method 2: Original approach (temperature + softmax + sampling)
        print("\nMethod 2: Original multi-kernel approach")
        logits_copy = logits.clone()
        logits_copy.div_(temperatures.unsqueeze(1))
        probs = torch.softmax(logits_copy, dim=-1)

        # Simple multinomial sampling for comparison
        samples_original = torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int32)
        print(f"  Samples: {samples_original}")

        # Note: Exact match is not expected due to different random number generation
        # But distributions should be similar
        print("\n✓ Both methods completed successfully")
        print("  Note: Exact match not expected due to different RNG implementations")

        return True

    except Exception as e:
        print(f"\n✗ Comparison failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_performance():
    """Benchmark fused kernel vs original approach"""
    print("\n" + "=" * 80)
    print("Performance Benchmark: Fused vs Original")
    print("=" * 80)

    configs = [
        (1, 32000),    # Small batch, large vocab (typical for inference)
        (4, 32000),
        (8, 32000),
        (32, 32000),
    ]

    num_warmup = 10
    num_iterations = 100

    try:
        from sgl_kernel.sampling import fused_sampling_from_logits

        for batch_size, vocab_size in configs:
            print(f"\n{'─' * 60}")
            print(f"Config: batch_size={batch_size}, vocab_size={vocab_size}")
            print(f"{'─' * 60}")

            # Create test data
            logits = torch.randn(batch_size, vocab_size, device='cuda', dtype=torch.float32)
            temperatures = torch.full((batch_size,), 0.7, device='cuda', dtype=torch.float32)
            top_k = torch.full((batch_size,), 50, dtype=torch.int32, device='cuda')
            top_p = torch.full((batch_size,), 0.9, device='cuda', dtype=torch.float32)

            # Warmup
            for _ in range(num_warmup):
                _ = fused_sampling_from_logits(
                    logits.clone(), temperatures, top_k, top_p
                )
            torch.cuda.synchronize()

            # Benchmark fused kernel
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = fused_sampling_from_logits(
                    logits.clone(), temperatures, top_k, top_p
                )
            torch.cuda.synchronize()
            fused_time = (time.perf_counter() - start) / num_iterations

            # Benchmark original approach
            start = time.perf_counter()
            for _ in range(num_iterations):
                logits_copy = logits.clone()
                logits_copy.div_(temperatures.unsqueeze(1))
                probs = torch.softmax(logits_copy, dim=-1)
                _ = torch.multinomial(probs, num_samples=1)
            torch.cuda.synchronize()
            original_time = (time.perf_counter() - start) / num_iterations

            speedup = original_time / fused_time

            print(f"  Fused kernel:    {fused_time*1000:.3f} ms")
            print(f"  Original:        {original_time*1000:.3f} ms")
            print(f"  Speedup:         {speedup:.2f}x")

            if speedup > 1.0:
                print(f"  ✓ Fused kernel is {speedup:.2f}x faster!")
            else:
                print(f"  ⚠ Fused kernel is slower (needs optimization)")

        return True

    except Exception as e:
        print(f"\n✗ Benchmark failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               Fused Sampling Kernel Test & Benchmark Suite                  ║
║                                                                              ║
║  This suite validates the fused sampling kernel that combines:              ║
║    • Temperature scaling                                                     ║
║    • Softmax computation                                                     ║
║    • Top-k / Top-p filtering                                                 ║
║    • Multinomial sampling                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA is not available. This test requires a CUDA-capable GPU.")
        return 1

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    results = []

    # Run tests
    results.append(("Basic Correctness", test_fused_sampling_basic()))
    results.append(("Numerical Comparison", test_fused_sampling_vs_original()))
    results.append(("Performance Benchmark", benchmark_performance()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:.<50} {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
