#!/usr/bin/env python3
"""
Realistic benchmark that includes tensor creation overhead.
This simulates the actual usage pattern in RadixCache.
"""

import time
import torch
import sys
sys.path.insert(0, '/home/user/sglang/python')

from sglang.srt.mem_cache.radix_cache import RadixKey, _key_match_page_size1

try:
    from sglang.srt.mem_cache.radix_cache_kernels import token_match_fast
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

def benchmark_with_tensor_creation():
    """Benchmark including tensor creation overhead (realistic scenario)."""
    print("Realistic Benchmark: Including Tensor Creation Overhead")
    print("=" * 80)
    print(f"{'Seq Len':<10} {'Python (ms)':<15} {'Triton+Create (ms)':<20} {'Speedup':<10}")
    print("-" * 80)

    sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
    num_iterations = 1000

    for seq_len in sequence_lengths:
        # Python implementation (baseline)
        python_times = []
        for _ in range(num_iterations):
            key0_tokens = list(range(seq_len))
            key1_tokens = list(range(seq_len))
            key0 = RadixKey(token_ids=key0_tokens, extra_key=None)
            key1 = RadixKey(token_ids=key1_tokens, extra_key=None)

            start = time.perf_counter()
            _key_match_page_size1(key0, key1)
            python_times.append((time.perf_counter() - start) * 1000)

        python_time = sum(python_times)

        # Triton implementation WITH tensor creation
        if TRITON_AVAILABLE and torch.cuda.is_available():
            triton_times = []
            for _ in range(num_iterations):
                key0_tokens = list(range(seq_len))
                key1_tokens = list(range(seq_len))

                start = time.perf_counter()
                # This is what actually happens in _key_match_optimized:
                key0_tensor = torch.tensor(key0_tokens, dtype=torch.int64)
                key1_tensor = torch.tensor(key1_tokens, dtype=torch.int64)
                token_match_fast(key0_tensor, key1_tensor)
                triton_times.append((time.perf_counter() - start) * 1000)

            triton_time = sum(triton_times)
            speedup = python_time / triton_time if triton_time > 0 else 0
            status = "✅" if speedup > 1.0 else "❌"

            print(f"{seq_len:<10} {python_time:<15.2f} {triton_time:<20.2f} {speedup:<10.2f}x {status}")
        else:
            print(f"{seq_len:<10} {python_time:<15.2f} {'N/A':<20} {'N/A':<10}")

    print("\n" + "=" * 80)
    print("Conclusion:")
    print("If speedup < 1.0, Triton is SLOWER than Python (overhead > benefit)")
    print("If speedup > 1.0, Triton is FASTER than Python")

if __name__ == "__main__":
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping benchmark")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        sys.exit(1)

    benchmark_with_tensor_creation()
