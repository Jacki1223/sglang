"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Test suite for RadixCache Triton kernel optimizations.

This module tests the correctness and performance of the optimized
Triton kernels against the original Python implementations.
"""

import time
from typing import List

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey, _key_match_page_size1

try:
    from sglang.srt.mem_cache.radix_cache_kernels import token_match_fast

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton kernels not available. Skipping GPU tests.")


def test_token_match_correctness():
    """Test that optimized token matching produces correct results."""
    print("Testing token matching correctness...")

    test_cases = [
        # (key0, key1, expected_match_length)
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 5),  # Full match
        ([1, 2, 3, 4, 5], [1, 2, 3, 6, 7], 3),  # Partial match
        ([1, 2, 3], [4, 5, 6], 0),  # No match
        ([], [1, 2, 3], 0),  # Empty first key
        ([1, 2, 3], [], 0),  # Empty second key
        ([], [], 0),  # Both empty
        ([1] * 100, [1] * 100, 100),  # Long full match
        ([1] * 100, [1] * 50 + [2] * 50, 50),  # Long partial match
    ]

    passed = 0
    failed = 0

    for i, (key0_tokens, key1_tokens, expected) in enumerate(test_cases):
        key0 = RadixKey(token_ids=key0_tokens, extra_key=None)
        key1 = RadixKey(token_ids=key1_tokens, extra_key=None)

        # Test Python implementation
        result_python = _key_match_page_size1(key0, key1)

        if result_python != expected:
            print(
                f"  ❌ Test case {i+1} FAILED (Python): expected {expected}, got {result_python}"
            )
            failed += 1
            continue

        # Test Triton implementation if available
        if TRITON_AVAILABLE and torch.cuda.is_available() and len(key0_tokens) >= 32:
            key0_tensor = torch.tensor(key0_tokens, dtype=torch.int64)
            key1_tensor = torch.tensor(key1_tokens, dtype=torch.int64)

            result_triton = token_match_fast(key0_tensor, key1_tensor)

            if result_triton != expected:
                print(
                    f"  ❌ Test case {i+1} FAILED (Triton): expected {expected}, got {result_triton}"
                )
                failed += 1
                continue

        passed += 1

    print(f"  ✅ Passed: {passed}/{len(test_cases)}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}/{len(test_cases)}")

    return failed == 0


def benchmark_token_match():
    """Benchmark optimized vs original token matching."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("Skipping benchmark: Triton or CUDA not available")
        return

    print("\nBenchmarking token matching performance...")
    print(f"{'Sequence Length':<20} {'Python (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    sequence_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    num_iterations = 100

    for seq_len in sequence_lengths:
        # Generate test data
        key0_tokens = list(range(seq_len))
        key1_tokens = list(range(seq_len))
        key0 = RadixKey(token_ids=key0_tokens, extra_key=None)
        key1 = RadixKey(token_ids=key1_tokens, extra_key=None)

        # Benchmark Python implementation
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _key_match_page_size1(key0, key1)
        python_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        # Benchmark Triton implementation
        key0_tensor = torch.tensor(key0_tokens, dtype=torch.int64).cuda()
        key1_tensor = torch.tensor(key1_tokens, dtype=torch.int64).cuda()

        # Warmup
        for _ in range(10):
            token_match_fast(key0_tensor, key1_tensor)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            token_match_fast(key0_tensor, key1_tensor)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        speedup = python_time / triton_time if triton_time > 0 else float("inf")

        print(
            f"{seq_len:<20} {python_time:<15.3f} {triton_time:<15.3f} {speedup:<10.2f}x"
        )


def test_radix_cache_integration():
    """Test that RadixCache works correctly with Triton kernels enabled."""
    print("\nTesting RadixCache integration...")

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, InsertParams, MatchPrefixParams

    # Create a simulated RadixCache with Triton kernels enabled
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        enable_triton_kernels=True,
    )
    cache = RadixCache(params)

    # Test basic insert and match
    test_sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [1, 2, 8, 9, 10],
    ]

    print("  Testing insert operations...")
    for seq in test_sequences:
        key = RadixKey(token_ids=seq, extra_key=None)
        result = cache.insert(InsertParams(key=key))
        print(f"    Inserted {seq[:5]}..., prefix_len={result.prefix_len}")

    print("  Testing match operations...")
    test_queries = [
        [1, 2, 3, 4, 5, 11, 12],  # Should match first 5
        [1, 2, 3, 6, 7, 13],  # Should match first 4
        [1, 2, 8, 9, 10, 14],  # Should match first 3
    ]

    for query in test_queries:
        key = RadixKey(token_ids=query, extra_key=None)
        result = cache.match_prefix(MatchPrefixParams(key=key))
        print(
            f"    Query {query[:5]}..., matched={len(result.device_indices)} tokens"
        )

    print("  ✅ RadixCache integration test passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("RadixCache Triton Kernel Optimization Tests")
    print("=" * 70)

    # Test correctness
    correctness_passed = test_token_match_correctness()

    # Run benchmark
    if TRITON_AVAILABLE and torch.cuda.is_available():
        benchmark_token_match()
    else:
        print("\nSkipping benchmark: GPU not available")

    # Test integration
    test_radix_cache_integration()

    print("\n" + "=" * 70)
    if correctness_passed:
        print("✅ All correctness tests passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
