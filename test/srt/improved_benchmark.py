#!/usr/bin/env python3
"""
Improved benchmark that better demonstrates eviction strategy differences.

This version creates a more realistic scenario with:
- Multiple competing prefix groups
- Smaller cache to create pressure
- Page size = 1 for token-level control
- Interleaved requests to trigger evictions

Usage:
    python test/srt/improved_benchmark.py
"""

import time
from collections import defaultdict
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


def generate_competing_prefixes_workload(
    num_groups: int = 5,
    requests_per_group: int = 20,
    prefix_len: int = 80,
    suffix_len: int = 40,
) -> list:
    """
    Generate workload with multiple competing prefix groups.

    Each group shares a common prefix, but groups compete for cache space.
    Requests are interleaved to create eviction pressure.
    """
    torch.manual_seed(42)

    # Generate prefix groups
    groups = []
    for i in range(num_groups):
        prefix = torch.randint(0, 50000, (prefix_len,)).tolist()

        # Generate requests for this group
        group_requests = []
        for j in range(requests_per_group):
            suffix = torch.randint(0, 50000, (suffix_len,)).tolist()
            group_requests.append(prefix + suffix)

        groups.append(group_requests)

    # Interleave requests from different groups
    # This creates competition: should we keep prefix A or prefix B?
    sequences = []
    for round_idx in range(requests_per_group):
        for group_idx in range(num_groups):
            sequences.append(groups[group_idx][round_idx])

    return sequences, num_groups, prefix_len, suffix_len


def test_policy(policy_name: str, sequences: list, cache_size: int) -> dict:
    """Test a single eviction policy."""
    # Create mock allocator
    mock_allocator = Mock()
    mock_allocator.device = torch.device("cpu")
    mock_allocator.free = Mock()

    # Create cache with page_size=1 for token-level control
    cache = RadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=mock_allocator,
        page_size=1,  # Token-level granularity
        eviction_policy=policy_name,
    )

    total_tokens = 0
    cached_tokens = 0
    eviction_count = 0
    eviction_time_total = 0.0

    # Process sequences
    for idx, seq in enumerate(sequences):
        key = RadixKey(seq)
        total_tokens += len(seq)

        # Match prefix
        result = cache.match_prefix(key)
        cached_tokens += len(result.device_indices)

        # Insert new tokens
        new_tokens = len(seq) - len(result.device_indices)
        if new_tokens > 0:
            # Evict if needed
            if cache.total_size() + new_tokens > cache_size:
                evict_start = time.perf_counter()
                tokens_to_evict = cache.total_size() + new_tokens - cache_size
                cache.evict(tokens_to_evict)
                eviction_time_total += time.perf_counter() - evict_start
                eviction_count += 1

            # Insert
            value = torch.tensor(seq, dtype=torch.int64)
            cache.insert(key, value)

    hit_rate = cached_tokens / total_tokens if total_tokens > 0 else 0.0
    avg_eviction_time = eviction_time_total / eviction_count if eviction_count > 0 else 0.0

    return {
        "policy": policy_name,
        "hit_rate": hit_rate,
        "evictions": eviction_count,
        "avg_eviction_ms": avg_eviction_time * 1000,
        "final_cache_size": cache.total_size(),
    }


def main():
    print("="*80)
    print("Improved Eviction Strategy Benchmark")
    print("="*80)

    # Configuration
    num_groups = 5          # 5 different prefix groups competing
    requests_per_group = 20 # 20 requests per group
    prefix_len = 80         # 80-token shared prefix per group
    suffix_len = 40         # 40-token unique suffix
    cache_size = 600        # Small cache: can hold ~5 full requests or ~7.5 prefixes

    print(f"\nWorkload Configuration:")
    print(f"  - {num_groups} competing prefix groups")
    print(f"  - {requests_per_group} requests per group")
    print(f"  - Total requests: {num_groups * requests_per_group}")
    print(f"  - Prefix length: {prefix_len} tokens (shared within group)")
    print(f"  - Suffix length: {suffix_len} tokens (unique per request)")
    print(f"  - Request size: {prefix_len + suffix_len} tokens")
    print(f"  - Cache size: {cache_size} tokens (~{cache_size/(prefix_len+suffix_len):.1f} requests)")
    print(f"  - Total unique tokens: {num_groups * prefix_len + num_groups * requests_per_group * suffix_len}")
    print(f"\nKey insight:")
    print(f"  - Cache can only hold ~{cache_size/prefix_len:.1f} prefixes")
    print(f"  - But there are {num_groups} competing prefixes")
    print(f"  - Smart strategies should protect high-value prefixes!")

    # Generate workload
    sequences, _, _, _ = generate_competing_prefixes_workload(
        num_groups=num_groups,
        requests_per_group=requests_per_group,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
    )

    # Test policies
    policies = [
        "lru",
        "lfu",
        "value_aware_lru",
        "adaptive_lfu",
        "value_aware_adaptive_lfu",
    ]

    print(f"\n{'='*80}")
    print("Running benchmarks...")
    print(f"{'='*80}\n")

    results = []
    for policy in policies:
        print(f"Testing {policy}...", end=" ", flush=True)
        result = test_policy(policy, sequences, cache_size)
        results.append(result)
        print(f"✓ Hit rate: {result['hit_rate']*100:.2f}%")

    # Print comparison
    print(f"\n{'='*80}")
    print("Results Comparison")
    print(f"{'='*80}")
    print(f"{'Policy':<30} {'Hit Rate':>12} {'Evictions':>12} {'Avg Evict Time':>15}")
    print(f"{'-'*80}")

    for result in results:
        print(
            f"{result['policy']:<30} "
            f"{result['hit_rate']*100:>11.2f}% "
            f"{result['evictions']:>12} "
            f"{result['avg_eviction_ms']:>14.3f}ms"
        )

    # Analysis
    baseline = next(r for r in results if r["policy"] == "lru")
    best = max(results, key=lambda x: x["hit_rate"])

    improvement = (best["hit_rate"] - baseline["hit_rate"]) / baseline["hit_rate"] * 100 if baseline["hit_rate"] > 0 else 0

    print(f"\n{'='*80}")
    print("Analysis")
    print(f"{'='*80}")
    print(f"Baseline (LRU):")
    print(f"  - Hit rate: {baseline['hit_rate']*100:.2f}%")
    print(f"  - Evictions: {baseline['evictions']}")

    print(f"\nBest Strategy ({best['policy']}):")
    print(f"  - Hit rate: {best['hit_rate']*100:.2f}%")
    print(f"  - Evictions: {best['evictions']}")
    print(f"  - Improvement: +{improvement:.1f}% over LRU")

    if improvement > 5:
        print(f"\n✅ Significant improvement! Value-aware strategies are working!")
    else:
        print(f"\n⚠️  Small improvement. This workload may not benefit much from value-awareness.")

    print(f"\n{'='*80}")
    print("\nWhy the difference?")
    print("  - Value-aware strategies protect common prefixes from eviction")
    print("  - When cache is under pressure, they keep high-value nodes")
    print("  - Adaptive strategies give new prefixes a fair chance")
    print("  - Combined strategy gets best of both worlds")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
