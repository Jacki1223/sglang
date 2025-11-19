#!/usr/bin/env python3
"""
Benchmark script for comparing KV cache eviction policies.

This script measures cache hit rate, eviction overhead, and throughput
for different eviction strategies under various workload patterns.

Usage:
    python benchmark_eviction_policies.py --workload shared_prefix
    python benchmark_eviction_policies.py --workload random
    python benchmark_eviction_policies.py --workload mixed --iterations 10000
"""

import argparse
import time
from collections import defaultdict
from typing import List, Tuple

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class WorkloadGenerator:
    """Generate different types of workloads for benchmarking."""

    def __init__(self, seed: int = 42):
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def shared_prefix_workload(
        self, num_requests: int, prefix_len: int = 100, suffix_len: int = 50
    ) -> List[List[int]]:
        """
        Generate requests with shared prefixes (simulates batch processing).

        Args:
            num_requests: Number of requests to generate
            prefix_len: Length of common prefix
            suffix_len: Length of unique suffix per request

        Returns:
            List of token sequences
        """
        # Common prefix for all requests
        common_prefix = torch.randint(
            0, 50000, (prefix_len,), generator=self.rng
        ).tolist()

        sequences = []
        for _ in range(num_requests):
            # Each request has the common prefix + unique suffix
            suffix = torch.randint(
                0, 50000, (suffix_len,), generator=self.rng
            ).tolist()
            sequences.append(common_prefix + suffix)

        return sequences

    def random_workload(
        self, num_requests: int, seq_len: int = 150
    ) -> List[List[int]]:
        """
        Generate completely random requests (worst case for caching).

        Args:
            num_requests: Number of requests to generate
            seq_len: Length of each sequence

        Returns:
            List of token sequences
        """
        sequences = []
        for _ in range(num_requests):
            seq = torch.randint(0, 50000, (seq_len,), generator=self.rng).tolist()
            sequences.append(seq)

        return sequences

    def repeated_workload(
        self, num_requests: int, num_unique: int = 10, seq_len: int = 150
    ) -> List[List[int]]:
        """
        Generate workload with repeated sequences (best case for caching).

        Args:
            num_requests: Number of requests to generate
            num_unique: Number of unique sequences to repeat
            seq_len: Length of each sequence

        Returns:
            List of token sequences
        """
        # Generate unique sequences
        unique_sequences = []
        for _ in range(num_unique):
            seq = torch.randint(0, 50000, (seq_len,), generator=self.rng).tolist()
            unique_sequences.append(seq)

        # Randomly select from unique sequences
        indices = torch.randint(
            0, num_unique, (num_requests,), generator=self.rng
        ).tolist()
        sequences = [unique_sequences[i] for i in indices]

        return sequences

    def mixed_workload(
        self, num_requests: int, shared_ratio: float = 0.5, seq_len: int = 150
    ) -> List[List[int]]:
        """
        Generate mixed workload with some shared prefixes and some random.

        Args:
            num_requests: Number of requests to generate
            shared_ratio: Ratio of requests with shared prefixes
            seq_len: Length of each sequence

        Returns:
            List of token sequences
        """
        num_shared = int(num_requests * shared_ratio)
        num_random = num_requests - num_shared

        sequences = []

        # Shared prefix portion
        if num_shared > 0:
            sequences.extend(
                self.shared_prefix_workload(
                    num_shared, prefix_len=int(seq_len * 0.7), suffix_len=int(seq_len * 0.3)
                )
            )

        # Random portion
        if num_random > 0:
            sequences.extend(self.random_workload(num_random, seq_len))

        # Shuffle
        indices = torch.randperm(len(sequences), generator=self.rng).tolist()
        sequences = [sequences[i] for i in indices]

        return sequences


class CacheBenchmark:
    """Benchmark cache performance with different eviction policies."""

    def __init__(
        self,
        cache_size: int = 10000,
        page_size: int = 16,
        eviction_policy: str = "lru",
    ):
        self.cache_size = cache_size
        self.page_size = page_size
        self.eviction_policy = eviction_policy
        self.stats = defaultdict(int)

    def run_benchmark(
        self, sequences: List[List[int]], verbose: bool = True
    ) -> dict:
        """
        Run benchmark on a list of sequences.

        Args:
            sequences: List of token sequences to process
            verbose: Print progress information

        Returns:
            Dictionary with benchmark results
        """
        # Create cache
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=self.page_size,
            eviction_policy=self.eviction_policy,
        )

        # Statistics
        total_tokens = 0
        cached_tokens = 0
        total_evictions = 0
        total_eviction_time = 0.0
        total_insert_time = 0.0
        total_match_time = 0.0

        # Process sequences
        for idx, seq in enumerate(sequences):
            if verbose and idx % 100 == 0:
                print(f"Processing request {idx}/{len(sequences)}...")

            key = RadixKey(seq)
            total_tokens += len(seq)

            # Match prefix
            match_start = time.perf_counter()
            result = cache.match_prefix(key)
            match_time = time.perf_counter() - match_start
            total_match_time += match_time

            cached_tokens += len(result.device_indices)
            new_tokens = len(seq) - len(result.device_indices)

            # Insert new tokens
            if new_tokens > 0:
                insert_start = time.perf_counter()

                # Check if eviction is needed
                cache_size = cache.total_size()
                if cache_size + new_tokens > self.cache_size:
                    evict_start = time.perf_counter()
                    tokens_to_evict = cache_size + new_tokens - self.cache_size
                    cache.evict(tokens_to_evict)
                    evict_time = time.perf_counter() - evict_start
                    total_eviction_time += evict_time
                    total_evictions += 1

                # Insert
                value = torch.tensor(seq, dtype=torch.int64)
                cache.insert(key, value)

                insert_time = time.perf_counter() - insert_start
                total_insert_time += insert_time

        # Calculate metrics
        hit_rate = cached_tokens / total_tokens if total_tokens > 0 else 0.0
        avg_eviction_time = (
            total_eviction_time / total_evictions if total_evictions > 0 else 0.0
        )
        avg_insert_time = total_insert_time / len(sequences)
        avg_match_time = total_match_time / len(sequences)

        results = {
            "policy": self.eviction_policy,
            "total_requests": len(sequences),
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "hit_rate": hit_rate,
            "total_evictions": total_evictions,
            "avg_eviction_time_ms": avg_eviction_time * 1000,
            "avg_insert_time_ms": avg_insert_time * 1000,
            "avg_match_time_ms": avg_match_time * 1000,
            "final_cache_size": cache.total_size(),
        }

        return results


def compare_policies(
    workload_type: str,
    num_requests: int = 1000,
    cache_size: int = 10000,
    page_size: int = 16,
) -> None:
    """
    Compare all eviction policies on a specific workload.

    Args:
        workload_type: Type of workload to generate
        num_requests: Number of requests to process
        cache_size: Maximum cache size in tokens
        page_size: Page size for cache
    """
    print(f"\n{'='*80}")
    print(f"Benchmark: {workload_type} workload")
    print(f"Requests: {num_requests}, Cache Size: {cache_size}, Page Size: {page_size}")
    print(f"{'='*80}\n")

    # Generate workload
    generator = WorkloadGenerator()
    if workload_type == "shared_prefix":
        sequences = generator.shared_prefix_workload(num_requests)
    elif workload_type == "random":
        sequences = generator.random_workload(num_requests)
    elif workload_type == "repeated":
        sequences = generator.repeated_workload(num_requests)
    elif workload_type == "mixed":
        sequences = generator.mixed_workload(num_requests)
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")

    # Policies to test
    policies = [
        "lru",
        "lfu",
        "value_aware_lru",
        "adaptive_lfu",
        "value_aware_adaptive_lfu",
    ]

    results = []

    for policy in policies:
        print(f"\nTesting {policy}...")
        benchmark = CacheBenchmark(
            cache_size=cache_size, page_size=page_size, eviction_policy=policy
        )
        result = benchmark.run_benchmark(sequences, verbose=False)
        results.append(result)

    # Print comparison table
    print(f"\n{'='*80}")
    print("Results Comparison")
    print(f"{'='*80}")
    print(
        f"{'Policy':<30} {'Hit Rate':>10} {'Evictions':>10} {'Evict Time':>12} {'Insert Time':>12}"
    )
    print(f"{'-'*80}")

    for result in results:
        print(
            f"{result['policy']:<30} "
            f"{result['hit_rate']*100:>9.2f}% "
            f"{result['total_evictions']:>10} "
            f"{result['avg_eviction_time_ms']:>11.3f}ms "
            f"{result['avg_insert_time_ms']:>11.3f}ms"
        )

    # Find best policy
    best_hit_rate = max(results, key=lambda x: x["hit_rate"])
    print(f"\n✓ Best hit rate: {best_hit_rate['policy']} ({best_hit_rate['hit_rate']*100:.2f}%)")

    best_eviction_time = min(
        results, key=lambda x: x["avg_eviction_time_ms"] if x["total_evictions"] > 0 else float("inf")
    )
    if best_eviction_time["total_evictions"] > 0:
        print(
            f"✓ Fastest eviction: {best_eviction_time['policy']} "
            f"({best_eviction_time['avg_eviction_time_ms']:.3f}ms)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KV cache eviction policies"
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["shared_prefix", "random", "repeated", "mixed"],
        default="shared_prefix",
        help="Type of workload to generate",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of requests to process",
    )
    parser.add_argument(
        "--cache-size", type=int, default=10000, help="Maximum cache size in tokens"
    )
    parser.add_argument(
        "--page-size", type=int, default=16, help="Page size for cache"
    )
    parser.add_argument(
        "--all-workloads",
        action="store_true",
        help="Run all workload types",
    )

    args = parser.parse_args()

    if args.all_workloads:
        workloads = ["shared_prefix", "random", "repeated", "mixed"]
        for workload in workloads:
            compare_policies(
                workload_type=workload,
                num_requests=args.iterations,
                cache_size=args.cache_size,
                page_size=args.page_size,
            )
    else:
        compare_policies(
            workload_type=args.workload,
            num_requests=args.iterations,
            cache_size=args.cache_size,
            page_size=args.page_size,
        )


if __name__ == "__main__":
    main()
