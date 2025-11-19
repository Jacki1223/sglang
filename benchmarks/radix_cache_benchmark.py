"""
Performance benchmark for RadixCache optimizations.

This script compares the performance of:
1. Original RadixCache
2. PersistentHeapRadixCache (optimized)

Benchmark scenarios:
- High eviction pressure (frequent evictions)
- Large trees (many nodes)
- Mixed workload (insert + evict + match)
- Lock/unlock operations

Usage:
    python benchmarks/radix_cache_benchmark.py
    python benchmarks/radix_cache_benchmark.py --quick  # Fast mode
    python benchmarks/radix_cache_benchmark.py --scenarios eviction_heavy  # Specific scenario
"""

import argparse
import random
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.radix_cache_optimized import PersistentHeapRadixCache


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.times = []
        self.operations_count = 0

    def add_measurement(self, duration: float, ops_count: int = 1):
        """Add a timing measurement."""
        self.times.append(duration)
        self.operations_count += ops_count

    def get_stats(self) -> Dict[str, float]:
        """Calculate statistics."""
        if not self.times:
            return {}

        return {
            'total_time': sum(self.times),
            'avg_time': sum(self.times) / len(self.times),
            'min_time': min(self.times),
            'max_time': max(self.times),
            'ops_count': self.operations_count,
            'ops_per_sec': self.operations_count / sum(self.times) if sum(self.times) > 0 else 0,
        }

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"{self.name}: "
            f"{stats.get('ops_count', 0)} ops in {stats.get('total_time', 0):.3f}s "
            f"({stats.get('ops_per_sec', 0):.1f} ops/s)"
        )


class RadixCacheBenchmark:
    """Benchmark suite for RadixCache implementations."""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results = defaultdict(dict)

        # Benchmark parameters
        if quick_mode:
            self.num_sequences = 100
            self.eviction_rounds = 10
            self.mixed_operations = 500
        else:
            self.num_sequences = 1000
            self.eviction_rounds = 100
            self.mixed_operations = 5000

    def create_cache(self, cache_class, **kwargs) -> RadixCache:
        """Create a cache instance."""
        return cache_class(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
            **kwargs,
        )

    def generate_random_sequence(self, min_len=5, max_len=50) -> List[int]:
        """Generate a random token sequence."""
        length = random.randint(min_len, max_len)
        return [random.randint(0, 1000) for _ in range(length)]

    def benchmark_insertion(
        self, cache_class, num_sequences: int, name: str
    ) -> BenchmarkResult:
        """Benchmark insertion performance."""
        result = BenchmarkResult(name)
        cache = self.create_cache(cache_class)

        sequences = [
            self.generate_random_sequence() for _ in range(num_sequences)
        ]

        start_time = time.perf_counter()

        for seq in sequences:
            cache.insert(RadixKey(token_ids=seq, extra_key=None))

        elapsed = time.perf_counter() - start_time
        result.add_measurement(elapsed, num_sequences)

        return result

    def benchmark_eviction_heavy(
        self, cache_class, name: str
    ) -> BenchmarkResult:
        """
        Benchmark eviction under high pressure.

        This is the key scenario where persistent heap should excel.
        """
        result = BenchmarkResult(name)
        cache = self.create_cache(cache_class)

        # Insert many sequences
        print(f"  Inserting {self.num_sequences} sequences...")
        for i in range(self.num_sequences):
            seq = self.generate_random_sequence(min_len=10, max_len=30)
            cache.insert(RadixKey(token_ids=seq, extra_key=None))

        initial_size = cache.total_size()
        print(f"  Tree size: {initial_size} tokens")
        print(f"  Running {self.eviction_rounds} eviction rounds...")

        # Perform many evictions
        eviction_times = []
        for i in range(self.eviction_rounds):
            # Evict 10-20% of cache each time
            evict_amount = random.randint(
                initial_size // 10, initial_size // 5
            )

            start_time = time.perf_counter()
            cache.evict(num_tokens=evict_amount)
            elapsed = time.perf_counter() - start_time

            eviction_times.append(elapsed)
            result.add_measurement(elapsed, 1)

            # Re-insert some sequences to maintain tree size
            if cache.total_size() < initial_size // 2:
                for _ in range(self.num_sequences // 10):
                    seq = self.generate_random_sequence(min_len=10, max_len=30)
                    cache.insert(RadixKey(token_ids=seq, extra_key=None))

        # Report eviction time statistics
        avg_eviction = sum(eviction_times) / len(eviction_times)
        max_eviction = max(eviction_times)
        min_eviction = min(eviction_times)

        print(f"  Eviction times: avg={avg_eviction*1000:.2f}ms, "
              f"min={min_eviction*1000:.2f}ms, max={max_eviction*1000:.2f}ms")

        return result

    def benchmark_mixed_workload(
        self, cache_class, name: str
    ) -> BenchmarkResult:
        """Benchmark mixed insert/match/evict operations."""
        result = BenchmarkResult(name)
        cache = self.create_cache(cache_class)

        # Pre-populate cache
        for _ in range(100):
            seq = self.generate_random_sequence()
            cache.insert(RadixKey(token_ids=seq, extra_key=None))

        start_time = time.perf_counter()

        for i in range(self.mixed_operations):
            op = random.choice(['insert', 'match', 'evict'])

            if op == 'insert':
                seq = self.generate_random_sequence()
                cache.insert(RadixKey(token_ids=seq, extra_key=None))

            elif op == 'match':
                seq = self.generate_random_sequence()
                cache.match_prefix(RadixKey(token_ids=seq, extra_key=None))

            elif op == 'evict':
                if cache.evictable_size() > 0:
                    evict_amount = min(10, cache.evictable_size())
                    cache.evict(num_tokens=evict_amount)

        elapsed = time.perf_counter() - start_time
        result.add_measurement(elapsed, self.mixed_operations)

        return result

    def benchmark_lock_unlock(
        self, cache_class, name: str
    ) -> BenchmarkResult:
        """Benchmark lock/unlock operations."""
        result = BenchmarkResult(name)
        cache = self.create_cache(cache_class)

        # Insert sequences
        nodes_to_lock = []
        for _ in range(100):
            seq = self.generate_random_sequence(min_len=5, max_len=15)
            cache.insert(RadixKey(token_ids=seq, extra_key=None))

        # Collect some leaf nodes
        def collect_leaves(node, leaves, max_count=50):
            if len(leaves) >= max_count:
                return
            if len(node.children) == 0 and node != cache.root_node:
                leaves.append(node)
                return
            for child in node.children.values():
                collect_leaves(child, leaves, max_count)

        leaves = []
        collect_leaves(cache.root_node, leaves)

        # Benchmark lock/unlock cycles
        start_time = time.perf_counter()

        for _ in range(100):
            # Lock some nodes
            for node in leaves[:10]:
                cache.inc_lock_ref(node)

            # Unlock them
            for node in leaves[:10]:
                cache.dec_lock_ref(node)

        elapsed = time.perf_counter() - start_time
        result.add_measurement(elapsed, 100 * 10 * 2)  # 100 rounds * 10 nodes * 2 ops

        return result

    def benchmark_large_tree(
        self, cache_class, name: str
    ) -> BenchmarkResult:
        """Benchmark with very large tree."""
        result = BenchmarkResult(name)

        large_size = 5000 if not self.quick_mode else 500
        print(f"  Building large tree with {large_size} sequences...")

        cache = self.create_cache(cache_class)

        start_time = time.perf_counter()

        # Insert many sequences
        for i in range(large_size):
            seq = self.generate_random_sequence(min_len=20, max_len=100)
            cache.insert(RadixKey(token_ids=seq, extra_key=None))

            # Periodic eviction
            if i % 100 == 0 and cache.evictable_size() > 10000:
                cache.evict(num_tokens=5000)

        elapsed = time.perf_counter() - start_time
        result.add_measurement(elapsed, large_size)

        print(f"  Final tree size: {cache.total_size()} tokens")

        return result

    def run_scenario(
        self, scenario_name: str, scenario_fn: Callable, cache_classes: Dict
    ):
        """Run a benchmark scenario for all cache implementations."""
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")

        for impl_name, cache_class in cache_classes.items():
            print(f"\nTesting {impl_name}...")
            try:
                result = scenario_fn(cache_class, impl_name)
                self.results[scenario_name][impl_name] = result
                print(f"  ✓ {result}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()

    def print_comparison(self):
        """Print comparison of all results."""
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")

        for scenario_name, scenario_results in self.results.items():
            print(f"\n{scenario_name}:")
            print("-" * 60)

            # Get baseline (original)
            baseline = scenario_results.get('Original RadixCache')
            if not baseline:
                print("  No baseline available")
                continue

            baseline_stats = baseline.get_stats()

            for impl_name, result in scenario_results.items():
                stats = result.get_stats()

                if impl_name == 'Original RadixCache':
                    print(f"  {impl_name} (baseline):")
                    print(f"    Time: {stats['total_time']:.3f}s")
                    print(f"    Ops/sec: {stats['ops_per_sec']:.1f}")
                else:
                    # Calculate speedup
                    speedup = baseline_stats['total_time'] / stats['total_time'] if stats['total_time'] > 0 else 0
                    improvement = (1 - stats['total_time'] / baseline_stats['total_time']) * 100 if baseline_stats['total_time'] > 0 else 0

                    print(f"  {impl_name}:")
                    print(f"    Time: {stats['total_time']:.3f}s")
                    print(f"    Ops/sec: {stats['ops_per_sec']:.1f}")
                    print(f"    Speedup: {speedup:.2f}x ({improvement:+.1f}%)")

                    # Color code the improvement
                    if speedup > 1.2:
                        print(f"    ✓ SIGNIFICANT IMPROVEMENT")
                    elif speedup > 1.05:
                        print(f"    ✓ Improvement")
                    elif speedup < 0.95:
                        print(f"    ✗ Slower")

    def run_all_benchmarks(self, scenarios: List[str] = None):
        """Run all benchmark scenarios."""
        cache_classes = {
            'Original RadixCache': RadixCache,
            'PersistentHeap RadixCache': PersistentHeapRadixCache,
        }

        # Define all scenarios
        all_scenarios = {
            'insertion': lambda c, n: self.benchmark_insertion(c, self.num_sequences, n),
            'eviction_heavy': self.benchmark_eviction_heavy,
            'mixed_workload': self.benchmark_mixed_workload,
            'lock_unlock': self.benchmark_lock_unlock,
            'large_tree': self.benchmark_large_tree,
        }

        # Filter scenarios if specified
        if scenarios:
            scenarios_to_run = {k: v for k, v in all_scenarios.items() if k in scenarios}
        else:
            scenarios_to_run = all_scenarios

        # Run each scenario
        for scenario_name, scenario_fn in scenarios_to_run.items():
            self.run_scenario(scenario_name, scenario_fn, cache_classes)

        # Print comparison
        self.print_comparison()


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark RadixCache implementations'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmarks with smaller datasets'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=['insertion', 'eviction_heavy', 'mixed_workload', 'lock_unlock', 'large_tree'],
        help='Specific scenarios to run (default: all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("="*60)
    print("RadixCache Performance Benchmark")
    print("="*60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    if args.scenarios:
        print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Random seed: {args.seed}")

    # Run benchmarks
    benchmark = RadixCacheBenchmark(quick_mode=args.quick)
    benchmark.run_all_benchmarks(scenarios=args.scenarios)

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
