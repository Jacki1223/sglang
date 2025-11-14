#!/usr/bin/env python3
# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark for expert load balancing.

This script benchmarks the impact of expert load balancing on MOE performance.

Usage:
    python benchmark/expert_load_balancing_benchmark.py \
        --num-experts 8 \
        --topk 2 \
        --batch-sizes 32,64,128,256 \
        --strategies none,local,adaptive
"""

import argparse
import time
from typing import List, Tuple

import torch

import sys
sys.path.insert(0, "python")

from sglang.srt.layers.moe.expert_load_balancer import (
    ExpertLoadBalancer,
    LoadBalancingConfig,
    LoadBalancingStrategy,
)


def create_imbalanced_topk_ids(
    batch_size: int,
    num_experts: int,
    topk: int,
    imbalance_ratio: float = 2.0,
) -> torch.Tensor:
    """Create topk_ids with controlled imbalance.

    Args:
        batch_size: Number of tokens
        num_experts: Total number of experts
        topk: Number of experts per token
        imbalance_ratio: Ratio of max load to avg load

    Returns:
        topk_ids tensor [batch_size, topk]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topk_ids = torch.zeros(batch_size, topk, dtype=torch.int32, device=device)

    # Create imbalance by making first few experts more popular
    num_popular_experts = max(1, num_experts // 4)
    popular_weight = imbalance_ratio / (imbalance_ratio - 1 + num_popular_experts / num_experts)

    for i in range(batch_size):
        for k in range(topk):
            if torch.rand(1).item() < popular_weight:
                # Choose popular expert
                expert = torch.randint(0, num_popular_experts, (1,)).item()
            else:
                # Choose non-popular expert
                expert = torch.randint(num_popular_experts, num_experts, (1,)).item()
            topk_ids[i, k] = expert

    return topk_ids


def benchmark_load_balancing(
    num_experts: int,
    topk: int,
    batch_sizes: List[int],
    strategies: List[str],
    imbalance_ratio: float = 2.0,
    num_iterations: int = 100,
):
    """Benchmark load balancing strategies.

    Args:
        num_experts: Number of experts
        topk: Number of experts per token
        batch_sizes: List of batch sizes to test
        strategies: List of strategies to test
        imbalance_ratio: Controlled imbalance ratio
        num_iterations: Number of iterations for timing
    """
    print(f"Benchmarking Expert Load Balancing")
    print(f"=" * 80)
    print(f"Configuration:")
    print(f"  num_experts: {num_experts}")
    print(f"  topk: {topk}")
    print(f"  imbalance_ratio: {imbalance_ratio}")
    print(f"  iterations: {num_iterations}")
    print(f"=" * 80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    results = []

    for batch_size in batch_sizes:
        print(f"Batch Size: {batch_size}")
        print("-" * 80)

        # Create imbalanced topk_ids
        topk_ids = create_imbalanced_topk_ids(
            batch_size, num_experts, topk, imbalance_ratio
        )
        topk_weights = torch.rand(batch_size, topk, dtype=torch.float32, device=device)

        # Compute initial statistics
        expert_counts = torch.bincount(topk_ids.view(-1), minlength=num_experts)
        max_load = expert_counts.max().item()
        avg_load = batch_size * topk / num_experts
        initial_imbalance = max_load / avg_load

        print(f"  Initial imbalance: {initial_imbalance:.3f}")
        print(f"  Max load: {max_load}, Avg load: {avg_load:.1f}")
        print()

        for strategy_name in strategies:
            try:
                strategy = LoadBalancingStrategy(strategy_name)
            except ValueError:
                print(f"  Invalid strategy: {strategy_name}, skipping")
                continue

            # Create load balancer
            config = LoadBalancingConfig(
                strategy=strategy,
                imbalance_threshold=1.3,
                redirect_fraction=0.2,
                enable_monitoring=False,
            )
            balancer = ExpertLoadBalancer(
                num_experts=num_experts,
                topk=topk,
                config=config,
            )

            # Warmup
            for _ in range(10):
                balanced_topk_ids, _ = balancer.balance(topk_ids.clone(), topk_weights)

            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            for _ in range(num_iterations):
                balanced_topk_ids, _ = balancer.balance(topk_ids.clone(), topk_weights)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            avg_latency = (end_time - start_time) / num_iterations * 1000  # ms

            # Compute final statistics
            balanced_expert_counts = torch.bincount(
                balanced_topk_ids.view(-1), minlength=num_experts
            )
            balanced_max_load = balanced_expert_counts.max().item()
            balanced_imbalance = balanced_max_load / avg_load

            improvement = (initial_imbalance - balanced_imbalance) / initial_imbalance * 100

            print(f"  Strategy: {strategy_name}")
            print(f"    Latency: {avg_latency:.3f} ms")
            print(f"    Final imbalance: {balanced_imbalance:.3f}")
            print(f"    Imbalance reduction: {improvement:.1f}%")
            print(f"    Max load: {balanced_max_load}")
            print()

            results.append({
                "batch_size": batch_size,
                "strategy": strategy_name,
                "latency_ms": avg_latency,
                "initial_imbalance": initial_imbalance,
                "final_imbalance": balanced_imbalance,
                "improvement_pct": improvement,
            })

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Batch':<10} {'Strategy':<15} {'Latency (ms)':<15} {'Imbalance':<20} {'Improvement'}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['batch_size']:<10} "
            f"{r['strategy']:<15} "
            f"{r['latency_ms']:<15.3f} "
            f"{r['initial_imbalance']:.3f} -> {r['final_imbalance']:.3f} "
            f"{r['improvement_pct']:>6.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark expert load balancing")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--topk", type=int, default=2, help="TopK value")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="32,64,128,256",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="none,local,adaptive",
        help="Comma-separated strategies to test",
    )
    parser.add_argument(
        "--imbalance-ratio",
        type=float,
        default=2.0,
        help="Target imbalance ratio",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of iterations"
    )

    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    strategies = [x.strip() for x in args.strategies.split(",")]

    benchmark_load_balancing(
        num_experts=args.num_experts,
        topk=args.topk,
        batch_sizes=batch_sizes,
        strategies=strategies,
        imbalance_ratio=args.imbalance_ratio,
        num_iterations=args.num_iterations,
    )


if __name__ == "__main__":
    main()
