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

"""Expert load balancing for MOE models.

This module implements dynamic load balancing strategies to improve
expert utilization and reduce performance degradation from load imbalance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategy."""

    NONE = "none"  # No load balancing (default)
    LOCAL = "local"  # Balance within single GPU/rank
    GLOBAL_EP = "global_ep"  # Balance across Expert Parallel ranks
    ADAPTIVE = "adaptive"  # Adaptive strategy based on imbalance ratio


@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing."""

    strategy: LoadBalancingStrategy = LoadBalancingStrategy.NONE
    imbalance_threshold: float = 1.3  # Trigger rebalancing if max/avg > threshold
    redirect_fraction: float = 0.2  # Fraction of tokens to redirect (0.0-0.5)
    enable_monitoring: bool = True  # Log load statistics
    monitor_interval: int = 100  # Log every N forward passes


@dataclass
class LoadStatistics:
    """Expert load statistics."""

    expert_counts: torch.Tensor  # [num_experts], tokens per expert
    total_tokens: int
    max_load: int
    min_load: int
    avg_load: float
    imbalance_ratio: float  # max_load / avg_load
    std_dev: float


class ExpertLoadBalancer:
    """Expert load balancer for MOE models.

    This class provides dynamic load balancing across experts to mitigate
    performance degradation from uneven expert utilization.

    Features:
    - Multiple balancing strategies (local, global EP, adaptive)
    - Load monitoring and statistics
    - Configurable imbalance thresholds
    - Minimal overhead when disabled

    Usage:
        balancer = ExpertLoadBalancer(
            num_experts=8,
            topk=2,
            config=LoadBalancingConfig(strategy=LoadBalancingStrategy.ADAPTIVE)
        )

        # In MOE forward pass:
        topk_ids = balancer.balance(topk_ids)
    """

    def __init__(
        self,
        num_experts: int,
        topk: int,
        config: Optional[LoadBalancingConfig] = None,
        ep_size: int = 1,
        ep_rank: int = 0,
    ):
        """Initialize load balancer.

        Args:
            num_experts: Total number of experts
            topk: Number of experts per token
            config: Load balancing configuration
            ep_size: Expert parallel size
            ep_rank: Expert parallel rank
        """
        self.num_experts = num_experts
        self.topk = topk
        self.config = config or LoadBalancingConfig()
        self.ep_size = ep_size
        self.ep_rank = ep_rank

        # Statistics tracking
        self.forward_count = 0
        self.rebalance_count = 0
        self.cumulative_imbalance = 0.0

        # EP configuration
        if ep_size > 1:
            self.experts_per_rank = num_experts // ep_size
            self.rank_expert_start = ep_rank * self.experts_per_rank
            self.rank_expert_end = (ep_rank + 1) * self.experts_per_rank
        else:
            self.experts_per_rank = num_experts
            self.rank_expert_start = 0
            self.rank_expert_end = num_experts

        logger.info(
            f"ExpertLoadBalancer initialized: "
            f"num_experts={num_experts}, topk={topk}, "
            f"strategy={config.strategy.value}, "
            f"ep_size={ep_size}, ep_rank={ep_rank}"
        )

    def balance(
        self,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply load balancing to topk_ids.

        Args:
            topk_ids: Expert IDs [M, topk]
            topk_weights: Optional routing weights [M, topk]

        Returns:
            Tuple of (balanced_topk_ids, balanced_topk_weights)
        """
        self.forward_count += 1

        if self.config.strategy == LoadBalancingStrategy.NONE:
            return topk_ids, topk_weights

        # Compute load statistics
        stats = self._compute_load_statistics(topk_ids)

        # Log statistics periodically
        if self.config.enable_monitoring and (
            self.forward_count % self.config.monitor_interval == 0
        ):
            self._log_statistics(stats)

        # Check if rebalancing is needed
        if stats.imbalance_ratio < self.config.imbalance_threshold:
            return topk_ids, topk_weights

        # Apply balancing strategy
        if self.config.strategy == LoadBalancingStrategy.LOCAL:
            balanced_topk_ids = self._balance_local(topk_ids, stats)
        elif self.config.strategy == LoadBalancingStrategy.GLOBAL_EP:
            balanced_topk_ids = self._balance_global_ep(topk_ids, stats)
        elif self.config.strategy == LoadBalancingStrategy.ADAPTIVE:
            balanced_topk_ids = self._balance_adaptive(topk_ids, stats)
        else:
            balanced_topk_ids = topk_ids

        self.rebalance_count += 1
        self.cumulative_imbalance += stats.imbalance_ratio

        return balanced_topk_ids, topk_weights

    def _compute_load_statistics(self, topk_ids: torch.Tensor) -> LoadStatistics:
        """Compute expert load statistics.

        Args:
            topk_ids: Expert IDs [M, topk]

        Returns:
            LoadStatistics object
        """
        # Count tokens per expert
        expert_counts = torch.bincount(
            topk_ids.view(-1), minlength=self.num_experts
        ).to(torch.int32)

        total_tokens = topk_ids.numel()
        max_load = expert_counts.max().item()
        min_load = expert_counts.min().item()
        avg_load = total_tokens / self.num_experts
        imbalance_ratio = max_load / (avg_load + 1e-6)
        std_dev = expert_counts.float().std().item()

        return LoadStatistics(
            expert_counts=expert_counts,
            total_tokens=total_tokens,
            max_load=max_load,
            min_load=min_load,
            avg_load=avg_load,
            imbalance_ratio=imbalance_ratio,
            std_dev=std_dev,
        )

    def _balance_local(
        self, topk_ids: torch.Tensor, stats: LoadStatistics
    ) -> torch.Tensor:
        """Local load balancing within a single rank.

        Strategy: Redirect tokens from overloaded experts to underloaded experts
        within the same rank.

        Args:
            topk_ids: Expert IDs [M, topk]
            stats: Load statistics

        Returns:
            Balanced topk_ids
        """
        balanced_topk_ids = topk_ids.clone()
        expert_counts = stats.expert_counts

        # Identify overloaded and underloaded experts
        threshold = stats.avg_load * 1.2
        overloaded_experts = (expert_counts > threshold).nonzero(as_tuple=False).view(-1)
        underloaded_experts = (
            (expert_counts < stats.avg_load).nonzero(as_tuple=False).view(-1)
        )

        if len(overloaded_experts) == 0 or len(underloaded_experts) == 0:
            return balanced_topk_ids

        # For each overloaded expert, redirect some tokens
        for expert_id in overloaded_experts:
            expert_id = expert_id.item()
            num_tokens = expert_counts[expert_id].item()
            num_to_redirect = int((num_tokens - stats.avg_load) * self.config.redirect_fraction)

            if num_to_redirect <= 0:
                continue

            # Find tokens assigned to this expert
            mask = topk_ids == expert_id
            token_positions = mask.nonzero(as_tuple=False)

            if len(token_positions) == 0:
                continue

            # Select tokens to redirect (randomly sample)
            num_to_redirect = min(num_to_redirect, len(token_positions))
            redirect_indices = torch.randperm(len(token_positions))[:num_to_redirect]
            redirect_positions = token_positions[redirect_indices]

            # Assign to underloaded experts (round-robin)
            for i, pos in enumerate(redirect_positions):
                token_idx, topk_idx = pos[0].item(), pos[1].item()
                alt_expert = underloaded_experts[i % len(underloaded_experts)].item()
                balanced_topk_ids[token_idx, topk_idx] = alt_expert

        return balanced_topk_ids

    def _balance_global_ep(
        self, topk_ids: torch.Tensor, stats: LoadStatistics
    ) -> torch.Tensor:
        """Global load balancing across Expert Parallel ranks.

        Strategy: When EP is enabled, balance load across different ranks
        by redirecting tokens to experts in less loaded ranks.

        Args:
            topk_ids: Expert IDs [M, topk]
            stats: Load statistics

        Returns:
            Balanced topk_ids
        """
        if self.ep_size <= 1:
            # No EP, fall back to local balancing
            return self._balance_local(topk_ids, stats)

        balanced_topk_ids = topk_ids.clone()
        expert_counts = stats.expert_counts

        # Compute load per EP rank
        rank_loads = []
        for rank in range(self.ep_size):
            rank_start = rank * self.experts_per_rank
            rank_end = (rank + 1) * self.experts_per_rank
            rank_load = expert_counts[rank_start:rank_end].sum().item()
            rank_loads.append(rank_load)

        avg_rank_load = sum(rank_loads) / self.ep_size
        max_rank_load = max(rank_loads)
        rank_imbalance = max_rank_load / (avg_rank_load + 1e-6)

        # If ranks are balanced, use local balancing
        if rank_imbalance < 1.2:
            return self._balance_local(topk_ids, stats)

        # Find overloaded and underloaded ranks
        overloaded_ranks = [
            r for r, load in enumerate(rank_loads) if load > avg_rank_load * 1.2
        ]
        underloaded_ranks = [
            r for r, load in enumerate(rank_loads) if load < avg_rank_load
        ]

        if not overloaded_ranks or not underloaded_ranks:
            return balanced_topk_ids

        # Redirect tokens from overloaded ranks to underloaded ranks
        for overloaded_rank in overloaded_ranks:
            # Find experts in this rank
            rank_start = overloaded_rank * self.experts_per_rank
            rank_end = (overloaded_rank + 1) * self.experts_per_rank
            rank_experts = list(range(rank_start, rank_end))

            # Find tokens assigned to experts in this rank
            mask = torch.zeros_like(topk_ids, dtype=torch.bool)
            for expert_id in rank_experts:
                mask |= topk_ids == expert_id

            token_positions = mask.nonzero(as_tuple=False)

            if len(token_positions) == 0:
                continue

            # Calculate number of tokens to redirect
            rank_load = rank_loads[overloaded_rank]
            num_to_redirect = int(
                (rank_load - avg_rank_load) * self.config.redirect_fraction
            )
            num_to_redirect = min(num_to_redirect, len(token_positions))

            if num_to_redirect <= 0:
                continue

            # Select tokens to redirect
            redirect_indices = torch.randperm(len(token_positions))[:num_to_redirect]
            redirect_positions = token_positions[redirect_indices]

            # Find underloaded experts in underloaded ranks
            underloaded_experts = []
            for under_rank in underloaded_ranks:
                rank_start = under_rank * self.experts_per_rank
                rank_end = (under_rank + 1) * self.experts_per_rank
                for expert_id in range(rank_start, rank_end):
                    if expert_counts[expert_id] < stats.avg_load:
                        underloaded_experts.append(expert_id)

            if not underloaded_experts:
                continue

            # Assign redirected tokens to underloaded experts
            for i, pos in enumerate(redirect_positions):
                token_idx, topk_idx = pos[0].item(), pos[1].item()
                alt_expert = underloaded_experts[i % len(underloaded_experts)]
                balanced_topk_ids[token_idx, topk_idx] = alt_expert

        return balanced_topk_ids

    def _balance_adaptive(
        self, topk_ids: torch.Tensor, stats: LoadStatistics
    ) -> torch.Tensor:
        """Adaptive load balancing.

        Selects strategy based on imbalance characteristics:
        - Low imbalance (< 1.5): No action
        - Medium imbalance (1.5-2.0): Local balancing with low redirect fraction
        - High imbalance (> 2.0): Aggressive local/global balancing

        Args:
            topk_ids: Expert IDs [M, topk]
            stats: Load statistics

        Returns:
            Balanced topk_ids
        """
        imbalance = stats.imbalance_ratio

        if imbalance < 1.5:
            # Low imbalance, no action needed
            return topk_ids

        # Save original redirect fraction
        original_redirect_fraction = self.config.redirect_fraction

        if imbalance < 2.0:
            # Medium imbalance: conservative local balancing
            self.config.redirect_fraction = 0.15
            balanced_topk_ids = self._balance_local(topk_ids, stats)
        elif imbalance < 3.0:
            # High imbalance: aggressive local balancing
            self.config.redirect_fraction = 0.25
            balanced_topk_ids = self._balance_local(topk_ids, stats)
        else:
            # Very high imbalance: global balancing if EP enabled
            self.config.redirect_fraction = 0.3
            if self.ep_size > 1:
                balanced_topk_ids = self._balance_global_ep(topk_ids, stats)
            else:
                balanced_topk_ids = self._balance_local(topk_ids, stats)

        # Restore original redirect fraction
        self.config.redirect_fraction = original_redirect_fraction

        return balanced_topk_ids

    def _log_statistics(self, stats: LoadStatistics):
        """Log load statistics.

        Args:
            stats: Load statistics
        """
        avg_imbalance = (
            self.cumulative_imbalance / self.rebalance_count
            if self.rebalance_count > 0
            else 0.0
        )

        logger.info(
            f"ExpertLoadBalancer Stats [forward={self.forward_count}]: "
            f"imbalance_ratio={stats.imbalance_ratio:.3f}, "
            f"max_load={stats.max_load}, "
            f"min_load={stats.min_load}, "
            f"avg_load={stats.avg_load:.1f}, "
            f"std_dev={stats.std_dev:.1f}, "
            f"rebalance_count={self.rebalance_count}, "
            f"avg_imbalance={avg_imbalance:.3f}"
        )

        # Detailed per-expert counts (only if very imbalanced)
        if stats.imbalance_ratio > 2.0:
            expert_counts_str = ", ".join(
                [f"E{i}:{stats.expert_counts[i].item()}" for i in range(min(10, self.num_experts))]
            )
            if self.num_experts > 10:
                expert_counts_str += ", ..."
            logger.debug(f"Expert counts: {expert_counts_str}")

    def get_statistics(self) -> dict:
        """Get balancer statistics.

        Returns:
            Dictionary of statistics
        """
        avg_imbalance = (
            self.cumulative_imbalance / self.rebalance_count
            if self.rebalance_count > 0
            else 0.0
        )

        return {
            "forward_count": self.forward_count,
            "rebalance_count": self.rebalance_count,
            "rebalance_rate": self.rebalance_count / max(self.forward_count, 1),
            "avg_imbalance_ratio": avg_imbalance,
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.forward_count = 0
        self.rebalance_count = 0
        self.cumulative_imbalance = 0.0


def create_load_balancer_from_env(
    num_experts: int,
    topk: int,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> ExpertLoadBalancer:
    """Create load balancer from environment variables.

    Environment variables:
        SGLANG_MOE_LOAD_BALANCE_STRATEGY: Strategy (none, local, global_ep, adaptive)
        SGLANG_MOE_LOAD_BALANCE_THRESHOLD: Imbalance threshold (default: 1.3)
        SGLANG_MOE_LOAD_BALANCE_REDIRECT: Redirect fraction (default: 0.2)
        SGLANG_MOE_LOAD_BALANCE_MONITORING: Enable monitoring (default: true)

    Args:
        num_experts: Total number of experts
        topk: Number of experts per token
        ep_size: Expert parallel size
        ep_rank: Expert parallel rank

    Returns:
        ExpertLoadBalancer instance
    """
    # Parse strategy from environment
    strategy_str = os.environ.get(
        "SGLANG_MOE_LOAD_BALANCE_STRATEGY", "none"
    ).lower()

    try:
        strategy = LoadBalancingStrategy(strategy_str)
    except ValueError:
        logger.warning(
            f"Invalid load balancing strategy: {strategy_str}. "
            f"Using 'none'. Valid options: {[s.value for s in LoadBalancingStrategy]}"
        )
        strategy = LoadBalancingStrategy.NONE

    # Parse other config from environment
    imbalance_threshold = float(
        os.environ.get("SGLANG_MOE_LOAD_BALANCE_THRESHOLD", "1.3")
    )
    redirect_fraction = float(
        os.environ.get("SGLANG_MOE_LOAD_BALANCE_REDIRECT", "0.2")
    )
    enable_monitoring = (
        os.environ.get("SGLANG_MOE_LOAD_BALANCE_MONITORING", "true").lower()
        == "true"
    )

    config = LoadBalancingConfig(
        strategy=strategy,
        imbalance_threshold=imbalance_threshold,
        redirect_fraction=redirect_fraction,
        enable_monitoring=enable_monitoring,
    )

    return ExpertLoadBalancer(
        num_experts=num_experts,
        topk=topk,
        config=config,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
