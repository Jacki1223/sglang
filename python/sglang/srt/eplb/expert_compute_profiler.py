# Copyright 2023-2024 SGLang Team
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

"""
Expert Compute Cost Profiler for MoE Load Balancing

This module profiles the actual computation cost (execution time, memory usage)
of each expert to enable compute-cost-aware load balancing instead of just
token-count-based balancing.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

logger = logging.getLogger(__name__)


@dataclass
class ExpertComputeCost:
    """Stores compute cost statistics for a single expert."""

    total_execution_time: float = 0.0  # Total time spent in this expert (ms)
    total_tokens_processed: int = 0  # Total tokens processed
    num_invocations: int = 0  # Number of times this expert was called

    @property
    def avg_time_per_token(self) -> float:
        """Average execution time per token (ms)."""
        if self.total_tokens_processed == 0:
            return 0.0
        return self.total_execution_time / self.total_tokens_processed

    @property
    def avg_time_per_invocation(self) -> float:
        """Average execution time per invocation (ms)."""
        if self.num_invocations == 0:
            return 0.0
        return self.total_execution_time / self.num_invocations

    def update(self, execution_time_ms: float, num_tokens: int):
        """Update statistics with new measurement."""
        self.total_execution_time += execution_time_ms
        self.total_tokens_processed += num_tokens
        self.num_invocations += 1


class ExpertComputeProfiler:
    """
    Profiles computation cost for each expert in MoE layers.

    This profiler tracks:
    1. Execution time per expert
    2. Token throughput per expert
    3. Compute cost variations across experts

    The profiled data is used to weight experts by actual compute cost
    rather than just token count for more accurate load balancing.
    """

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        enable_profiling: bool = True,
        warmup_steps: int = 10,
        profiling_interval: int = 1,  # Profile every N forward passes
    ):
        """
        Args:
            expert_location_metadata: Metadata about expert locations
            enable_profiling: Whether to enable profiling
            warmup_steps: Number of warmup steps before profiling
            profiling_interval: Profile every N forward passes (1 = every pass)
        """
        self.expert_location_metadata = expert_location_metadata
        self.enable_profiling = enable_profiling
        self.warmup_steps = warmup_steps
        self.profiling_interval = profiling_interval

        self.num_layers = expert_location_metadata.num_layers
        self.num_physical_experts = expert_location_metadata.num_physical_experts

        # Statistics per (layer, expert)
        # Dict[layer_idx, Dict[expert_idx, ExpertComputeCost]]
        self.expert_costs: Dict[int, Dict[int, ExpertComputeCost]] = defaultdict(
            lambda: defaultdict(ExpertComputeCost)
        )

        self.forward_pass_count = 0
        self.profiling_enabled = False

        # CUDA events for timing
        self.device_module = torch.get_device_module()
        self.use_cuda_events = torch.cuda.is_available()

        logger.info(
            f"[ExpertComputeProfiler] Initialized with warmup_steps={warmup_steps}, "
            f"profiling_interval={profiling_interval}"
        )

    def start_profiling(self):
        """Start profiling after warmup."""
        if self.enable_profiling and not self.profiling_enabled:
            self.profiling_enabled = True
            logger.info("[ExpertComputeProfiler] Profiling started")

    def should_profile_this_pass(self) -> bool:
        """Check if we should profile the current forward pass."""
        if not self.enable_profiling or not self.profiling_enabled:
            return False

        # Profile every N passes
        return self.forward_pass_count % self.profiling_interval == 0

    def profile_expert_execution(
        self,
        layer_idx: int,
        expert_idx: int,
        num_tokens: int,
        execution_time_ms: Optional[float] = None,
        start_event: Optional[torch.cuda.Event] = None,
        end_event: Optional[torch.cuda.Event] = None,
    ):
        """
        Record execution statistics for a single expert.

        Args:
            layer_idx: Layer index
            expert_idx: Physical expert index
            num_tokens: Number of tokens processed
            execution_time_ms: Execution time in milliseconds (if pre-computed)
            start_event: CUDA start event (if using CUDA events)
            end_event: CUDA end event (if using CUDA events)
        """
        if not self.should_profile_this_pass():
            return

        # Calculate execution time
        if execution_time_ms is not None:
            exec_time = execution_time_ms
        elif start_event is not None and end_event is not None:
            # Use CUDA events
            end_event.synchronize()
            exec_time = start_event.elapsed_time(end_event)  # Returns ms
        else:
            # No timing info available
            return

        # Update statistics
        self.expert_costs[layer_idx][expert_idx].update(exec_time, num_tokens)

    def on_forward_pass_start(self):
        """Called at the start of each forward pass."""
        self.forward_pass_count += 1

        # Enable profiling after warmup
        if self.forward_pass_count == self.warmup_steps:
            self.start_profiling()

    def get_expert_compute_weights(
        self, layer_idx: Optional[int] = None, use_time_per_token: bool = True
    ) -> torch.Tensor:
        """
        Get compute cost weights for experts.

        Args:
            layer_idx: If specified, return weights for this layer only.
                      Otherwise, return weights for all layers.
            use_time_per_token: If True, use avg time per token.
                               If False, use avg time per invocation.

        Returns:
            Tensor of shape [num_layers, num_physical_experts] or
            [num_physical_experts] if layer_idx is specified.
            Values represent relative compute cost (higher = more expensive).
        """
        if layer_idx is not None:
            # Single layer
            weights = torch.ones(
                self.num_physical_experts, dtype=torch.float32, device="cpu"
            )
            for expert_idx in range(self.num_physical_experts):
                cost = self.expert_costs[layer_idx].get(
                    expert_idx, ExpertComputeCost()
                )
                if use_time_per_token:
                    weights[expert_idx] = max(cost.avg_time_per_token, 1e-6)
                else:
                    weights[expert_idx] = max(cost.avg_time_per_invocation, 1e-6)
            return weights
        else:
            # All layers
            weights = torch.ones(
                (self.num_layers, self.num_physical_experts),
                dtype=torch.float32,
                device="cpu",
            )
            for layer_idx in range(self.num_layers):
                for expert_idx in range(self.num_physical_experts):
                    cost = self.expert_costs[layer_idx].get(
                        expert_idx, ExpertComputeCost()
                    )
                    if use_time_per_token:
                        weights[layer_idx, expert_idx] = max(
                            cost.avg_time_per_token, 1e-6
                        )
                    else:
                        weights[layer_idx, expert_idx] = max(
                            cost.avg_time_per_invocation, 1e-6
                        )
            return weights

    def get_adjusted_expert_load(
        self, token_count: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust expert load based on compute cost.

        Args:
            token_count: Tensor of shape [num_layers, num_logical_experts]
                        representing token counts per expert.

        Returns:
            compute_cost_weight: Tensor of shape [num_layers, num_logical_experts]
                                 representing compute-cost-adjusted load.
            token_count: Original token count (unchanged).
        """
        if not self.profiling_enabled:
            # Return token count as-is if profiling not enabled
            return token_count.clone(), token_count

        num_layers, num_logical_experts = token_count.shape

        # Get compute cost per physical expert
        physical_compute_weights = self.get_expert_compute_weights(
            use_time_per_token=True
        )

        # Convert to logical expert compute weights by averaging
        # across physical replicas
        logical_compute_weights = torch.zeros(
            (num_layers, num_logical_experts), dtype=torch.float32, device="cpu"
        )

        physical_to_logical_map = (
            self.expert_location_metadata.physical_to_logical_map_cpu
        )

        for layer_idx in range(num_layers):
            for physical_idx in range(self.num_physical_experts):
                logical_idx = physical_to_logical_map[layer_idx, physical_idx].item()
                logical_compute_weights[layer_idx, logical_idx] += (
                    physical_compute_weights[layer_idx, physical_idx]
                )

        # Normalize by number of replicas
        for layer_idx in range(num_layers):
            for logical_idx in range(num_logical_experts):
                num_replicas = (physical_to_logical_map[layer_idx] == logical_idx).sum()
                if num_replicas > 0:
                    logical_compute_weights[layer_idx, logical_idx] /= num_replicas

        # Compute-cost-adjusted load = token_count * compute_weight
        compute_cost_weight = token_count.float() * logical_compute_weights.to(
            token_count.device
        )

        return compute_cost_weight, token_count

    def get_statistics_summary(self) -> Dict:
        """Get summary statistics for logging/debugging."""
        summary = {
            "forward_pass_count": self.forward_pass_count,
            "profiling_enabled": self.profiling_enabled,
            "layers": {},
        }

        for layer_idx in range(self.num_layers):
            layer_stats = {}
            for expert_idx in range(self.num_physical_experts):
                cost = self.expert_costs[layer_idx].get(
                    expert_idx, ExpertComputeCost()
                )
                if cost.num_invocations > 0:
                    layer_stats[f"expert_{expert_idx}"] = {
                        "avg_time_per_token_ms": cost.avg_time_per_token,
                        "avg_time_per_invocation_ms": cost.avg_time_per_invocation,
                        "total_tokens": cost.total_tokens_processed,
                        "num_invocations": cost.num_invocations,
                    }
            if layer_stats:
                summary["layers"][f"layer_{layer_idx}"] = layer_stats

        return summary

    def reset_statistics(self):
        """Reset all profiling statistics."""
        self.expert_costs.clear()
        self.forward_pass_count = 0
        self.profiling_enabled = False
        logger.info("[ExpertComputeProfiler] Statistics reset")


# Global profiler instance
_global_expert_compute_profiler: Optional[ExpertComputeProfiler] = None


def get_global_expert_compute_profiler() -> Optional[ExpertComputeProfiler]:
    """Get the global expert compute profiler."""
    return _global_expert_compute_profiler


def set_global_expert_compute_profiler(profiler: Optional[ExpertComputeProfiler]):
    """Set the global expert compute profiler."""
    global _global_expert_compute_profiler
    _global_expert_compute_profiler = profiler


def create_expert_compute_profiler(
    expert_location_metadata: ExpertLocationMetadata,
    enable_profiling: bool = True,
    warmup_steps: int = 10,
    profiling_interval: int = 1,
) -> ExpertComputeProfiler:
    """
    Create and set a global expert compute profiler.

    Args:
        expert_location_metadata: Expert location metadata
        enable_profiling: Whether to enable profiling
        warmup_steps: Number of warmup steps
        profiling_interval: Profile every N forward passes

    Returns:
        Created profiler instance
    """
    profiler = ExpertComputeProfiler(
        expert_location_metadata=expert_location_metadata,
        enable_profiling=enable_profiling,
        warmup_steps=warmup_steps,
        profiling_interval=profiling_interval,
    )
    set_global_expert_compute_profiler(profiler)
    return profiler
