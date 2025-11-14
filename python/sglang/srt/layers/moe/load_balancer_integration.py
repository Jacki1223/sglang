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

"""Integration helper for expert load balancer in fused_moe.

This module provides helper functions to easily integrate load balancing
into existing MOE implementations without modifying core files.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from .expert_load_balancer import ExpertLoadBalancer, create_load_balancer_from_env


# Global load balancer cache (per num_experts, topk)
_LOAD_BALANCER_CACHE = {}


def get_or_create_load_balancer(
    num_experts: int,
    topk: int,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> Optional[ExpertLoadBalancer]:
    """Get or create load balancer instance.

    This function caches load balancer instances to avoid repeated initialization.
    Load balancing is only enabled if SGLANG_MOE_ENABLE_LOAD_BALANCING=1.

    Args:
        num_experts: Total number of experts
        topk: Number of experts per token
        ep_size: Expert parallel size
        ep_rank: Expert parallel rank

    Returns:
        ExpertLoadBalancer instance if enabled, None otherwise
    """
    # Check if load balancing is enabled
    if os.environ.get("SGLANG_MOE_ENABLE_LOAD_BALANCING", "0") != "1":
        return None

    # Create cache key
    cache_key = (num_experts, topk, ep_size, ep_rank)

    # Return cached instance if exists
    if cache_key in _LOAD_BALANCER_CACHE:
        return _LOAD_BALANCER_CACHE[cache_key]

    # Create new instance
    balancer = create_load_balancer_from_env(
        num_experts=num_experts,
        topk=topk,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )

    # Cache it
    _LOAD_BALANCER_CACHE[cache_key] = balancer

    return balancer


def apply_load_balancing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply load balancing to topk_ids and topk_weights.

    This is a convenience function that can be called directly in MOE forward pass.

    Args:
        topk_ids: Expert IDs [M, topk]
        topk_weights: Routing weights [M, topk]
        num_experts: Total number of experts
        ep_size: Expert parallel size
        ep_rank: Expert parallel rank

    Returns:
        Tuple of (balanced_topk_ids, balanced_topk_weights)

    Example:
        # In MOE forward pass:
        topk_weights, topk_ids = router(hidden_states)

        # Apply load balancing
        from sglang.srt.layers.moe.load_balancer_integration import apply_load_balancing
        topk_ids, topk_weights = apply_load_balancing(
            topk_ids, topk_weights, num_experts=8
        )

        # Continue with expert computation
        output = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, ...)
    """
    topk = topk_ids.shape[1]

    # Get or create load balancer
    balancer = get_or_create_load_balancer(
        num_experts=num_experts,
        topk=topk,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )

    # Apply balancing if enabled
    if balancer is not None:
        topk_ids, topk_weights = balancer.balance(topk_ids, topk_weights)

    return topk_ids, topk_weights


def clear_load_balancer_cache():
    """Clear load balancer cache.

    Call this when you want to reset load balancer instances,
    e.g., after changing configuration.
    """
    global _LOAD_BALANCER_CACHE
    _LOAD_BALANCER_CACHE.clear()
