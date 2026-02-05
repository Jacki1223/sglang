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
Compute-Cost-Aware Load Balancing Algorithm for MoE

This algorithm extends the standard token-count-based load balancing
to consider actual expert computation costs, providing more accurate
GPU utilization and better performance.
"""

from typing import Optional, Tuple

import torch

from .deepseek import balanced_packing, replicate_experts


def compute_cost_aware_balanced_packing(
    token_weight: torch.Tensor,
    compute_cost_weight: torch.Tensor,
    num_packs: int,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack experts to GPUs considering both token count and compute cost.

    This function balances experts across GPUs by considering:
    1. Token count: How many tokens each expert processes
    2. Compute cost: Actual computation time per expert

    Parameters:
        token_weight: [num_layers, num_experts], token count per expert
        compute_cost_weight: [num_layers, num_experts], compute cost per expert
        num_packs: Number of GPU packs
        alpha: Blending factor (0.0 = only tokens, 1.0 = only compute cost)

    Returns:
        pack_index: [num_layers, num_experts], pack assignment for each expert
        rank_in_pack: [num_layers, num_experts], rank within pack
    """
    # Blend token weight and compute cost weight
    # combined_weight = alpha * compute_cost_weight + (1 - alpha) * token_weight
    combined_weight = alpha * compute_cost_weight + (1 - alpha) * token_weight

    # Use the standard balanced packing with combined weight
    return balanced_packing(combined_weight, num_packs)


def compute_cost_aware_replicate_experts(
    token_weight: torch.Tensor,
    compute_cost_weight: torch.Tensor,
    num_phy: int,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts considering both token count and compute cost.

    This decides which experts to replicate by considering:
    1. Token count: Experts with many tokens should be replicated
    2. Compute cost: Experts with high compute cost should be replicated

    Parameters:
        token_weight: [num_layers, num_logical_experts], token count
        compute_cost_weight: [num_layers, num_logical_experts], compute cost
        num_phy: Number of physical experts after replication
        alpha: Blending factor (0.0 = only tokens, 1.0 = only compute cost)

    Returns:
        phy2log: [num_layers, num_phy], logical expert id of each physical expert
        rank: [num_layers, num_phy], replica rank
        logcnt: [num_layers, num_logical_experts], number of replicas per expert
    """
    # Blend token weight and compute cost weight
    combined_weight = alpha * compute_cost_weight + (1 - alpha) * token_weight

    # Use the standard replication with combined weight
    return replicate_experts(combined_weight, num_phy)


def compute_cost_aware_rebalance_experts_hierarchical(
    token_weight: torch.Tensor,
    compute_cost_weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hierarchical expert rebalancing with compute cost awareness.

    This is the compute-cost-aware version of the hierarchical rebalancing
    algorithm. It considers both token count and actual compute cost when
    making replication and placement decisions.

    Parameters:
        token_weight: [num_layers, num_logical_experts], token count
        compute_cost_weight: [num_layers, num_logical_experts], compute cost
        num_physical_experts: Number of physical experts after replication
        num_groups: Number of expert groups
        num_nodes: Number of server nodes
        num_gpus: Number of GPUs
        alpha: Blending factor (0.0 = token-only, 1.0 = compute-cost-only)

    Returns:
        physical_to_logical_map: [num_layers, num_physical_experts]
        phyrank: [num_layers, num_physical_experts], replica rank
        logcnt: [num_layers, num_logical_experts], replica count per expert
    """
    num_layers, num_logical_experts = token_weight.shape
    assert token_weight.shape == compute_cost_weight.shape

    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Blend weights
    combined_weight = alpha * compute_cost_weight + (1 - alpha) * token_weight

    # Step 1: Pack groups to nodes
    tokens_per_group = combined_weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: Construct redundant experts within nodes
    tokens_per_mlog = combined_weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: Pack physical experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def compute_cost_aware_rebalance_experts(
    token_weight: torch.Tensor,
    compute_cost_weight: Optional[torch.Tensor],
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for compute-cost-aware expert load balancing.

    This algorithm improves upon token-count-based balancing by considering
    the actual computation cost of each expert. This leads to better GPU
    utilization since experts with higher compute cost are replicated more
    and distributed more evenly.

    Parameters:
        token_weight: [layers, num_logical_experts], token count per expert
        compute_cost_weight: [layers, num_logical_experts], compute cost per expert
                            If None, falls back to token-based balancing
        num_replicas: Number of physical experts after replication
        num_groups: Number of expert groups
        num_nodes: Number of server nodes
        num_gpus: Number of GPUs
        enable_hierarchical: Use hierarchical balancing
        alpha: Blending factor (0.0 = token-only, 1.0 = compute-cost-only)
                Recommended: 0.5-0.7 for balanced consideration

    Returns:
        physical_to_logical_map: [layers, num_replicas], expert mapping
        logical_to_physical_map: [layers, num_logical_experts, X], replica indices
        expert_count: [layers, num_logical_experts], replica count per expert

    Example:
        >>> # Token count from profiling
        >>> token_count = torch.tensor([[100, 200, 150], [120, 180, 200]])
        >>> # Compute cost from profiler (ms per token)
        >>> compute_cost = torch.tensor([[1.0, 2.5, 1.2], [1.1, 2.0, 2.3]])
        >>> # Rebalance with alpha=0.6 (60% weight on compute cost)
        >>> phy2log, log2phy, logcnt = compute_cost_aware_rebalance_experts(
        ...     token_count, compute_cost, num_replicas=6, num_groups=1,
        ...     num_nodes=1, num_gpus=2, enable_hierarchical=False, alpha=0.6
        ... )
        >>> # Expert 1 (high compute cost) will be replicated more
    """
    num_layers, num_logical_experts = token_weight.shape

    # Convert to CPU and float for computation
    token_weight = token_weight.float().cpu()

    # Fallback to token-based if no compute cost available
    if compute_cost_weight is None:
        compute_cost_weight = torch.ones_like(token_weight)
    else:
        compute_cost_weight = compute_cost_weight.float().cpu()

    # Normalize weights to prevent numerical issues
    # Normalize per layer
    token_weight_norm = token_weight / (token_weight.sum(dim=-1, keepdim=True) + 1e-8)
    compute_cost_weight_norm = compute_cost_weight / (
        compute_cost_weight.sum(dim=-1, keepdim=True) + 1e-8
    )

    # Scale back to original magnitude (use token weight magnitude)
    token_magnitude = token_weight.sum(dim=-1, keepdim=True)
    token_weight_norm = token_weight_norm * token_magnitude
    compute_cost_weight_norm = compute_cost_weight_norm * token_magnitude

    # Apply hierarchical or flat balancing
    if enable_hierarchical:
        phy2log, phyrank, logcnt = compute_cost_aware_rebalance_experts_hierarchical(
            token_weight_norm,
            compute_cost_weight_norm,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
            alpha,
        )
    else:
        # Flat balancing (treat as single node/group)
        phy2log, phyrank, logcnt = compute_cost_aware_rebalance_experts_hierarchical(
            token_weight_norm,
            compute_cost_weight_norm,
            num_replicas,
            1,  # num_groups = 1
            1,  # num_nodes = 1
            num_gpus,
            alpha,
        )

    # Build logical to physical map
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )

    return phy2log, log2phy, logcnt


__all__ = [
    "compute_cost_aware_rebalance_experts",
    "compute_cost_aware_balanced_packing",
    "compute_cost_aware_replicate_experts",
]
