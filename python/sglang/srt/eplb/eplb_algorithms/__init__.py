from enum import Enum, auto
from typing import Optional

import torch

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.eplb.eplb_algorithms import (
    compute_aware,
    deepseek,
    deepseek_vec,
    elasticity_aware,
)


class EplbAlgorithm(Enum):
    deepseek = auto()
    deepseek_hierarchical = auto()
    deepseek_vec = auto()
    deepseek_vec_hierarchical = auto()
    elasticity_aware = auto()
    compute_aware = auto()
    compute_aware_hierarchical = auto()
    # TODO may have more algorithm later


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    algorithm: EplbAlgorithm,
    compute_cost_per_expert: Optional[torch.Tensor] = None,
    compute_cost_alpha: float = 0.5,
):
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        return deepseek.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
        )

    if algorithm in [
        EplbAlgorithm.deepseek_vec,
        EplbAlgorithm.deepseek_vec_hierarchical,
    ]:
        return deepseek_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_vec_hierarchical,
        )

    if algorithm == EplbAlgorithm.elasticity_aware:
        return elasticity_aware.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=False,
            active_ranks=(
                ElasticEPStateManager.instance().active_ranks
                if ElasticEPStateManager.instance() is not None
                else ElasticEPStateManager.healthy_rank_state()
            ),
        )

    if algorithm in [
        EplbAlgorithm.compute_aware,
        EplbAlgorithm.compute_aware_hierarchical,
    ]:
        return compute_aware.compute_cost_aware_rebalance_experts(
            token_weight=tokens_per_expert.sum(dim=0),
            compute_cost_weight=(
                compute_cost_per_expert.sum(dim=0)
                if compute_cost_per_expert is not None
                else None
            ),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.compute_aware_hierarchical,
            alpha=compute_cost_alpha,
        )

    raise NotImplementedError


def compute_algorithm(
    raw_algorithm: str,
    num_groups: Optional[int],
    num_nodes: int,
) -> EplbAlgorithm:
    if raw_algorithm != "auto":
        return EplbAlgorithm[raw_algorithm]

    # TODO test on real scenarios and know which ones perform better
    if (num_groups is not None) and (num_groups % num_nodes == 0):
        return EplbAlgorithm.deepseek_hierarchical
    else:
        return EplbAlgorithm.deepseek
