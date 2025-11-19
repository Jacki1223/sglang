from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class ValueAwareLRUStrategy(EvictionStrategy):
    """
    Value-aware LRU eviction strategy that considers node value in addition to recency.

    This strategy protects high-value nodes from eviction by considering:
    1. Prefix length: Longer prefixes are more valuable as they may be shared
    2. Subtree size: Nodes with more children are likely common prefixes
    3. Recency: Standard LRU based on last_access_time

    The priority is calculated as:
        priority = recency - value_score * value_weight

    Lower priority values are evicted first, so high-value nodes get lower priority
    (protected from eviction).

    Args:
        prefix_weight: Weight for prefix length contribution to value score
        subtree_weight: Weight for subtree size contribution to value score
        value_weight: Overall multiplier for value score impact on priority
    """

    def __init__(
        self,
        prefix_weight: float = 0.001,
        subtree_weight: float = 0.01,
        value_weight: float = 1000.0,
    ):
        self.prefix_weight = prefix_weight
        self.subtree_weight = subtree_weight
        self.value_weight = value_weight

    def get_priority(self, node: "TreeNode") -> float:
        # Calculate value score based on node characteristics
        value_score = self._calculate_value_score(node)

        # Recency component (older nodes have lower priority)
        recency = node.last_access_time

        # Combine recency and value (high value nodes get protected)
        # Lower priority = evicted first
        priority = recency - (value_score * self.value_weight)

        return priority

    def _calculate_value_score(self, node: "TreeNode") -> float:
        """Calculate the value score of a node based on its characteristics."""
        # Prefix length: longer prefixes may be shared by multiple requests
        prefix_len = self._get_prefix_length(node)

        # Subtree size: more children means it's a common prefix
        subtree_size = len(node.children)

        # Combined value score
        value_score = (
            self.prefix_weight * prefix_len + self.subtree_weight * subtree_size
        )

        return value_score

    def _get_prefix_length(self, node: "TreeNode") -> int:
        """Calculate the total prefix length from root to this node."""
        length = 0
        current = node
        while current.parent is not None:
            if current.key is not None:
                length += len(current.key)
            current = current.parent
        return length


class AdaptiveLFUStrategy(EvictionStrategy):
    """
    Adaptive LFU strategy that addresses the cold-start problem of traditional LFU.

    This strategy uses a two-phase approach:
    1. Protection phase: New nodes use LRU for a protection period or until they
       reach a minimum hit threshold
    2. Mature phase: After protection, nodes use LFU based on hit_count

    This ensures new requests get fair treatment while still benefiting from
    frequency-based eviction for established content.

    Args:
        protection_period: Time in seconds to protect new nodes (default: 60s)
        min_hit_threshold: Minimum hits before graduating to LFU phase (default: 3)
    """

    def __init__(self, protection_period: float = 60.0, min_hit_threshold: int = 3):
        self.protection_period = protection_period
        self.min_hit_threshold = min_hit_threshold

    def get_priority(self, node: "TreeNode") -> Tuple:
        age = time.monotonic() - node.creation_time

        # Phase 0: New nodes in protection period use LRU
        if age < self.protection_period or node.hit_count < self.min_hit_threshold:
            return (0, node.last_access_time)

        # Phase 1: Mature nodes use LFU
        return (1, node.hit_count, node.last_access_time)


class ValueAwareAdaptiveLFUStrategy(EvictionStrategy):
    """
    Combined strategy that incorporates both value-awareness and adaptive LFU.

    This strategy provides the best of both worlds:
    - Protects new nodes during a protection period (adaptive)
    - Considers node value (prefix length, subtree size)
    - Uses frequency for mature nodes

    Priority tuple format:
    - Phase 0 (new nodes): (0, value_adjusted_recency)
    - Phase 1 (mature nodes): (1, hit_count, value_adjusted_recency)

    Args:
        protection_period: Time in seconds to protect new nodes (default: 60s)
        min_hit_threshold: Minimum hits before graduating to LFU phase (default: 3)
        prefix_weight: Weight for prefix length contribution to value score
        subtree_weight: Weight for subtree size contribution to value score
        value_weight: Overall multiplier for value score impact on priority
    """

    def __init__(
        self,
        protection_period: float = 60.0,
        min_hit_threshold: int = 3,
        prefix_weight: float = 0.001,
        subtree_weight: float = 0.01,
        value_weight: float = 1000.0,
    ):
        self.protection_period = protection_period
        self.min_hit_threshold = min_hit_threshold
        self.prefix_weight = prefix_weight
        self.subtree_weight = subtree_weight
        self.value_weight = value_weight

    def get_priority(self, node: "TreeNode") -> Tuple:
        age = time.monotonic() - node.creation_time
        value_score = self._calculate_value_score(node)

        # Adjust recency by value (high value nodes get protected)
        value_adjusted_recency = node.last_access_time - (
            value_score * self.value_weight
        )

        # Phase 0: New nodes in protection period use value-aware LRU
        if age < self.protection_period or node.hit_count < self.min_hit_threshold:
            return (0, value_adjusted_recency)

        # Phase 1: Mature nodes use value-aware LFU
        return (1, node.hit_count, value_adjusted_recency)

    def _calculate_value_score(self, node: "TreeNode") -> float:
        """Calculate the value score of a node based on its characteristics."""
        prefix_len = self._get_prefix_length(node)
        subtree_size = len(node.children)

        value_score = (
            self.prefix_weight * prefix_len + self.subtree_weight * subtree_size
        )

        return value_score

    def _get_prefix_length(self, node: "TreeNode") -> int:
        """Calculate the total prefix length from root to this node."""
        length = 0
        current = node
        while current.parent is not None:
            if current.key is not None:
                length += len(current.key)
            current = current.parent
        return length
