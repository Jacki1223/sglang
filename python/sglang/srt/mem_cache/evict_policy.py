from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Union

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


class ARCManager:
    """
    Adaptive Replacement Cache (ARC) Manager.

    ARC maintains four lists:
    - T1: Recent cache (pages accessed once recently)
    - T2: Frequent cache (pages accessed multiple times)
    - B1: Ghost entries evicted from T1 (metadata only)
    - B2: Ghost entries evicted from T2 (metadata only)

    The algorithm adaptively adjusts the target size of T1 (parameter p)
    based on the workload pattern.
    """

    def __init__(self, cache_size: int):
        """
        Initialize ARC manager.

        Args:
            cache_size: Maximum number of pages in cache (|T1| + |T2| <= cache_size)
        """
        self.cache_size = cache_size
        # Adaptive parameter: target size for T1
        self.p = 0

        # Lists to track nodes
        self.T1: Set[int] = set()  # Recent cache (node IDs)
        self.T2: Set[int] = set()  # Frequent cache (node IDs)
        self.B1: Set[int] = set()  # Ghost entries from T1 (node IDs)
        self.B2: Set[int] = set()  # Ghost entries from T2 (node IDs)

        # Track node access patterns
        self.node_registry: Dict[int, "TreeNode"] = {}

    def register_node(self, node: "TreeNode"):
        """Register a node with the ARC manager."""
        self.node_registry[node.id] = node

    def unregister_node(self, node: "TreeNode"):
        """Unregister a node from the ARC manager."""
        if node.id in self.node_registry:
            del self.node_registry[node.id]
        # Remove from all lists
        self.T1.discard(node.id)
        self.T2.discard(node.id)
        self.B1.discard(node.id)
        self.B2.discard(node.id)

    def on_cache_hit(self, node: "TreeNode"):
        """
        Handle cache hit: move node from T1 to T2 if needed.

        Args:
            node: The node that was accessed
        """
        self.register_node(node)
        node_id = node.id

        if node_id in self.T1:
            # Move from T1 to T2 (becomes frequent)
            self.T1.remove(node_id)
            self.T2.add(node_id)
            node.arc_list_type = "T2"
        elif node_id in self.T2:
            # Already in T2, just update access time (handled by caller)
            pass
        else:
            # New hit, add to T1
            self.T1.add(node_id)
            node.arc_list_type = "T1"

    def on_cache_miss(self, node: "TreeNode") -> None:
        """
        Handle cache miss: update ghost lists and adjust p.

        Args:
            node: The node that was missed
        """
        self.register_node(node)
        node_id = node.id

        if node_id in self.B1:
            # Ghost hit in B1: increase preference for recent pages
            delta = max(1, len(self.B2) / max(1, len(self.B1)))
            self.p = min(self.cache_size, self.p + delta)
            # Move from B1 to T2
            self.B1.remove(node_id)
            self.T2.add(node_id)
            node.arc_list_type = "T2"
            node.in_ghost = False

        elif node_id in self.B2:
            # Ghost hit in B2: increase preference for frequent pages
            delta = max(1, len(self.B1) / max(1, len(self.B2)))
            self.p = max(0, self.p - delta)
            # Move from B2 to T2
            self.B2.remove(node_id)
            self.T2.add(node_id)
            node.arc_list_type = "T2"
            node.in_ghost = False

        else:
            # Complete miss: add to T1
            self.T1.add(node_id)
            node.arc_list_type = "T1"
            node.in_ghost = False

    def should_evict_from_t1(self) -> bool:
        """
        Determine whether to evict from T1 or T2.

        Returns:
            True if should evict from T1, False if should evict from T2
        """
        t1_size = len(self.T1)
        t2_size = len(self.T2)

        # If T1 exceeds its target size, evict from T1
        if t1_size > 0 and (t1_size > self.p or (t1_size == self.p and t2_size == 0)):
            return True
        return False

    def on_eviction(self, node: "TreeNode", keep_ghost: bool = True):
        """
        Handle eviction: move node to appropriate ghost list.

        Args:
            node: The node being evicted
            keep_ghost: Whether to keep the node as a ghost entry
        """
        node_id = node.id

        if node_id in self.T1:
            self.T1.remove(node_id)
            if keep_ghost:
                self.B1.add(node_id)
                node.arc_list_type = "B1"
                node.in_ghost = True
                # Limit B1 size to cache_size
                while len(self.B1) > self.cache_size:
                    oldest_id = min(self.B1, key=lambda nid: self.node_registry.get(nid).last_access_time if nid in self.node_registry else float('inf'))
                    self.B1.remove(oldest_id)
                    if oldest_id in self.node_registry:
                        self.node_registry[oldest_id].arc_list_type = None
                        self.node_registry[oldest_id].in_ghost = False
            else:
                node.arc_list_type = None
                node.in_ghost = False

        elif node_id in self.T2:
            self.T2.remove(node_id)
            if keep_ghost:
                self.B2.add(node_id)
                node.arc_list_type = "B2"
                node.in_ghost = True
                # Limit B2 size to cache_size
                while len(self.B2) > self.cache_size:
                    oldest_id = min(self.B2, key=lambda nid: self.node_registry.get(nid).last_access_time if nid in self.node_registry else float('inf'))
                    self.B2.remove(oldest_id)
                    if oldest_id in self.node_registry:
                        self.node_registry[oldest_id].arc_list_type = None
                        self.node_registry[oldest_id].in_ghost = False
            else:
                node.arc_list_type = None
                node.in_ghost = False

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics of the ARC cache."""
        return {
            "T1_size": len(self.T1),
            "T2_size": len(self.T2),
            "B1_size": len(self.B1),
            "B2_size": len(self.B2),
            "p": self.p,
            "cache_size": self.cache_size,
        }


class ARCStrategy(EvictionStrategy):
    """
    ARC (Adaptive Replacement Cache) eviction strategy.

    This strategy uses an ARCManager to track cache state and adaptively
    balances between recency (LRU-like) and frequency (LFU-like) based
    on the workload pattern.
    """

    def __init__(self, arc_manager: ARCManager):
        """
        Initialize ARC strategy.

        Args:
            arc_manager: The ARC manager instance to use
        """
        self.arc_manager = arc_manager

    def get_priority(self, node: "TreeNode") -> Tuple[int, float, float]:
        """
        Get eviction priority for a node.

        Priority is a tuple: (list_priority, access_time, hit_count)
        - list_priority: 0 for T1 (evict first), 1 for T2 (evict later)
        - access_time: last_access_time (older = lower priority)
        - hit_count: number of hits (lower = higher eviction priority)

        Args:
            node: The node to get priority for

        Returns:
            Priority tuple for heap-based eviction
        """
        # Determine which list the node is in
        if node.arc_list_type == "T1":
            list_priority = 0  # Evict T1 first if needed
        elif node.arc_list_type == "T2":
            list_priority = 1  # Evict T2 later
        else:
            # Not in ARC lists, use default priority
            list_priority = -1

        # Check if we should prioritize T1 or T2 for eviction
        if self.arc_manager.should_evict_from_t1():
            # Prioritize evicting from T1
            if list_priority == 0:  # T1
                list_priority = 0
            elif list_priority == 1:  # T2
                list_priority = 2  # Make T2 lower priority for eviction
        else:
            # Prioritize evicting from T2
            if list_priority == 1:  # T2
                list_priority = 0
            elif list_priority == 0:  # T1
                list_priority = 2  # Make T1 lower priority for eviction

        return (list_priority, node.last_access_time, -node.hit_count)
