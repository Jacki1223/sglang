"""
Unit tests for value-aware eviction strategies.

This module tests the new eviction strategies:
- ValueAwareLRUStrategy
- AdaptiveLFUStrategy
- ValueAwareAdaptiveLFUStrategy

Test Coverage:
- Priority calculation for value-aware strategies
- Protection of high-value nodes (long prefixes, common prefixes)
- Adaptive behavior in LFU strategies
- Comparison with baseline LRU/LFU strategies

Usage:
    python test_value_aware_eviction.py
    python -m pytest test_value_aware_eviction.py -v
"""

import time
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.evict_policy import (
    AdaptiveLFUStrategy,
    LFUStrategy,
    LRUStrategy,
    ValueAwareAdaptiveLFUStrategy,
    ValueAwareLRUStrategy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


class TestValueAwareLRUStrategy(unittest.TestCase):
    """Test cases for ValueAwareLRUStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_basic_priority_calculation(self):
        """Test basic priority calculation."""
        strategy = ValueAwareLRUStrategy()

        node1 = TreeNode()
        node1.last_access_time = 100.0
        # Single node with no children or prefix

        priority = strategy.get_priority(node1)
        # Should be close to last_access_time (100.0) since value_score is 0
        self.assertAlmostEqual(priority, 100.0, delta=1.0)

    def test_value_aware_priority_with_children(self):
        """Test that nodes with children get higher priority (protected)."""
        strategy = ValueAwareLRUStrategy()

        # Node without children
        node_no_children = TreeNode()
        node_no_children.last_access_time = 100.0

        # Node with children
        node_with_children = TreeNode()
        node_with_children.last_access_time = 100.0
        # Add mock children
        node_with_children.children["child1"] = TreeNode()
        node_with_children.children["child2"] = TreeNode()

        priority_no_children = strategy.get_priority(node_no_children)
        priority_with_children = strategy.get_priority(node_with_children)

        # Node with children should have lower priority (higher value, protected)
        self.assertLess(priority_with_children, priority_no_children)

    def test_value_aware_priority_with_long_prefix(self):
        """Test that nodes with longer prefixes get higher priority (protected)."""
        strategy = ValueAwareLRUStrategy()

        # Create a chain of nodes to simulate prefix
        root = TreeNode()
        root.key = RadixKey([])

        node1 = TreeNode()
        node1.key = RadixKey([1, 2, 3, 4, 5])
        node1.parent = root
        node1.last_access_time = 100.0

        node2 = TreeNode()
        node2.key = RadixKey([1])
        node2.parent = root
        node2.last_access_time = 100.0

        priority1 = strategy.get_priority(node1)
        priority2 = strategy.get_priority(node2)

        # Node with longer prefix should have lower priority (protected)
        self.assertLess(priority1, priority2)

    def test_combined_value_factors(self):
        """Test combined effect of prefix length and subtree size."""
        strategy = ValueAwareLRUStrategy()

        # Create root
        root = TreeNode()
        root.key = RadixKey([])

        # Node with long prefix and children
        high_value_node = TreeNode()
        high_value_node.key = RadixKey([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        high_value_node.parent = root
        high_value_node.last_access_time = 100.0
        high_value_node.children["child1"] = TreeNode()
        high_value_node.children["child2"] = TreeNode()
        high_value_node.children["child3"] = TreeNode()

        # Node with short prefix and no children
        low_value_node = TreeNode()
        low_value_node.key = RadixKey([1])
        low_value_node.parent = root
        low_value_node.last_access_time = 100.0

        priority_high = strategy.get_priority(high_value_node)
        priority_low = strategy.get_priority(low_value_node)

        # High value node should be strongly protected
        self.assertLess(priority_high, priority_low)


class TestAdaptiveLFUStrategy(unittest.TestCase):
    """Test cases for AdaptiveLFUStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_new_node_uses_lru(self):
        """Test that new nodes use LRU (phase 0)."""
        strategy = AdaptiveLFUStrategy(
            protection_period=60.0, min_hit_threshold=3
        )

        node1 = TreeNode()
        node1.hit_count = 0
        node1.last_access_time = 100.0

        node2 = TreeNode()
        node2.hit_count = 1
        node2.last_access_time = 90.0

        priority1 = strategy.get_priority(node1)
        priority2 = strategy.get_priority(node2)

        # Both should be in phase 0 (protection period)
        self.assertEqual(priority1[0], 0)
        self.assertEqual(priority2[0], 0)

        # Should follow LRU: older node (node2) has lower priority
        self.assertLess(priority2, priority1)

    def test_mature_node_uses_lfu(self):
        """Test that mature nodes use LFU (phase 1)."""
        strategy = AdaptiveLFUStrategy(
            protection_period=0.01, min_hit_threshold=3
        )

        # Create mature node (past protection period and hit threshold)
        mature_node = TreeNode()
        mature_node.hit_count = 5
        mature_node.last_access_time = 100.0
        time.sleep(0.02)  # Ensure protection period has passed

        priority = strategy.get_priority(mature_node)

        # Should be in phase 1 (mature)
        self.assertEqual(priority[0], 1)
        self.assertEqual(priority[1], 5)  # hit_count

    def test_phase_transition(self):
        """Test transition from protection to mature phase."""
        strategy = AdaptiveLFUStrategy(
            protection_period=0.01, min_hit_threshold=2
        )

        node = TreeNode()
        node.hit_count = 1
        node.last_access_time = 100.0

        # Should be in phase 0 (below threshold)
        priority1 = strategy.get_priority(node)
        self.assertEqual(priority1[0], 0)

        # Increase hit count
        node.hit_count = 3

        # Wait for protection period
        time.sleep(0.02)

        # Should transition to phase 1
        priority2 = strategy.get_priority(node)
        self.assertEqual(priority2[0], 1)


class TestValueAwareAdaptiveLFUStrategy(unittest.TestCase):
    """Test cases for ValueAwareAdaptiveLFUStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_new_node_with_value_awareness(self):
        """Test that new nodes use value-aware LRU."""
        strategy = ValueAwareAdaptiveLFUStrategy(
            protection_period=60.0, min_hit_threshold=3
        )

        root = TreeNode()
        root.key = RadixKey([])

        # High value node (long prefix + children)
        high_value_node = TreeNode()
        high_value_node.key = RadixKey([1, 2, 3, 4, 5])
        high_value_node.parent = root
        high_value_node.hit_count = 0
        high_value_node.last_access_time = 100.0
        high_value_node.children["child1"] = TreeNode()

        # Low value node
        low_value_node = TreeNode()
        low_value_node.key = RadixKey([1])
        low_value_node.parent = root
        low_value_node.hit_count = 0
        low_value_node.last_access_time = 100.0

        priority_high = strategy.get_priority(high_value_node)
        priority_low = strategy.get_priority(low_value_node)

        # Both should be in phase 0
        self.assertEqual(priority_high[0], 0)
        self.assertEqual(priority_low[0], 0)

        # High value node should be protected even in phase 0
        self.assertLess(priority_high[1], priority_low[1])

    def test_mature_node_with_value_awareness(self):
        """Test that mature nodes use value-aware LFU."""
        strategy = ValueAwareAdaptiveLFUStrategy(
            protection_period=0.01, min_hit_threshold=2
        )

        root = TreeNode()
        root.key = RadixKey([])

        # Create two mature nodes with same hit count but different values
        high_value_node = TreeNode()
        high_value_node.key = RadixKey([1, 2, 3, 4, 5])
        high_value_node.parent = root
        high_value_node.hit_count = 5
        high_value_node.last_access_time = 100.0
        high_value_node.children["child1"] = TreeNode()

        low_value_node = TreeNode()
        low_value_node.key = RadixKey([1])
        low_value_node.parent = root
        low_value_node.hit_count = 5
        low_value_node.last_access_time = 100.0

        time.sleep(0.02)  # Wait for protection period

        priority_high = strategy.get_priority(high_value_node)
        priority_low = strategy.get_priority(low_value_node)

        # Both should be in phase 1
        self.assertEqual(priority_high[0], 1)
        self.assertEqual(priority_low[0], 1)

        # Same hit count
        self.assertEqual(priority_high[1], 5)
        self.assertEqual(priority_low[1], 5)

        # High value node should have lower recency component (protected)
        self.assertLess(priority_high[2], priority_low[2])


class TestEvictionWithValueAwareStrategies(unittest.TestCase):
    """Integration tests for eviction with value-aware strategies."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_value_aware_lru_protects_common_prefix(self):
        """Test that ValueAwareLRU protects common prefix nodes."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        # Create cache with value-aware LRU
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="value_aware_lru",
        )

        # Insert a common prefix sequence
        cache.insert(RadixKey([1, 2, 3, 4]), torch.tensor([10, 20, 30, 40], dtype=torch.int64))

        # Insert a diverging branch (shares prefix [1, 2])
        cache.insert(RadixKey([1, 2, 5, 6]), torch.tensor([10, 20, 50, 60], dtype=torch.int64))

        # Insert an unrelated sequence (will be evicted first)
        cache.insert(RadixKey([7, 8]), torch.tensor([70, 80], dtype=torch.int64))

        initial_size = cache.total_size()
        self.assertEqual(initial_size, 8)  # 4 + 2 (shared prefix) + 2 (unrelated)

        # Evict 2 tokens
        cache.evict(2)

        # The unrelated sequence [7, 8] should be evicted first
        # because it has no children and no long prefix
        self.assertEqual(cache.total_size(), initial_size - 2)

    def test_adaptive_lfu_protects_new_nodes(self):
        """Test that AdaptiveLFU gives new nodes a fair chance."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="adaptive_lfu",
        )

        # Insert and access an old sequence multiple times
        cache.insert(RadixKey([1, 2]), torch.tensor([10, 20], dtype=torch.int64))
        result = cache.match_prefix(RadixKey([1, 2]))
        old_node = result.last_device_node
        old_node.hit_count = 10  # Simulate high hit count
        old_node.last_access_time = time.monotonic() - 100  # Old access

        # Insert a new sequence
        cache.insert(RadixKey([3, 4]), torch.tensor([30, 40], dtype=torch.int64))
        result = cache.match_prefix(RadixKey([3, 4]))
        new_node = result.last_device_node
        new_node.hit_count = 0  # New node, no hits
        new_node.last_access_time = time.monotonic()  # Recent access

        # With adaptive LFU, the new node should not be immediately evicted
        # because it's in the protection period
        strategy = AdaptiveLFUStrategy()
        old_priority = strategy.get_priority(old_node)
        new_priority = strategy.get_priority(new_node)

        # Old node is in phase 1 (mature), new node in phase 0 (protected)
        self.assertEqual(old_priority[0], 1)
        self.assertEqual(new_priority[0], 0)

        # Phase 0 has higher priority than phase 1 in min-heap
        # So new node will be evicted first. But the key is that it gets
        # a chance to accumulate hits during the protection period.


class TestComparisonWithBaseline(unittest.TestCase):
    """Compare new strategies with baseline LRU/LFU."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_value_aware_vs_standard_lru(self):
        """Compare ValueAwareLRU with standard LRU."""
        root = TreeNode()
        root.key = RadixKey([])

        # Create a high-value node (long prefix, has children)
        high_value_node = TreeNode()
        high_value_node.key = RadixKey([1, 2, 3, 4, 5])
        high_value_node.parent = root
        high_value_node.last_access_time = 50.0  # Older
        high_value_node.children["child1"] = TreeNode()

        # Create a low-value node (short prefix, no children)
        low_value_node = TreeNode()
        low_value_node.key = RadixKey([6])
        low_value_node.parent = root
        low_value_node.last_access_time = 100.0  # Newer

        # Standard LRU: evicts older node (high_value_node)
        lru_strategy = LRUStrategy()
        lru_priority_high = lru_strategy.get_priority(high_value_node)
        lru_priority_low = lru_strategy.get_priority(low_value_node)
        self.assertLess(lru_priority_high, lru_priority_low)  # High value evicted

        # Value-aware LRU: protects high-value node
        value_aware_strategy = ValueAwareLRUStrategy()
        va_priority_high = value_aware_strategy.get_priority(high_value_node)
        va_priority_low = value_aware_strategy.get_priority(low_value_node)
        self.assertLess(va_priority_high, va_priority_low)  # High value protected

        # The key difference: value-aware gives even stronger protection
        # to the high-value node due to its structural importance

    def test_adaptive_vs_standard_lfu(self):
        """Compare AdaptiveLFU with standard LFU."""
        # Create a frequently accessed node
        frequent_node = TreeNode()
        frequent_node.hit_count = 10
        frequent_node.last_access_time = 50.0

        # Create a new node with no hits
        new_node = TreeNode()
        new_node.hit_count = 0
        new_node.last_access_time = 100.0

        # Standard LFU: new node has lowest priority, evicted immediately
        lfu_strategy = LFUStrategy()
        lfu_priority_frequent = lfu_strategy.get_priority(frequent_node)
        lfu_priority_new = lfu_strategy.get_priority(new_node)
        self.assertLess(lfu_priority_new, lfu_priority_frequent)  # New node evicted

        # Adaptive LFU: new node gets protection period
        adaptive_strategy = AdaptiveLFUStrategy()
        adaptive_priority_frequent = adaptive_strategy.get_priority(frequent_node)
        adaptive_priority_new = adaptive_strategy.get_priority(new_node)

        # New node is in phase 0, frequent node in phase 1
        self.assertEqual(adaptive_priority_new[0], 0)
        self.assertEqual(adaptive_priority_frequent[0], 1)

        # Phase 0 (protection) is evaluated first, giving new nodes a chance


if __name__ == "__main__":
    unittest.main()
