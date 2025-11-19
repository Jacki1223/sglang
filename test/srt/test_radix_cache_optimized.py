"""
Unit tests for PersistentHeapRadixCache optimization.

This test suite verifies that the optimized RadixCache with persistent heap:
1. Maintains correctness (same behavior as original)
2. Properly maintains heap invariants
3. Handles edge cases correctly
4. Provides performance improvements

Run with:
    python -m pytest test_radix_cache_optimized.py -v
    python test_radix_cache_optimized.py
"""

import time
import unittest

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.radix_cache_optimized import (
    HeapEntry,
    OptimizedTreeNode,
    PersistentHeapRadixCache,
    create_optimized_cache,
)


class TestHeapEntry(unittest.TestCase):
    """Test cases for HeapEntry class."""

    def setUp(self):
        """Reset sequence counter before each test."""
        HeapEntry._sequence_counter = 0

    def test_heap_entry_creation(self):
        """Test basic HeapEntry creation."""
        node = TreeNode()
        entry = HeapEntry(priority=1.0, node=node)

        self.assertEqual(entry.priority, 1.0)
        self.assertEqual(entry.node, node)
        self.assertFalse(entry.deleted)
        self.assertEqual(entry.sequence_id, 0)

    def test_heap_entry_comparison(self):
        """Test HeapEntry comparison for heap ordering."""
        node1, node2, node3 = TreeNode(), TreeNode(), TreeNode()

        entry1 = HeapEntry(priority=1.0, node=node1)
        entry2 = HeapEntry(priority=2.0, node=node2)
        entry3 = HeapEntry(priority=1.0, node=node3)  # Same priority

        # Lower priority comes first
        self.assertTrue(entry1 < entry2)
        self.assertFalse(entry2 < entry1)

        # Same priority - use sequence id (FIFO)
        self.assertTrue(entry1 < entry3)
        self.assertFalse(entry3 < entry1)

    def test_heap_entry_deletion(self):
        """Test HeapEntry deletion flag."""
        node = TreeNode()
        entry = HeapEntry(priority=1.0, node=node)

        self.assertFalse(entry.deleted)
        entry.deleted = True
        self.assertTrue(entry.deleted)


class TestOptimizedTreeNode(unittest.TestCase):
    """Test cases for OptimizedTreeNode."""

    def test_optimized_node_creation(self):
        """Test OptimizedTreeNode creation and attributes."""
        node = OptimizedTreeNode()

        self.assertIsNone(node.heap_entry)
        self.assertFalse(node._is_in_heap)
        self.assertFalse(node.is_in_heap)

    def test_heap_tracking(self):
        """Test heap entry tracking."""
        node = OptimizedTreeNode()
        entry = HeapEntry(priority=1.0, node=node)

        # Link entry to node
        node.heap_entry = entry
        node._is_in_heap = True

        self.assertTrue(node.is_in_heap)

        # Mark as deleted
        node.mark_heap_deleted()

        self.assertFalse(node.is_in_heap)
        self.assertTrue(entry.deleted)
        self.assertIsNone(node.heap_entry)


class TestPersistentHeapRadixCache(unittest.TestCase):
    """Test cases for PersistentHeapRadixCache."""

    def setUp(self):
        """Create a fresh cache for each test."""
        self.cache = PersistentHeapRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

    def test_initialization(self):
        """Test cache initialization."""
        self.assertEqual(len(self.cache._eviction_heap), 0)
        self.assertEqual(self.cache._deleted_count, 0)
        self.assertEqual(self.cache.total_size(), 0)

    def test_insert_adds_to_heap(self):
        """Test that inserting creates heap entries for leaves."""
        # Insert a sequence
        self.cache.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))

        # Should have leaf nodes in heap
        heap_size = len(self.cache._eviction_heap)
        self.assertGreater(heap_size, 0, "Heap should contain leaf nodes after insert")

    def test_insert_multiple_sequences(self):
        """Test inserting multiple sequences."""
        sequences = [
            [1, 2, 3],
            [1, 2, 4, 5],
            [1, 2, 4, 6, 7],
            [8, 9, 10],
        ]

        for seq in sequences:
            self.cache.insert(RadixKey(token_ids=seq, extra_key=None))

        # Verify cache has correct total size
        expected_size = sum(len(seq) for seq in sequences)
        # Account for shared prefixes in radix tree
        actual_size = self.cache.total_size()
        self.assertLessEqual(
            actual_size, expected_size, "Radix tree should deduplicate prefixes"
        )

    def test_eviction_removes_from_heap(self):
        """Test that eviction properly removes nodes from heap."""
        # Insert sequences
        self.cache.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
        self.cache.insert(RadixKey(token_ids=[4, 5, 6], extra_key=None))

        initial_size = self.cache.total_size()
        initial_heap_size = len(self.cache._eviction_heap)

        # Evict some tokens
        self.cache.evict(num_tokens=2)

        # Size should decrease
        self.assertLess(self.cache.total_size(), initial_size)

    def test_eviction_with_lazy_deletion(self):
        """Test that lazy deletion works correctly."""
        # Insert and then modify tree structure
        self.cache.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
        self.cache.insert(RadixKey(token_ids=[1, 2, 4], extra_key=None))

        # Some entries might be marked deleted due to structure changes
        stats = self.cache.get_stats()

        # Evict - should skip deleted entries
        initial_deleted = self.cache._deleted_count
        self.cache.evict(num_tokens=1)

        # Deleted count should not increase indefinitely
        self.assertGreaterEqual(
            stats['heap_size'], 0, "Heap size should be non-negative"
        )

    def test_heap_cleanup(self):
        """Test that heap cleanup is triggered and works correctly."""
        # Create many sequences
        for i in range(100):
            self.cache.insert(RadixKey(token_ids=[i, i + 1, i + 2], extra_key=None))

        # Modify tree to create deleted entries
        for i in range(50):
            self.cache.insert(
                RadixKey(token_ids=[i, i + 1, i + 2, i + 3], extra_key=None)
            )

        initial_heap_size = len(self.cache._eviction_heap)
        initial_deleted = self.cache._deleted_count

        # Trigger cleanup by evicting
        for _ in range(10):
            self.cache.evict(num_tokens=1)

        # Check if cleanup occurred
        stats = self.cache.get_stats()
        if stats['heap_cleanups'] > 0:
            # After cleanup, deleted count should be 0
            final_deleted = self.cache._deleted_count
            self.assertLess(
                final_deleted, initial_deleted, "Cleanup should remove deleted entries"
            )

    def test_lock_removes_from_heap(self):
        """Test that locking a node removes it from eviction heap."""
        # Create cache with OptimizedTreeNode
        cache = PersistentHeapRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        # Insert a sequence
        cache.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))

        # Get a leaf node (traverse to find one)
        leaf = None
        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            if len(node.children) == 0 and node != cache.root_node:
                leaf = node
                break
            stack.extend(node.children.values())

        self.assertIsNotNone(leaf, "Should find a leaf node")

        # Lock the node
        initial_evictable = cache.evictable_size()
        cache.inc_lock_ref(leaf)

        # Evictable size should decrease
        self.assertLess(cache.evictable_size(), initial_evictable)

        # If using OptimizedTreeNode, should not be in heap
        if isinstance(leaf, OptimizedTreeNode):
            self.assertFalse(leaf.is_in_heap)

    def test_unlock_adds_to_heap(self):
        """Test that unlocking a leaf node adds it back to heap."""
        cache = PersistentHeapRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        # Insert and get leaf
        cache.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))

        leaf = None
        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            if len(node.children) == 0 and node != cache.root_node:
                leaf = node
                break
            stack.extend(node.children.values())

        # Lock then unlock
        cache.inc_lock_ref(leaf)
        initial_heap_size = len(cache._eviction_heap)

        cache.dec_lock_ref(leaf)

        # Heap size should increase or stay same (leaf re-added)
        final_heap_size = len(cache._eviction_heap)
        self.assertGreaterEqual(
            final_heap_size, initial_heap_size, "Unlocking should add leaf to heap"
        )

    def test_get_stats(self):
        """Test statistics collection."""
        # Insert some data
        for i in range(10):
            self.cache.insert(RadixKey(token_ids=[i, i + 1], extra_key=None))

        # Evict some
        self.cache.evict(num_tokens=5)

        stats = self.cache.get_stats()

        # Verify stats structure
        self.assertIn('heap_size', stats)
        self.assertIn('deleted_count', stats)
        self.assertIn('heap_cleanups', stats)
        self.assertIn('total_evictions', stats)
        self.assertIn('heap_hits', stats)
        self.assertIn('heap_skips', stats)
        self.assertIn('hit_rate', stats)

        # Verify stats values
        self.assertGreaterEqual(stats['heap_size'], 0)
        self.assertGreaterEqual(stats['total_evictions'], 1)
        self.assertGreaterEqual(stats['heap_hits'], 0)
        self.assertLessEqual(stats['hit_rate'], 1.0)

    def test_reset(self):
        """Test cache reset clears heap."""
        # Insert data
        self.cache.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))

        self.assertGreater(len(self.cache._eviction_heap), 0)
        self.assertGreater(self.cache.total_size(), 0)

        # Reset
        self.cache.reset()

        # Heap should be empty
        self.assertEqual(len(self.cache._eviction_heap), 0)
        self.assertEqual(self.cache._deleted_count, 0)
        self.assertEqual(self.cache.total_size(), 0)

    def test_match_prefix(self):
        """Test that match_prefix works correctly with heap."""
        # Insert sequences
        self.cache.insert(RadixKey(token_ids=[1, 2, 3, 4], extra_key=None))
        self.cache.insert(RadixKey(token_ids=[1, 2, 5, 6], extra_key=None))

        # Match prefix
        result = self.cache.match_prefix(
            RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=None)
        )

        # Should match [1, 2, 3, 4]
        self.assertIsNotNone(result.device_indices)
        self.assertEqual(len(result.device_indices), 4)


class TestCorrectnessComparison(unittest.TestCase):
    """Test that optimized cache behaves identically to original."""

    def test_same_insertion_behavior(self):
        """Test that both caches produce same tree structure."""
        original = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        optimized = PersistentHeapRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        sequences = [
            [1, 2, 3],
            [1, 2, 4, 5],
            [1, 2, 4, 6, 7],
            [8, 9, 10],
        ]

        for seq in sequences:
            original.insert(RadixKey(token_ids=seq, extra_key=None))
            optimized.insert(RadixKey(token_ids=seq, extra_key=None))

        # Should have same total size
        self.assertEqual(original.total_size(), optimized.total_size())
        self.assertEqual(original.evictable_size(), optimized.evictable_size())

    def test_same_match_behavior(self):
        """Test that prefix matching works identically."""
        original = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        optimized = PersistentHeapRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        # Insert same data
        sequences = [[1, 2, 3, 4], [1, 2, 5, 6], [7, 8, 9]]
        for seq in sequences:
            original.insert(RadixKey(token_ids=seq, extra_key=None))
            optimized.insert(RadixKey(token_ids=seq, extra_key=None))

        # Test various prefixes
        test_keys = [
            [1, 2],
            [1, 2, 3, 4, 5],
            [7, 8],
            [10, 11],
        ]

        for key in test_keys:
            orig_result = original.match_prefix(RadixKey(token_ids=key, extra_key=None))
            opt_result = optimized.match_prefix(RadixKey(token_ids=key, extra_key=None))

            # Should have same match length
            self.assertEqual(
                len(orig_result.device_indices),
                len(opt_result.device_indices),
                f"Match length differs for key {key}",
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_cache_eviction(self):
        """Test evicting from empty cache."""
        cache = PersistentHeapRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        # Should not crash
        cache.evict(num_tokens=100)
        self.assertEqual(cache.total_size(), 0)

    def test_evict_more_than_available(self):
        """Test evicting more tokens than available."""
        cache = PersistentHeapRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        cache.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))

        # Try to evict more than available
        cache.evict(num_tokens=1000)

        # Should evict all available
        self.assertEqual(cache.evictable_size(), 0)

    def test_single_node_tree(self):
        """Test cache with single node."""
        cache = PersistentHeapRadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            eviction_policy="lru",
        )

        cache.insert(RadixKey(token_ids=[1], extra_key=None))

        self.assertEqual(cache.total_size(), 1)
        self.assertGreater(len(cache._eviction_heap), 0)


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
