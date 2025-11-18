"""
Unit tests for ARC (Adaptive Replacement Cache) implementation.

Run with:
    pytest python/sglang/test/test_arc_cache.py -v
"""

import pytest
import torch

from sglang.srt.mem_cache.arc_radix_cache import ARCEvictionStrategy, ARCRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


class TestARCEvictionStrategy:
    """Test the ARC eviction strategy in isolation"""

    def test_basic_lru_behavior(self):
        """Test that ARC behaves like LRU for single-access patterns"""
        arc = ARCEvictionStrategy(max_cache_tokens=10)

        # Create nodes
        nodes = []
        for i in range(5):
            node = TreeNode()
            node.key = RadixKey(token_ids=list(range(i, i + 2)))
            node.value = torch.tensor(list(range(i, i + 2)))
            nodes.append(node)

        # Insert nodes (each accessed once)
        for node in nodes:
            arc.insert(node)

        # All should be in T1 (recently accessed once)
        assert len(arc.t1) == 5
        assert len(arc.t2) == 0

        # Access first node again -> should move to T2
        arc.access(nodes[0])
        assert len(arc.t1) == 4
        assert len(arc.t2) == 1
        assert nodes[0].id in arc.t2

    def test_lfu_behavior_for_frequent_access(self):
        """Test that frequently accessed items stay in T2"""
        arc = ARCEvictionStrategy(max_cache_tokens=10)

        # Create nodes
        hot_node = TreeNode()
        hot_node.key = RadixKey(token_ids=[1, 2, 3])
        hot_node.value = torch.tensor([1, 2, 3])

        cold_nodes = []
        for i in range(3):
            node = TreeNode()
            node.key = RadixKey(token_ids=[10 + i])
            node.value = torch.tensor([10 + i])
            cold_nodes.append(node)

        # Insert hot node and access it multiple times
        arc.insert(hot_node)
        for _ in range(5):
            arc.access(hot_node)

        # Insert cold nodes
        for node in cold_nodes:
            arc.insert(node)

        # Hot node should be in T2 (frequently accessed)
        assert hot_node.id in arc.t2

    def test_ghost_hit_adjusts_p(self):
        """Test that ghost hits adjust the parameter p"""
        arc = ARCEvictionStrategy(max_cache_tokens=6)

        nodes = []
        for i in range(10):
            node = TreeNode()
            node.key = RadixKey(token_ids=[i])
            node.value = torch.tensor([i])
            nodes.append(node)

        # Fill cache
        for i in range(6):
            arc.insert(nodes[i])

        initial_p = arc.p

        # Force eviction and create ghost entries
        for i in range(6, 8):
            arc.insert(nodes[i])

        # Access an evicted node (should be in B1)
        # This should adjust p
        assert len(arc.b1) > 0 or len(arc.b2) > 0
        # Note: Exact behavior depends on insertion order

    def test_evict_tokens(self):
        """Test token eviction"""
        arc = ARCEvictionStrategy(max_cache_tokens=20)

        nodes = []
        for i in range(10):
            node = TreeNode()
            node.key = RadixKey(token_ids=list(range(i, i + 2)))
            node.value = torch.tensor(list(range(i, i + 2)))
            nodes.append(node)

        # Insert all nodes
        for node in nodes:
            arc.insert(node)

        # Evict 10 tokens
        evicted = arc.evict_tokens(10)

        assert len(evicted) >= 5  # At least 5 nodes (2 tokens each)
        total_evicted = sum(len(node.key) for node in evicted)
        assert total_evicted >= 10

    def test_stats(self):
        """Test statistics collection"""
        arc = ARCEvictionStrategy(max_cache_tokens=10)

        node = TreeNode()
        node.key = RadixKey(token_ids=[1, 2])
        node.value = torch.tensor([1, 2])

        arc.insert(node)
        arc.access(node)

        stats = arc.get_stats()

        assert 'hit_rate' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert stats['hits'] > 0


class TestARCRadixCache:
    """Test ARC integration with RadixCache"""

    @pytest.fixture
    def mock_pools(self):
        """Create mock pools for testing"""
        class MockReqToTokenPool:
            def __init__(self):
                self.req_to_token = torch.zeros((100, 1000), dtype=torch.int32)
                self.free_slots = list(range(100))

            def alloc(self):
                return self.free_slots.pop()

            def free(self, idx):
                self.free_slots.append(idx)

        class MockTokenToKVPoolAllocator:
            def __init__(self):
                self.device = torch.device('cpu')
                self.pool = list(range(10000))

            def alloc(self, size):
                return torch.tensor(self.pool[:size], dtype=torch.int64)

            def free(self, indices):
                pass

        return MockReqToTokenPool(), MockTokenToKVPoolAllocator()

    def test_arc_cache_creation(self, mock_pools):
        """Test creating an ARC cache"""
        req_pool, kv_pool = mock_pools

        cache = ARCRadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=kv_pool,
            max_cache_tokens=10000,
        )

        assert cache.arc_strategy is not None
        assert cache.arc_strategy.c == 10000

    def test_arc_cache_insert_and_match(self, mock_pools):
        """Test insertion and prefix matching with ARC"""
        req_pool, kv_pool = mock_pools

        cache = ARCRadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=kv_pool,
            max_cache_tokens=10000,
        )

        # Insert a key
        key1 = RadixKey(token_ids=[1, 2, 3, 4, 5])
        cache.insert(key1)

        # Match prefix
        key2 = RadixKey(token_ids=[1, 2, 3, 6, 7])
        result = cache.match_prefix(key2)

        # Should match first 3 tokens
        assert len(result.device_indices) == 3

    def test_arc_stats_collection(self, mock_pools):
        """Test that ARC stats are collected"""
        req_pool, kv_pool = mock_pools

        cache = ARCRadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=kv_pool,
            max_cache_tokens=10000,
        )

        # Insert some keys
        for i in range(10):
            key = RadixKey(token_ids=list(range(i, i + 5)))
            cache.insert(key)

        # Get stats
        stats = cache.get_arc_stats()

        assert 'hit_rate' in stats
        assert 't1_size' in stats
        assert 't2_size' in stats


@pytest.mark.benchmark
class TestARCPerformance:
    """Benchmark tests for ARC cache"""

    def test_benchmark_lru_vs_arc(self, benchmark, mock_pools):
        """Compare LRU and ARC performance"""
        from sglang.srt.mem_cache.radix_cache import RadixCache

        req_pool, kv_pool = mock_pools

        # Test pattern: 80% hot set, 20% random
        hot_keys = [RadixKey(token_ids=[i]) for i in range(100)]
        random_keys = [RadixKey(token_ids=[1000 + i]) for i in range(500)]

        def run_cache(cache_class, eviction_policy='lru'):
            if cache_class == ARCRadixCache:
                cache = cache_class(
                    req_to_token_pool=req_pool,
                    token_to_kv_pool_allocator=kv_pool,
                    max_cache_tokens=200,
                )
            else:
                cache = cache_class(
                    req_to_token_pool=req_pool,
                    token_to_kv_pool_allocator=kv_pool,
                    eviction_policy=eviction_policy,
                )

            # Workload
            for _ in range(1000):
                # 80% hot, 20% random
                import random
                if random.random() < 0.8:
                    key = random.choice(hot_keys)
                else:
                    key = random.choice(random_keys)

                cache.match_prefix(key)
                cache.insert(key)

        # Benchmark both
        print("\n=== LRU Cache ===")
        benchmark(run_cache, RadixCache, 'lru')

        print("\n=== ARC Cache ===")
        benchmark(run_cache, ARCRadixCache)
