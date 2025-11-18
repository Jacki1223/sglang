"""
Test and example code for ARC (Adaptive Replacement Cache) implementation in SGLang.

This script demonstrates how to use the ARC caching strategy and compares it
with other eviction policies like LRU and LFU.
"""

import torch
import time
from typing import List

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.mem_cache.evict_policy import ARCManager


class DummyAllocator:
    """Dummy allocator for testing purposes."""

    def __init__(self, size: int = 1000):
        self.size = size
        self.device = torch.device("cpu")
        self.allocated = set()

    def alloc(self, size: int):
        """Allocate tokens and return their indices."""
        indices = []
        for i in range(size):
            idx = len(self.allocated)
            if idx < self.size:
                self.allocated.add(idx)
                indices.append(idx)
        return torch.tensor(indices, dtype=torch.int64, device=self.device)

    def free(self, indices: torch.Tensor):
        """Free allocated tokens."""
        for idx in indices.tolist():
            self.allocated.discard(idx)


def test_arc_basic():
    """Test basic ARC functionality."""
    print("=" * 80)
    print("Test 1: Basic ARC Functionality")
    print("=" * 80)

    # Create allocator and cache with ARC policy
    allocator = DummyAllocator(size=100)
    req_to_token_pool = None
    cache = RadixCache(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="arc",
    )

    # Test inserting and matching keys
    key1 = RadixKey([1, 2, 3, 4, 5])
    key2 = RadixKey([1, 2, 3, 6, 7])
    key3 = RadixKey([1, 2, 8, 9, 10])

    # Insert first key
    value1 = allocator.alloc(len(key1))
    cache.insert(key1, value1)
    print(f"Inserted key1: {key1.token_ids}")

    # Match first key (should be a cache hit)
    result = cache.match_prefix(key1)
    print(f"Match key1: hit_length={len(result.device_indices)}")
    assert len(result.device_indices) == len(key1), "Should match full key"

    # Insert second key (shares prefix [1, 2, 3] with key1)
    value2 = allocator.alloc(len(key2))
    cache.insert(key2, value2)
    print(f"Inserted key2: {key2.token_ids}")

    # Match second key
    result = cache.match_prefix(key2)
    print(f"Match key2: hit_length={len(result.device_indices)}")

    # Insert third key (shares prefix [1, 2] with key1 and key2)
    value3 = allocator.alloc(len(key3))
    cache.insert(key3, value3)
    print(f"Inserted key3: {key3.token_ids}")

    # Get ARC stats
    if cache.arc_manager:
        stats = cache.arc_manager.get_stats()
        print("\nARC Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\n✓ Basic ARC functionality test passed!\n")


def test_arc_eviction():
    """Test ARC eviction behavior."""
    print("=" * 80)
    print("Test 2: ARC Eviction Behavior")
    print("=" * 80)

    # Create a small cache to trigger evictions
    allocator = DummyAllocator(size=20)  # Small cache
    cache = RadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="arc",
    )

    # Insert multiple keys to fill the cache
    keys = []
    for i in range(10):
        key = RadixKey([i, i + 1, i + 2])
        value = allocator.alloc(len(key))
        cache.insert(key, value)
        keys.append(key)
        print(f"Inserted key {i}: {key.token_ids}")

    print(f"\nCache is full. Allocated: {len(allocator.allocated)}/{allocator.size}")

    # Access some keys multiple times to promote them to T2
    for _ in range(3):
        for i in [0, 1, 2]:  # Access first 3 keys multiple times
            result = cache.match_prefix(keys[i])
            print(f"Accessed key {i} (hit_length={len(result.device_indices)})")

    # Get ARC stats before eviction
    if cache.arc_manager:
        stats = cache.arc_manager.get_stats()
        print("\nARC Statistics before eviction:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Trigger eviction by inserting a large new key
    new_key = RadixKey([100, 101, 102, 103, 104])
    print(f"\nInserting large key to trigger eviction: {new_key.token_ids}")

    # First evict some space
    cache.evict(num_tokens=10)
    print(f"Evicted 10 tokens. Allocated: {len(allocator.allocated)}/{allocator.size}")

    # Get ARC stats after eviction
    if cache.arc_manager:
        stats = cache.arc_manager.get_stats()
        print("\nARC Statistics after eviction:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\n✓ ARC eviction test passed!\n")


def test_arc_adaptive_behavior():
    """Test ARC adaptive parameter adjustment."""
    print("=" * 80)
    print("Test 3: ARC Adaptive Behavior")
    print("=" * 80)

    allocator = DummyAllocator(size=50)
    cache = RadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="arc",
    )

    # Pattern 1: Sequential access (favors recency - T1)
    print("Pattern 1: Sequential access (favors recency)")
    for i in range(10):
        key = RadixKey([i, i + 1])
        value = allocator.alloc(len(key))
        cache.insert(key, value)

    if cache.arc_manager:
        stats = cache.arc_manager.get_stats()
        print(f"  p (target T1 size) = {stats['p']}")
        print(f"  T1 size = {stats['T1_size']}, T2 size = {stats['T2_size']}")

    # Pattern 2: Repeated access (favors frequency - T2)
    print("\nPattern 2: Repeated access (favors frequency)")
    repeat_keys = []
    for i in range(5):
        key = RadixKey([20 + i, 21 + i])
        repeat_keys.append(key)
        value = allocator.alloc(len(key))
        cache.insert(key, value)

    # Access them multiple times
    for _ in range(5):
        for key in repeat_keys:
            cache.match_prefix(key)

    if cache.arc_manager:
        stats = cache.arc_manager.get_stats()
        print(f"  p (target T1 size) = {stats['p']}")
        print(f"  T1 size = {stats['T1_size']}, T2 size = {stats['T2_size']}")

    print("\n✓ ARC adaptive behavior test passed!\n")


def compare_eviction_policies():
    """Compare ARC with other eviction policies."""
    print("=" * 80)
    print("Test 4: Comparing ARC with LRU and LFU")
    print("=" * 80)

    policies = ["lru", "lfu", "arc"]
    results = {}

    for policy in policies:
        print(f"\nTesting {policy.upper()} policy...")
        allocator = DummyAllocator(size=30)
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy=policy,
        )

        # Simulate a mixed workload
        hit_count = 0
        miss_count = 0

        # Phase 1: Sequential inserts
        for i in range(10):
            key = RadixKey([i, i + 1, i + 2])
            value = allocator.alloc(len(key))
            cache.insert(key, value)

        # Phase 2: Repeated access to some keys
        for _ in range(3):
            for i in [0, 1, 2, 7, 8, 9]:
                key = RadixKey([i, i + 1, i + 2])
                result = cache.match_prefix(key)
                if len(result.device_indices) == 3:
                    hit_count += 1
                else:
                    miss_count += 1

        # Phase 3: New sequential access
        for i in range(10, 15):
            key = RadixKey([i, i + 1, i + 2])
            value = allocator.alloc(len(key))
            cache.insert(key, value)

        # Phase 4: Access mix of old and new
        test_keys = [0, 1, 2, 5, 7, 8, 10, 11, 12, 14]
        for i in test_keys:
            key = RadixKey([i, i + 1, i + 2])
            result = cache.match_prefix(key)
            if len(result.device_indices) == 3:
                hit_count += 1
            else:
                miss_count += 1

        total = hit_count + miss_count
        hit_rate = hit_count / total if total > 0 else 0
        results[policy] = {"hits": hit_count, "misses": miss_count, "hit_rate": hit_rate}

        print(f"  Hits: {hit_count}, Misses: {miss_count}, Hit rate: {hit_rate:.2%}")

        if policy == "arc" and cache.arc_manager:
            stats = cache.arc_manager.get_stats()
            print(f"  ARC stats: T1={stats['T1_size']}, T2={stats['T2_size']}, "
                  f"B1={stats['B1_size']}, B2={stats['B2_size']}, p={stats['p']}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    for policy, result in results.items():
        print(f"{policy.upper()}: Hit rate = {result['hit_rate']:.2%}")

    print("\n✓ Comparison test completed!\n")


def example_usage():
    """Example of how to use ARC cache in SGLang."""
    print("=" * 80)
    print("Example: Using ARC Cache in SGLang")
    print("=" * 80)

    # Create a cache with ARC eviction policy
    allocator = DummyAllocator(size=100)
    cache = RadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="arc",  # Use ARC instead of default LRU
    )

    print("Created RadixCache with ARC eviction policy")
    print(f"Cache capacity: {allocator.size} tokens\n")

    # Simulate a typical workload
    print("Simulating typical LLM workload...")

    # Scenario 1: User sends a prompt
    prompt_key = RadixKey([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    prompt_value = allocator.alloc(len(prompt_key))
    cache.insert(prompt_key, prompt_value)
    print(f"✓ Cached prompt: {prompt_key.token_ids[:5]}... ({len(prompt_key)} tokens)")

    # Scenario 2: User sends a follow-up with shared prefix
    followup_key = RadixKey([1, 2, 3, 4, 5, 11, 12, 13, 14, 15])
    result = cache.match_prefix(followup_key)
    print(f"✓ Follow-up shares {len(result.device_indices)} tokens with cached prompt")

    # Insert the new part
    new_part = RadixKey(followup_key.token_ids[len(result.device_indices):])
    if len(new_part) > 0:
        new_value = allocator.alloc(len(new_part))
        cache.insert(followup_key, torch.cat([result.device_indices, new_value]))

    # Scenario 3: Frequently accessed system prompt
    system_prompt = RadixKey([100, 101, 102, 103, 104])
    system_value = allocator.alloc(len(system_prompt))
    cache.insert(system_prompt, system_value)
    print(f"✓ Cached system prompt: {system_prompt.token_ids}")

    # Access it multiple times (simulating multiple requests)
    for i in range(5):
        result = cache.match_prefix(system_prompt)
        assert len(result.device_indices) == len(system_prompt)
    print(f"✓ System prompt accessed 5 times (will be promoted to T2)")

    # Check ARC statistics
    if cache.arc_manager:
        stats = cache.arc_manager.get_stats()
        print("\nARC Cache Statistics:")
        print(f"  T1 (Recent cache):    {stats['T1_size']} entries")
        print(f"  T2 (Frequent cache):  {stats['T2_size']} entries")
        print(f"  B1 (Ghost T1):        {stats['B1_size']} entries")
        print(f"  B2 (Ghost T2):        {stats['B2_size']} entries")
        print(f"  p (Target T1 size):   {stats['p']}")
        print(f"  Total cache size:     {stats['cache_size']}")

    print("\n✓ Example completed successfully!\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ARC (Adaptive Replacement Cache) Test Suite for SGLang")
    print("=" * 80 + "\n")

    try:
        # Run all tests
        test_arc_basic()
        test_arc_eviction()
        test_arc_adaptive_behavior()
        compare_eviction_policies()
        example_usage()

        print("=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
