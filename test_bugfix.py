#!/usr/bin/env python3
"""
Quick test to verify the bug fixes work.
"""
import sys
import os

# Add python directory to path
python_dir = os.path.join(os.path.dirname(__file__), 'python')
sys.path.insert(0, python_dir)

# Import directly without going through sglang package
import importlib.util

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules directly
radix_cache_path = os.path.join(python_dir, 'sglang/srt/mem_cache/radix_cache.py')
radix_cache_optimized_path = os.path.join(python_dir, 'sglang/srt/mem_cache/radix_cache_optimized.py')

radix_cache_module = import_from_path('radix_cache', radix_cache_path)
radix_cache_optimized_module = import_from_path('radix_cache_optimized', radix_cache_optimized_path)

RadixCache = radix_cache_module.RadixCache
RadixKey = radix_cache_module.RadixKey
PersistentHeapRadixCache = radix_cache_optimized_module.PersistentHeapRadixCache

def test_initialization():
    """Test that PersistentHeapRadixCache initializes correctly."""
    print("Testing PersistentHeapRadixCache initialization...")

    cache = PersistentHeapRadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        disable=False,
        eviction_policy='lru'
    )

    assert hasattr(cache, '_eviction_heap'), "Missing _eviction_heap attribute"
    assert len(cache._eviction_heap) == 0, "Heap should be empty initially"
    print("  ✓ Initialization successful")

def test_insert_and_evict():
    """Test insert and evict without allocator."""
    print("\nTesting insert and evict operations...")

    # Test original RadixCache
    print("  Testing Original RadixCache...")
    cache1 = RadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        disable=False,
        eviction_policy='lru'
    )

    cache1.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    cache1.insert(RadixKey(token_ids=[4, 5, 6], extra_key=None))

    # Should not crash even with None allocator
    cache1.evict(num_tokens=2)
    print("    ✓ Original RadixCache works with None allocator")

    # Test PersistentHeapRadixCache
    print("  Testing PersistentHeapRadixCache...")
    cache2 = PersistentHeapRadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        disable=False,
        eviction_policy='lru'
    )

    cache2.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    cache2.insert(RadixKey(token_ids=[4, 5, 6], extra_key=None))

    initial_heap_size = len(cache2._eviction_heap)
    print(f"    Heap size after insert: {initial_heap_size}")

    # Should not crash even with None allocator
    cache2.evict(num_tokens=2)
    print("    ✓ PersistentHeapRadixCache works with None allocator")

def test_reset():
    """Test reset operation."""
    print("\nTesting reset operation...")

    cache = PersistentHeapRadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        disable=False,
        eviction_policy='lru'
    )

    cache.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    assert len(cache._eviction_heap) > 0, "Heap should not be empty after insert"

    cache.reset()
    assert len(cache._eviction_heap) == 0, "Heap should be empty after reset"
    print("  ✓ Reset works correctly")

def main():
    print("="*60)
    print("Bug Fix Verification Test")
    print("="*60)

    try:
        test_initialization()
        test_insert_and_evict()
        test_reset()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
