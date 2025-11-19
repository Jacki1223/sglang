"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Tests for PreallocatedKVBlockPool and PreallocatedPagedTokenToKVPoolAllocator.
"""

import torch

from sglang.srt.mem_cache.preallocated_pool import PreallocatedKVBlockPool


def test_preallocated_pool_basic():
    """Test basic allocation and deallocation."""
    print("Test 1: Basic allocation and deallocation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = PreallocatedKVBlockPool(
        total_pages=1000,
        page_size=16,
        device=device,
        bucket_sizes=[1, 2, 4, 8, 16, 32],
        debug_mode=True,
    )

    print(f"Initial pool: {pool}")
    print(f"Available pages: {pool.available_pages()}")

    # Test allocation
    pages_1 = pool.allocate(4)
    assert pages_1 is not None, "Failed to allocate 4 pages"
    assert len(pages_1) == 4, f"Expected 4 pages, got {len(pages_1)}"
    print(f"Allocated 4 pages: {pages_1}")

    pages_2 = pool.allocate(8)
    assert pages_2 is not None, "Failed to allocate 8 pages"
    assert len(pages_2) == 8, f"Expected 8 pages, got {len(pages_2)}"
    print(f"Allocated 8 pages: {pages_2}")

    # Test deallocation
    pool.free(pages_1)
    print(f"Freed 4 pages")

    pool.free(pages_2)
    print(f"Freed 8 pages")

    stats = pool.get_statistics()
    print(f"\nStatistics after test:")
    print(f"  Total allocations: {stats['total_allocations']}")
    print(f"  Total frees: {stats['total_frees']}")
    print(f"  Available pages: {stats['available_pages']}")
    print(f"  Utilization: {stats['utilization']:.2%}")

    print("✓ Test 1 passed\n")


def test_preallocated_pool_splitting():
    """Test block splitting when exact size unavailable."""
    print("Test 2: Block splitting")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = PreallocatedKVBlockPool(
        total_pages=500,
        page_size=16,
        device=device,
        bucket_sizes=[1, 4, 16, 32],
        enable_splitting=True,
        debug_mode=True,
    )

    print(f"Initial pool: {pool}")

    # Allocate 5 pages (should split from larger bucket)
    pages = pool.allocate(5)
    assert pages is not None, "Failed to allocate 5 pages"
    assert len(pages) == 5, f"Expected 5 pages, got {len(pages)}"
    print(f"Allocated 5 pages (with splitting): {pages}")

    stats = pool.get_statistics()
    print(f"\nStatistics after splitting:")
    print(f"  Split operations: {stats['split_operations']}")
    print(f"  Fallback allocations: {stats['fallback_allocations']}")

    print("✓ Test 2 passed\n")


def test_preallocated_pool_stress():
    """Stress test with many allocations and deallocations."""
    print("Test 3: Stress test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = PreallocatedKVBlockPool(
        total_pages=2000,
        page_size=16,
        device=device,
        bucket_sizes=[1, 2, 4, 8, 16, 32, 64],
        debug_mode=False,  # Disable debug for performance
    )

    allocated_blocks = []

    # Allocate many blocks
    for i in range(50):
        size = (i % 8) + 1  # Sizes from 1 to 8
        pages = pool.allocate(size)
        if pages is not None:
            allocated_blocks.append(pages)

    print(f"Allocated {len(allocated_blocks)} blocks")

    # Free half of them
    for block in allocated_blocks[:25]:
        pool.free(block)

    print(f"Freed 25 blocks")

    # Allocate more
    for i in range(25):
        size = ((i * 2) % 16) + 1
        pages = pool.allocate(size)
        if pages is not None:
            allocated_blocks.append(pages)

    stats = pool.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total allocations: {stats['total_allocations']}")
    print(f"  Total frees: {stats['total_frees']}")
    print(f"  Available pages: {stats['available_pages']}")
    print(f"  Utilization: {stats['utilization']:.2%}")

    # Print per-bucket statistics
    print(f"\nPer-bucket statistics:")
    for bucket_size, bucket_stats in stats['buckets'].items():
        if bucket_stats['allocations'] > 0 or bucket_stats['frees'] > 0:
            print(f"  Bucket {bucket_size:3d}: "
                  f"allocs={bucket_stats['allocations']:3d}, "
                  f"frees={bucket_stats['frees']:3d}, "
                  f"free_blocks={bucket_stats['free_blocks']:3d}")

    print("✓ Test 3 passed\n")


def test_clear():
    """Test clear functionality."""
    print("Test 4: Clear functionality")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = PreallocatedKVBlockPool(
        total_pages=1000,
        page_size=16,
        device=device,
        bucket_sizes=[1, 2, 4, 8],
        debug_mode=True,
    )

    # Allocate some pages
    for _ in range(10):
        pool.allocate(4)

    stats_before = pool.get_statistics()
    print(f"Before clear - available pages: {stats_before['available_pages']}")

    # Clear the pool
    pool.clear()

    stats_after = pool.get_statistics()
    print(f"After clear - available pages: {stats_after['available_pages']}")

    assert stats_after['total_allocations'] == 0, "Allocations not reset"
    assert stats_after['total_frees'] == 0, "Frees not reset"

    print("✓ Test 4 passed\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running PreallocatedKVBlockPool Tests")
    print("=" * 60)

    try:
        test_preallocated_pool_basic()
        test_preallocated_pool_splitting()
        test_preallocated_pool_stress()
        test_clear()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    run_all_tests()
