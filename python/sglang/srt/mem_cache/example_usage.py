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
Example usage of PreallocatedKVBlockPool and PreallocatedPagedTokenToKVPoolAllocator.

This script demonstrates how to use the KV cache preallocation pool in various scenarios.
"""

import torch

from sglang.srt.mem_cache.preallocated_pool import PreallocatedKVBlockPool


def example_basic_usage():
    """Example 1: Basic usage of PreallocatedKVBlockPool."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create a preallocation pool
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = PreallocatedKVBlockPool(
        total_pages=1000,       # Total pages available
        page_size=16,           # Tokens per page
        device=device,          # Device to use
        bucket_sizes=[1, 2, 4, 8, 16, 32, 64],  # Bucket sizes in pages
        enable_splitting=True,  # Enable block splitting
        debug_mode=False        # Disable debug for production
    )

    print(f"Created pool: {pool}")
    print(f"Available pages: {pool.available_pages()}")

    # Allocate some pages
    print("\nAllocating pages:")
    pages_4 = pool.allocate(4)   # Allocate 4 pages
    print(f"  Allocated 4 pages: {pages_4}")

    pages_8 = pool.allocate(8)   # Allocate 8 pages
    print(f"  Allocated 8 pages: {pages_8}")

    pages_16 = pool.allocate(16) # Allocate 16 pages
    print(f"  Allocated 16 pages: {pages_16}")

    # Check statistics
    stats = pool.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total allocations: {stats['total_allocations']}")
    print(f"  Available pages: {stats['available_pages']}")
    print(f"  Utilization: {stats['utilization']:.2%}")

    # Free pages
    print("\nFreeing pages:")
    pool.free(pages_4)
    print(f"  Freed 4 pages")

    pool.free(pages_8)
    print(f"  Freed 8 pages")

    # Check statistics after freeing
    stats = pool.get_statistics()
    print(f"\nStatistics after freeing:")
    print(f"  Available pages: {stats['available_pages']}")
    print(f"  Utilization: {stats['utilization']:.2%}")


def example_block_splitting():
    """Example 2: Demonstrating block splitting."""
    print("\n" + "=" * 60)
    print("Example 2: Block Splitting")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = PreallocatedKVBlockPool(
        total_pages=500,
        page_size=16,
        device=device,
        bucket_sizes=[1, 4, 16, 32],  # Sparse bucket sizes
        enable_splitting=True,
        debug_mode=True  # Enable debug to see splitting
    )

    print(f"Bucket sizes: {pool.bucket_sizes}")
    print(f"Enable splitting: {pool.enable_splitting}")

    # Allocate sizes that don't match bucket sizes
    print("\nAllocating non-standard sizes:")

    # Request 5 pages - should split from 16-page bucket
    pages_5 = pool.allocate(5)
    print(f"  Allocated 5 pages (split from larger bucket)")

    # Request 10 pages - should split from 16-page bucket
    pages_10 = pool.allocate(10)
    print(f"  Allocated 10 pages (split from larger bucket)")

    # Check split statistics
    stats = pool.get_statistics()
    print(f"\nSplit statistics:")
    print(f"  Split operations: {stats['split_operations']}")
    print(f"  Fallback allocations: {stats['fallback_allocations']}")


def example_workload_simulation():
    """Example 3: Simulating a realistic workload."""
    print("\n" + "=" * 60)
    print("Example 3: Realistic Workload Simulation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = PreallocatedKVBlockPool(
        total_pages=2000,
        page_size=16,
        device=device,
        bucket_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
        enable_splitting=True,
        debug_mode=False
    )

    print("Simulating workload:")
    print("  - Phase 1: Prefill operations (large allocations)")
    print("  - Phase 2: Decode operations (small allocations)")
    print("  - Phase 3: Mixed workload")

    allocated_blocks = []

    # Phase 1: Prefill (large allocations)
    print("\nPhase 1: Prefill")
    for i in range(10):
        size = 64 + (i % 3) * 16  # 64, 80, 96 pages
        pages = pool.allocate(size)
        if pages is not None:
            allocated_blocks.append(pages)

    stats = pool.get_statistics()
    print(f"  After prefill - Utilization: {stats['utilization']:.2%}")

    # Phase 2: Decode (small allocations)
    print("\nPhase 2: Decode")
    for i in range(50):
        size = 1  # 1 page per decode
        pages = pool.allocate(size)
        if pages is not None:
            allocated_blocks.append(pages)

    stats = pool.get_statistics()
    print(f"  After decode - Utilization: {stats['utilization']:.2%}")

    # Phase 3: Free some and allocate more
    print("\nPhase 3: Mixed workload")
    # Free prefill blocks
    for block in allocated_blocks[:10]:
        pool.free(block)

    # Allocate mixed sizes
    for i in range(20):
        size = (i % 4 + 1) * 4  # 4, 8, 12, 16 pages
        pages = pool.allocate(size)
        if pages is not None:
            allocated_blocks.append(pages)

    # Final statistics
    stats = pool.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total allocations: {stats['total_allocations']}")
    print(f"  Total frees: {stats['total_frees']}")
    print(f"  Utilization: {stats['utilization']:.2%}")
    print(f"  Split operations: {stats['split_operations']}")

    # Per-bucket statistics
    print(f"\nPer-bucket statistics:")
    for bucket_size in sorted(stats['buckets'].keys()):
        bucket_stats = stats['buckets'][bucket_size]
        if bucket_stats['allocations'] > 0:
            print(f"  Bucket {bucket_size:3d}: "
                  f"allocs={bucket_stats['allocations']:3d}, "
                  f"frees={bucket_stats['frees']:3d}, "
                  f"free_pages={bucket_stats['free_pages']:4d}")


def example_integrated_usage():
    """Example 4: Using PreallocatedPagedTokenToKVPoolAllocator."""
    print("\n" + "=" * 60)
    print("Example 4: Integrated Allocator Usage")
    print("=" * 60)

    # Note: This example shows the API usage
    # In actual deployment, this would be integrated with the full SGLang system

    from sglang.srt.mem_cache.allocator import PreallocatedPagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create KV cache
    kvcache = MHATokenToKVPool(
        size=10000,
        page_size=16,
        dtype=torch.float16,
        head_num=32,
        head_dim=128,
        layer_num=32,
        device=device,
        enable_memory_saver=False
    )

    # Create allocator with preallocation
    allocator = PreallocatedPagedTokenToKVPoolAllocator(
        size=10000,
        page_size=16,
        dtype=torch.int64,
        device=device,
        kvcache=kvcache,
        need_sort=True,
        enable_prealloc=True,
        prealloc_bucket_sizes=[1, 2, 4, 8, 16, 32, 64],
        prealloc_ratio=0.8
    )

    print(f"Created allocator: {allocator}")

    # Allocate tokens
    print("\nAllocating tokens:")
    indices_64 = allocator.alloc(need_size=64)  # 64 tokens (4 pages)
    print(f"  Allocated 64 tokens: {indices_64 is not None}")

    indices_128 = allocator.alloc(need_size=128)  # 128 tokens (8 pages)
    print(f"  Allocated 128 tokens: {indices_128 is not None}")

    # Get statistics
    stats = allocator.get_statistics()
    print(f"\nAllocator statistics:")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Total available size: {stats['total_available_size']}")
    if 'prealloc' in stats:
        print(f"  Prealloc utilization: {stats['prealloc']['utilization']:.2%}")

    # Free tokens
    allocator.free(indices_64)
    allocator.free(indices_128)

    print("\nTokens freed")


def run_all_examples():
    """Run all examples."""
    print("KV Cache Preallocation Pool - Usage Examples")
    print()

    try:
        example_basic_usage()
        example_block_splitting()
        example_workload_simulation()
        example_integrated_usage()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
