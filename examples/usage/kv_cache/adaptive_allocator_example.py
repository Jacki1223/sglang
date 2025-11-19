"""
Adaptive Page Size Allocator Example

This example demonstrates how to use the AdaptivePagedTokenToKVPoolAllocator
to reduce memory fragmentation and improve memory utilization.
"""

import torch
from sglang.srt.mem_cache.allocator_adaptive import AdaptivePagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


def main():
    """Main example function."""
    print("=" * 70)
    print("Adaptive Page Size Allocator Example")
    print("=" * 70)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_size = 10000  # Total tokens
    dtype = torch.float16

    print(f"\nDevice: {device}")
    print(f"Total size: {total_size} tokens")
    print(f"Dtype: {dtype}")

    # Step 1: Create KV Cache
    print("\n1. Creating KV Cache...")
    kvcache = MHATokenToKVPool(
        size=total_size,
        page_size=16,  # Base page size
        dtype=dtype,
        head_num=32,
        head_dim=128,
        layer_num=1,  # Simplified for example
        device=device,
        enable_memory_saver=False,
    )
    print("   ✓ KV Cache created")

    # Step 2: Create Adaptive Allocator
    print("\n2. Creating Adaptive Allocator...")
    print("   Page sizes: [16, 64, 256]")
    print("   Ratios: {16: 25%, 64: 50%, 256: 25%}")

    allocator = AdaptivePagedTokenToKVPoolAllocator(
        size=total_size,
        page_sizes=[16, 64, 256],
        dtype=dtype,
        device=device,
        kvcache=kvcache,
        need_sort=True,
        page_size_ratios={
            16: 0.25,
            64: 0.50,
            256: 0.25,
        }
    )
    print("   ✓ Allocator created")

    # Step 3: Perform allocations
    print("\n3. Performing sample allocations...")
    allocations = []

    # Small request (uses 16-token pages)
    print("\n   Allocating 10 tokens...")
    indices_10 = allocator.alloc(10)
    print(f"   ✓ Allocated {len(indices_10)} tokens")
    print(f"     Chosen page size: {allocator.choose_page_size(10)}")
    allocations.append(('small', indices_10))

    # Medium request (uses 64-token pages)
    print("\n   Allocating 100 tokens...")
    indices_100 = allocator.alloc(100)
    print(f"   ✓ Allocated {len(indices_100)} tokens")
    print(f"     Chosen page size: {allocator.choose_page_size(100)}")
    allocations.append(('medium', indices_100))

    # Large request (uses 256-token pages)
    print("\n   Allocating 500 tokens...")
    indices_500 = allocator.alloc(500)
    print(f"   ✓ Allocated {len(indices_500)} tokens")
    print(f"     Chosen page size: {allocator.choose_page_size(500)}")
    allocations.append(('large', indices_500))

    # Step 4: Show statistics
    print("\n4. Allocator Statistics:")
    print("   " + "-" * 60)
    stats = allocator.get_stats()

    print(f"   Total allocations: {stats['total_allocations']}")
    print(f"   Average fragmentation: {stats['average_fragmentation']:.2%}")
    print(f"   Memory utilization: {stats['memory_utilization']:.2%}")
    print(f"   Page splits: {stats['split_count']}")

    print("\n   Allocations by page size:")
    for ps, count in stats['alloc_by_size'].items():
        print(f"     {ps:3d}-token pages: {count:3d} allocations")

    print("\n   Free pages distribution:")
    for ps, count in stats['free_pages_distribution'].items():
        print(f"     {ps:3d}-token pages: {count:3d} free")

    # Step 5: Free some allocations
    print("\n5. Freeing allocations...")
    for name, indices in allocations[:2]:
        allocator.free(indices)
        print(f"   ✓ Freed {name} allocation ({len(indices)} tokens)")

    # Show updated statistics
    print("\n6. Updated Statistics:")
    print("   " + "-" * 60)
    stats = allocator.get_stats()
    print(f"   Available size: {allocator.available_size()} tokens")

    print("\n   Free pages distribution:")
    for ps, count in stats['free_pages_distribution'].items():
        print(f"     {ps:3d}-token pages: {count:3d} free")

    # Step 7: Demonstrate page splitting
    print("\n7. Demonstrating page splitting...")
    print("   Allocating all 64-token pages...")

    count_64 = 0
    while True:
        indices = allocator.alloc(64)
        if indices is None:
            break
        count_64 += 1

    print(f"   ✓ Allocated {count_64} 64-token allocations")

    print("\n   Now trying to allocate 16 tokens...")
    print("   (This should trigger splitting from larger pages)")

    indices_16 = allocator.alloc(16)
    if indices_16 is not None:
        print(f"   ✓ Successfully allocated via page splitting")
        stats = allocator.get_stats()
        print(f"   Split count: {stats['split_count']}")
    else:
        print(f"   ✗ Allocation failed (no available memory)")

    # Final statistics
    print("\n8. Final Statistics:")
    print("   " + "=" * 60)
    print(allocator.debug_print())

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def compare_with_fixed():
    """Compare adaptive allocator with fixed page size allocator."""
    print("\n" + "=" * 70)
    print("Comparison: Adaptive vs Fixed Page Size")
    print("=" * 70)

    from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_size = 10000
    dtype = torch.float16

    # Create KV caches
    kvcache_fixed = MHATokenToKVPool(
        size=total_size, page_size=64, dtype=dtype,
        head_num=32, head_dim=128, layer_num=1,
        device=device, enable_memory_saver=False,
    )

    kvcache_adaptive = MHATokenToKVPool(
        size=total_size, page_size=16, dtype=dtype,
        head_num=32, head_dim=128, layer_num=1,
        device=device, enable_memory_saver=False,
    )

    # Create allocators
    fixed_alloc = PagedTokenToKVPoolAllocator(
        size=total_size, page_size=64, dtype=dtype,
        device=device, kvcache=kvcache_fixed, need_sort=True,
    )

    adaptive_alloc = AdaptivePagedTokenToKVPoolAllocator(
        size=total_size, page_sizes=[16, 64, 256],
        dtype=dtype, device=device,
        kvcache=kvcache_adaptive, need_sort=True,
    )

    # Test with various request sizes
    test_sizes = [10, 30, 50, 90, 150, 500]

    print("\nTest allocation sizes:", test_sizes)

    # Fixed allocator (must align to page size)
    print("\n1. Fixed Page Size Allocator (64 tokens/page):")
    total_waste_fixed = 0
    for size in test_sizes:
        aligned_size = ((size + 63) // 64) * 64
        indices = fixed_alloc.alloc(aligned_size)
        waste = aligned_size - size
        total_waste_fixed += waste
        waste_pct = waste / size * 100
        print(f"   Request: {size:3d} -> Allocated: {aligned_size:3d} "
              f"(waste: {waste:2d} = {waste_pct:5.1f}%)")

    avg_waste_fixed = total_waste_fixed / sum(test_sizes)
    print(f"\n   Total waste: {total_waste_fixed} tokens")
    print(f"   Average waste ratio: {avg_waste_fixed:.2%}")

    # Adaptive allocator (exact size)
    print("\n2. Adaptive Page Size Allocator:")
    for size in test_sizes:
        indices = adaptive_alloc.alloc(size)
        page_size = adaptive_alloc.choose_page_size(size)
        aligned_size = ((size + page_size - 1) // page_size) * page_size
        waste = aligned_size - size
        waste_pct = waste / size * 100
        print(f"   Request: {size:3d} -> Page size: {page_size:3d} -> "
              f"Allocated: {aligned_size:3d} (waste: {waste:2d} = {waste_pct:5.1f}%)")

    stats = adaptive_alloc.get_stats()
    print(f"\n   Average fragmentation: {stats['average_fragmentation']:.2%}")
    print(f"   Memory utilization: {stats['memory_utilization']:.2%}")

    # Comparison
    print("\n3. Improvement:")
    improvement = (avg_waste_fixed - stats['average_fragmentation']) / avg_waste_fixed
    print(f"   Fragmentation reduction: {improvement:.1%}")
    print(f"   Memory saved: {total_waste_fixed - stats['total_wasted_tokens']} tokens")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run main example
    main()

    # Run comparison
    compare_with_fixed()

    print("\nFor more information, see:")
    print("  - docs/adaptive_page_allocator_guide.md")
    print("  - test/srt/mem_cache/test_adaptive_allocator.py")
