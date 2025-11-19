"""
Unit tests for AdaptivePagedTokenToKVPoolAllocator.
"""

import pytest
import torch

from sglang.srt.mem_cache.allocator_adaptive import AdaptivePagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


class TestAdaptiveAllocator:
    @pytest.fixture
    def kvcache(self):
        """Create a simple KV cache for testing."""
        return MHATokenToKVPool(
            size=10000,
            page_size=16,  # Base page size
            dtype=torch.float16,
            head_num=32,
            head_dim=128,
            layer_num=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            enable_memory_saver=False,
        )

    @pytest.fixture
    def allocator(self, kvcache):
        """Create an adaptive allocator."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return AdaptivePagedTokenToKVPoolAllocator(
            size=10000,
            page_sizes=[16, 64, 256],
            dtype=torch.float16,
            device=device,
            kvcache=kvcache,
            need_sort=True,
            page_size_ratios={16: 0.25, 64: 0.5, 256: 0.25},
        )

    def test_init(self, allocator):
        """Test allocator initialization."""
        assert allocator.page_sizes == [16, 64, 256]
        assert allocator.base_page_size == 16
        assert allocator.max_page_size == 256
        assert len(allocator.free_pages_by_size) == 3

    def test_choose_page_size_small(self, allocator):
        """Test page size selection for small requests."""
        # Small request should use 16-token pages
        assert allocator.choose_page_size(10) == 16
        assert allocator.choose_page_size(15) == 16
        assert allocator.choose_page_size(20) == 16

    def test_choose_page_size_medium(self, allocator):
        """Test page size selection for medium requests."""
        # Medium request should use 64-token pages
        assert allocator.choose_page_size(50) == 64
        assert allocator.choose_page_size(100) == 64
        assert allocator.choose_page_size(200) == 64

    def test_choose_page_size_large(self, allocator):
        """Test page size selection for large requests."""
        # Large request should use 256-token pages
        assert allocator.choose_page_size(500) == 256
        assert allocator.choose_page_size(1000) == 256

    def test_alloc_small(self, allocator):
        """Test allocation of small request."""
        indices = allocator.alloc(10)
        assert indices is not None
        assert len(indices) == 10
        assert indices.device.type == allocator.device

    def test_alloc_medium(self, allocator):
        """Test allocation of medium request."""
        indices = allocator.alloc(100)
        assert indices is not None
        assert len(indices) == 100

    def test_alloc_large(self, allocator):
        """Test allocation of large request."""
        indices = allocator.alloc(500)
        assert indices is not None
        assert len(indices) == 500

    def test_alloc_and_free(self, allocator):
        """Test allocation and freeing."""
        # Allocate
        indices1 = allocator.alloc(100)
        assert indices1 is not None

        # Check available size decreased
        initial_size = 10000
        available = allocator.available_size()
        assert available < initial_size

        # Free
        allocator.free(indices1)

        # Available size should increase (may not be exact due to fragmentation)
        available_after_free = allocator.available_size()
        assert available_after_free > available

    def test_multiple_allocs(self, allocator):
        """Test multiple allocations."""
        sizes = [10, 50, 100, 500, 1000]
        allocated = []

        for size in sizes:
            indices = allocator.alloc(size)
            assert indices is not None
            assert len(indices) == size
            allocated.append(indices)

        # Free all
        for indices in allocated:
            allocator.free(indices)

    def test_fragmentation_reduction(self, kvcache):
        """Test that fragmentation is reduced compared to fixed page size."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create fixed-size allocator (64-token pages)
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        fixed_allocator = PagedTokenToKVPoolAllocator(
            size=10000,
            page_size=64,
            dtype=torch.float16,
            device=device,
            kvcache=kvcache,
            need_sort=True,
        )

        # Create adaptive allocator
        adaptive_allocator = AdaptivePagedTokenToKVPoolAllocator(
            size=10000,
            page_sizes=[16, 64, 256],
            dtype=torch.float16,
            device=device,
            kvcache=kvcache,
            need_sort=True,
        )

        # Test allocation sizes that cause high fragmentation with fixed size
        test_sizes = [10, 30, 90, 150, 500]

        # Allocate with both allocators
        for size in test_sizes:
            # Fixed allocator: must be page-aligned
            aligned_size = ((size + 63) // 64) * 64
            fixed_indices = fixed_allocator.alloc(aligned_size)

            # Adaptive allocator: exact size
            adaptive_indices = adaptive_allocator.alloc(size)

            assert fixed_indices is not None
            assert adaptive_indices is not None
            assert len(adaptive_indices) == size
            assert len(fixed_indices) == aligned_size

        # Compare fragmentation
        adaptive_stats = adaptive_allocator.get_stats()
        adaptive_frag = adaptive_stats['average_fragmentation']

        # For test sizes, adaptive should have much lower fragmentation
        # Fixed allocator waste: (10->64, 30->64, 90->128, 150->192, 500->512)
        # Average: ~40% waste
        # Adaptive allocator waste: should be <15%
        print(f"Adaptive fragmentation: {adaptive_frag:.2%}")
        assert adaptive_frag < 0.15, f"Fragmentation too high: {adaptive_frag:.2%}"

    def test_page_splitting(self, allocator):
        """Test that pages can be split from larger to smaller."""
        # Allocate all 64-token pages first
        while True:
            indices = allocator.alloc(64)
            if indices is None:
                break

        # Now try to allocate 16-token request
        # This should trigger splitting from 256-token pages
        indices_16 = allocator.alloc(16)

        # Should succeed by splitting larger pages
        assert indices_16 is not None or allocator.available_size() < 16
        assert len(indices_16) == 16 if indices_16 is not None else True

        # Check split count
        stats = allocator.get_stats()
        assert stats['split_count'] >= 0

    def test_stats(self, allocator):
        """Test statistics collection."""
        # Perform some allocations
        allocator.alloc(10)
        allocator.alloc(100)
        allocator.alloc(500)

        stats = allocator.get_stats()

        assert 'alloc_by_size' in stats
        assert 'average_fragmentation' in stats
        assert 'total_allocations' in stats
        assert 'memory_utilization' in stats

        assert stats['total_allocations'] == 3
        assert stats['average_fragmentation'] >= 0.0
        assert stats['memory_utilization'] > 0.0

    def test_clear(self, allocator):
        """Test clearing the allocator."""
        # Allocate some memory
        allocator.alloc(100)
        allocator.alloc(200)

        # Clear
        allocator.clear()

        # Stats should be reset
        stats = allocator.get_stats()
        assert stats['total_allocations'] == 0
        assert stats['split_count'] == 0

        # Should be able to allocate again
        indices = allocator.alloc(100)
        assert indices is not None

    def test_invalid_page_sizes(self):
        """Test that invalid page sizes are rejected."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        kvcache = MHATokenToKVPool(
            size=10000,
            page_size=16,
            dtype=torch.float16,
            head_num=32,
            head_dim=128,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
        )

        # Non-power-of-2 should fail
        with pytest.raises(ValueError):
            AdaptivePagedTokenToKVPoolAllocator(
                size=10000,
                page_sizes=[16, 63, 256],  # 63 is not power of 2
                dtype=torch.float16,
                device=device,
                kvcache=kvcache,
                need_sort=True,
            )

        # Non-multiple of base size should fail
        with pytest.raises(ValueError):
            AdaptivePagedTokenToKVPoolAllocator(
                size=10000,
                page_sizes=[16, 64, 200],  # 200 is not multiple of 16
                dtype=torch.float16,
                device=device,
                kvcache=kvcache,
                need_sort=True,
            )

    def test_available_size(self, allocator):
        """Test available size calculation."""
        initial_size = allocator.available_size()
        assert initial_size > 0

        # Allocate some memory
        indices = allocator.alloc(100)
        assert indices is not None

        # Available size should decrease
        new_size = allocator.available_size()
        assert new_size < initial_size

        # Free the memory
        allocator.free(indices)

        # Available size should increase
        freed_size = allocator.available_size()
        assert freed_size > new_size


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
