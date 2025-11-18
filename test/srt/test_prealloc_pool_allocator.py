"""
KV Cache预分配池的单元测试
"""

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.prealloc_pool_allocator import PreallocPoolAllocator


class TestPreallocPoolAllocator:
    """预分配池分配器测试"""

    @pytest.fixture
    def setup_allocator(self):
        """创建测试用的allocator"""
        # 配置
        total_tokens = 32768  # 32K tokens
        page_size = 16  # 每个page 16 tokens
        num_pages = total_tokens // page_size  # 2048 pages
        head_num = 32
        head_dim = 128
        layer_num = 32
        dtype = torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 创建KV Cache
        kv_pool = MHATokenToKVPool(
            size=total_tokens,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=False,
        )

        # 创建预分配池allocator
        allocator = PreallocPoolAllocator(
            size=total_tokens,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=True,
            enable_prealloc=True,
            prealloc_ratio=0.3,  # 使用30%空间做预分配
        )

        return allocator, page_size, device

    def test_basic_allocation(self, setup_allocator):
        """测试基本分配功能"""
        allocator, page_size, device = setup_allocator

        # 测试1: 分配64 tokens (4 pages)
        indices = allocator.alloc(64)
        assert indices is not None
        assert len(indices) == 64
        assert indices.device.type == device

        # 测试2: 分配128 tokens (8 pages)
        indices2 = allocator.alloc(128)
        assert indices2 is not None
        assert len(indices2) == 128

        # 测试3: 检查统计信息
        stats = allocator.get_stats()
        assert len(stats) > 0

        # 打印统计信息
        allocator.print_stats()

    def test_allocation_and_free(self, setup_allocator):
        """测试分配和释放"""
        allocator, page_size, device = setup_allocator

        # 分配
        indices1 = allocator.alloc(64)
        indices2 = allocator.alloc(128)
        indices3 = allocator.alloc(256)

        assert indices1 is not None
        assert indices2 is not None
        assert indices3 is not None

        # 记录分配前的统计
        stats_before = allocator.get_stats()
        total_allocated_before = sum(s.allocated_blocks for s in stats_before.values())

        # 释放
        allocator.free(indices1)
        allocator.free(indices2)

        # 检查统计
        stats_after = allocator.get_stats()
        total_allocated_after = sum(s.allocated_blocks for s in stats_after.values())

        # 释放后，已分配块数应该减少
        assert total_allocated_after < total_allocated_before

        # 再次分配，应该能复用
        indices4 = allocator.alloc(64)
        assert indices4 is not None

    def test_different_sizes(self, setup_allocator):
        """测试不同大小的分配"""
        allocator, page_size, device = setup_allocator

        # 测试各种大小
        sizes = [64, 128, 256, 512, 1024]
        allocated_indices = []

        for size in sizes:
            indices = allocator.alloc(size)
            assert indices is not None, f"Failed to allocate {size} tokens"
            assert len(indices) == size
            allocated_indices.append(indices)

        # 打印统计
        allocator.print_stats()

        # 清理
        for indices in allocated_indices:
            allocator.free(indices)

    def test_pool_exhaustion_fallback(self, setup_allocator):
        """测试池耗尽后的fallback"""
        allocator, page_size, device = setup_allocator

        # 大量分配，耗尽预分配池
        allocated = []
        for i in range(100):
            indices = allocator.alloc(256)  # 16 pages each
            if indices is None:
                break
            allocated.append(indices)

        print(f"Successfully allocated {len(allocated)} blocks before exhaustion")

        # 即使预分配池耗尽，fallback到标准分配应该还能工作
        # (取决于总容量)
        allocator.print_stats()

        # 清理
        for indices in allocated:
            allocator.free(indices)

    def test_clear(self, setup_allocator):
        """测试清空功能"""
        allocator, page_size, device = setup_allocator

        # 分配一些块
        indices1 = allocator.alloc(64)
        indices2 = allocator.alloc(128)

        # 清空
        allocator.clear()

        # 检查统计
        stats = allocator.get_stats()
        for stat in stats.values():
            assert stat.allocated_blocks == 0
            assert stat.free_blocks == stat.total_blocks

    def test_disabled_prealloc(self):
        """测试禁用预分配池的情况"""
        total_tokens = 32768
        page_size = 16
        head_num = 32
        head_dim = 128
        layer_num = 32
        dtype = torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"

        kv_pool = MHATokenToKVPool(
            size=total_tokens,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=False,
        )

        # 禁用预分配池
        allocator = PreallocPoolAllocator(
            size=total_tokens,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=True,
            enable_prealloc=False,  # 禁用
        )

        # 应该fallback到标准分配
        indices = allocator.alloc(64)
        assert indices is not None
        assert len(indices) == 64

        # 统计应该为空
        stats = allocator.get_stats()
        assert len(stats) == 0


if __name__ == "__main__":
    # 简单的测试运行
    import sys

    if torch.cuda.is_available():
        print("Running tests on CUDA...")
    else:
        print("CUDA not available, running on CPU...")

    # 创建测试实例
    test = TestPreallocPoolAllocator()

    # 运行测试
    setup = test.setup_allocator()

    print("\n=== Test 1: Basic Allocation ===")
    test.test_basic_allocation(setup)

    setup = test.setup_allocator()
    print("\n=== Test 2: Allocation and Free ===")
    test.test_allocation_and_free(setup)

    setup = test.setup_allocator()
    print("\n=== Test 3: Different Sizes ===")
    test.test_different_sizes(setup)

    setup = test.setup_allocator()
    print("\n=== Test 4: Pool Exhaustion ===")
    test.test_pool_exhaustion_fallback(setup)

    setup = test.setup_allocator()
    print("\n=== Test 5: Clear ===")
    test.test_clear(setup)

    print("\n=== Test 6: Disabled Prealloc ===")
    test.test_disabled_prealloc()

    print("\n✅ All tests passed!")
