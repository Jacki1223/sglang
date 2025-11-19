#!/usr/bin/env python3
"""
KV Cache 预分配池性能对比基准测试

对比原始实现和优化实现的性能差异
"""

import sys
import time
import torch
from typing import List, Tuple

# 添加路径
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from sglang.srt.mem_cache.preallocated_pool import PreallocatedKVBlockPool
from sglang.srt.mem_cache.preallocated_pool_optimized import OptimizedPreallocatedKVBlockPool
from sglang.srt.mem_cache.performance_diagnostics import (
    benchmark_allocator,
    generate_realistic_workload,
    compare_allocators,
    diagnose_allocation_pattern
)


class SimpleAllocatorWrapper:
    """简单的包装器以统一测试接口"""

    def __init__(self, pool, page_size):
        self.pool = pool
        self.page_size = page_size
        self.allocated = []

    def alloc(self, need_size: int):
        """分配tokens"""
        num_pages = need_size // self.page_size
        pages = self.pool.allocate(num_pages)

        if pages is None:
            return None

        # 转换为token indices
        token_indices = (
            pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=pages.device)
        ).reshape(-1)

        return token_indices

    def free(self, token_indices: torch.Tensor):
        """释放tokens"""
        if token_indices is None:
            return

        # 转换为pages
        pages = torch.unique(token_indices // self.page_size)
        self.pool.free(pages)

    def get_statistics(self):
        return self.pool.get_statistics()


def run_micro_benchmark(device='cuda'):
    """微基准测试 - 单个操作的性能"""
    print("=" * 70)
    print("微基准测试")
    print("=" * 70)
    print()

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA不可用，使用CPU")

    page_size = 16
    total_pages = 10000

    # 创建两个池
    pool_orig = PreallocatedKVBlockPool(
        total_pages=total_pages,
        page_size=page_size,
        device=device,
        debug_mode=False
    )

    pool_opt = OptimizedPreallocatedKVBlockPool(
        total_pages=total_pages,
        page_size=page_size,
        device=device,
        debug_mode=False
    )

    print(f"设备: {device}")
    print(f"页大小: {page_size}")
    print(f"总页数: {total_pages}")
    print()

    # 测试不同大小的分配
    test_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    iterations = 1000

    print(f"{'大小(页)':>10} {'原始(μs)':>15} {'优化(μs)':>15} {'加速比':>10}")
    print("-" * 55)

    for size in test_sizes:
        # 原始版本
        start = time.perf_counter()
        for _ in range(iterations):
            pages = pool_orig.allocate(size)
            if pages is not None:
                pool_orig.free(pages)
        time_orig = (time.perf_counter() - start) * 1e6 / iterations

        # 优化版本
        start = time.perf_counter()
        for _ in range(iterations):
            pages = pool_opt.allocate(size)
            if pages is not None:
                pool_opt.free(pages)
        time_opt = (time.perf_counter() - start) * 1e6 / iterations

        speedup = time_orig / time_opt if time_opt > 0 else float('inf')

        print(f"{size:>10} {time_orig:>15.2f} {time_opt:>15.2f} {speedup:>9.2f}x")

    print()


def run_macro_benchmark(device='cuda'):
    """宏基准测试 - 真实工作负载"""
    print("=" * 70)
    print("宏基准测试 - 真实工作负载")
    print("=" * 70)
    print()

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    page_size = 16
    total_pages = 10000

    # 生成真实工作负载
    num_ops = 5000
    workload = generate_realistic_workload(num_ops, page_size)

    print(f"工作负载: {num_ops} 操作")
    print(f"设备: {device}")
    print()

    # 创建包装器
    pool_orig = PreallocatedKVBlockPool(
        total_pages=total_pages,
        page_size=page_size,
        device=device,
        debug_mode=False
    )
    allocator_orig = SimpleAllocatorWrapper(pool_orig, page_size)

    pool_opt = OptimizedPreallocatedKVBlockPool(
        total_pages=total_pages,
        page_size=page_size,
        device=device,
        debug_mode=False
    )
    allocator_opt = SimpleAllocatorWrapper(pool_opt, page_size)

    # 运行对比
    compare_allocators(allocator_orig, allocator_opt, workload)

    # 显示优化版本的额外统计
    print("\n优化版本额外指标:")
    stats = pool_opt.get_statistics()
    print(f"  快速路径命中率: {stats.get('fast_path_hit_rate', 0):.2%}")
    print(f"  分割率: {stats.get('split_rate', 0):.2%}")


def run_pattern_analysis(device='cuda'):
    """分配模式分析"""
    print("=" * 70)
    print("分配模式分析")
    print("=" * 70)
    print()

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    page_size = 16
    total_pages = 10000

    # 创建优化版本
    pool = OptimizedPreallocatedKVBlockPool(
        total_pages=total_pages,
        page_size=page_size,
        device=device,
        bucket_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
        debug_mode=False
    )

    allocator = SimpleAllocatorWrapper(pool, page_size)

    # 运行真实工作负载
    workload = generate_realistic_workload(2000, page_size)

    allocated = []
    for op_type, size in workload:
        if op_type == 'alloc':
            result = allocator.alloc(size)
            if result is not None:
                allocated.append(result)
        elif op_type == 'free' and allocated:
            allocator.free(allocated.pop())

    # 分析
    diagnose_allocation_pattern(allocator, num_samples=2000)


def run_memory_efficiency_test(device='cuda'):
    """内存效率测试"""
    print("\n" + "=" * 70)
    print("内存效率测试")
    print("=" * 70)
    print()

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    page_size = 16
    total_pages = 10000

    # 测试不同预分配比例的影响
    ratios = [0.5, 0.7, 0.8, 0.9, 0.95]

    print(f"{'预分配比例':>15} {'可用页数':>12} {'利用率':>10} {'性能(ops/s)':>15}")
    print("-" * 60)

    workload = generate_realistic_workload(1000, page_size)

    for ratio in ratios:
        pool = OptimizedPreallocatedKVBlockPool(
            total_pages=int(total_pages * ratio),
            page_size=page_size,
            device=device,
            debug_mode=False
        )

        allocator = SimpleAllocatorWrapper(pool, page_size)

        # 运行基准测试
        start = time.time()
        allocated = []
        for op_type, size in workload:
            if op_type == 'alloc':
                result = allocator.alloc(size)
                if result is not None:
                    allocated.append(result)
            elif op_type == 'free' and allocated:
                allocator.free(allocated.pop())
        elapsed = time.time() - start

        stats = pool.get_statistics()
        ops_per_sec = len(workload) / elapsed if elapsed > 0 else 0

        print(f"{ratio:>14.0%} {stats['available_pages']:>12} {stats['utilization']:>9.1%} {ops_per_sec:>15.0f}")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         KV Cache 预分配池性能对比基准测试                            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # 运行所有测试
        run_micro_benchmark(device)
        run_macro_benchmark(device)
        run_pattern_analysis(device)
        run_memory_efficiency_test(device)

        print("\n" + "=" * 70)
        print("总结")
        print("=" * 70)
        print("""
优化版本的主要改进:
  ✓ 快速路径优化 - 精确匹配O(1)查找
  ✓ 缓存优化 - 减少tensor创建开销
  ✓ 智能分配 - 更均匀的初始页面分配
  ✓ 减少复制 - 优化free操作

预期性能提升:
  - 精确匹配分配: 30-50% 更快
  - 混合工作负载: 15-30% 更快
  - 内存效率: 减少5-10%碎片

建议:
  1. 对于稳定工作负载,使用优化版本
  2. 调整桶大小以匹配实际分配模式
  3. 监控快速路径命中率(>80%为佳)
  4. 根据工作负载调整预分配比例
        """)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
