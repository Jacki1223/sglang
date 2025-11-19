"""
KV Cache 预分配池性能诊断和优化工具

用于分析性能瓶颈并提供优化建议
"""

import time
import torch
from typing import Dict, List, Tuple
from collections import defaultdict


class PerformanceProfiler:
    """性能分析器"""

    def __init__(self):
        self.timings = defaultdict(list)
        self.counts = defaultdict(int)
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def clear(self):
        self.timings.clear()
        self.counts.clear()

    def record(self, operation: str, duration: float):
        if self.enabled:
            self.timings[operation].append(duration)
            self.counts[operation] += 1

    def get_stats(self) -> Dict:
        stats = {}
        for op, times in self.timings.items():
            if times:
                stats[op] = {
                    'count': len(times),
                    'total': sum(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                }
        return stats

    def print_report(self):
        print("=" * 70)
        print("性能分析报告")
        print("=" * 70)

        stats = self.get_stats()
        if not stats:
            print("没有收集到性能数据")
            return

        # 按平均时间排序
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['avg'], reverse=True)

        print(f"{'操作':<30} {'次数':>8} {'总时间(ms)':>12} {'平均(ms)':>12} {'最小(ms)':>12} {'最大(ms)':>12}")
        print("-" * 70)

        for op, s in sorted_ops:
            print(f"{op:<30} {s['count']:>8} {s['total']*1000:>12.3f} {s['avg']*1000:>12.3f} {s['min']*1000:>12.3f} {s['max']*1000:>12.3f}")

        print("-" * 70)
        total_time = sum(s['total'] for s in stats.values())
        print(f"总时间: {total_time*1000:.3f} ms")


def benchmark_allocator(allocator, operations: List[Tuple[str, int]], warmup: int = 10):
    """
    基准测试allocator性能

    Args:
        allocator: 要测试的allocator
        operations: [(op_type, size), ...] 其中op_type是'alloc'或'free'
        warmup: 预热次数
    """
    print(f"基准测试: {allocator.__class__.__name__}")
    print(f"操作数: {len(operations)}, 预热: {warmup}")

    # 预热
    allocated = []
    for _ in range(warmup):
        for op_type, size in operations[:10]:
            if op_type == 'alloc':
                result = allocator.alloc(size)
                if result is not None:
                    allocated.append(result)
            elif op_type == 'free' and allocated:
                allocator.free(allocated.pop())

    # 清理
    for indices in allocated:
        allocator.free(indices)
    allocated.clear()

    # 实际测试
    start_time = time.time()

    for op_type, size in operations:
        if op_type == 'alloc':
            result = allocator.alloc(size)
            if result is not None:
                allocated.append(result)
        elif op_type == 'free' and allocated:
            allocator.free(allocated.pop())

    end_time = time.time()

    # 清理剩余
    for indices in allocated:
        allocator.free(indices)

    elapsed = (end_time - start_time) * 1000  # ms
    ops_per_sec = len(operations) / (end_time - start_time)

    print(f"总时间: {elapsed:.2f} ms")
    print(f"吞吐量: {ops_per_sec:.0f} ops/sec")
    print(f"平均延迟: {elapsed/len(operations):.3f} ms/op")
    print()

    return {
        'elapsed_ms': elapsed,
        'ops_per_sec': ops_per_sec,
        'avg_latency_ms': elapsed / len(operations)
    }


def diagnose_allocation_pattern(allocator, num_samples: int = 1000):
    """
    诊断分配模式，找出最常用的大小

    Args:
        allocator: allocator实例
        num_samples: 采样次数
    """
    print("诊断分配模式...")

    if not hasattr(allocator, 'get_statistics'):
        print("Allocator不支持统计信息")
        return

    # 获取统计信息
    stats = allocator.get_statistics()

    if 'prealloc' not in stats:
        print("未启用预分配池")
        return

    prealloc = stats['prealloc']
    buckets = prealloc['buckets']

    # 分析桶使用情况
    print("\n桶使用分析:")
    print(f"{'桶大小':>10} {'分配次数':>12} {'释放次数':>12} {'空闲块':>10} {'空闲页':>10} {'利用率':>10}")
    print("-" * 70)

    total_allocs = sum(b['allocations'] for b in buckets.values())

    for size in sorted(buckets.keys()):
        b = buckets[size]
        if b['allocations'] > 0 or b['free_blocks'] < 10:  # 显示活跃的桶
            utilization = b['allocations'] / total_allocs * 100 if total_allocs > 0 else 0
            print(f"{size:>10} {b['allocations']:>12} {b['frees']:>12} {b['free_blocks']:>10} {b['free_pages']:>10} {utilization:>9.1f}%")

    # 建议
    print("\n优化建议:")

    # 找出未使用的桶
    unused_buckets = [size for size, b in buckets.items() if b['allocations'] == 0]
    if unused_buckets:
        print(f"- 未使用的桶: {unused_buckets}")
        print(f"  建议: 移除这些桶以减少管理开销")

    # 找出高使用率的桶
    high_usage = [(size, b['allocations']) for size, b in buckets.items() if b['allocations'] > total_allocs * 0.2]
    if high_usage:
        print(f"- 高使用率桶 (>20%): {[size for size, _ in high_usage]}")
        print(f"  建议: 为这些大小增加预分配块数")

    # 检查分割操作
    if prealloc['split_operations'] > total_allocs * 0.3:
        print(f"- 块分割次数过多: {prealloc['split_operations']} ({prealloc['split_operations']/total_allocs*100:.1f}%)")
        print(f"  建议: 调整桶大小以更好地匹配实际分配模式")

    # 检查回退分配
    if prealloc['fallback_allocations'] > total_allocs * 0.1:
        print(f"- 回退分配较多: {prealloc['fallback_allocations']} ({prealloc['fallback_allocations']/total_allocs*100:.1f}%)")
        print(f"  建议: 增加预分配比例或调整桶大小")


def compare_allocators(allocator1, allocator2, operations: List[Tuple[str, int]]):
    """
    比较两个allocator的性能

    Args:
        allocator1: 第一个allocator (baseline)
        allocator2: 第二个allocator (optimized)
        operations: 操作序列
    """
    print("=" * 70)
    print("Allocator性能对比")
    print("=" * 70)
    print()

    print("Baseline (原始实现):")
    baseline = benchmark_allocator(allocator1, operations)

    print("Optimized (优化实现):")
    optimized = benchmark_allocator(allocator2, operations)

    print("=" * 70)
    print("对比结果:")
    print("=" * 70)

    speedup = baseline['ops_per_sec'] / optimized['ops_per_sec']
    latency_improvement = (baseline['avg_latency_ms'] - optimized['avg_latency_ms']) / baseline['avg_latency_ms'] * 100

    print(f"吞吐量变化: {speedup:.2f}x")
    print(f"延迟改进: {latency_improvement:+.1f}%")

    if speedup > 1.1:
        print("✓ 优化版本更快")
    elif speedup < 0.9:
        print("✗ 优化版本更慢")
    else:
        print("≈ 性能相近")


def generate_realistic_workload(num_ops: int = 1000, page_size: int = 16) -> List[Tuple[str, int]]:
    """
    生成真实的工作负载模式

    Args:
        num_ops: 操作数
        page_size: 页大小
    """
    import random

    operations = []
    allocated_sizes = []

    # 模拟真实场景：
    # - 70% prefill (大块分配)
    # - 20% decode (小块分配)
    # - 10% 释放

    for _ in range(num_ops):
        op_type = random.choices(['prefill', 'decode', 'free'], weights=[0.7, 0.2, 0.1])[0]

        if op_type == 'prefill':
            # 大块分配: 64-512 tokens
            size = random.choice([64, 128, 256, 512])
            # 确保page对齐
            size = (size // page_size) * page_size
            operations.append(('alloc', size))
            allocated_sizes.append(size)

        elif op_type == 'decode':
            # 小块分配: 16-64 tokens
            size = random.choice([16, 32, 64])
            operations.append(('alloc', size))
            allocated_sizes.append(size)

        elif op_type == 'free' and allocated_sizes:
            # 释放
            operations.append(('free', allocated_sizes.pop()))

    return operations


if __name__ == "__main__":
    print("KV Cache预分配池性能诊断工具")
    print()
    print("使用方法:")
    print("1. 从你的代码中导入此模块")
    print("2. 使用 benchmark_allocator() 测试性能")
    print("3. 使用 diagnose_allocation_pattern() 分析分配模式")
    print("4. 使用 compare_allocators() 对比不同实现")
    print()
    print("示例:")
    print("""
    from performance_diagnostics import benchmark_allocator, generate_realistic_workload

    # 生成工作负载
    ops = generate_realistic_workload(1000)

    # 测试allocator
    results = benchmark_allocator(your_allocator, ops)

    # 分析模式
    diagnose_allocation_pattern(your_allocator)
    """)
