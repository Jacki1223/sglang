#!/usr/bin/env python3
"""
KV Cache分配器性能对比脚本

对比三种实现:
1. 原始PagedTokenToKVPoolAllocator (Baseline)
2. PreallocatedPagedTokenToKVPoolAllocator (标准预分配池)
3. OptimizedPreallocatedKVBlockPool (优化预分配池)

运行方式:
    python compare_allocators.py

环境要求:
    - PyTorch
    - SGLang源码
"""

import sys
import os
import time
import torch
from typing import List, Tuple, Dict
from collections import defaultdict
import random

# 添加SGLang路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

try:
    from sglang.srt.mem_cache.allocator import (
        PagedTokenToKVPoolAllocator,
        PreallocatedPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.preallocated_pool import PreallocatedKVBlockPool
    from sglang.srt.mem_cache.preallocated_pool_optimized import OptimizedPreallocatedKVBlockPool
except ImportError as e:
    print(f"错误: 无法导入SGLang模块: {e}")
    print("请确保在SGLang根目录运行此脚本")
    sys.exit(1)


class WorkloadGenerator:
    """工作负载生成器"""

    def __init__(self, page_size: int = 16):
        self.page_size = page_size

    def generate_chat_workload(self, num_requests: int = 100) -> List[Tuple[str, int]]:
        """
        生成聊天场景工作负载
        特点: 多为小到中等大小的分配 (16-256 tokens)
        """
        operations = []
        allocated_sizes = []

        for i in range(num_requests):
            # 80% 分配, 20% 释放
            if random.random() < 0.8 or not allocated_sizes:
                # 分配
                # 聊天场景: 用户输入(短) + 模型输出(中等)
                if random.random() < 0.6:
                    # 短输入
                    tokens = random.randint(10, 100)
                else:
                    # 长输出
                    tokens = random.randint(100, 300)

                # 对齐到page
                size = ((tokens + self.page_size - 1) // self.page_size) * self.page_size
                operations.append(('alloc', size))
                allocated_sizes.append(size)
            else:
                # 释放
                operations.append(('free', allocated_sizes.pop()))

        return operations

    def generate_long_context_workload(self, num_requests: int = 100) -> List[Tuple[str, int]]:
        """
        生成长文本场景工作负载
        特点: 大块分配 (512-4096 tokens)
        """
        operations = []
        allocated_sizes = []

        for i in range(num_requests):
            if random.random() < 0.7 or not allocated_sizes:
                # 长文本prefill
                tokens = random.choice([512, 1024, 2048, 4096, 8192])
                size = ((tokens + self.page_size - 1) // self.page_size) * self.page_size
                operations.append(('alloc', size))
                allocated_sizes.append(size)
            else:
                operations.append(('free', allocated_sizes.pop()))

        return operations

    def generate_mixed_workload(self, num_requests: int = 1000) -> List[Tuple[str, int]]:
        """
        生成混合场景工作负载
        特点: 70% prefill (大), 20% decode (小), 10% 释放
        """
        operations = []
        allocated_sizes = []

        for i in range(num_requests):
            op_type = random.choices(['prefill', 'decode', 'free'], weights=[0.7, 0.2, 0.1])[0]

            if op_type == 'prefill':
                # 大块分配
                tokens = random.choice([64, 128, 256, 512, 1024])
                size = ((tokens + self.page_size - 1) // self.page_size) * self.page_size
                operations.append(('alloc', size))
                allocated_sizes.append(size)

            elif op_type == 'decode':
                # 小块分配
                tokens = random.choice([1, 2, 4, 8, 16, 32])
                size = ((tokens + self.page_size - 1) // self.page_size) * self.page_size
                operations.append(('alloc', size))
                allocated_sizes.append(size)

            elif op_type == 'free' and allocated_sizes:
                operations.append(('free', allocated_sizes.pop()))

        return operations


class PerformanceBenchmark:
    """性能基准测试"""

    def __init__(self, device: str = 'cuda'):
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.results = defaultdict(dict)

    def create_kvcache(self, size: int, page_size: int) -> MHATokenToKVPool:
        """创建KV Cache池"""
        return MHATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=torch.float16,
            head_num=32,
            head_dim=128,
            layer_num=1,  # 简化，只用1层
            device=self.device,
            enable_memory_saver=False,
        )

    def benchmark_allocator(
        self,
        allocator,
        operations: List[Tuple[str, int]],
        name: str,
        warmup: int = 10
    ) -> Dict:
        """
        测试allocator性能

        返回:
            dict: {
                'total_time': float,  # 总时间(秒)
                'ops_per_sec': float,  # 每秒操作数
                'avg_alloc_time': float,  # 平均分配时间(ms)
                'avg_free_time': float,  # 平均释放时间(ms)
                'success_rate': float,  # 成功率
            }
        """
        # 预热
        allocated = []
        for i, (op_type, size) in enumerate(operations[:warmup]):
            if op_type == 'alloc':
                result = allocator.alloc(size)
                if result is not None:
                    allocated.append(result)
            elif op_type == 'free' and allocated:
                allocator.free(allocated.pop())

        # 清理预热数据
        for indices in allocated:
            allocator.free(indices)
        allocated.clear()

        # 实际测试
        alloc_times = []
        free_times = []
        success_count = 0
        fail_count = 0

        if self.device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for op_type, size in operations:
            if op_type == 'alloc':
                t0 = time.perf_counter()
                result = allocator.alloc(size)
                t1 = time.perf_counter()

                alloc_times.append((t1 - t0) * 1000)  # ms

                if result is not None:
                    allocated.append(result)
                    success_count += 1
                else:
                    fail_count += 1

            elif op_type == 'free' and allocated:
                t0 = time.perf_counter()
                allocator.free(allocated.pop())
                t1 = time.perf_counter()

                free_times.append((t1 - t0) * 1000)  # ms

        if self.device == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # 清理剩余
        for indices in allocated:
            allocator.free(indices)

        total_time = end_time - start_time
        ops_per_sec = len(operations) / total_time if total_time > 0 else 0

        return {
            'name': name,
            'total_time': total_time,
            'ops_per_sec': ops_per_sec,
            'avg_alloc_time': sum(alloc_times) / len(alloc_times) if alloc_times else 0,
            'avg_free_time': sum(free_times) / len(free_times) if free_times else 0,
            'p50_alloc_time': sorted(alloc_times)[len(alloc_times)//2] if alloc_times else 0,
            'p95_alloc_time': sorted(alloc_times)[int(len(alloc_times)*0.95)] if alloc_times else 0,
            'p99_alloc_time': sorted(alloc_times)[int(len(alloc_times)*0.99)] if alloc_times else 0,
            'success_rate': success_count / (success_count + fail_count) if (success_count + fail_count) > 0 else 0,
        }

    def compare_allocators(
        self,
        workload: List[Tuple[str, int]],
        workload_name: str,
        size: int = 50000,
        page_size: int = 16
    ):
        """对比三种allocator"""
        print(f"\n{'='*70}")
        print(f"工作负载: {workload_name}")
        print(f"总操作数: {len(workload)}")
        print(f"内存池大小: {size} tokens, 页大小: {page_size}")
        print(f"设备: {self.device}")
        print(f"{'='*70}\n")

        # 1. 原始PagedTokenToKVPoolAllocator (Baseline)
        print("测试 1/3: 原始PagedTokenToKVPoolAllocator (Baseline)...")
        kvcache1 = self.create_kvcache(size, page_size)
        allocator1 = PagedTokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=torch.int64,
            device=self.device,
            kvcache=kvcache1,
            need_sort=True,
        )
        result1 = self.benchmark_allocator(allocator1, workload, "Original (Baseline)")

        # 2. PreallocatedPagedTokenToKVPoolAllocator (标准)
        print("测试 2/3: PreallocatedPagedTokenToKVPoolAllocator (标准)...")
        kvcache2 = self.create_kvcache(size, page_size)
        allocator2 = PreallocatedPagedTokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=torch.int64,
            device=self.device,
            kvcache=kvcache2,
            need_sort=True,
            enable_prealloc=True,
            use_optimized=False,  # 使用标准版本
            prealloc_ratio=0.8,
        )
        result2 = self.benchmark_allocator(allocator2, workload, "Prealloc (Standard)")

        # 3. PreallocatedPagedTokenToKVPoolAllocator (优化)
        print("测试 3/3: PreallocatedPagedTokenToKVPoolAllocator (优化)...")
        kvcache3 = self.create_kvcache(size, page_size)
        allocator3 = PreallocatedPagedTokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=torch.int64,
            device=self.device,
            kvcache=kvcache3,
            need_sort=True,
            enable_prealloc=True,
            use_optimized=True,  # 使用优化版本
            prealloc_ratio=0.8,
        )
        result3 = self.benchmark_allocator(allocator3, workload, "Prealloc (Optimized)")

        # 打印结果
        self._print_comparison(result1, result2, result3)

        # 保存结果
        self.results[workload_name] = {
            'baseline': result1,
            'standard': result2,
            'optimized': result3,
        }

    def _print_comparison(self, baseline, standard, optimized):
        """打印对比结果"""
        print(f"\n{'指标':<25} {'Baseline':<15} {'Standard':<15} {'Optimized':<15} {'Std vs Base':<15} {'Opt vs Base':<15}")
        print("-" * 100)

        # 总时间
        print(f"{'总时间 (秒)':<25} {baseline['total_time']:<15.3f} {standard['total_time']:<15.3f} {optimized['total_time']:<15.3f} "
              f"{self._speedup_str(baseline['total_time'], standard['total_time']):<15} "
              f"{self._speedup_str(baseline['total_time'], optimized['total_time']):<15}")

        # 吞吐量
        print(f"{'吞吐量 (ops/s)':<25} {baseline['ops_per_sec']:<15.0f} {standard['ops_per_sec']:<15.0f} {optimized['ops_per_sec']:<15.0f} "
              f"{self._speedup_str(standard['ops_per_sec'], baseline['ops_per_sec'], inverse=True):<15} "
              f"{self._speedup_str(optimized['ops_per_sec'], baseline['ops_per_sec'], inverse=True):<15}")

        # 平均分配时间
        print(f"{'平均分配时间 (ms)':<25} {baseline['avg_alloc_time']:<15.4f} {standard['avg_alloc_time']:<15.4f} {optimized['avg_alloc_time']:<15.4f} "
              f"{self._speedup_str(baseline['avg_alloc_time'], standard['avg_alloc_time']):<15} "
              f"{self._speedup_str(baseline['avg_alloc_time'], optimized['avg_alloc_time']):<15}")

        # P95分配时间
        print(f"{'P95分配时间 (ms)':<25} {baseline['p95_alloc_time']:<15.4f} {standard['p95_alloc_time']:<15.4f} {optimized['p95_alloc_time']:<15.4f} "
              f"{self._speedup_str(baseline['p95_alloc_time'], standard['p95_alloc_time']):<15} "
              f"{self._speedup_str(baseline['p95_alloc_time'], optimized['p95_alloc_time']):<15}")

        # 平均释放时间
        print(f"{'平均释放时间 (ms)':<25} {baseline['avg_free_time']:<15.4f} {standard['avg_free_time']:<15.4f} {optimized['avg_free_time']:<15.4f} "
              f"{self._speedup_str(baseline['avg_free_time'], standard['avg_free_time']):<15} "
              f"{self._speedup_str(baseline['avg_free_time'], optimized['avg_free_time']):<15}")

        # 成功率
        print(f"{'成功率':<25} {baseline['success_rate']:<15.2%} {standard['success_rate']:<15.2%} {optimized['success_rate']:<15.2%} "
              f"{'=':<15} {'=':<15}")

        print()

        # 总结
        opt_speedup = optimized['ops_per_sec'] / baseline['ops_per_sec'] if baseline['ops_per_sec'] > 0 else 0
        std_speedup = standard['ops_per_sec'] / baseline['ops_per_sec'] if baseline['ops_per_sec'] > 0 else 0

        print("📊 总结:")
        print(f"  标准预分配池: {std_speedup:.2f}x 吞吐量 ({self._improvement_str(std_speedup)})")
        print(f"  优化预分配池: {opt_speedup:.2f}x 吞吐量 ({self._improvement_str(opt_speedup)})")

        if opt_speedup > 1.1:
            print(f"  ✅ 优化版本性能提升显著 ({(opt_speedup-1)*100:.1f}% 更快)")
        elif opt_speedup > 1.0:
            print(f"  ✓ 优化版本略有提升 ({(opt_speedup-1)*100:.1f}% 更快)")
        else:
            print(f"  ⚠️ 优化版本未见提升")

    def _speedup_str(self, baseline: float, new: float, inverse: bool = False) -> str:
        """计算加速比字符串"""
        if baseline == 0 or new == 0:
            return "N/A"

        if inverse:
            speedup = new / baseline
        else:
            speedup = baseline / new

        if speedup > 1.1:
            return f"{speedup:.2f}x ✅"
        elif speedup > 1.0:
            return f"{speedup:.2f}x ✓"
        elif speedup < 0.9:
            return f"{speedup:.2f}x ❌"
        else:
            return f"{speedup:.2f}x ≈"

    def _improvement_str(self, speedup: float) -> str:
        """改进百分比字符串"""
        improvement = (speedup - 1) * 100
        if improvement > 10:
            return f"+{improvement:.1f}% ✅"
        elif improvement > 0:
            return f"+{improvement:.1f}% ✓"
        elif improvement < -10:
            return f"{improvement:.1f}% ❌"
        else:
            return f"{improvement:.1f}% ≈"

    def print_summary(self):
        """打印所有测试的总结"""
        print(f"\n{'='*70}")
        print("总结报告")
        print(f"{'='*70}\n")

        for workload_name, results in self.results.items():
            baseline = results['baseline']
            optimized = results['optimized']

            speedup = optimized['ops_per_sec'] / baseline['ops_per_sec'] if baseline['ops_per_sec'] > 0 else 0
            improvement = (speedup - 1) * 100

            print(f"{workload_name}:")
            print(f"  Baseline: {baseline['ops_per_sec']:.0f} ops/s")
            print(f"  Optimized: {optimized['ops_per_sec']:.0f} ops/s")
            print(f"  加速比: {speedup:.2f}x ({improvement:+.1f}%)")
            print()


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           KV Cache分配器性能对比测试                                 ║
║                                                                      ║
║  对比:                                                               ║
║    1. PagedTokenToKVPoolAllocator (原始)                            ║
║    2. PreallocatedPagedTokenToKVPoolAllocator (标准)                ║
║    3. PreallocatedPagedTokenToKVPoolAllocator (优化)                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # 创建基准测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    benchmark = PerformanceBenchmark(device=device)

    # 生成工作负载
    generator = WorkloadGenerator(page_size=16)

    # 测试1: 聊天场景
    chat_workload = generator.generate_chat_workload(num_requests=500)
    benchmark.compare_allocators(
        chat_workload,
        "聊天场景 (小到中等分配)",
        size=50000,
        page_size=16
    )

    # 测试2: 长文本场景
    long_workload = generator.generate_long_context_workload(num_requests=200)
    benchmark.compare_allocators(
        long_workload,
        "长文本场景 (大块分配)",
        size=100000,
        page_size=16
    )

    # 测试3: 混合场景
    mixed_workload = generator.generate_mixed_workload(num_requests=1000)
    benchmark.compare_allocators(
        mixed_workload,
        "混合场景 (Prefill + Decode)",
        size=100000,
        page_size=16
    )

    # 打印总结
    benchmark.print_summary()

    print(f"\n{'='*70}")
    print("建议:")
    print(f"{'='*70}")
    print("""
1. 如果优化版本显示 >1.2x 加速比:
   ✅ 强烈建议在生产环境使用优化版本
   设置: use_optimized=True

2. 如果优化版本显示 1.0-1.2x 加速比:
   ✓ 建议使用优化版本
   根据具体场景测试验证

3. 如果优化版本显示 <1.0x (性能下降):
   ⚠️ 检查工作负载特点,可能需要调整桶配置
   或继续使用原始实现

4. 查看详细优化指南:
   PERFORMANCE_OPTIMIZATION_GUIDE.md
   QUICK_FIX_PERFORMANCE.md
    """)


if __name__ == "__main__":
    main()
