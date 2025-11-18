#!/usr/bin/env python3
"""
调度策略切换性能断崖验证脚本

用于验证和对比原始策略与优化策略的性能。
"""

import argparse
import json
import time
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch

# Mock imports for testing without full SGLang setup
try:
    from sglang.srt.managers.schedule_policy import SchedulePolicy, CacheAwarePolicy
    from sglang.srt.managers.schedule_policy_optimized import (
        AdaptiveSchedulePolicy,
        SamplingLPMPolicy,
    )
    from sglang.srt.managers.schedule_policy_validator import PolicyPerformanceMonitor, PolicySwitchMetrics
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.managers.schedule_batch import Req
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    print("Warning: SGLang not fully available, using mock objects")


class MockReq:
    """模拟请求对象"""
    def __init__(self, rid, prompt_len, output_len):
        self.rid = rid
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = []
        self.extra_key = None
        self.prefix_indices = torch.tensor([], dtype=torch.long)
        self.last_node = None
        self.last_host_node = None
        self.host_hit_length = 0
        self.time_stats = type('obj', (object,), {'wait_queue_entry_time': time.time()})()
        self.priority = 0
        self.sampling_params = type('obj', (object,), {'max_new_tokens': output_len})()


class MockRadixCache:
    """模拟RadixCache"""
    def __init__(self):
        self.disable = False
        self.root_node = type('obj', (object,), {'children': {}})()

    def match_prefix(self, rid, key):
        # 模拟前缀匹配，返回随机长度的前缀
        prefix_len = np.random.randint(0, len(key.token_ids) // 2)
        indices = torch.tensor(list(range(prefix_len)), dtype=torch.long)
        return indices, self.root_node, None, 0

    def reset(self):
        pass

    def insert(self, key, value):
        pass


def generate_test_workload(
    num_requests: int,
    avg_prompt_len: int = 512,
    std_prompt_len: int = 128,
    avg_output_len: int = 128,
) -> List:
    """生成测试工作负载"""
    requests = []
    for i in range(num_requests):
        prompt_len = max(1, int(np.random.normal(avg_prompt_len, std_prompt_len)))
        output_len = max(1, int(np.random.normal(avg_output_len, 32)))

        if SGLANG_AVAILABLE:
            # 使用真实Req对象
            req = MockReq(i, prompt_len, output_len)
        else:
            req = MockReq(i, prompt_len, output_len)

        requests.append(req)

    return requests


def benchmark_policy(
    policy,
    workload_sizes: List[int],
    num_iterations: int = 10,
) -> dict:
    """
    对单个策略进行性能测试

    Args:
        policy: 策略实例
        workload_sizes: 测试的队列长度列表
        num_iterations: 每个大小重复测试次数

    Returns:
        包含性能数据的字典
    """
    results = {
        'workload_sizes': workload_sizes,
        'mean_times': [],
        'std_times': [],
        'p95_times': [],
        'p99_times': [],
    }

    for size in workload_sizes:
        times = []

        for _ in range(num_iterations):
            # 生成工作负载
            waiting_queue = generate_test_workload(size)

            # 计时
            start = time.perf_counter()
            policy.calc_priority(waiting_queue)
            elapsed = (time.perf_counter() - start) * 1000  # 转换为ms

            times.append(elapsed)

        # 统计
        results['mean_times'].append(np.mean(times))
        results['std_times'].append(np.std(times))
        results['p95_times'].append(np.percentile(times, 95))
        results['p99_times'].append(np.percentile(times, 99))

    return results


def visualize_results(results_dict: dict, output_path: str = "policy_performance.png"):
    """
    可视化性能对比结果

    Args:
        results_dict: {策略名称: benchmark结果}
        output_path: 输出图片路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 获取workload sizes（所有策略应该相同）
    workload_sizes = list(results_dict.values())[0]['workload_sizes']

    # 图1: 平均调度时间
    ax = axes[0, 0]
    for policy_name, results in results_dict.items():
        ax.plot(workload_sizes, results['mean_times'], marker='o', label=policy_name)
    ax.set_xlabel('Queue Length')
    ax.set_ylabel('Mean Schedule Time (ms)')
    ax.set_title('Mean Schedule Time vs Queue Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=128, color='red', linestyle='--', alpha=0.5, label='Original Threshold')

    # 图2: P95延迟
    ax = axes[0, 1]
    for policy_name, results in results_dict.items():
        ax.plot(workload_sizes, results['p95_times'], marker='s', label=policy_name)
    ax.set_xlabel('Queue Length')
    ax.set_ylabel('P95 Schedule Time (ms)')
    ax.set_title('P95 Schedule Time vs Queue Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=128, color='red', linestyle='--', alpha=0.5)

    # 图3: 性能比率（相对于Original）
    ax = axes[1, 0]
    if 'Original' in results_dict:
        baseline = np.array(results_dict['Original']['mean_times'])
        for policy_name, results in results_dict.items():
            if policy_name != 'Original':
                ratio = np.array(results['mean_times']) / baseline
                ax.plot(workload_sizes, ratio, marker='o', label=policy_name)
        ax.set_xlabel('Queue Length')
        ax.set_ylabel('Speedup vs Original')
        ax.set_title('Performance Ratio (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Baseline')
        ax.axvline(x=128, color='red', linestyle='--', alpha=0.5)

    # 图4: 性能断崖检测
    ax = axes[1, 1]
    for policy_name, results in results_dict.items():
        mean_times = np.array(results['mean_times'])
        # 计算相邻点的比率
        ratios = mean_times[1:] / mean_times[:-1]
        ax.plot(workload_sizes[1:], ratios, marker='o', label=policy_name)
    ax.set_xlabel('Queue Length')
    ax.set_ylabel('Time Ratio (Current/Previous)')
    ax.set_title('Performance Cliff Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Cliff Threshold (2x)')
    ax.axvline(x=128, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark scheduling policy performance")
    parser.add_argument('--workload-sizes', type=str, default='20,40,60,80,100,120,140,160,180,200',
                       help='Comma-separated list of queue sizes to test')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per workload size')
    parser.add_argument('--output-json', type=str, default='policy_benchmark_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--output-chart', type=str, default='policy_performance.png',
                       help='Output chart file')
    parser.add_argument('--policies', type=str, default='original,adaptive,sampling',
                       help='Comma-separated list of policies to test')

    args = parser.parse_args()

    # 解析参数
    workload_sizes = [int(x) for x in args.workload_sizes.split(',')]
    policy_names = args.policies.split(',')

    print("=" * 80)
    print("调度策略性能断崖验证实验")
    print("=" * 80)
    print(f"测试队列长度: {workload_sizes}")
    print(f"每个长度迭代次数: {args.iterations}")
    print(f"测试策略: {policy_names}")
    print("=" * 80)

    # 创建模拟的tree_cache
    tree_cache = MockRadixCache()

    # 准备策略
    policies = {}

    if 'original' in policy_names:
        if SGLANG_AVAILABLE:
            policies['Original'] = SchedulePolicy(
                policy='lpm',
                tree_cache=tree_cache,
                enable_hierarchical_cache=False,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
            )
        else:
            print("Warning: Skipping original policy (SGLang not available)")

    if 'adaptive' in policy_names:
        if SGLANG_AVAILABLE:
            policies['Adaptive'] = AdaptiveSchedulePolicy(
                policy='lpm',
                tree_cache=tree_cache,
                enable_hierarchical_cache=False,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
                adaptive_threshold=True,
            )
        else:
            print("Warning: Skipping adaptive policy (SGLang not available)")

    if 'sampling' in policy_names:
        if SGLANG_AVAILABLE:
            policies['Sampling'] = SamplingLPMPolicy(
                policy='lpm',
                tree_cache=tree_cache,
                enable_hierarchical_cache=False,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
                adaptive_threshold=True,
                sampling_ratio=0.3,
            )
        else:
            print("Warning: Skipping sampling policy (SGLang not available)")

    if not policies:
        print("Error: No policies to test!")
        return

    # 运行benchmark
    results_dict = {}
    for policy_name, policy in policies.items():
        print(f"\n测试策略: {policy_name}")
        results = benchmark_policy(policy, workload_sizes, args.iterations)
        results_dict[policy_name] = results

        # 打印结果摘要
        print(f"  队列长度范围: {min(workload_sizes)} - {max(workload_sizes)}")
        print(f"  平均调度时间: {np.mean(results['mean_times']):.2f} ms")
        print(f"  P95调度时间: {np.mean(results['p95_times']):.2f} ms")

    # 保存结果到JSON
    with open(args.output_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n结果已保存到: {args.output_json}")

    # 生成可视化
    try:
        visualize_results(results_dict, args.output_chart)
    except Exception as e:
        print(f"Warning: Failed to generate chart: {e}")

    # 性能对比分析
    print("\n" + "=" * 80)
    print("性能对比分析")
    print("=" * 80)

    if 'Original' in results_dict:
        baseline = results_dict['Original']
        for policy_name, results in results_dict.items():
            if policy_name != 'Original':
                # 计算平均加速比
                baseline_times = np.array(baseline['mean_times'])
                policy_times = np.array(results['mean_times'])
                speedup = baseline_times / policy_times
                avg_speedup = np.mean(speedup)

                print(f"\n{policy_name} vs Original:")
                print(f"  平均加速比: {avg_speedup:.2f}x")
                print(f"  最大加速比: {np.max(speedup):.2f}x (at queue={workload_sizes[np.argmax(speedup)]})")
                print(f"  最小加速比: {np.min(speedup):.2f}x (at queue={workload_sizes[np.argmin(speedup)]})")

                # 检测性能断崖
                baseline_ratios = baseline_times[1:] / baseline_times[:-1]
                policy_ratios = policy_times[1:] / policy_times[:-1]

                baseline_cliffs = np.where(baseline_ratios > 2.0)[0]
                policy_cliffs = np.where(policy_ratios > 2.0)[0]

                print(f"  原始策略性能断崖: {len(baseline_cliffs)} 处")
                if len(baseline_cliffs) > 0:
                    for cliff_idx in baseline_cliffs:
                        print(f"    - 队列长度 {workload_sizes[cliff_idx]} -> {workload_sizes[cliff_idx+1]}: "
                              f"{baseline_ratios[cliff_idx]:.2f}x增长")

                print(f"  {policy_name}策略性能断崖: {len(policy_cliffs)} 处")
                if len(policy_cliffs) > 0:
                    for cliff_idx in policy_cliffs:
                        print(f"    - 队列长度 {workload_sizes[cliff_idx]} -> {workload_sizes[cliff_idx+1]}: "
                              f"{policy_ratios[cliff_idx]:.2f}x增长")

    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
