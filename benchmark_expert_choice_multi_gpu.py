#!/usr/bin/env python3
"""
多GPU Expert Parallelism场景下的性能测试

这个脚本模拟真实的EP场景：
1. Experts分片到多个GPU
2. 每个GPU并行处理其负责的experts
3. 总时间 = max(所有GPU的时间) ← 关键！
4. 负载不均会导致GPU等待

用法：
python benchmark_expert_choice_multi_gpu.py --batch-size 512 --num-experts 64 --ep-size 4
"""

import torch
import torch.nn as nn
import time
import numpy as np
import argparse
from sglang.srt.layers.moe.topk import expert_choice_topk, fused_topk


class SimpleMLP(nn.Module):
    """简单的expert MLP"""
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.up(x)))


def simulate_ep_expert_computation(expert_loads, experts, hidden_states, topk_ids, ep_size):
    """
    模拟Expert Parallelism场景下的expert计算

    关键：
    - Experts被分片到ep_size个GPU
    - 每个GPU并行处理其负责的experts
    - 总时间 = max(所有GPU的时间)
    """
    num_experts = len(experts)
    experts_per_gpu = num_experts // ep_size

    # 每个GPU的计算时间
    gpu_times = []
    gpu_loads = []

    for gpu_id in range(ep_size):
        # 这个GPU负责的experts
        start_expert = gpu_id * experts_per_gpu
        end_expert = start_expert + experts_per_gpu

        # 计算这个GPU的总负载
        gpu_load = expert_loads[start_expert:end_expert].sum().item()
        gpu_loads.append(gpu_load)

        # 模拟这个GPU的计算时间
        start = time.perf_counter()

        for expert_id in range(start_expert, end_expert):
            load = expert_loads[expert_id].item()
            if load > 0:
                # 构造该expert需要处理的tokens
                mask = (topk_ids == expert_id).any(dim=1)
                expert_tokens = hidden_states[mask][:load]
                if len(expert_tokens) > 0:
                    _ = experts[expert_id](expert_tokens)

        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) * 1000
        gpu_times.append(gpu_time)

    # 关键：总时间由最慢的GPU决定！
    total_time = max(gpu_times)

    return total_time, gpu_times, gpu_loads


def simulate_moe_forward_standard_ep(
    hidden_states,
    router_logits,
    experts,
    top_k,
    ep_size
):
    """
    模拟标准MoE + Expert Parallelism
    """
    batch_size, hidden_dim = hidden_states.shape
    num_experts = len(experts)
    device = hidden_states.device

    # 1. Routing (标准)
    start = time.perf_counter()
    topk_weights, topk_ids = fused_topk(
        hidden_states, router_logits, top_k, True, scoring_func="softmax"
    )
    torch.cuda.synchronize()
    routing_time = (time.perf_counter() - start) * 1000

    # 2. 统计每个expert的负载
    expert_loads = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for expert_id in range(num_experts):
        expert_loads[expert_id] = (topk_ids == expert_id).sum()

    # 3. Expert计算（EP场景）
    expert_time, gpu_times, gpu_loads = simulate_ep_expert_computation(
        expert_loads, experts, hidden_states, topk_ids, ep_size
    )

    return routing_time, expert_time, expert_loads, gpu_times, gpu_loads


def simulate_moe_forward_expert_choice_ep(
    hidden_states,
    router_logits,
    experts,
    top_k,
    ep_size,
    expert_capacity_factor=1.25
):
    """
    模拟Expert Choice + Expert Parallelism
    """
    batch_size, hidden_dim = hidden_states.shape
    num_experts = len(experts)
    device = hidden_states.device

    # 1. Routing (Expert Choice)
    start = time.perf_counter()
    topk_weights, topk_ids = expert_choice_topk(
        hidden_states, router_logits, top_k, True,
        expert_capacity_factor=expert_capacity_factor,
        scoring_func="softmax"
    )
    torch.cuda.synchronize()
    routing_time = (time.perf_counter() - start) * 1000

    # 2. 统计每个expert的负载
    expert_loads = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for expert_id in range(num_experts):
        expert_loads[expert_id] = (topk_ids == expert_id).sum()

    # 3. Expert计算（EP场景）
    expert_time, gpu_times, gpu_loads = simulate_ep_expert_computation(
        expert_loads, experts, hidden_states, topk_ids, ep_size
    )

    return routing_time, expert_time, expert_loads, gpu_times, gpu_loads


def benchmark_multi_gpu_ep(
    batch_size,
    num_experts,
    ep_size,
    hidden_dim=4096,
    intermediate_dim=14336,
    top_k=8,
    num_iterations=50,
    expert_capacity_factor=1.25,
):
    """
    测试多GPU EP场景下的性能
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建experts
    experts = nn.ModuleList([
        SimpleMLP(hidden_dim, intermediate_dim).to(device)
        for _ in range(num_experts)
    ])

    # 准备输入
    hidden_states = torch.randn(batch_size, hidden_dim, device=device)
    router_logits = torch.randn(batch_size, num_experts, device=device)

    # Warmup
    for _ in range(10):
        _ = simulate_moe_forward_standard_ep(
            hidden_states, router_logits, experts, top_k, ep_size
        )

    # 测试标准routing
    routing_times_std = []
    expert_times_std = []

    for _ in range(num_iterations):
        routing_time, expert_time, expert_loads_std, gpu_times_std, gpu_loads_std = \
            simulate_moe_forward_standard_ep(
                hidden_states, router_logits, experts, top_k, ep_size
            )
        routing_times_std.append(routing_time)
        expert_times_std.append(expert_time)

    # 测试Expert Choice
    routing_times_ec = []
    expert_times_ec = []

    for _ in range(num_iterations):
        routing_time, expert_time, expert_loads_ec, gpu_times_ec, gpu_loads_ec = \
            simulate_moe_forward_expert_choice_ep(
                hidden_states, router_logits, experts, top_k, ep_size,
                expert_capacity_factor
            )
        routing_times_ec.append(routing_time)
        expert_times_ec.append(expert_time)

    # 计算统计
    std_routing = np.mean(routing_times_std)
    std_expert = np.mean(expert_times_std)
    std_total = std_routing + std_expert

    ec_routing = np.mean(routing_times_ec)
    ec_expert = np.mean(expert_times_ec)
    ec_total = ec_routing + ec_expert

    # 负载统计
    std_loads = expert_loads_std.cpu().numpy()
    ec_loads = expert_loads_ec.cpu().numpy()

    std_cv = np.std(std_loads) / (np.mean(std_loads) + 1e-6)
    ec_cv = np.std(ec_loads) / (np.mean(ec_loads) + 1e-6)

    # 打印结果
    print("\n" + "="*80)
    print(f"多GPU Expert Parallelism性能测试")
    print(f"配置: batch={batch_size}, experts={num_experts}, top_k={top_k}, EP_size={ep_size}")
    print("="*80)

    print(f"\n标准Routing:")
    print(f"  Routing时间:   {std_routing:.3f} ms")
    print(f"  Expert时间:    {std_expert:.3f} ms")
    print(f"  总时间:        {std_total:.3f} ms")
    print(f"  负载CV:        {std_cv:.3f}")
    print(f"  负载分布:      min={std_loads.min()}, max={std_loads.max()}, mean={std_loads.mean():.1f}")
    print(f"\n  各GPU负载:")
    for i, (gpu_time, gpu_load) in enumerate(zip(gpu_times_std, gpu_loads_std)):
        print(f"    GPU {i}: {gpu_load} tokens, {gpu_time:.3f} ms")
    print(f"  GPU负载CV:     {np.std(gpu_loads_std) / (np.mean(gpu_loads_std) + 1e-6):.3f}")
    print(f"  最慢GPU:       {max(gpu_times_std):.3f} ms (决定总时间)")

    print(f"\nExpert Choice Routing:")
    print(f"  Routing时间:   {ec_routing:.3f} ms")
    print(f"  Expert时间:    {ec_expert:.3f} ms")
    print(f"  总时间:        {ec_total:.3f} ms")
    print(f"  负载CV:        {ec_cv:.3f}")
    print(f"  负载分布:      min={ec_loads.min()}, max={ec_loads.max()}, mean={ec_loads.mean():.1f}")
    print(f"\n  各GPU负载:")
    for i, (gpu_time, gpu_load) in enumerate(zip(gpu_times_ec, gpu_loads_ec)):
        print(f"    GPU {i}: {gpu_load} tokens, {gpu_time:.3f} ms")
    print(f"  GPU负载CV:     {np.std(gpu_loads_ec) / (np.mean(gpu_loads_ec) + 1e-6):.3f}")
    print(f"  最慢GPU:       {max(gpu_times_ec):.3f} ms (决定总时间)")

    print(f"\n性能对比:")
    routing_overhead = (ec_routing / std_routing - 1) * 100
    expert_speedup = (1 - ec_expert / std_expert) * 100
    total_speedup = (1 - ec_total / std_total) * 100
    load_balance_improvement = (1 - ec_cv / (std_cv + 1e-6)) * 100
    gpu_balance_improvement = (
        1 - (np.std(gpu_loads_ec) / (np.mean(gpu_loads_ec) + 1e-6)) /
        (np.std(gpu_loads_std) / (np.mean(gpu_loads_std) + 1e-6) + 1e-6)
    ) * 100

    print(f"  Routing开销:      {routing_overhead:+.1f}%")
    print(f"  Expert计算加速:   {expert_speedup:+.1f}%")
    print(f"  端到端加速:       {total_speedup:+.1f}%")
    print(f"  负载均衡改善:     {load_balance_improvement:+.1f}%")
    print(f"  GPU负载均衡改善:  {gpu_balance_improvement:+.1f}%")

    if total_speedup > 5:
        print(f"\n✅ Expert Choice带来 {total_speedup:.1f}% 端到端加速，推荐启用")
    elif total_speedup > 0:
        print(f"\n⚠️ 性能改善较小 ({total_speedup:.1f}%)，收益不明显")
    else:
        print(f"\n❌ 性能下降 ({total_speedup:.1f}%)，不推荐启用")

    print(f"\n关键洞察:")
    print(f"  - Routing开销: {ec_routing:.2f}ms (expert choice) vs {std_routing:.2f}ms (standard)")
    print(f"  - Expert计算节省: {std_expert - ec_expert:.2f}ms")
    print(f"  - 净收益: {std_total - ec_total:.2f}ms")
    print(f"  - GPU等待时间节省: {max(gpu_times_std) - max(gpu_times_ec):.2f}ms")
    print()


def main():
    parser = argparse.ArgumentParser(description="多GPU EP场景性能测试")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--ep-size", type=int, default=4, help="Expert Parallelism size (模拟的GPU数量)")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--intermediate-dim", type=int, default=14336)
    parser.add_argument("--expert-capacity-factor", type=float, default=1.25)
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--test-all", action="store_true", help="测试多种配置")

    args = parser.parse_args()

    if args.test_all:
        configs = [
            (256, 64, 2),  # 小batch, EP=2
            (512, 64, 4),  # 中batch, EP=4
            (1024, 64, 4), # 大batch, EP=4
            (1024, 64, 8), # 大batch, EP=8
        ]
        for batch_size, num_experts, ep_size in configs:
            benchmark_multi_gpu_ep(
                batch_size=batch_size,
                num_experts=num_experts,
                ep_size=ep_size,
                hidden_dim=args.hidden_dim,
                intermediate_dim=args.intermediate_dim,
                top_k=args.top_k,
                num_iterations=args.num_iterations,
                expert_capacity_factor=args.expert_capacity_factor,
            )
    else:
        benchmark_multi_gpu_ep(
            batch_size=args.batch_size,
            num_experts=args.num_experts,
            ep_size=args.ep_size,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            top_k=args.top_k,
            num_iterations=args.num_iterations,
            expert_capacity_factor=args.expert_capacity_factor,
        )


if __name__ == "__main__":
    main()
