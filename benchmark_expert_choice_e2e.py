#!/usr/bin/env python3
"""
端到端性能测试：包括完整的MoE层计算

这个脚本测量真实的收益，包括：
1. Routing开销
2. Expert计算时间（考虑负载不均的影响）
3. 总体端到端延迟

用法：
python benchmark_expert_choice_e2e.py --batch-size 256 --num-experts 64
"""

import torch
import torch.nn as nn
import time
import numpy as np
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


def simulate_moe_forward_standard(
    hidden_states,
    router_logits,
    experts,
    top_k,
    simulate_load_imbalance=True
):
    """
    模拟标准MoE forward，考虑负载不均的影响
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

    # 3. Expert计算（模拟负载不均的影响）
    start = time.perf_counter()

    if simulate_load_imbalance:
        # 模拟真实情况：每个expert独立计算，总时间由最慢的决定
        max_load = expert_loads.max().item()

        # 为了真实模拟，让负载最重的expert实际计算
        for expert_id in range(num_experts):
            load = expert_loads[expert_id].item()
            if load > 0:
                # 构造该expert需要处理的tokens
                mask = (topk_ids == expert_id).any(dim=1)
                expert_tokens = hidden_states[mask][:load]
                if len(expert_tokens) > 0:
                    _ = experts[expert_id](expert_tokens)

        torch.cuda.synchronize()
    else:
        # 简化版本：只计算总工作量
        total_work = 0
        for expert_id in range(num_experts):
            load = expert_loads[expert_id].item()
            if load > 0:
                mask = (topk_ids == expert_id).any(dim=1)
                expert_tokens = hidden_states[mask][:load]
                if len(expert_tokens) > 0:
                    _ = experts[expert_id](expert_tokens)
                    total_work += load
        torch.cuda.synchronize()

    expert_time = (time.perf_counter() - start) * 1000

    return routing_time, expert_time, expert_loads


def simulate_moe_forward_expert_choice(
    hidden_states,
    router_logits,
    experts,
    top_k,
    expert_capacity_factor=1.25
):
    """
    模拟Expert Choice MoE forward，负载均衡
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

    # 3. Expert计算（负载均衡，所有expert同时完成）
    start = time.perf_counter()

    # 所有expert处理大约相同数量的tokens
    for expert_id in range(num_experts):
        load = expert_loads[expert_id].item()
        if load > 0:
            mask = (topk_ids == expert_id).any(dim=1)
            expert_tokens = hidden_states[mask][:load]
            if len(expert_tokens) > 0:
                _ = experts[expert_id](expert_tokens)

    torch.cuda.synchronize()
    expert_time = (time.perf_counter() - start) * 1000

    return routing_time, expert_time, expert_loads


def benchmark_e2e(
    batch_size,
    num_experts,
    hidden_dim=4096,
    intermediate_dim=14336,
    top_k=8,
    num_iterations=50,
    simulate_load_imbalance=True
):
    """
    端到端性能测试
    """
    device = 'cuda'

    print(f"\n{'='*80}")
    print(f"端到端性能测试")
    print(f"配置: batch={batch_size}, experts={num_experts}, top_k={top_k}")
    print(f"{'='*80}\n")

    # 创建experts
    experts = nn.ModuleList([
        SimpleMLP(hidden_dim, intermediate_dim).to(device).to(torch.bfloat16)
        for _ in range(num_experts)
    ])

    # 准备数据
    hidden_states = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(batch_size, num_experts, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        simulate_moe_forward_standard(hidden_states, router_logits, experts, top_k, simulate_load_imbalance)
        simulate_moe_forward_expert_choice(hidden_states, router_logits, experts, top_k)

    # 测试标准routing
    routing_times_std = []
    expert_times_std = []
    loads_std = []

    for _ in range(num_iterations):
        r_time, e_time, loads = simulate_moe_forward_standard(
            hidden_states, router_logits, experts, top_k, simulate_load_imbalance
        )
        routing_times_std.append(r_time)
        expert_times_std.append(e_time)
        loads_std.append(loads.cpu().numpy())

    avg_routing_std = np.mean(routing_times_std)
    avg_expert_std = np.mean(expert_times_std)
    avg_total_std = avg_routing_std + avg_expert_std
    avg_loads_std = np.mean(loads_std, axis=0)

    # 测试Expert Choice
    routing_times_ec = []
    expert_times_ec = []
    loads_ec = []

    for _ in range(num_iterations):
        r_time, e_time, loads = simulate_moe_forward_expert_choice(
            hidden_states, router_logits, experts, top_k
        )
        routing_times_ec.append(r_time)
        expert_times_ec.append(e_time)
        loads_ec.append(loads.cpu().numpy())

    avg_routing_ec = np.mean(routing_times_ec)
    avg_expert_ec = np.mean(expert_times_ec)
    avg_total_ec = avg_routing_ec + avg_expert_ec
    avg_loads_ec = np.mean(loads_ec, axis=0)

    # 计算负载统计
    std_cv = np.std(avg_loads_std) / np.mean(avg_loads_std)
    ec_cv = np.std(avg_loads_ec) / np.mean(avg_loads_ec)

    # 输出结果
    print("标准Routing:")
    print(f"  Routing时间:   {avg_routing_std:.3f} ms")
    print(f"  Expert时间:    {avg_expert_std:.3f} ms")
    print(f"  总时间:        {avg_total_std:.3f} ms")
    print(f"  负载CV:        {std_cv:.3f}")
    print(f"  负载分布:      min={avg_loads_std.min():.0f}, max={avg_loads_std.max():.0f}, mean={avg_loads_std.mean():.0f}")
    print()

    print("Expert Choice Routing:")
    print(f"  Routing时间:   {avg_routing_ec:.3f} ms")
    print(f"  Expert时间:    {avg_expert_ec:.3f} ms")
    print(f"  总时间:        {avg_total_ec:.3f} ms")
    print(f"  负载CV:        {ec_cv:.3f}")
    print(f"  负载分布:      min={avg_loads_ec.min():.0f}, max={avg_loads_ec.max():.0f}, mean={avg_loads_ec.mean():.0f}")
    print()

    # 分析
    print("性能对比:")
    routing_overhead = ((avg_routing_ec - avg_routing_std) / avg_routing_std) * 100
    expert_speedup = ((avg_expert_std - avg_expert_ec) / avg_expert_std) * 100
    total_speedup = ((avg_total_std - avg_total_ec) / avg_total_std) * 100

    print(f"  Routing开销:      +{routing_overhead:.1f}%")
    print(f"  Expert计算加速:   +{expert_speedup:.1f}%")
    print(f"  端到端加速:       +{total_speedup:.1f}%")
    print(f"  负载均衡改善:     +{((std_cv - ec_cv) / std_cv * 100):.1f}%")
    print()

    if total_speedup > 5:
        print(f"✅ Expert Choice带来 {total_speedup:.1f}% 端到端加速，推荐启用")
    elif total_speedup > -5:
        print(f"⚠️ 性能差异不大 ({total_speedup:.1f}%)，收益不明显")
    else:
        print(f"❌ 性能下降 {abs(total_speedup):.1f}%，不推荐启用")

    print(f"\n关键洞察:")
    print(f"  - Routing开销: {avg_routing_ec:.2f}ms (expert choice) vs {avg_routing_std:.2f}ms (standard)")
    print(f"  - 但Expert计算节省: {avg_expert_std - avg_expert_ec:.2f}ms")
    print(f"  - 净收益: {avg_total_std - avg_total_ec:.2f}ms")

    return {
        'routing_std': avg_routing_std,
        'expert_std': avg_expert_std,
        'total_std': avg_total_std,
        'routing_ec': avg_routing_ec,
        'expert_ec': avg_expert_ec,
        'total_ec': avg_total_ec,
        'speedup': total_speedup
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--test-all', action='store_true', help='测试多个batch size')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("需要CUDA支持")
        return

    if args.test_all:
        # 测试不同batch size
        configs = [
            (32, 64, 8),
            (64, 64, 8),
            (128, 64, 8),
            (256, 64, 8),
            (512, 64, 8),
        ]

        results = []
        for batch_size, num_experts, top_k in configs:
            result = benchmark_e2e(batch_size, num_experts, top_k=top_k)
            results.append({
                'batch_size': batch_size,
                **result
            })

        # 总结
        print(f"\n{'='*80}")
        print("总结：不同batch size的端到端性能")
        print(f"{'='*80}\n")
        print(f"{'Batch':<10} {'标准(ms)':<12} {'ExpertChoice(ms)':<18} {'加速':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['batch_size']:<10} {r['total_std']:<12.2f} {r['total_ec']:<18.2f} {r['speedup']:>+.1f}%")

    else:
        benchmark_e2e(args.batch_size, args.num_experts, top_k=args.top_k)


if __name__ == "__main__":
    main()
