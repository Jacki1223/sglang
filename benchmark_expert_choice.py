#!/usr/bin/env python3
"""
测试Expert Choice Routing在不同配置下的性能

用法：
python benchmark_expert_choice.py --batch-size 256 --num-experts 64 --ep-size 4
"""

import torch
import time
import numpy as np
from sglang.srt.layers.moe.topk import expert_choice_topk, fused_topk


def benchmark_routing(
    batch_size,
    num_experts,
    hidden_dim=4096,
    top_k=8,
    num_iterations=100
):
    """
    对比标准routing和expert choice routing的性能
    """
    device = 'cuda'

    # 准备数据
    hidden_states = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(batch_size, num_experts, device=device, dtype=torch.bfloat16)

    print(f"\n{'='*80}")
    print(f"配置: batch_size={batch_size}, num_experts={num_experts}, top_k={top_k}")
    print(f"{'='*80}\n")

    # Warmup
    for _ in range(10):
        _ = fused_topk(hidden_states, router_logits, top_k, True, scoring_func="softmax")
        _ = expert_choice_topk(hidden_states, router_logits, top_k, True, scoring_func="softmax")

    torch.cuda.synchronize()

    # 测试标准routing
    start = time.perf_counter()
    for _ in range(num_iterations):
        topk_weights_std, topk_ids_std = fused_topk(
            hidden_states, router_logits, top_k, True, scoring_func="softmax"
        )
    torch.cuda.synchronize()
    time_standard = (time.perf_counter() - start) / num_iterations * 1000

    # 测量负载不均
    expert_loads_std = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for expert_id in range(num_experts):
        expert_loads_std[expert_id] = (topk_ids_std == expert_id).sum()

    std_cv = expert_loads_std.float().std() / expert_loads_std.float().mean()
    std_max = expert_loads_std.max().item()
    std_min = expert_loads_std.min().item()

    # 测试expert choice
    start = time.perf_counter()
    for _ in range(num_iterations):
        topk_weights_ec, topk_ids_ec = expert_choice_topk(
            hidden_states, router_logits, top_k, True,
            expert_capacity_factor=1.25, scoring_func="softmax"
        )
    torch.cuda.synchronize()
    time_expert_choice = (time.perf_counter() - start) / num_iterations * 1000

    # 测量负载不均
    expert_loads_ec = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for expert_id in range(num_experts):
        expert_loads_ec[expert_id] = (topk_ids_ec == expert_id).sum()

    ec_cv = expert_loads_ec.float().std() / expert_loads_ec.float().mean()
    ec_max = expert_loads_ec.max().item()
    ec_min = expert_loads_ec.min().item()

    # 输出结果
    print(f"标准Routing:")
    print(f"  延迟: {time_standard:.3f} ms")
    print(f"  负载CV: {std_cv:.3f}")
    print(f"  负载范围: [{std_min}, {std_max}]")
    print()

    print(f"Expert Choice Routing:")
    print(f"  延迟: {time_expert_choice:.3f} ms")
    print(f"  负载CV: {ec_cv:.3f}")
    print(f"  负载范围: [{ec_min}, {ec_max}]")
    print()

    overhead = (time_expert_choice - time_standard) / time_standard * 100
    balance_improvement = (std_cv - ec_cv) / std_cv * 100

    print(f"性能对比:")
    print(f"  时间开销: +{overhead:.1f}%")
    print(f"  负载均衡改善: +{balance_improvement:.1f}%")

    # 评估是否值得启用
    expected_load_benefit = 0
    if std_cv > 0.3:  # 严重不均
        expected_load_benefit = 20 * (std_cv - ec_cv)  # 经验公式
    elif std_cv > 0.15:  # 中等不均
        expected_load_benefit = 10 * (std_cv - ec_cv)

    net_benefit = expected_load_benefit - overhead

    print(f"\n预估净收益: {net_benefit:.1f}%")

    if net_benefit > 5:
        print(f"✅ 推荐启用Expert Choice Routing")
    elif net_benefit > -5:
        print(f"⚠️ 收益不明显，需要实际测试")
    else:
        print(f"❌ 不推荐启用，会降低性能")

    return {
        'time_standard': time_standard,
        'time_expert_choice': time_expert_choice,
        'std_cv': std_cv,
        'ec_cv': ec_cv,
        'net_benefit': net_benefit
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("需要CUDA支持")
        return

    # 测试不同配置
    configs = [
        (16, 64, 8),    # 小batch
        (64, 64, 8),    # 中batch
        (128, 64, 8),   # 大batch
        (256, 64, 8),   # 超大batch
    ]

    if args.batch_size:
        configs = [(args.batch_size, args.num_experts, args.top_k)]

    results = []
    for batch_size, num_experts, top_k in configs:
        result = benchmark_routing(batch_size, num_experts, top_k=top_k)
        results.append({
            'batch_size': batch_size,
            **result
        })

    # 总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}\n")

    for r in results:
        print(f"Batch={r['batch_size']}: "
              f"开销+{(r['time_expert_choice']/r['time_standard']-1)*100:.0f}%, "
              f"净收益{r['net_benefit']:.0f}%")


if __name__ == "__main__":
    main()
