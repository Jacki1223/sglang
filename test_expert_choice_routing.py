#!/usr/bin/env python3
"""
Test script for Expert Choice Routing implementation.

This script demonstrates how to use the new expert choice routing feature
for better load balancing in MoE models.
"""

import torch
from sglang.srt.layers.moe.topk import expert_choice_topk, fused_topk


def test_expert_choice_vs_standard():
    """Compare expert choice routing with standard token-choose-expert routing."""

    # Setup
    num_tokens = 128
    hidden_dim = 512
    num_experts = 8
    top_k = 2

    # Create test data
    hidden_states = torch.randn(num_tokens, hidden_dim, device='cuda')
    router_logits = torch.randn(num_tokens, num_experts, device='cuda')

    print("=" * 80)
    print("Expert Choice Routing vs Standard Routing Comparison")
    print("=" * 80)
    print(f"Number of tokens: {num_tokens}")
    print(f"Number of experts: {num_experts}")
    print(f"Top-k per token: {top_k}")
    print()

    # Test 1: Standard routing (token chooses expert)
    print("1. Standard Routing (Token-Choose-Expert):")
    print("-" * 80)

    topk_weights_std, topk_ids_std = fused_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=top_k,
        renormalize=True,
        scoring_func="softmax"
    )

    # Calculate expert load distribution
    expert_loads_std = torch.zeros(num_experts, dtype=torch.int32, device='cuda')
    for expert_id in range(num_experts):
        expert_loads_std[expert_id] = (topk_ids_std == expert_id).sum()

    print(f"Expert load distribution: {expert_loads_std.cpu().numpy()}")
    print(f"Max load: {expert_loads_std.max().item()}")
    print(f"Min load: {expert_loads_std.min().item()}")
    print(f"Std dev: {expert_loads_std.float().std().item():.2f}")
    print(f"Load imbalance ratio: {expert_loads_std.max().item() / (expert_loads_std.float().mean().item() + 1e-6):.2f}x")
    print()

    # Test 2: Expert choice routing (expert chooses token)
    print("2. Expert Choice Routing (Expert-Choose-Token):")
    print("-" * 80)

    topk_weights_ec, topk_ids_ec = expert_choice_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=top_k,
        renormalize=True,
        expert_capacity_factor=1.25,
        scoring_func="softmax"
    )

    # Calculate expert load distribution
    expert_loads_ec = torch.zeros(num_experts, dtype=torch.int32, device='cuda')
    for expert_id in range(num_experts):
        expert_loads_ec[expert_id] = (topk_ids_ec == expert_id).sum()

    print(f"Expert load distribution: {expert_loads_ec.cpu().numpy()}")
    print(f"Max load: {expert_loads_ec.max().item()}")
    print(f"Min load: {expert_loads_ec.min().item()}")
    print(f"Std dev: {expert_loads_ec.float().std().item():.2f}")
    print(f"Load imbalance ratio: {expert_loads_ec.max().item() / (expert_loads_ec.float().mean().item() + 1e-6):.2f}x")
    print()

    # Compare
    print("3. Comparison:")
    print("-" * 80)
    std_imbalance = expert_loads_std.float().std().item()
    ec_imbalance = expert_loads_ec.float().std().item()
    improvement = ((std_imbalance - ec_imbalance) / std_imbalance) * 100

    print(f"Load balancing improvement: {improvement:.1f}%")
    print(f"Expected load per expert: {num_tokens * top_k / num_experts:.1f}")
    print()

    print("=" * 80)
    print("Conclusion:")
    print("-" * 80)
    if ec_imbalance < std_imbalance:
        print("✓ Expert Choice Routing provides better load balancing!")
    else:
        print("Note: Results may vary based on router_logits distribution")
    print("=" * 80)


def test_expert_choice_direct():
    """Direct test of expert_choice_topk function."""

    print("\n" + "=" * 80)
    print("Direct Expert Choice TopK Test")
    print("=" * 80)

    num_tokens = 64
    num_experts = 4
    hidden_dim = 256
    top_k = 2

    hidden_states = torch.randn(num_tokens, hidden_dim, device='cuda')
    router_logits = torch.randn(num_tokens, num_experts, device='cuda')

    topk_weights, topk_ids = expert_choice_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=top_k,
        renormalize=True,
        expert_capacity_factor=1.25,
        scoring_func="softmax"
    )

    print(f"Input shape: {router_logits.shape}")
    print(f"Output topk_weights shape: {topk_weights.shape}")
    print(f"Output topk_ids shape: {topk_ids.shape}")
    print()

    # Verify each token has exactly top_k experts
    valid_assignments = (topk_ids >= 0).sum(dim=1)
    print(f"Tokens with {top_k} expert assignments: {(valid_assignments == top_k).sum().item()}/{num_tokens}")

    # Show expert load distribution
    expert_loads = torch.zeros(num_experts, dtype=torch.int32, device='cuda')
    for expert_id in range(num_experts):
        expert_loads[expert_id] = (topk_ids == expert_id).sum()

    print(f"Expert loads: {expert_loads.cpu().numpy()}")
    print(f"Average load: {expert_loads.float().mean().item():.1f}")
    print(f"Expected load: {num_tokens * top_k / num_experts:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Running tests...\n")

        # Run direct test
        test_expert_choice_direct()

        # Run comparison test
        test_expert_choice_vs_standard()

    else:
        print("CUDA is not available. Please run on a CUDA-enabled device.")
        print("\nTo use expert choice routing in your model:")
        print("1. Set use_expert_choice=True when creating TopK layer")
        print("2. Optionally adjust expert_capacity_factor (default: 1.25)")
        print("\nExample:")
        print("  topk = TopK(")
        print("      top_k=2,")
        print("      use_expert_choice=True,  # Enable expert choice routing")
        print("      expert_capacity_factor=1.25,  # Expert capacity multiplier")
        print("      renormalize=True,")
        print("      scoring_func='softmax'")
        print("  )")
