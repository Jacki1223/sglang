#!/usr/bin/env python3
# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for expert load balancer."""

import os
import unittest

import torch

from sglang.srt.layers.moe.expert_load_balancer import (
    ExpertLoadBalancer,
    LoadBalancingConfig,
    LoadBalancingStrategy,
    create_load_balancer_from_env,
)


class TestExpertLoadBalancer(unittest.TestCase):
    """Test cases for ExpertLoadBalancer."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_no_balancing(self):
        """Test that no balancing leaves topk_ids unchanged."""
        config = LoadBalancingConfig(strategy=LoadBalancingStrategy.NONE)
        balancer = ExpertLoadBalancer(num_experts=8, topk=2, config=config)

        topk_ids = torch.randint(0, 8, (32, 2), dtype=torch.int32, device=self.device)
        original_topk_ids = topk_ids.clone()

        balanced_topk_ids, _ = balancer.balance(topk_ids)

        self.assertTrue(torch.equal(balanced_topk_ids, original_topk_ids))

    def test_local_balancing_reduces_imbalance(self):
        """Test that local balancing reduces imbalance."""
        config = LoadBalancingConfig(
            strategy=LoadBalancingStrategy.LOCAL,
            imbalance_threshold=1.3,
            redirect_fraction=0.3,
        )
        balancer = ExpertLoadBalancer(num_experts=8, topk=2, config=config)

        # Create highly imbalanced topk_ids
        # All tokens go to first 2 experts
        topk_ids = torch.zeros(128, 2, dtype=torch.int32, device=self.device)
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 1

        # Compute initial imbalance
        initial_counts = torch.bincount(topk_ids.view(-1), minlength=8)
        initial_max = initial_counts.max().item()
        initial_avg = topk_ids.numel() / 8
        initial_imbalance = initial_max / initial_avg

        # Apply balancing
        balanced_topk_ids, _ = balancer.balance(topk_ids)

        # Compute final imbalance
        final_counts = torch.bincount(balanced_topk_ids.view(-1), minlength=8)
        final_max = final_counts.max().item()
        final_imbalance = final_max / initial_avg

        # Verify imbalance is reduced
        self.assertLess(final_imbalance, initial_imbalance)

    def test_adaptive_strategy(self):
        """Test adaptive strategy selection."""
        config = LoadBalancingConfig(
            strategy=LoadBalancingStrategy.ADAPTIVE,
            imbalance_threshold=1.3,
        )
        balancer = ExpertLoadBalancer(num_experts=8, topk=2, config=config)

        # Low imbalance - should not trigger rebalancing
        topk_ids_balanced = torch.randint(0, 8, (64, 2), dtype=torch.int32, device=self.device)
        result, _ = balancer.balance(topk_ids_balanced)
        self.assertTrue(torch.equal(result, topk_ids_balanced))

        # High imbalance - should trigger rebalancing
        topk_ids_imbalanced = torch.zeros(128, 2, dtype=torch.int32, device=self.device)
        topk_ids_imbalanced[:, 0] = 0
        topk_ids_imbalanced[:, 1] = 1

        result_imbalanced, _ = balancer.balance(topk_ids_imbalanced)
        self.assertFalse(torch.equal(result_imbalanced, topk_ids_imbalanced))

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        config = LoadBalancingConfig(
            strategy=LoadBalancingStrategy.LOCAL,
            imbalance_threshold=1.3,
        )
        balancer = ExpertLoadBalancer(num_experts=8, topk=2, config=config)

        # Create imbalanced topk_ids that will trigger rebalancing
        topk_ids = torch.zeros(128, 2, dtype=torch.int32, device=self.device)
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 1

        # Apply balancing
        balancer.balance(topk_ids)

        # Check statistics
        stats = balancer.get_statistics()
        self.assertEqual(stats["forward_count"], 1)
        self.assertEqual(stats["rebalance_count"], 1)
        self.assertEqual(stats["rebalance_rate"], 1.0)

    def test_ep_balancing(self):
        """Test global EP balancing."""
        config = LoadBalancingConfig(
            strategy=LoadBalancingStrategy.GLOBAL_EP,
            imbalance_threshold=1.3,
            redirect_fraction=0.3,
        )

        # Simulate 2 EP ranks, 4 experts per rank
        balancer = ExpertLoadBalancer(
            num_experts=8, topk=2, config=config, ep_size=2, ep_rank=0
        )

        # Create imbalance across ranks
        # All tokens go to experts in rank 0 (experts 0-3)
        topk_ids = torch.zeros(128, 2, dtype=torch.int32, device=self.device)
        topk_ids[:, 0] = torch.randint(0, 4, (128,))
        topk_ids[:, 1] = torch.randint(0, 4, (128,))

        # Compute initial rank load
        initial_counts = torch.bincount(topk_ids.view(-1), minlength=8)
        initial_rank0_load = initial_counts[0:4].sum().item()
        initial_rank1_load = initial_counts[4:8].sum().item()

        # Apply balancing
        balanced_topk_ids, _ = balancer.balance(topk_ids)

        # Compute final rank load
        final_counts = torch.bincount(balanced_topk_ids.view(-1), minlength=8)
        final_rank0_load = final_counts[0:4].sum().item()
        final_rank1_load = final_counts[4:8].sum().item()

        # Verify that rank 1 gets some load
        self.assertGreater(final_rank1_load, initial_rank1_load)

    def test_create_from_env(self):
        """Test creating balancer from environment variables."""
        # Set environment variables
        os.environ["SGLANG_MOE_LOAD_BALANCE_STRATEGY"] = "adaptive"
        os.environ["SGLANG_MOE_LOAD_BALANCE_THRESHOLD"] = "1.5"
        os.environ["SGLANG_MOE_LOAD_BALANCE_REDIRECT"] = "0.25"

        balancer = create_load_balancer_from_env(num_experts=8, topk=2)

        self.assertEqual(balancer.config.strategy, LoadBalancingStrategy.ADAPTIVE)
        self.assertEqual(balancer.config.imbalance_threshold, 1.5)
        self.assertEqual(balancer.config.redirect_fraction, 0.25)

        # Clean up
        del os.environ["SGLANG_MOE_LOAD_BALANCE_STRATEGY"]
        del os.environ["SGLANG_MOE_LOAD_BALANCE_THRESHOLD"]
        del os.environ["SGLANG_MOE_LOAD_BALANCE_REDIRECT"]

    def test_weights_preservation(self):
        """Test that topk_weights are preserved."""
        config = LoadBalancingConfig(strategy=LoadBalancingStrategy.LOCAL)
        balancer = ExpertLoadBalancer(num_experts=8, topk=2, config=config)

        topk_ids = torch.randint(0, 8, (32, 2), dtype=torch.int32, device=self.device)
        topk_weights = torch.rand(32, 2, dtype=torch.float32, device=self.device)

        original_weights = topk_weights.clone()

        _, balanced_weights = balancer.balance(topk_ids, topk_weights)

        # Weights should be returned unchanged
        if balanced_weights is not None:
            self.assertTrue(torch.allclose(balanced_weights, original_weights))


if __name__ == "__main__":
    unittest.main()
