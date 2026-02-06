"""
Tests for the fused Activation+GEMM2 MoE kernel.

Verifies that the fused kernel (Activation + GEMM2 in one pass)
produces results matching the original separate Activation + GEMM2 path.
The fused kernel reads gate and up halves from intermediate_cache1,
applies activation in registers during A-tile loading, then performs
the standard GEMM2 dot product.
"""

import os
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_experts_impl,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class TestFusedMoEActGemm2(unittest.TestCase):
    """Test fused Activation+GEMM2 against the original separate path."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _run_fused_experts(
        self,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        use_fused=True,
    ):
        """Run fused_experts_impl with or without the fused Act+GEMM2 path."""
        os.environ["SGLANG_FUSED_MOE_ACT_GEMM2"] = "1" if use_fused else "0"
        # Reimport to pick up env change
        import importlib

        import sglang.srt.layers.moe.fused_moe_triton.fused_moe as mod

        importlib.reload(mod)
        return mod.fused_experts_impl(
            hidden_states.clone(),
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation=activation,
            is_gated=True,
        )

    def _create_moe_inputs(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        topk,
        dtype=torch.bfloat16,
    ):
        """Create random MoE inputs."""
        device = "cuda"
        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=dtype, device=device
        )
        w1 = torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        w2 = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=dtype,
            device=device,
        )

        router_logits = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device=device
        )
        scores = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(scores, topk)
        topk_weights = topk_weights.to(torch.float32)

        return hidden_states, w1, w2, topk_weights, topk_ids

    def _assert_close(self, result_fused, result_original, dtype):
        """Assert results are close within tolerance for the given dtype."""
        if dtype == torch.bfloat16:
            rtol, atol = 1e-2, 1e-2
        elif dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-3, 1e-3

        torch.testing.assert_close(
            result_fused,
            result_original,
            rtol=rtol,
            atol=atol,
            msg=f"Fused and original results differ (dtype={dtype})",
        )

    def test_basic_silu_bf16(self):
        """Test basic SiLU activation with BF16."""
        torch.manual_seed(42)
        inputs = self._create_moe_inputs(
            num_tokens=32,
            num_experts=8,
            hidden_size=256,
            intermediate_size=512,
            topk=2,
            dtype=torch.bfloat16,
        )

        result_original = self._run_fused_experts(*inputs, activation="silu", use_fused=False)
        result_fused = self._run_fused_experts(*inputs, activation="silu", use_fused=True)
        self._assert_close(result_fused, result_original, torch.bfloat16)

    def test_basic_gelu_bf16(self):
        """Test basic GELU activation with BF16."""
        torch.manual_seed(42)
        inputs = self._create_moe_inputs(
            num_tokens=32,
            num_experts=8,
            hidden_size=256,
            intermediate_size=512,
            topk=2,
            dtype=torch.bfloat16,
        )

        result_original = self._run_fused_experts(*inputs, activation="gelu", use_fused=False)
        result_fused = self._run_fused_experts(*inputs, activation="gelu", use_fused=True)
        self._assert_close(result_fused, result_original, torch.bfloat16)

    def test_basic_silu_fp16(self):
        """Test basic SiLU activation with FP16."""
        torch.manual_seed(42)
        inputs = self._create_moe_inputs(
            num_tokens=32,
            num_experts=8,
            hidden_size=256,
            intermediate_size=512,
            topk=2,
            dtype=torch.float16,
        )

        result_original = self._run_fused_experts(*inputs, activation="silu", use_fused=False)
        result_fused = self._run_fused_experts(*inputs, activation="silu", use_fused=True)
        self._assert_close(result_fused, result_original, torch.float16)

    def test_single_token(self):
        """Test with a single token (decode scenario)."""
        torch.manual_seed(42)
        inputs = self._create_moe_inputs(
            num_tokens=1,
            num_experts=8,
            hidden_size=256,
            intermediate_size=512,
            topk=2,
            dtype=torch.bfloat16,
        )

        result_original = self._run_fused_experts(*inputs, activation="silu", use_fused=False)
        result_fused = self._run_fused_experts(*inputs, activation="silu", use_fused=True)
        self._assert_close(result_fused, result_original, torch.bfloat16)

    def test_large_batch(self):
        """Test with a large batch (prefill scenario)."""
        torch.manual_seed(42)
        inputs = self._create_moe_inputs(
            num_tokens=1024,
            num_experts=8,
            hidden_size=256,
            intermediate_size=512,
            topk=2,
            dtype=torch.bfloat16,
        )

        result_original = self._run_fused_experts(*inputs, activation="silu", use_fused=False)
        result_fused = self._run_fused_experts(*inputs, activation="silu", use_fused=True)
        self._assert_close(result_fused, result_original, torch.bfloat16)

    def test_many_experts(self):
        """Test with many experts (e.g. DeepSeek-like)."""
        torch.manual_seed(42)
        inputs = self._create_moe_inputs(
            num_tokens=16,
            num_experts=64,
            hidden_size=256,
            intermediate_size=256,
            topk=6,
            dtype=torch.bfloat16,
        )

        result_original = self._run_fused_experts(*inputs, activation="silu", use_fused=False)
        result_fused = self._run_fused_experts(*inputs, activation="silu", use_fused=True)
        self._assert_close(result_fused, result_original, torch.bfloat16)

    def test_topk_1(self):
        """Test with topk=1 (single expert per token)."""
        torch.manual_seed(42)
        inputs = self._create_moe_inputs(
            num_tokens=32,
            num_experts=8,
            hidden_size=256,
            intermediate_size=512,
            topk=1,
            dtype=torch.bfloat16,
        )

        result_original = self._run_fused_experts(*inputs, activation="silu", use_fused=False)
        result_fused = self._run_fused_experts(*inputs, activation="silu", use_fused=True)
        self._assert_close(result_fused, result_original, torch.bfloat16)

    def test_various_sizes(self):
        """Test multiple hidden/intermediate size combinations."""
        torch.manual_seed(42)
        size_configs = [
            (128, 256),
            (256, 512),
            (512, 1024),
        ]

        for hidden_size, intermediate_size in size_configs:
            with self.subTest(
                hidden_size=hidden_size, intermediate_size=intermediate_size
            ):
                inputs = self._create_moe_inputs(
                    num_tokens=16,
                    num_experts=8,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    topk=2,
                    dtype=torch.bfloat16,
                )

                result_original = self._run_fused_experts(
                    *inputs, activation="silu", use_fused=False
                )
                result_fused = self._run_fused_experts(
                    *inputs, activation="silu", use_fused=True
                )
                self._assert_close(
                    result_fused, result_original, torch.bfloat16
                )


if __name__ == "__main__":
    unittest.main()
