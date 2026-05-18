"""Unit tests for HISA helpers (block-pool maintenance + Stage-2 scatter).

These tests exercise the pure-PyTorch parts of `hisa.py` and do not depend
on deep_gemm. Kernel-level end-to-end correctness (Stage-1 + Stage-2 vs.
the dense indexer) is covered by the existing NSA indexer tests once HISA
is enabled via the SGLANG_NSA_HISA_ENABLE env var on a CUDA host.
"""

from __future__ import annotations

import os
import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _make_pool_stub(num_pages: int, device: str = "cpu"):
    """Build a minimal stub that exposes the methods HISA needs."""

    class _Pool:
        hisa_enabled = True

        def __init__(self):
            # Full per-token buffer (page_size=64, head_dim=128, 4B scale per token).
            self.index_k_with_scale_buffer = [
                torch.zeros((num_pages, 64 * 128 + 64 * 4), dtype=torch.uint8, device=device)
            ]
            # Per-page representative buffer (1 synthetic "token" per page).
            self.index_k_block_buffer = [
                torch.zeros((num_pages, 128 + 4), dtype=torch.uint8, device=device)
            ]

        def get_index_k_with_scale_buffer(self, layer_id: int):
            return self.index_k_with_scale_buffer[layer_id]

        def get_index_k_block_buffer(self, layer_id: int):
            return self.index_k_block_buffer[layer_id]

    return _Pool()


def _fill_page_with_known_bf16(
    pool, layer_id: int, page_id: int, values_bf16: torch.Tensor
):
    """Quantize `values_bf16 [64, 128]` per-token and write into the page."""
    from sglang.srt.layers.attention.nsa.hisa import (
        FP8_BYTES,
        INDEX_HEAD_DIM,
        PAGE_SIZE,
    )

    assert values_bf16.shape == (PAGE_SIZE, INDEX_HEAD_DIM)
    buf = pool.get_index_k_with_scale_buffer(layer_id)

    fp8_max = 448.0
    x = values_bf16.to(torch.float32)
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = absmax / fp8_max  # [64, 1]
    fp8 = (x / scale).to(torch.float8_e4m3fn)

    page = buf[page_id]
    page[: PAGE_SIZE * INDEX_HEAD_DIM] = fp8.view(torch.uint8).reshape(-1)
    page[PAGE_SIZE * INDEX_HEAD_DIM :] = (
        scale.to(torch.float32).view(torch.uint8).reshape(-1)
    )


class TestHISABlockPool(CustomTestCase):
    def setUp(self):
        os.environ["SGLANG_NSA_HISA_ENABLE"] = "1"

    def test_update_block_pool_recovers_mean(self):
        """After update, the block representative should dequant to the
        per-page mean of the underlying tokens (up to fp8 quant error)."""
        from sglang.srt.layers.attention.nsa.hisa import (
            INDEX_HEAD_DIM,
            PAGE_SIZE,
            update_block_pool_for_locs,
        )

        torch.manual_seed(0)
        num_pages = 4
        pool = _make_pool_stub(num_pages)

        # Fill page 1 with a tensor whose mean is a known unit vector.
        tokens = torch.randn(PAGE_SIZE, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        target_mean = torch.zeros(INDEX_HEAD_DIM, dtype=torch.bfloat16)
        target_mean[0] = 1.0
        tokens[0] = target_mean * PAGE_SIZE - tokens[1:].to(torch.float32).sum(0).to(
            torch.bfloat16
        )
        _fill_page_with_known_bf16(pool, 0, page_id=1, values_bf16=tokens)

        # Trigger pool update for tokens in page 1 (out_cache_loc in [64, 128)).
        out_cache_loc = torch.arange(64, 128, dtype=torch.int32)
        update_block_pool_for_locs(pool, 0, out_cache_loc)

        # Decode the representative.
        repr_row = pool.get_index_k_block_buffer(0)[1]
        fp8 = repr_row[:INDEX_HEAD_DIM].view(torch.float8_e4m3fn).to(torch.float32)
        scale = repr_row[INDEX_HEAD_DIM:].view(torch.float32).item()
        recovered = fp8 * scale

        expected_mean = tokens.to(torch.float32).mean(dim=0)
        # FP8 absmax quant gives ~1/127 relative error; allow generous slack.
        torch.testing.assert_close(
            recovered, expected_mean, rtol=5e-2, atol=5e-2
        )

    def test_update_block_pool_only_touches_listed_pages(self):
        from sglang.srt.layers.attention.nsa.hisa import (
            update_block_pool_for_locs,
        )

        num_pages = 4
        pool = _make_pool_stub(num_pages)
        # Pre-stamp untouched pages with a sentinel.
        pool.get_index_k_block_buffer(0).fill_(0xAB)

        # Touch only page 2 (locs in [128, 192)).
        out_cache_loc = torch.arange(128, 192, dtype=torch.int32)
        update_block_pool_for_locs(pool, 0, out_cache_loc)

        blk = pool.get_index_k_block_buffer(0)
        self.assertTrue((blk[0] == 0xAB).all())
        self.assertTrue((blk[1] == 0xAB).all())
        self.assertFalse((blk[2] == 0xAB).all())  # should have been rewritten
        self.assertTrue((blk[3] == 0xAB).all())

    def test_update_block_pool_disabled_pool_is_noop(self):
        """If pool.hisa_enabled is False the helper must return without
        touching anything."""
        from sglang.srt.layers.attention.nsa.hisa import (
            update_block_pool_for_locs,
        )

        pool = _make_pool_stub(2)
        pool.hisa_enabled = False
        sentinel = pool.get_index_k_block_buffer(0).clone()

        out_cache_loc = torch.arange(0, 64, dtype=torch.int32)
        update_block_pool_for_locs(pool, 0, out_cache_loc)

        torch.testing.assert_close(
            pool.get_index_k_block_buffer(0), sentinel
        )

    def test_update_block_pool_empty_locs_is_noop(self):
        from sglang.srt.layers.attention.nsa.hisa import (
            update_block_pool_for_locs,
        )

        pool = _make_pool_stub(2)
        sentinel = pool.get_index_k_block_buffer(0).clone()
        update_block_pool_for_locs(
            pool, 0, torch.empty((0,), dtype=torch.int32)
        )
        torch.testing.assert_close(
            pool.get_index_k_block_buffer(0), sentinel
        )


if __name__ == "__main__":
    unittest.main()
