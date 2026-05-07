"""Unit tests for the HISA hierarchical indexer.

Verifies:
    1. The PyTorch reference matches dense scoring on positions inside the
       selected blocks (and is -inf elsewhere).
    2. The recall of the dense top-k tokens within HISA's selected blocks
       is high on synthetic data.
    3. (CUDA-only) the Triton kernel matches the reference numerically.
"""

import unittest

import torch

from sglang.srt.layers.attention.nsa.hisa_kernel import (
    HISAConfig,
    dense_paged_logits_reference,
    hisa_paged_logits_reference,
    hisa_paged_logits_triton,
)


def _make_inputs(
    seed: int = 0,
    B: int = 2,
    H: int = 4,
    D: int = 128,
    P: int = 64,
    seq_lens=(800, 1100),
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    g = torch.Generator(device=device).manual_seed(seed)
    max_len = max(seq_lens)
    max_blocks = (max_len + P - 1) // P
    # Each batch row gets `n_blocks` distinct page indices.
    seqlens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    used = []
    block_tables = torch.zeros((B, max_blocks), dtype=torch.int32, device=device)
    next_page = 1  # leave page 0 as a "junk" page so unique-page gather is exercised
    for b in range(B):
        n_blocks = (seq_lens[b] + P - 1) // P
        ids = torch.arange(next_page, next_page + n_blocks, dtype=torch.int32, device=device)
        block_tables[b, :n_blocks] = ids
        used.append(int(ids.max().item()))
        next_page += n_blocks
    num_pages = max(used) + 1

    k_pages = torch.randn((num_pages, P, D), generator=g, device=device, dtype=dtype)

    N_q = B
    q = torch.randn((N_q, H, D), generator=g, device=device, dtype=dtype)
    weights = torch.randn((N_q, H), generator=g, device=device, dtype=torch.float32)
    q_to_batch = torch.arange(B, dtype=torch.int32, device=device)

    return q, k_pages, weights, seqlens_t, block_tables, q_to_batch, max_len


class TestHISAReference(unittest.TestCase):
    def test_inside_selected_blocks_matches_dense(self):
        """Within the blocks HISA selects, every token logit must equal dense."""
        q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len = _make_inputs(
            seed=42, B=2, seq_lens=(640, 832), P=64, H=2
        )
        config = HISAConfig(block_size=64, top_blocks=4, min_seq_len=0)

        dense = dense_paged_logits_reference(
            q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len
        )
        hisa = hisa_paged_logits_reference(
            q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len, config
        )

        # For each query, find positions where HISA is finite; those must
        # match dense exactly.
        finite = torch.isfinite(hisa)
        torch.testing.assert_close(
            hisa[finite], dense[finite], rtol=1e-5, atol=1e-4
        )

    def test_planted_needle_block_is_selected(self):
        """When one block contains a key strongly aligned with q, HISA must keep it."""
        B, H, D, P = 1, 2, 128, 64
        n_blocks = 16
        seq_len = n_blocks * P
        device, dtype = "cpu", torch.float32

        torch.manual_seed(123)
        # Random low-magnitude background keys.
        k_pages = 0.01 * torch.randn(n_blocks + 1, P, D, dtype=dtype)
        # Random unit query.
        q = torch.randn(B, H, D, dtype=dtype)
        q = q / q.norm(dim=-1, keepdim=True)
        # Plant a strongly-aligned key in block index `needle_block` (logical 5).
        needle_block = 5
        k_pages[needle_block + 1, 7, :] = 100.0 * q[0, 0]  # +1 because page 0 is junk
        weights = torch.ones(B, H, dtype=torch.float32)
        seqlens = torch.tensor([seq_len], dtype=torch.int32)
        block_tables = torch.arange(1, n_blocks + 1, dtype=torch.int32).unsqueeze(0)
        q_to_batch = torch.zeros(B, dtype=torch.int32)

        # With top_blocks=2 (out of 16), HISA must still keep the needle.
        config = HISAConfig(block_size=P, top_blocks=2, min_seq_len=0)
        hisa = hisa_paged_logits_reference(
            q, k_pages, weights, seqlens, block_tables, q_to_batch, seq_len, config
        )
        # The needle position must be finite (i.e. its block was selected).
        needle_pos = needle_block * P + 7
        self.assertTrue(
            torch.isfinite(hisa[0, needle_pos]),
            f"HISA failed to keep the planted needle block; "
            f"finite block count = {(torch.isfinite(hisa[0]).view(n_blocks, P).any(dim=-1)).sum().item()}",
        )

    def test_min_blocks_per_request(self):
        """Top_blocks larger than n_blocks must not crash; output covers every block."""
        q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len = _make_inputs(
            seed=1, B=1, seq_lens=(128,), P=64, H=2
        )
        config = HISAConfig(block_size=64, top_blocks=128, min_seq_len=0)

        dense = dense_paged_logits_reference(
            q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len
        )
        hisa = hisa_paged_logits_reference(
            q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len, config
        )

        # All sequence positions should be finite under HISA (we kept all blocks).
        valid_positions = torch.isfinite(dense)
        torch.testing.assert_close(
            hisa[valid_positions], dense[valid_positions], rtol=1e-5, atol=1e-4
        )

    def test_invalid_positions_remain_minus_inf(self):
        """Positions past seq_len must be -inf under HISA, like the dense kernel."""
        q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len = _make_inputs(
            seed=2, B=2, seq_lens=(70, 130), P=64, H=2
        )
        config = HISAConfig(block_size=64, top_blocks=4, min_seq_len=0)
        hisa = hisa_paged_logits_reference(
            q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len, config
        )
        for i in range(q.shape[0]):
            s = int(seqlens[i].item())
            self.assertTrue(torch.isinf(hisa[i, s:]).all())

    def test_config_parsing(self):
        cfg = HISAConfig.from_json('{"top_blocks": 17, "block_size": 64, "min_seq_len": 1024}')
        self.assertEqual(cfg.top_blocks, 17)
        self.assertEqual(cfg.block_size, 64)
        self.assertEqual(cfg.min_seq_len, 1024)
        self.assertEqual(HISAConfig.from_json(None), HISAConfig())
        # Aliased key.
        cfg2 = HISAConfig.from_json('{"top_k_blocks": 5}')
        self.assertEqual(cfg2.top_blocks, 5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernel test")
class TestHISATritonMatchesReference(unittest.TestCase):
    def test_triton_matches_reference_bf16(self):
        device = "cuda"
        q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len = _make_inputs(
            seed=11, B=2, seq_lens=(640, 832), P=64, H=2,
            device=device, dtype=torch.bfloat16,
        )
        config = HISAConfig(block_size=64, top_blocks=4, min_seq_len=0)
        ref = hisa_paged_logits_reference(
            q.to(torch.float32), k_pages.to(torch.float32),
            weights, seqlens, block_tables, q_to_batch, max_len, config,
        )
        triton_out = hisa_paged_logits_triton(
            q, k_pages, weights, seqlens, block_tables, q_to_batch, max_len, config
        )
        finite = torch.isfinite(ref) & torch.isfinite(triton_out)
        # bf16 dot products: loose tolerances.
        torch.testing.assert_close(
            triton_out[finite], ref[finite], rtol=5e-2, atol=5e-2
        )
        # Same set of selected positions.
        self.assertEqual(
            torch.isfinite(triton_out).sum().item(),
            torch.isfinite(ref).sum().item(),
        )


if __name__ == "__main__":
    unittest.main()
