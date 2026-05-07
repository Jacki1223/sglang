"""Wiring layer that adapts SGLang's NSA indexer storage to the HISA kernel.

The NSA indexer stores keys in a packed FP8 cache (per-token interleaved
[128 byte FP8 || 4 byte FP32 scale]). The HISA Triton kernels in
`hisa_kernel.py` consume BF16 keys for clarity and ease of testing. This
module:

1. Gathers the set of unique pages referenced by the current batch's
   block table (avoiding a full-cache dequant).
2. Dequantizes those pages to BF16 once.
3. Calls the HISA two-stage kernel and returns logits with the same
   `[N_q, max_seq_len]` shape and semantics as
   `deep_gemm.fp8_paged_mqa_logits(..., clean_logits=False)`.

Activation is gated by `--enable-hisparse` and `--hisparse-config`. If
HISA cannot run (max_kv_len < min_seq_len, unsupported next_n>1, missing
Triton, etc.) the caller falls back to the dense kernel.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.attention.nsa.hisa_kernel import (
    HISAConfig,
    hisa_paged_logits,
)
from sglang.srt.server_args import get_global_server_args


_HISA_CONFIG_CACHE: Optional[HISAConfig] = None


def get_hisa_config() -> HISAConfig:
    """Cached parser for ``--hisparse-config``.

    The parsed config is process-global; server args are immutable after
    startup so this is safe.
    """
    global _HISA_CONFIG_CACHE
    if _HISA_CONFIG_CACHE is None:
        srv = get_global_server_args()
        _HISA_CONFIG_CACHE = HISAConfig.from_json(
            getattr(srv, "hisparse_config", None)
        )
    return _HISA_CONFIG_CACHE


def is_hisa_enabled() -> bool:
    """True if `--enable-hisparse` was passed on launch."""
    srv = get_global_server_args()
    return bool(getattr(srv, "enable_hisparse", False))


def _dequant_active_pages(
    kv_cache_fp8: torch.Tensor,    # [num_pages, P, 1, 132] uint8 (P=64)
    block_tables: torch.Tensor,    # [B, max_blocks] i32, page indices
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather unique active pages and dequantize them to BF16.

    Returns:
        k_pages_bf16: ``[U, P, D=128]`` BF16 keys for the active pages.
        local_block_tables: ``[B, max_blocks]`` i32 indices into ``k_pages_bf16``.
    """
    assert kv_cache_fp8.dim() == 4 and kv_cache_fp8.shape[-1] == 132, (
        f"unexpected indexer KV cache shape {tuple(kv_cache_fp8.shape)}"
    )
    P = kv_cache_fp8.shape[1]
    assert P == 64, "HISA indexer wiring assumes page_size=64 (CUDA path)."

    flat_bt = block_tables.reshape(-1)
    unique_pages, inverse = torch.unique(flat_bt, return_inverse=True)
    local_block_tables = inverse.view_as(block_tables).to(torch.int32)

    # [U, 64, 1, 132] uint8
    active = kv_cache_fp8.index_select(0, unique_pages.to(torch.long)).contiguous()
    # Split byte layout into FP8 K and FP32 scale.
    k_bytes = active[..., :128].contiguous()                  # [U, 64, 1, 128] uint8
    scale_bytes = active[..., 128:132].contiguous()           # [U, 64, 1, 4]   uint8

    k_fp8 = k_bytes.view(torch.float8_e4m3fn).squeeze(2)      # [U, 64, 128] fp8
    k_scale = (
        scale_bytes.view(torch.float32).squeeze(-1).squeeze(-1)
    )                                                          # [U, 64] fp32

    # Dequantize: BF16 K = FP8 K * per-token scale.
    k_bf16 = (k_fp8.to(torch.float32) * k_scale.unsqueeze(-1)).to(torch.bfloat16)
    return k_bf16, local_block_tables


def hisa_paged_logits_from_indexer_cache(
    q_fp8: torch.Tensor,           # [N_q, 1, H, D] fp8e4m3fn (post-unsqueeze)
    kv_cache_fp8: torch.Tensor,    # [num_pages, 64, 1, 132] uint8
    weights: torch.Tensor,         # [N_q, H] f32 (q_scale already folded in)
    seqlens_2d: torch.Tensor,      # [B, next_n] i32
    block_tables: torch.Tensor,    # [B, max_blocks] i32
    max_seq_len: int,
    config: Optional[HISAConfig] = None,
) -> torch.Tensor:
    """Compute HISA paged logits from SGLang's existing indexer state.

    Output shape and semantics match ``deep_gemm.fp8_paged_mqa_logits(...,
    clean_logits=False)`` so the existing ``topk_transform`` path can be
    used unchanged.

    Constraints (caller must enforce; otherwise fall back to dense):
        * Page size must be 64 (CUDA path).
        * ``next_n`` (second dim of q_fp8 / seqlens_2d) must be 1.
    """
    if config is None:
        config = get_hisa_config()

    assert q_fp8.dim() == 4 and q_fp8.shape[1] == 1, (
        f"HISA wiring expects q_fp8 [N_q, 1, H, D]; got {tuple(q_fp8.shape)}"
    )
    N_q, _, H, D = q_fp8.shape
    assert D == 128, f"HISA wiring expects head_dim=128; got {D}"
    assert seqlens_2d.dim() == 2 and seqlens_2d.shape[1] == 1, (
        f"HISA wiring requires next_n=1; got seqlens_2d {tuple(seqlens_2d.shape)}"
    )
    B = seqlens_2d.shape[0]
    assert N_q == B, f"HISA wiring expects N_q==B for next_n=1; got N_q={N_q} B={B}"

    device = q_fp8.device

    # 1. Dequantize active pages once.
    k_pages_bf16, local_block_tables = _dequant_active_pages(
        kv_cache_fp8, block_tables
    )

    # 2. Cast q to BF16 (q_scale is folded into weights).
    q_bf16 = q_fp8.squeeze(1).to(torch.bfloat16)               # [N_q, H, D]

    # 3. q_to_batch is identity for next_n=1.
    q_to_batch = torch.arange(N_q, device=device, dtype=torch.int32)

    # 4. seqlens flatten.
    seqlens = seqlens_2d.reshape(-1).to(torch.int32)

    return hisa_paged_logits(
        q=q_bf16,
        k_pages=k_pages_bf16,
        weights=weights.to(torch.float32),
        seqlens=seqlens,
        block_tables=local_block_tables,
        q_to_batch=q_to_batch,
        max_seq_len=max_seq_len,
        config=config,
    )


__all__ = [
    "get_hisa_config",
    "is_hisa_enabled",
    "hisa_paged_logits_from_indexer_cache",
]
