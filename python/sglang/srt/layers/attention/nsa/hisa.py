"""HISA: Hierarchical Indexed Sparse Attention (arXiv 2603.28458).

Drop-in accelerator for the DSA indexer. Rewrites the indexer's flat
per-query token scan into a two-stage hierarchical scan:

  Stage-1 (block-level coarse filter):
      Score Q against per-page mean-pooled K representatives. Pick the
      top-`m_pages` pages per query, giving a candidate token pool of
      size `m * B` (default 8192).

  Stage-2 (token-level refinement):
      Run the original FP8 MQA logits within the candidate pages only,
      then top-`k` (default index_topk=2048) to produce the final
      token-level indices. Output shape and semantics are identical to
      the dense indexer, so downstream Sparse MLA is unchanged.

v1 scope:
  * CUDA-only; falls back to dense indexer otherwise.
  * `block_size = page_size = 64` (no repacking).
  * Stage-2 uses path 1 (per-query batch-flatten of fp8_paged_mqa_logits).
  * Block representatives are maintained eagerly via
    `update_block_pool_for_locs` after every indexer K store.
  * Only wired into the paged (decode / MTP) code path; ragged prefill
    falls back to the dense indexer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool


PAGE_SIZE = 64
INDEX_HEAD_DIM = 128
FP8_BYTES = INDEX_HEAD_DIM  # 1 byte per element
SCALE_BYTES = 4  # one fp32 scale per page_size*head_dim block (head_dim==quant_block)
PER_TOKEN_BYTES = FP8_BYTES + SCALE_BYTES  # 132
# Per-page representative slot layout in `index_k_block_buffer`:
#   [0 : 128)  -> fp8 mean of the page's 64 tokens
#   [128: 132) -> fp32 absmax-scale used for the fp8 mean
REPR_BYTES = INDEX_HEAD_DIM + SCALE_BYTES  # 132


def _dequant_page_fp8_to_bf16(
    page_buf: torch.Tensor,
) -> torch.Tensor:
    """Dequantize the 64-token fp8 segment of one page back to bf16.

    Args:
        page_buf: uint8 tensor [page_size * head_dim + page_size * 4] for one page.

    Returns:
        bf16 tensor [page_size, head_dim].
    """
    fp8 = page_buf[: PAGE_SIZE * INDEX_HEAD_DIM].view(torch.float8_e4m3fn)
    fp8 = fp8.view(PAGE_SIZE, INDEX_HEAD_DIM)
    scale = page_buf[PAGE_SIZE * INDEX_HEAD_DIM :].view(torch.float32)
    # head_dim==quant_block_size==128, so there is exactly one scale per token.
    scale = scale.view(PAGE_SIZE, 1)
    return (fp8.to(torch.float32) * scale).to(torch.bfloat16)


def _quant_bf16_to_fp8_single(
    x_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax fp8 quantization for one vector.

    Args:
        x_bf16: bf16 tensor [..., head_dim].

    Returns:
        (fp8 uint8 view tensor [..., head_dim], fp32 scale [..., 1]).
    """
    # E4M3 max representable magnitude.
    fp8_max = 448.0
    x_f32 = x_bf16.to(torch.float32)
    absmax = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = absmax / fp8_max
    fp8 = (x_f32 / scale).to(torch.float8_e4m3fn)
    return fp8, scale.to(torch.float32)


def update_block_pool_for_locs(
    pool: "NSATokenToKVPool",
    layer_id: int,
    out_cache_loc: torch.Tensor,
) -> None:
    """Recompute the per-page representative for every page touched by
    `out_cache_loc`.

    Called after `_store_index_k_cache` writes the freshly-quantized
    indexer K for the current step. Per step the set of touched pages is
    small (one per request in decode, ceil(seq_len/64) in prefill), so we
    simply rebuild those pages' representatives from scratch.

    Operations are scalar-light: dequant the 64 fp8 values of the page,
    take a mean along the token axis, then re-quantize to fp8 with a
    single absmax scale.
    """
    if not getattr(pool, "hisa_enabled", False):
        return
    if out_cache_loc.numel() == 0:
        return

    page_ids = torch.unique(out_cache_loc // PAGE_SIZE)
    if page_ids.numel() == 0:
        return

    full_buf = pool.get_index_k_with_scale_buffer(layer_id)  # [N_pages, 132*64]
    blk_buf = pool.get_index_k_block_buffer(layer_id)  # [N_pages, 132]

    # Gather the touched pages (small tensor; copy to make per-page math easy).
    pages = full_buf.index_select(0, page_ids)  # [P, 132*64] uint8

    # Decompose into fp8 data + fp32 scale per token, then dequant.
    P = pages.shape[0]
    fp8 = pages[:, : PAGE_SIZE * INDEX_HEAD_DIM].view(torch.float8_e4m3fn)
    fp8 = fp8.view(P, PAGE_SIZE, INDEX_HEAD_DIM).to(torch.float32)
    scale = pages[:, PAGE_SIZE * INDEX_HEAD_DIM :].view(torch.float32)
    scale = scale.view(P, PAGE_SIZE, 1)
    bf16 = (fp8 * scale).to(torch.bfloat16)  # [P, 64, 128]

    # Mean-pool across tokens. Zero-quantized tail slots in partially-filled
    # pages dequant to 0, so the mean is naturally biased toward small magnitudes
    # for short tails; that's acceptable since Stage-1 only ranks pages and
    # invalid-tail page logits are masked downstream via seqlens_32.
    mean_bf16 = bf16.mean(dim=1)  # [P, 128]
    fp8_repr, scale_repr = _quant_bf16_to_fp8_single(mean_bf16)  # [P,128] u8, [P,1] f32

    # Pack into the block buffer rows.
    out = torch.empty((P, REPR_BYTES), dtype=torch.uint8, device=blk_buf.device)
    out[:, :INDEX_HEAD_DIM] = fp8_repr.view(torch.uint8)
    out[:, INDEX_HEAD_DIM:] = scale_repr.view(P, 1).view(torch.uint8).reshape(P, 4)
    blk_buf.index_copy_(0, page_ids.to(torch.long), out)


def _stage1_block_logits(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    block_repr_buf: torch.Tensor,
    seqlens_pages_2d: torch.Tensor,
    block_tables: torch.Tensor,
    sm_count: int,
):
    """Stage-1 block-level MQA logits via deep_gemm.fp8_paged_mqa_logits with
    KVBlockSize=1.

    Args:
        q_fp8: [Q, 1, H, D] fp8 queries (already unsqueezed for next_n=1).
        weights: [Q, H] head gates (squeezed).
        block_repr_buf: [N_pages, 132] uint8 per-page representative buffer.
        seqlens_pages_2d: [B, 1] int32, number of valid pages per batch entry.
        block_tables: [B, max_pages] int32, page-id table.
        sm_count: int, deep_gemm SM count.

    Returns:
        block_logits: [Q, max_pages] float32. Invalid pages are -inf (the
        downstream topk_transform / topk handles this).
    """
    import deep_gemm  # local import for CUDA-only path

    max_pages = block_tables.shape[1]
    # block_repr_buf has one synthetic "token" per page: view as
    # [N_pages, 1, 1, 132] to match fp8_paged_mqa_logits' (num_pages, block_kv,
    # num_heads_kv, head_dim_with_sf) expectation with block_kv=1.
    kv_cache_fp8 = block_repr_buf.view(block_repr_buf.shape[0], 1, 1, REPR_BYTES)

    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        seqlens_pages_2d, 1, sm_count
    )
    block_logits = deep_gemm.fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        seqlens_pages_2d,
        block_tables,
        schedule_metadata,
        max_pages,
        clean_logits=True,  # mask invalid pages to -inf
    )
    return block_logits  # [Q, max_pages]


def _stage2_token_logits(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    full_buf: torch.Tensor,
    cand_pages_per_query: torch.Tensor,
    seqlens_tokens_2d: torch.Tensor,
    sm_count: int,
):
    """Stage-2 token-level MQA logits restricted to candidate pages.

    Path 1 implementation: flatten the batch so every query becomes its own
    batch entry with its own per-query block_tables. Reuses
    `deep_gemm.fp8_paged_mqa_logits` unchanged.

    Args:
        q_fp8: [Q, 1, H, D] fp8 queries.
        weights: [Q, H] head gates.
        full_buf: [N_pages, 132*64] uint8 full indexer K buffer.
        cand_pages_per_query: [Q, m_pages] int32 candidate page ids per query.
        seqlens_tokens_2d: [Q, 1] int32, valid token count within the
            m_pages*64 candidate slots for each query (= min(seq_len_q,
            m_pages * 64)).
        sm_count: deep_gemm SM count.

    Returns:
        fine_logits: [Q, m_pages * 64] float32.
    """
    import deep_gemm

    m_pages = cand_pages_per_query.shape[1]
    max_seq_len = m_pages * PAGE_SIZE
    kv_cache_fp8 = full_buf.view(full_buf.shape[0], PAGE_SIZE, 1, PER_TOKEN_BYTES)

    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        seqlens_tokens_2d, PAGE_SIZE, sm_count
    )
    fine_logits = deep_gemm.fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        seqlens_tokens_2d,
        cand_pages_per_query,
        schedule_metadata,
        max_seq_len,
        clean_logits=True,
    )
    return fine_logits  # [Q, m_pages * 64]


def hisa_paged_logits(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    pool: "NSATokenToKVPool",
    layer_id: int,
    seqlens_32_2d: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    candidate_size: int,
    sm_count: int,
) -> torch.Tensor:
    """HISA paged indexer logits, in per-batch local token space.

    Drop-in replacement for `deep_gemm.fp8_paged_mqa_logits(...)` in the
    paged path: returns a `[Q, max_seq_len]` float32 tensor with the same
    semantics (logit per token in the batch's local sequence, -inf for
    invalid positions). The dense path's `topk_transform` consumes the
    result unchanged.

    Implementation:
      Stage-1: score per-page mean-pooled K and pick top-`m_pages`
               candidate page slots (per-batch local).
      Stage-2: score tokens only within the candidate pages.
      Scatter: fill a [Q, max_seq_len] -inf tensor and place Stage-2
               logits at the per-batch-local positions of each candidate
               page.

    Args:
        q_fp8: [Q, 1, H, D] fp8 queries (already next_n-unsqueezed,
            matching the dense `_get_topk_paged` call site).
        weights: [Q, H] head gates (already squeezed, matching dense path).
        pool: NSATokenToKVPool with HISA block buffer allocated.
        layer_id: indexer layer id.
        seqlens_32_2d: [B, 1] int32 token-level seqlens.
        block_tables: [B, max_pages] int32 per-batch page-id table.
        max_seq_len: the dense path's `block_tables.shape[1] * page_size`.
        candidate_size: m*B for Stage-1; must be multiple of 64.
        sm_count: deep_gemm SM count.

    Returns:
        logits: [Q, max_seq_len] float32. Per-batch local token-space.
        Caller passes this through `metadata.topk_transform(logits, k)`.
    """
    assert q_fp8.dim() == 4, f"expected [Q, 1, H, D], got {q_fp8.shape}"
    assert weights.dim() == 2, f"expected [Q, H], got {weights.shape}"
    assert candidate_size % PAGE_SIZE == 0
    m_pages = candidate_size // PAGE_SIZE

    Q = q_fp8.shape[0]
    device = q_fp8.device
    B = seqlens_32_2d.shape[0]
    # Decode / MTP both produce one query per batch entry.
    assert Q == B, (
        "HISA paged path currently requires Q == B. "
        f"Got Q={Q}, B={B}; fall back to dense indexer for this batch."
    )

    q_fp8_4d = q_fp8
    weights_2d = weights

    # ---- Stage 1: block-level coarse filter -----------------------------
    seqlens_tokens = seqlens_32_2d.squeeze(-1).to(torch.int32)  # [B]
    seqlens_pages = ((seqlens_tokens + PAGE_SIZE - 1) // PAGE_SIZE).to(torch.int32)
    seqlens_pages_2d = seqlens_pages.unsqueeze(-1)  # [B, 1]

    block_repr_buf = pool.get_index_k_block_buffer(layer_id)
    block_logits = _stage1_block_logits(
        q_fp8_4d,
        weights_2d,
        block_repr_buf,
        seqlens_pages_2d,
        block_tables,
        sm_count,
    )  # [Q, max_pages]

    max_pages_avail = block_logits.shape[1]
    take_m = min(m_pages, max_pages_avail)
    # `cand_local` holds **per-batch local** page-slot indices into
    # block_tables[q, :]. We keep this representation throughout so we can
    # scatter Stage-2 logits back into per-batch flat token space without
    # an extra global->local lookup.
    _, cand_local = block_logits.topk(take_m, dim=-1)  # [Q, take_m]

    # Resolve per-query candidate page ids (global) for the Stage-2 kernel.
    cand_pages = torch.gather(
        block_tables, 1, cand_local.to(torch.int64)
    ).to(torch.int32)  # [Q, take_m]

    # Pad to m_pages if the sequence has fewer valid pages. We replicate
    # the last selected page; Stage-2's seqlen cap masks any redundant
    # tokens, and the scatter step writes -inf for duplicate slots
    # naturally (overwrites with the same value, harmless).
    if take_m < m_pages:
        pad_pages = cand_pages[:, -1:].expand(Q, m_pages - take_m).contiguous()
        cand_pages = torch.cat([cand_pages, pad_pages], dim=1)
        pad_local = cand_local[:, -1:].expand(Q, m_pages - take_m).contiguous()
        cand_local = torch.cat([cand_local, pad_local], dim=1)

    # ---- Stage 2: token-level refinement --------------------------------
    cap = m_pages * PAGE_SIZE
    seqlens_tokens_q = torch.minimum(
        seqlens_tokens,
        torch.full_like(seqlens_tokens, cap),
    )
    seqlens_tokens_2d = seqlens_tokens_q.unsqueeze(-1)  # [Q, 1]

    full_buf = pool.get_index_k_with_scale_buffer(layer_id)
    fine_logits = _stage2_token_logits(
        q_fp8_4d,
        weights_2d,
        full_buf,
        cand_pages,
        seqlens_tokens_2d,
        sm_count,
    )  # [Q, m_pages * 64]

    # ---- Scatter into [Q, max_seq_len] per-batch local space -----------
    # Per-batch local position of token (page_slot=s, intra_offset=p):
    #     pos = cand_local[q, s] * 64 + p
    # `cand_local` is per-batch (since Q==B). Build position indices once
    # and scatter Stage-2 logits.
    logits_full = torch.full(
        (Q, max_seq_len),
        float("-inf"),
        dtype=fine_logits.dtype,
        device=device,
    )

    # Build [Q, m_pages*64] of target positions in [0, max_seq_len).
    base = cand_local.to(torch.int64) * PAGE_SIZE  # [Q, m_pages]
    base = base.unsqueeze(-1).expand(Q, m_pages, PAGE_SIZE)  # [Q, m_pages, 64]
    offs = torch.arange(PAGE_SIZE, device=device, dtype=torch.int64)
    positions = (base + offs.view(1, 1, PAGE_SIZE)).reshape(Q, m_pages * PAGE_SIZE)
    # Clamp to within bounds (paranoia; padded slots may produce huge ids).
    positions = positions.clamp_(0, max_seq_len - 1)

    logits_full.scatter_(1, positions, fine_logits)
    return logits_full
