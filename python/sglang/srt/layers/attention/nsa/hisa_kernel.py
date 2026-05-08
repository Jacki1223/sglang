"""
HISA: Hierarchical Indexed Sparse Attention.

Triton implementation of the two-stage indexer described in
"HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention"
(arXiv:2603.28458).

Stage 1 (block coarse): for each query, score every page-sized block by its
mean-pooled key representation, and select the top-N_b blocks.

Stage 2 (token refine): for each query, compute fine-grained per-token logits
only inside the selected blocks; positions outside selected blocks remain
-inf.

The output of `hisa_paged_logits` has the same shape and semantics as
`deep_gemm.fp8_paged_mqa_logits(..., clean_logits=False)`, so the downstream
`topk_transform`, sparse MLA kernel, and the rest of the NSA stack do not
need any changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton always present in CUDA builds
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HISAConfig:
    """Runtime knobs for the hierarchical indexer.

    Attributes:
        block_size: page size used for stage-1 coarse blocks. Must equal the
            indexer-K cache page size (64 on CUDA).
        top_blocks: number of blocks kept after stage 1 per query. Final
            candidate token pool is `top_blocks * block_size`.
        min_seq_len: if the longest sequence in the batch is no longer than
            this threshold, fall back to dense scoring (HISA's coarse stage
            has a fixed cost that does not pay off for short context).
    """

    block_size: int = 64
    top_blocks: int = 64
    min_seq_len: int = 4096

    @staticmethod
    def from_json(s: Optional[str]) -> "HISAConfig":
        if not s:
            return HISAConfig()
        d = json.loads(s)
        return HISAConfig(
            block_size=int(d.get("block_size", 64)),
            top_blocks=int(d.get("top_blocks", d.get("top_k_blocks", 64))),
            min_seq_len=int(d.get("min_seq_len", 4096)),
        )


# ---------------------------------------------------------------------------
# PyTorch reference (used in unit tests and as the no-Triton fallback)
# ---------------------------------------------------------------------------


def hisa_paged_logits_reference(
    q: torch.Tensor,                 # [N_q, H, D] bf16 / fp32
    k_pages: torch.Tensor,           # [num_pages, P, D] bf16 / fp32 (P = block_size)
    weights: torch.Tensor,           # [N_q, H] fp32
    seqlens: torch.Tensor,           # [B] int32, batch sequence lengths
    block_tables: torch.Tensor,      # [B, max_blocks] int32, page indices
    q_to_batch: torch.Tensor,        # [N_q] int32, which batch row each query belongs to
    max_seq_len: int,
    config: HISAConfig,
) -> torch.Tensor:
    """Pure PyTorch reference for HISA's two-stage paged logits.

    Returns logits of shape `[N_q, max_seq_len]` with `-inf` outside selected
    blocks and beyond seq_len. Within selected blocks the values match the
    dense per-token logits.
    """
    assert k_pages.dim() == 3
    num_pages, P, D = k_pages.shape
    assert P == config.block_size
    N_q, H, Dq = q.shape
    assert Dq == D

    device = q.device
    out = torch.full((N_q, max_seq_len), float("-inf"), dtype=torch.float32, device=device)

    # Compute per-page mean K (stage-1 representation).
    pooled_k = k_pages.to(torch.float32).mean(dim=1)  # [num_pages, D]

    # Per-query batch mapping.
    q_batch = q_to_batch.to(torch.long)
    seqs = seqlens.to(torch.long)

    # max blocks per request.
    max_blocks = block_tables.shape[1]
    top_b = min(config.top_blocks, max_blocks)

    q_f32 = q.to(torch.float32)
    w_f32 = weights.to(torch.float32)

    for i in range(N_q):
        b = int(q_batch[i].item())
        s_len = int(seqs[b].item())
        if s_len <= 0:
            continue
        n_blocks = (s_len + P - 1) // P
        # Page indices for this request, shape [n_blocks].
        pages = block_tables[b, :n_blocks].to(torch.long)

        # Stage 1: block scores = sum_h w[h] * dot(q[h], pooled_k[page])
        # q[h] @ pooled_k[page]^T  -> [H, n_blocks]
        block_logits_h = q_f32[i] @ pooled_k[pages].T  # [H, n_blocks]
        block_scores = (w_f32[i].unsqueeze(-1) * block_logits_h).sum(dim=0)  # [n_blocks]

        # Mask blocks past sequence length already excluded by n_blocks.
        k = min(top_b, n_blocks)
        # Top-k block indices (within this request's logical block range).
        sel = torch.topk(block_scores, k=k, dim=-1).indices  # [k]
        sel_pages = pages[sel]  # physical pages

        # Stage 2: per-token logits inside selected blocks only.
        # token logits = sum_h w[h] * dot(q[h], k[t]) for t in selected blocks
        # Gather K for those pages: [k, P, D]
        k_sel = k_pages[sel_pages].to(torch.float32)
        # token_logits_h: [H, k*P] = q[h] @ k_sel.reshape(k*P, D).T
        token_logits_h = q_f32[i] @ k_sel.reshape(k * P, D).T  # [H, k*P]
        token_scores = (w_f32[i].unsqueeze(-1) * token_logits_h).sum(dim=0)  # [k*P]
        token_scores = token_scores.view(k, P)  # [k, P]

        # Scatter into the output. Position of token (logical_block, offset) is
        # `logical_block * P + offset`; logical_block = sel[j].
        for j in range(k):
            lb = int(sel[j].item())
            base = lb * P
            end = min(base + P, s_len)
            n_valid = end - base
            if n_valid <= 0:
                continue
            out[i, base:base + n_valid] = token_scores[j, :n_valid]

    return out


def dense_paged_logits_reference(
    q: torch.Tensor,                 # [N_q, H, D] bf16 / fp32
    k_pages: torch.Tensor,           # [num_pages, P, D]
    weights: torch.Tensor,           # [N_q, H]
    seqlens: torch.Tensor,           # [B] int32
    block_tables: torch.Tensor,      # [B, max_blocks] int32
    q_to_batch: torch.Tensor,        # [N_q] int32
    max_seq_len: int,
) -> torch.Tensor:
    """Reference implementation of the dense indexer for cross-checking."""
    N_q, H, D = q.shape
    P = k_pages.shape[1]
    device = q.device
    out = torch.full((N_q, max_seq_len), float("-inf"), dtype=torch.float32, device=device)
    q_f32 = q.to(torch.float32)
    w_f32 = weights.to(torch.float32)
    seqs = seqlens.to(torch.long)
    q_batch = q_to_batch.to(torch.long)

    for i in range(N_q):
        b = int(q_batch[i].item())
        s_len = int(seqs[b].item())
        if s_len <= 0:
            continue
        n_blocks = (s_len + P - 1) // P
        pages = block_tables[b, :n_blocks].to(torch.long)
        k_full = k_pages[pages].to(torch.float32).reshape(n_blocks * P, D)
        # logits per head: [H, n_blocks*P]
        logits_h = q_f32[i] @ k_full.T
        scores = (w_f32[i].unsqueeze(-1) * logits_h).sum(dim=0)  # [n_blocks*P]
        out[i, : n_blocks * P] = scores
        # mask tail past seq_len
        if n_blocks * P > s_len:
            out[i, s_len:n_blocks * P] = float("-inf")
    return out


# ---------------------------------------------------------------------------
# Triton kernels (BF16 K). FP8 K kernels are a follow-up; the wiring layer
# dequantizes the active pages once and reuses bf16 K for both stages, which
# keeps the kernel simple and verifiable while still avoiding the O(L^2)
# scan over the full prefix in stage 2.
# ---------------------------------------------------------------------------


if _HAS_TRITON:

    @triton.jit
    def _hisa_block_score_kernel(
        Q_ptr,                # [N_q, H, D] bf16
        Kp_ptr,               # [num_pages, D] bf16 -- precomputed mean per page
        W_ptr,                # [N_q, H] f32
        BT_ptr,               # [B, max_blocks] i32
        SL_ptr,               # [B] i32
        QB_ptr,               # [N_q] i32, q -> batch
        Out_ptr,              # [N_q, max_blocks] f32, set to -inf for invalid
        N_q,
        H: tl.constexpr,
        D: tl.constexpr,
        max_blocks: tl.constexpr,
        P: tl.constexpr,        # block_size, used to compute valid block count
        BLOCK_BL: tl.constexpr,  # blocks per program (tile size along block dim)
    ):
        pid_q = tl.program_id(0)
        pid_b = tl.program_id(1)
        if pid_q >= N_q:
            return

        # Load batch mapping and sequence length.
        b = tl.load(QB_ptr + pid_q).to(tl.int32)
        s_len = tl.load(SL_ptr + b).to(tl.int32)
        n_blocks = (s_len + P - 1) // P

        offs_bl = pid_b * BLOCK_BL + tl.arange(0, BLOCK_BL)
        bl_mask = offs_bl < max_blocks
        valid_bl = (offs_bl < n_blocks) & bl_mask

        # Load page indices for this batch's logical blocks.
        pages = tl.load(BT_ptr + b * max_blocks + offs_bl, mask=bl_mask, other=0).to(tl.int32)

        # Compute block scores by accumulating over heads.
        offs_d = tl.arange(0, D)
        score = tl.zeros([BLOCK_BL], dtype=tl.float32)
        # Non-unrolled head loop: DSA indexer can have H=64, which would
        # blow up code size with ``tl.static_range``. ``range`` over a
        # constexpr lets the compiler decide.
        for h in range(H):
            # q [D] for this head
            q_vec = tl.load(Q_ptr + pid_q * H * D + h * D + offs_d).to(tl.float32)
            # pooled K [BLOCK_BL, D]: load using gathered page indices
            kp_ptrs = Kp_ptr + pages[:, None].to(tl.int64) * D + offs_d[None, :]
            kp = tl.load(kp_ptrs, mask=bl_mask[:, None], other=0.0).to(tl.float32)
            dots = tl.sum(kp * q_vec[None, :], axis=1)  # [BLOCK_BL]
            w = tl.load(W_ptr + pid_q * H + h).to(tl.float32)
            score += w * dots

        # Write -inf for invalid blocks.
        score = tl.where(valid_bl, score, float("-inf"))
        out_ptrs = Out_ptr + pid_q * max_blocks + offs_bl
        tl.store(out_ptrs, score, mask=bl_mask)


    @triton.jit
    def _hisa_token_score_kernel(
        Q_ptr,                # [N_q, H, D] bf16
        K_ptr,                # [num_pages, P, D] bf16
        W_ptr,                # [N_q, H] f32
        Sel_ptr,              # [N_q, top_b] i32, logical block indices
        SelPages_ptr,         # [N_q, top_b] i32, physical page indices
        SL_ptr,               # [B] i32
        QB_ptr,               # [N_q] i32
        Out_ptr,              # [N_q, max_seq_len] f32, pre-initialized to -inf
        N_q,
        max_seq_len: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
        top_b: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_j = tl.program_id(1)  # which selected block (0..top_b)
        if pid_q >= N_q or pid_j >= top_b:
            return

        # Sentinel: -1 means invalid selection (request had fewer blocks than top_b).
        lb = tl.load(Sel_ptr + pid_q * top_b + pid_j).to(tl.int32)
        if lb < 0:
            return

        page = tl.load(SelPages_ptr + pid_q * top_b + pid_j).to(tl.int32)
        b = tl.load(QB_ptr + pid_q).to(tl.int32)
        s_len = tl.load(SL_ptr + b).to(tl.int32)

        offs_t = tl.arange(0, P)
        token_pos = lb * P + offs_t                        # global position
        valid_t = token_pos < s_len

        offs_d = tl.arange(0, D)
        score = tl.zeros([P], dtype=tl.float32)
        # Non-unrolled head loop: DSA indexer can have H=64, which would
        # blow up code size with ``tl.static_range``. ``range`` over a
        # constexpr lets the compiler decide.
        for h in range(H):
            q_vec = tl.load(Q_ptr + pid_q * H * D + h * D + offs_d).to(tl.float32)
            k_ptrs = (
                K_ptr
                + page.to(tl.int64) * P * D
                + offs_t[:, None] * D
                + offs_d[None, :]
            )
            k_tile = tl.load(k_ptrs).to(tl.float32)        # [P, D]
            dots = tl.sum(k_tile * q_vec[None, :], axis=1)  # [P]
            w = tl.load(W_ptr + pid_q * H + h).to(tl.float32)
            score += w * dots

        score = tl.where(valid_t, score, float("-inf"))
        out_ptrs = Out_ptr + pid_q * max_seq_len + token_pos
        tl.store(out_ptrs, score, mask=valid_t)


def _ensure_pooled_k(k_pages: torch.Tensor) -> torch.Tensor:
    # k_pages: [num_pages, P, D] -> [num_pages, D] mean-pool.
    return k_pages.to(torch.float32).mean(dim=1).to(k_pages.dtype)


def hisa_paged_logits_triton(
    q: torch.Tensor,                 # [N_q, H, D] bf16
    k_pages: torch.Tensor,           # [num_pages, P, D] bf16
    weights: torch.Tensor,           # [N_q, H] f32
    seqlens: torch.Tensor,           # [B] i32
    block_tables: torch.Tensor,      # [B, max_blocks] i32
    q_to_batch: torch.Tensor,        # [N_q] i32
    max_seq_len: int,
    config: HISAConfig,
    pooled_k: Optional[torch.Tensor] = None,  # [num_pages, D] bf16
) -> torch.Tensor:
    """Two-stage Triton implementation.

    Returns `[N_q, max_seq_len]` f32 logits, `-inf` outside selected blocks /
    beyond sequence length.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available; cannot run HISA Triton path.")
    assert q.is_cuda and k_pages.is_cuda
    assert q.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert q.dim() == 3 and k_pages.dim() == 3
    N_q, H, D = q.shape
    num_pages, P, Dk = k_pages.shape
    assert Dk == D
    assert P == config.block_size, (
        f"k_pages block_size {P} != config.block_size {config.block_size}"
    )
    B, max_blocks = block_tables.shape
    device = q.device

    # Stage 1: pooled K + block scoring.
    if pooled_k is None:
        pooled_k = _ensure_pooled_k(k_pages)
    assert pooled_k.shape == (num_pages, D)

    block_scores = torch.full(
        (N_q, max_blocks), float("-inf"), dtype=torch.float32, device=device
    )

    # Choose tile size along blocks dim. Triton wants power-of-2.
    BLOCK_BL = 1
    while BLOCK_BL < max_blocks and BLOCK_BL < 128:
        BLOCK_BL <<= 1
    BLOCK_BL = min(BLOCK_BL, 128)

    grid_b = (N_q, triton.cdiv(max_blocks, BLOCK_BL))
    _hisa_block_score_kernel[grid_b](
        q,
        pooled_k,
        weights,
        block_tables,
        seqlens,
        q_to_batch,
        block_scores,
        N_q,
        H=H,
        D=D,
        max_blocks=max_blocks,
        P=P,
        BLOCK_BL=BLOCK_BL,
    )

    # Top-N_b blocks per query.
    top_b = min(config.top_blocks, max_blocks)
    sel_scores, sel_logical = torch.topk(block_scores, k=top_b, dim=-1)
    # Mark invalid selections (-inf score => block doesn't exist) with -1.
    sel_logical = torch.where(
        torch.isinf(sel_scores) & (sel_scores < 0),
        torch.full_like(sel_logical, -1),
        sel_logical,
    )

    # Resolve to physical page indices via per-query batch.
    q_batch_long = q_to_batch.to(torch.long)
    safe_sel = sel_logical.clamp_min(0).to(torch.long)
    sel_pages = block_tables[q_batch_long.unsqueeze(-1).expand_as(safe_sel), safe_sel]
    sel_pages = sel_pages.to(torch.int32)
    # Where selection is invalid, page index is irrelevant (kernel checks lb<0).
    sel_logical = sel_logical.to(torch.int32)

    # Stage 2: token-level scoring inside selected blocks.
    out = torch.full(
        (N_q, max_seq_len), float("-inf"), dtype=torch.float32, device=device
    )
    grid_t = (N_q, top_b)
    _hisa_token_score_kernel[grid_t](
        q,
        k_pages,
        weights,
        sel_logical,
        sel_pages,
        seqlens,
        q_to_batch,
        out,
        N_q,
        max_seq_len=max_seq_len,
        H=H,
        D=D,
        P=P,
        top_b=top_b,
    )
    return out


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------


def hisa_paged_logits(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    weights: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    q_to_batch: torch.Tensor,
    max_seq_len: int,
    config: HISAConfig,
    pooled_k: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch to the Triton implementation when available.

    Raises if Triton is not present and the input is on CUDA.
    """
    if q.is_cuda and _HAS_TRITON:
        return hisa_paged_logits_triton(
            q,
            k_pages,
            weights,
            seqlens,
            block_tables,
            q_to_batch,
            max_seq_len,
            config,
            pooled_k=pooled_k,
        )
    return hisa_paged_logits_reference(
        q,
        k_pages,
        weights,
        seqlens,
        block_tables,
        q_to_batch,
        max_seq_len,
        config,
    )


__all__ = [
    "HISAConfig",
    "hisa_paged_logits",
    "hisa_paged_logits_triton",
    "hisa_paged_logits_reference",
    "dense_paged_logits_reference",
]
