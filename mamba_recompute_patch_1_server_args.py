"""
Patch 1: Add configuration parameters to ServerArgs

Add these fields to the ServerArgs dataclass in python/sglang/srt/server_args.py
Insert after line ~478 (after disable_radix_cache)
"""

# ==================== ADD THESE FIELDS TO ServerArgs ====================

    # Mamba Radix Cache Recomputation Settings
    enable_mamba_state_recomputation: bool = False
    """Enable recomputation of mamba states from tombstone nodes.
    When enabled, if a cache node has full KV cache but missing mamba state,
    the system will attempt to recompute the mamba state from the nearest
    valid mamba state node. This can significantly improve cache hit rate
    for hybrid GDN models like Qwen3-Next.
    """

    mamba_recompute_max_tokens: int = 512
    """Maximum number of tokens to recompute for mamba state reconstruction.
    If the distance from the last valid mamba state exceeds this threshold,
    recomputation will be skipped to avoid excessive overhead.
    Recommended: 256-1024 depending on your latency tolerance.
    """

    mamba_recompute_batch_size: int = 1
    """Batch size for mamba state recomputation.
    Currently only batch_size=1 is supported.
    """

    prioritize_mamba_retention: bool = True
    """When evicting cache, prioritize retaining mamba states over full KV cache.
    This reduces tombstone node creation and improves cache hit rate.
    """

    mamba_eviction_threshold: float = 0.8
    """Threshold for triggering full KV eviction instead of mamba eviction.
    When mamba usage is below this threshold, the system will try to evict
    full KV cache first to preserve mamba states.
    Range: 0.0-1.0, where 0.8 means evict full KV when mamba usage < 80%.
    """


# ==================== ADD CLI ARGUMENTS ====================
# Add to ServerArgs.add_cli_args() method around line ~1500

def add_mamba_recompute_args(parser):
    """Add mamba recomputation arguments to argument parser"""

    parser.add_argument(
        "--enable-mamba-state-recomputation",
        action="store_true",
        default=False,
        help=(
            "Enable recomputation of mamba states from tombstone nodes. "
            "This can significantly improve cache hit rate for hybrid GDN models "
            "like Qwen3-Next by reconstructing missing mamba states from cached KV data."
        ),
    )

    parser.add_argument(
        "--mamba-recompute-max-tokens",
        type=int,
        default=512,
        help=(
            "Maximum number of tokens to recompute for mamba state reconstruction. "
            "If the distance exceeds this threshold, recomputation is skipped. "
            "Recommended: 256-1024. Default: 512."
        ),
    )

    parser.add_argument(
        "--mamba-recompute-batch-size",
        type=int,
        default=1,
        help="Batch size for mamba state recomputation. Currently only 1 is supported.",
    )

    parser.add_argument(
        "--prioritize-mamba-retention",
        action="store_true",
        default=True,
        help=(
            "When evicting cache, prioritize retaining mamba states over full KV cache. "
            "This reduces tombstone node creation."
        ),
    )

    parser.add_argument(
        "--mamba-eviction-threshold",
        type=float,
        default=0.8,
        help=(
            "Threshold for triggering full KV eviction instead of mamba eviction. "
            "Range: 0.0-1.0. Default: 0.8 (80%%)."
        ),
    )
