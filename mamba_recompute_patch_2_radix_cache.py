"""
Patch 2: Enhanced MambaRadixCache with State Recomputation

This patch modifies python/sglang/srt/mem_cache/mamba_radix_cache.py
to support mamba state recomputation from tombstone nodes.

Key Changes:
1. Add model_runner reference to cache
2. Modify __init__ to accept recomputation parameters
3. Enhance _match_prefix_helper to support recomputation
4. Add _try_rebuild_mamba_state method
5. Improve eviction strategy to prioritize mamba retention
"""

# ==================== MODIFICATIONS TO __init__ ====================

def __init__(
    self,
    req_to_token_pool: HybridReqToTokenPool,
    token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    page_size: int,
    disable: bool = False,
    enable_metrics: bool = False,
    # ===== NEW PARAMETERS =====
    enable_recomputation: bool = False,
    recompute_max_tokens: int = 512,
    prioritize_mamba_retention: bool = True,
    mamba_eviction_threshold: float = 0.8,
    model_runner=None,  # Reference to ModelRunner for recomputation
):
    # ... existing initialization code ...

    # ===== NEW FIELDS =====
    self.enable_recomputation = enable_recomputation
    self.recompute_max_tokens = recompute_max_tokens
    self.prioritize_mamba_retention = prioritize_mamba_retention
    self.mamba_eviction_threshold = mamba_eviction_threshold
    self.model_runner = model_runner

    # Statistics
    self.recompute_hit_count = 0
    self.recompute_miss_count = 0
    self.recompute_skip_count = 0

    if enable_recomputation and model_runner is None:
        logger.warning(
            "Mamba state recomputation is enabled but model_runner is not provided. "
            "Recomputation will be disabled."
        )
        self.enable_recomputation = False


# ==================== ENHANCED _match_prefix_helper ====================

def _match_prefix_helper(
    self, key: RadixKey
) -> Tuple[List[torch.Tensor], TreeNode]:
    """
    Enhanced Mamba prefix matching with state recomputation support.

    When encountering tombstone nodes (nodes with full KV but no mamba state),
    this method can optionally recompute the mamba state from the nearest
    valid mamba state node.
    """
    node = self.root_node
    child_key = self.get_child_key_fn(key)

    value = []
    best_value_len = 0
    best_last_node = node

    # Track tombstone path for potential recomputation
    last_valid_mamba_node = None
    last_valid_mamba_len = 0
    tombstone_encountered = False
    tombstone_start_len = 0

    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]

        # Check if current node has valid mamba state
        if node.mamba_value is not None:
            best_value_len = len(value)
            best_last_node = node
            last_valid_mamba_node = node
            last_valid_mamba_len = len(value)
            tombstone_encountered = False  # Reset tombstone tracking
        elif node != self.root_node and not tombstone_encountered:
            # First tombstone node encountered
            tombstone_encountered = True
            tombstone_start_len = len(value)

        prefix_len = self.key_match_fn(child.key, key)
        if prefix_len < len(child.key):
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

    # Check the final node
    if node.mamba_value is not None:
        best_value_len = len(value)
        best_last_node = node

    # ===== NEW: ATTEMPT RECOMPUTATION =====
    if self.enable_recomputation and tombstone_encountered:
        recompute_len = len(value) - last_valid_mamba_len

        if (recompute_len > 0 and
            recompute_len <= self.recompute_max_tokens and
            last_valid_mamba_node is not None):

            # Try to rebuild mamba state
            rebuilt_node = self._try_rebuild_mamba_state(
                last_valid_mamba_node,
                value[last_valid_mamba_len:],
                node,
            )

            if rebuilt_node is not None:
                # Recomputation successful!
                best_value_len = len(value)
                best_last_node = rebuilt_node
                self.recompute_hit_count += 1
                logger.debug(
                    f"Mamba state recomputed successfully: "
                    f"{recompute_len} tokens, "
                    f"total hits: {self.recompute_hit_count}"
                )
            else:
                self.recompute_miss_count += 1
                logger.debug(
                    f"Mamba state recomputation failed: {recompute_len} tokens"
                )
        elif recompute_len > self.recompute_max_tokens:
            self.recompute_skip_count += 1
            logger.debug(
                f"Mamba state recomputation skipped: "
                f"{recompute_len} tokens exceeds threshold {self.recompute_max_tokens}"
            )

    # Update LRU lists
    node_update = best_last_node
    self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
    self.mamba_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

    # Update last access time for sanity check
    cur_time = get_last_access_time()
    while node_update:
        node_update.last_access_time = cur_time
        cur_time -= 0.00001
        node_update = node_update.parent

    return value[:best_value_len], best_last_node


# ==================== NEW: _try_rebuild_mamba_state ====================

def _try_rebuild_mamba_state(
    self,
    start_node: TreeNode,
    kv_indices_list: List[torch.Tensor],
    target_node: TreeNode,
) -> Optional[TreeNode]:
    """
    Attempt to rebuild mamba state from start_node to target_node.

    Args:
        start_node: Node with valid mamba_value to start from
        kv_indices_list: List of KV cache index tensors to recompute over
        target_node: Target node to assign the rebuilt mamba state

    Returns:
        target_node if successful, None otherwise
    """
    if self.model_runner is None:
        return None

    try:
        # Concatenate all KV indices
        if not kv_indices_list:
            return None

        kv_indices = torch.cat(kv_indices_list)
        num_tokens = len(kv_indices)

        if num_tokens == 0:
            return None

        # Get starting mamba state
        start_mamba_idx = start_node.mamba_value
        if start_mamba_idx is None:
            return None

        # Allocate new mamba state slot
        new_mamba_idx = self.req_to_token_pool.mamba_pool.alloc(1)
        if new_mamba_idx is None:
            # Try evicting and allocate again
            self.evict_mamba(1)
            new_mamba_idx = self.req_to_token_pool.mamba_pool.alloc(1)
            if new_mamba_idx is None:
                logger.warning("Failed to allocate mamba state for recomputation")
                return None

        # Call model_runner to recompute mamba state
        success = self.model_runner.recompute_mamba_state(
            start_mamba_idx=start_mamba_idx[0].item(),
            target_mamba_idx=new_mamba_idx[0].item(),
            kv_indices=kv_indices,
        )

        if not success:
            # Recomputation failed, free the allocated slot
            self.req_to_token_pool.mamba_pool.free(new_mamba_idx)
            return None

        # Update target node
        target_node.mamba_value = new_mamba_idx
        self.mamba_lru_list.insert_mru(target_node)
        self.mamba_evictable_size_ += 1

        return target_node

    except Exception as e:
        logger.warning(f"Mamba state recomputation failed with error: {e}")
        return None


# ==================== ENHANCED evict_mamba ====================

def evict_mamba(self, mamba_num: int) -> None:
    """Enhanced mamba eviction with prioritization strategy"""
    if self.disable or mamba_num <= 0:
        return

    # ===== NEW: PRIORITIZE MAMBA RETENTION =====
    if self.prioritize_mamba_retention:
        mamba_total = self.mamba_evictable_size_ + self.mamba_protected_size_
        if mamba_total > 0:
            mamba_usage = self.mamba_protected_size_ / mamba_total

            # If mamba usage is below threshold, try evicting full KV first
            if mamba_usage < self.mamba_eviction_threshold:
                # Estimate full KV tokens equivalent
                # Heuristic: 1 mamba state ≈ 10 full KV tokens
                full_tokens_to_evict = mamba_num * 10

                logger.debug(
                    f"Mamba usage {mamba_usage:.2%} < threshold {self.mamba_eviction_threshold:.2%}, "
                    f"evicting {full_tokens_to_evict} full KV tokens first"
                )

                self.evict(full_tokens_to_evict)

                # Check if we now have enough space
                if self.req_to_token_pool.mamba_pool.available_size() >= mamba_num:
                    return

    # Original eviction logic
    x = self.mamba_lru_list.get_lru_no_lock()
    mamba_num_evicted = 0

    while mamba_num_evicted < mamba_num and self.mamba_lru_list.in_list(x):
        assert x.mamba_value is not None
        assert len(x.mamba_value) == 1
        assert x != self.root_node
        assert x.mamba_lock_ref == 0

        if len(x.children) > 0:
            # Internal node - tombstone it
            self.req_to_token_pool.mamba_pool.free(x.mamba_value)
            mamba_num_evicted += len(x.mamba_value)
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
            self.mamba_lru_list.remove_node(x)
            self._tombstone_internal_node(x)
        else:
            # Leaf node - fully evict
            _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
            mamba_num_evicted += mamba_evicted_delta

        x = x_next


# ==================== NEW: Statistics Methods ====================

def get_recomputation_stats(self) -> Dict[str, int]:
    """Get statistics about mamba state recomputation"""
    return {
        "recompute_hit_count": self.recompute_hit_count,
        "recompute_miss_count": self.recompute_miss_count,
        "recompute_skip_count": self.recompute_skip_count,
        "total_attempts": self.recompute_hit_count + self.recompute_miss_count,
        "hit_rate": (
            self.recompute_hit_count / (self.recompute_hit_count + self.recompute_miss_count)
            if (self.recompute_hit_count + self.recompute_miss_count) > 0
            else 0.0
        ),
    }

def reset_recomputation_stats(self):
    """Reset recomputation statistics"""
    self.recompute_hit_count = 0
    self.recompute_miss_count = 0
    self.recompute_skip_count = 0
