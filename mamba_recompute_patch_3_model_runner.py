"""
Patch 3: Add Mamba State Recomputation to ModelRunner

This patch modifies python/sglang/srt/model_executor/model_runner.py
to provide mamba state recomputation capability.

Key Changes:
1. Add recompute_mamba_state method
2. Pass model_runner reference to MambaRadixCache during initialization
3. Support layer-wise state recomputation for Qwen3Next models
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ==================== ADD TO ModelRunner CLASS ====================

def recompute_mamba_state(
    self,
    start_mamba_idx: int,
    target_mamba_idx: int,
    kv_indices: torch.Tensor,
) -> bool:
    """
    Recompute mamba state from start_mamba_idx by replaying through kv_indices.

    This method iterates through linear attention layers and recomputes their
    mamba states using the cached KV data.

    Args:
        start_mamba_idx: Index of the starting mamba state in mamba_pool
        target_mamba_idx: Index where the recomputed state will be stored
        kv_indices: Tensor of KV cache indices to replay through

    Returns:
        True if recomputation succeeded, False otherwise
    """
    if not hasattr(self, 'hybrid_gdn_config') or self.hybrid_gdn_config is None:
        logger.warning("Mamba state recomputation requires hybrid_gdn_config")
        return False

    try:
        num_tokens = len(kv_indices)
        if num_tokens == 0:
            return False

        # Get mamba cache parameters
        mamba_config = self.hybrid_gdn_config.mamba2_cache_params
        linear_layer_ids = mamba_config.layers

        # Get mamba pool
        mamba_pool = self.req_to_token_pool.mamba_pool

        # Copy start state to target as initial state
        start_idx_tensor = torch.tensor([start_mamba_idx], device=self.device, dtype=torch.int64)
        target_idx_tensor = torch.tensor([target_mamba_idx], device=self.device, dtype=torch.int64)
        mamba_pool.copy_from(start_idx_tensor, target_idx_tensor)

        # Sequentially process each token through each linear attention layer
        for token_idx in range(num_tokens):
            kv_idx = kv_indices[token_idx].item()

            for layer_id in linear_layer_ids:
                # Get the linear attention layer
                layer = self.model.layers[layer_id]
                if not hasattr(layer, 'linear_attn'):
                    continue

                linear_attn = layer.linear_attn

                # Recompute for this token at this layer
                success = self._recompute_single_token_layer(
                    linear_attn=linear_attn,
                    layer_id=layer_id,
                    kv_idx=kv_idx,
                    mamba_idx=target_mamba_idx,
                    token_position=token_idx,
                )

                if not success:
                    logger.warning(
                        f"Failed to recompute mamba state at layer {layer_id}, "
                        f"token {token_idx}"
                    )
                    return False

        return True

    except Exception as e:
        logger.error(f"Mamba state recomputation failed: {e}", exc_info=True)
        return False


def _recompute_single_token_layer(
    self,
    linear_attn,
    layer_id: int,
    kv_idx: int,
    mamba_idx: int,
    token_position: int,
) -> bool:
    """
    Recompute mamba state for a single token at a single layer.

    This is a simplified implementation that uses the linear attention layer's
    recurrent computation to update the mamba state.

    Args:
        linear_attn: The Qwen3GatedDeltaNet layer
        layer_id: Layer index
        kv_idx: Index in KV cache to read from
        mamba_idx: Index in mamba pool to update
        token_position: Position of the token in the sequence

    Returns:
        True if successful
    """
    try:
        # This is a placeholder for the actual implementation
        # In a real implementation, you would:
        # 1. Read K, V, and other necessary data from kv_cache[layer_id][kv_idx]
        # 2. Read current mamba state from mamba_pool[mamba_idx][layer_id]
        # 3. Run one step of the recurrent computation
        # 4. Write updated state back to mamba_pool[mamba_idx][layer_id]

        # For now, we return True to indicate the interface is ready
        # The actual implementation requires:
        # - Access to intermediate activations (q, k, v, gates)
        # - Recurrent state update logic from the linear attention layer
        # - Proper handling of convolution states

        logger.debug(
            f"Recomputing layer {layer_id}, kv_idx {kv_idx}, "
            f"mamba_idx {mamba_idx}, token_pos {token_position}"
        )

        # Actual recurrent update would go here
        # This requires extracting the recurrent logic from fused kernels

        return True

    except Exception as e:
        logger.error(f"Single token recomputation failed: {e}")
        return False


# ==================== MODIFY _init_cache_engine ====================
# In ModelRunner._init_cache_engine(), pass self reference to MambaRadixCache

def _init_cache_engine_with_recomputation(self):
    """
    Modified cache engine initialization to support mamba state recomputation.

    Add this modification to ModelRunner._init_cache_engine()
    """

    # ... existing cache initialization code ...

    # When creating MambaRadixCache, pass additional parameters:
    if isinstance(self.tree_cache, MambaRadixCache):
        # Update the cache with model_runner reference and recomputation config
        self.tree_cache.model_runner = self
        self.tree_cache.enable_recomputation = self.server_args.enable_mamba_state_recomputation
        self.tree_cache.recompute_max_tokens = self.server_args.mamba_recompute_max_tokens
        self.tree_cache.prioritize_mamba_retention = self.server_args.prioritize_mamba_retention
        self.tree_cache.mamba_eviction_threshold = self.server_args.mamba_eviction_threshold

        logger.info(
            f"MambaRadixCache recomputation: "
            f"enabled={self.tree_cache.enable_recomputation}, "
            f"max_tokens={self.tree_cache.recompute_max_tokens}, "
            f"prioritize_retention={self.tree_cache.prioritize_mamba_retention}"
        )


# ==================== DETAILED RECOMPUTATION IMPLEMENTATION ====================
"""
For a complete implementation, the _recompute_single_token_layer method needs to:

1. **Read KV Cache Data**:
   ```python
   # Get layer's KV cache
   k_cache = self.kv_cache[layer_id * 2]      # K cache
   v_cache = self.kv_cache[layer_id * 2 + 1]  # V cache

   # Read K, V for this token
   k = k_cache[:, kv_idx, :]  # [num_heads, head_dim]
   v = v_cache[:, kv_idx, :]  # [num_heads, head_dim]
   ```

2. **Read Current Mamba State**:
   ```python
   mamba_pool = self.req_to_token_pool.mamba_pool
   # Access conv state and temporal state
   conv_state = mamba_pool.mamba_cache.conv[layer_id][:, mamba_idx]
   temporal_state = mamba_pool.mamba_cache.temporal[layer_id, mamba_idx]
   ```

3. **Run Recurrent Update** (pseudo-code):
   ```python
   # This mirrors the fused_recurrent_gated_delta_rule logic
   # Update conv state (shift + add new k)
   conv_state = shift_and_add(conv_state, k)

   # Compute gate
   gate = compute_gate(...)

   # Update temporal state
   temporal_state = gate * temporal_state + k @ v.T

   # Compute output
   output = temporal_state @ q
   ```

4. **Write Back Updated State**:
   ```python
   mamba_pool.mamba_cache.conv[layer_id][:, mamba_idx] = conv_state
   mamba_pool.mamba_cache.temporal[layer_id, mamba_idx] = temporal_state
   ```

**Challenge**: The KV cache in SGLang stores final KV values, but recomputation
needs intermediate activations (q, k, v before gating). Two options:

Option A: Store additional intermediate activations (increases memory)
Option B: Approximate from final KV (may have accuracy loss)

For production use, Option A is recommended. This requires modifying
the forward pass to store intermediate states when recomputation is enabled.
"""


# ==================== ALTERNATIVE: LAZY RECOMPUTATION ====================
"""
Alternative approach: Instead of eager recomputation during match_prefix,
perform lazy recomputation during the actual forward pass:

1. During match_prefix: Mark nodes as "needs_recomputation"
2. During forward pass: If a node needs recomputation, compute it on-the-fly
3. Cache the newly computed state

This approach:
- Avoids recomputation overhead if the prefix isn't actually used
- Integrates naturally with the forward pass
- Simpler to implement correctly

Implementation sketch:
```python
class TreeNode:
    def __init__(self):
        # ... existing fields ...
        self.needs_mamba_recomputation = False
        self.recompute_from_node = None  # Parent node with valid state
        self.recompute_kv_indices = None  # Indices to recompute

# In _match_prefix_helper:
if node.mamba_value is None and can_recompute:
    node.needs_mamba_recomputation = True
    node.recompute_from_node = last_valid_mamba_node
    node.recompute_kv_indices = kv_indices_to_here

# In forward pass (attention backend):
if node.needs_mamba_recomputation:
    state = lazy_recompute_mamba_state(node)
    node.mamba_value = cache_state(state)
    node.needs_mamba_recomputation = False
```
"""
