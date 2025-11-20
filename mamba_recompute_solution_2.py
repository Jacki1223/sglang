"""
方案 2: 从 token IDs 重新运行前向传播到 Mamba layers

核心思路:
- 从 radix tree node 的 key (token IDs) 重新运行
- Embedding → LayerNorm → 每个 Mamba Layer 的前向
- 只计算必要的层，跳过全注意力层

优点:
- 不需要额外内存缓存
- 计算准确

缺点:
- 计算开销较大 (但比重新生成整个序列小得多)
- 需要访问 token IDs (已经在 radix tree 中)
"""

import torch
from typing import List

def recompute_mamba_state_v2(
    self,
    start_mamba_idx: int,
    target_mamba_idx: int,
    kv_indices: torch.Tensor,
) -> bool:
    """
    方案 2: 从 token IDs 重新计算 Mamba state

    关键挑战: 如何从 kv_indices 反推出 token IDs？
    答案: 我们需要修改 radix tree 节点存储，或者在重计算时传递 token IDs
    """
    linear_layer_ids = self.hybrid_gdn_config.mamba2_cache_params.layers
    mamba_pool = self.req_to_token_pool.mamba_pool

    # ❌ 问题：kv_indices 不能直接得到 token IDs
    # 我们需要从 RadixCache 传递 token_ids 过来

    # 假设我们修改接口，传递 token_ids:
    # def recompute_mamba_state(self, ..., token_ids: List[int])

    # 这个方案不可行，因为 kv_indices 指向的是 KV cache slots，
    # 而不是 token IDs！

    # 我们需要方案 3...
    return False


def alternative_approach_with_token_ids(
    self,
    start_mamba_idx: int,
    target_mamba_idx: int,
    token_ids: List[int],  # ← 从 radix tree 获取
) -> bool:
    """
    如果我们能获取 token IDs，这个方案就可行
    """
    linear_layer_ids = self.hybrid_gdn_config.mamba2_cache_params.layers
    mamba_pool = self.req_to_token_pool.mamba_pool

    # 1. Embedding
    token_ids_tensor = torch.tensor(token_ids, device=self.device)
    hidden_states = self.model.embed_tokens(token_ids_tensor)

    # 2. 初始化 mamba state
    if start_mamba_idx == -1:
        # 零初始化所有层的 mamba state
        temp_mamba_cache = self._create_zero_mamba_cache()
    else:
        # 克隆起始状态
        temp_mamba_cache = self._clone_mamba_cache(start_mamba_idx)

    # 3. 逐层处理
    for layer_idx in range(len(self.model.layers)):
        layer = self.model.layers[layer_idx]

        # Pre-LayerNorm
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # 判断是 Mamba layer 还是 Attention layer
        if layer_idx in linear_layer_ids:
            # === Mamba Layer ===
            output = torch.zeros_like(hidden_states)

            # 创建临时的 Mamba2Metadata (用于单 token decode)
            metadata = self._create_single_token_metadata(len(token_ids))

            # 调用 Mamba forward (会更新 temp_mamba_cache)
            layer.self_attn.forward(
                hidden_states=hidden_states,
                output=output,
                layer_cache=temp_mamba_cache[layer_idx],
                metadata=metadata,
            )
            hidden_states = output
        else:
            # === Attention Layer ===
            # 跳过！我们只关心 Mamba state
            # 但这里有个问题：hidden_states 会不一致...
            pass

        # Post-LayerNorm and residual
        hidden_states = residual + hidden_states

    # 4. 存储最终的 mamba state
    mamba_pool[target_mamba_idx] = temp_mamba_cache

    return True


# ❌ 问题: 这个方案也有缺陷
"""
主要问题:
1. 跳过 Attention layers 会导致 hidden_states 不准确
2. 后续的 Mamba layers 会基于错误的输入
3. 计算的 mamba state 不正确

结论: 需要完整运行整个模型，这与"重计算"的初衷相悖
"""
