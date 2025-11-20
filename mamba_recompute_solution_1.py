"""
方案 1: 缓存每个 Mamba layer 的输入 hidden states

优点:
- 实现最直接
- 重计算准确且快速

缺点:
- 内存开销巨大 (每个 token 每个 layer 都要存 hidden_state)
- 对于 80B 模型，每个 token 可能需要额外 几十 MB

实现思路:
"""

import torch

class EnhancedMambaCache:
    """增强的 Mamba Cache，额外存储每层的输入 hidden states"""

    def __init__(self, num_layers, hidden_size, max_tokens, device):
        # 原有的 mamba state
        self.conv_state = ...
        self.ssm_state = ...

        # 新增：每层的输入 hidden states 缓存
        # Shape: [max_tokens, num_layers, hidden_size]
        self.layer_inputs = torch.zeros(
            max_tokens, num_layers, hidden_size,
            dtype=torch.bfloat16,
            device=device
        )
        self.valid_mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)

    def store_layer_input(self, token_idx, layer_idx, hidden_state):
        """存储某个 token 在某层的输入 hidden state"""
        self.layer_inputs[token_idx, layer_idx] = hidden_state
        self.valid_mask[token_idx] = True

    def get_layer_input(self, token_idx, layer_idx):
        """获取某个 token 在某层的输入 hidden state"""
        if not self.valid_mask[token_idx]:
            return None
        return self.layer_inputs[token_idx, layer_idx]


def recompute_mamba_state_v1(
    self,
    start_mamba_idx: int,
    target_mamba_idx: int,
    kv_indices: torch.Tensor,
) -> bool:
    """
    方案 1 的实现：使用缓存的 hidden states 重计算
    """
    linear_layer_ids = self.hybrid_gdn_config.mamba2_cache_params.layers
    mamba_pool = self.req_to_token_pool.mamba_pool

    # 初始化状态
    if start_mamba_idx == -1:
        # 零初始化
        current_conv_state = torch.zeros_like(mamba_pool.conv_state[0])
        current_ssm_state = torch.zeros_like(mamba_pool.ssm_state[0])
    else:
        # 复制起始状态
        current_conv_state = mamba_pool.conv_state[start_mamba_idx].clone()
        current_ssm_state = mamba_pool.ssm_state[start_mamba_idx].clone()

    # 逐 token 逐层重计算
    for token_idx in kv_indices:
        for layer_idx in linear_layer_ids:
            # 获取缓存的 hidden state (这需要新的缓存机制)
            hidden_state = self.enhanced_cache.get_layer_input(token_idx, layer_idx)
            if hidden_state is None:
                return False  # 缓存不完整，重计算失败

            # 获取该层的 mamba layer
            mamba_layer = self.model.layers[layer_idx].self_attn.mamba_layer

            # 重新执行前向传播
            # 1. Projection
            projected_states, _ = mamba_layer.in_proj(hidden_state.unsqueeze(0))
            gate, hidden_states_B_C, dt = torch.split(projected_states, ...)

            # 2. Convolution (使用当前的 conv_state)
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                current_conv_state[layer_idx],  # 使用临时的 conv state
                mamba_layer.conv1d.weight,
                mamba_layer.conv1d.bias,
                ...
            )
            hidden_states, B, C = torch.split(hidden_states_B_C, ...)

            # 3. 更新 SSM state
            selective_state_update(
                current_ssm_state[layer_idx],  # 临时 ssm state (会被就地更新)
                hidden_states,
                dt,
                mamba_layer.A,
                B, C,
                mamba_layer.D,
                ...
            )

    # 存储最终状态
    mamba_pool.conv_state[target_mamba_idx] = current_conv_state
    mamba_pool.ssm_state[target_mamba_idx] = current_ssm_state

    return True


# 内存开销估算
def estimate_memory_overhead():
    """
    Qwen3-Next-80B-A3B-Instruct 的参数:
    - hidden_size = 8192
    - num_layers = 80
    - Mamba layers = 假设 40 层
    - max_cached_tokens = 100,000

    额外内存 = 100,000 * 40 * 8192 * 2 bytes (bfloat16)
             = 65 GB ❌ 不可接受！
    """
    pass
