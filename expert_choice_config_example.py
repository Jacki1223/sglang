"""
Example: 如何在 qwen2_moe.py 中添加可配置的 Expert Choice Routing

在 python/sglang/srt/models/qwen2_moe.py 的 Qwen2MoeSparseMoeBlock.__init__ 中修改：
"""

# 在文件顶部添加
import os

# 在 __init__ 方法中（第164行左右）修改为：

# 通过环境变量控制是否启用Expert Choice
# 默认False（性能优先），设置环境变量后启用（负载均衡优先）
use_expert_choice = os.getenv("SGLANG_EXPERT_CHOICE", "0") == "1"

self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    layer_id=layer_id,
    use_expert_choice=use_expert_choice,        # 可配置
    expert_capacity_factor=1.25,                 # 仅当启用时有效
)

# 使用方法：

# 1. 默认模式（性能优先，不启用Expert Choice）
# python -m sglang.launch_server --model-path ...

# 2. 启用Expert Choice（负载均衡优先）
# SGLANG_EXPERT_CHOICE=1 python -m sglang.launch_server --model-path ...
