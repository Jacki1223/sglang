# Expert Choice Routing 使用决策指南

## 🚨 重要提醒

**当前Expert Choice实现会降低性能！**
- 单次推理延迟：**增加 20-50%**
- 吞吐量（小batch）：**降低 20-30%**
- 吞吐量（大batch）：可能持平或略有提升（如果原本负载严重不均）

## ⚖️ 什么时候应该启用？

### ✅ 应该启用的场景

| 场景 | 原因 |
|------|------|
| **大batch离线推理** (batch > 128) | 负载均衡收益 > 算法开销 |
| **多GPU Expert Parallelism** | 通信开销降低 > 计算开销增加 |
| **原始负载极度不均** | 某些expert超载严重时 |
| **关注总吞吐量** | 可以容忍单次延迟增加 |

### ❌ 不应该启用的场景

| 场景 | 原因 |
|------|------|
| **在线推理服务** | 延迟敏感 |
| **小batch推理** (batch < 32) | 负载不均影响小，开销不值得 |
| **单GPU推理** | 负载均衡收益有限 |
| **离线推理（你的场景）** | **性能下降明显** ⬅️ |

## 🎯 推荐配置

### 对于你的离线推理场景

**推荐：不启用 Expert Choice**

```python
# python/sglang/srt/models/qwen2_moe.py
# 保持简单，不添加 use_expert_choice 参数

self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    layer_id=layer_id,
    # 不添加 use_expert_choice=True  ← 保持这样
)
```

### 如果想灵活切换

添加环境变量控制：

```python
import os

use_expert_choice = os.getenv("SGLANG_EXPERT_CHOICE", "0") == "1"

self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    layer_id=layer_id,
    use_expert_choice=use_expert_choice,  # 默认False
    expert_capacity_factor=1.25,
)
```

使用：
```bash
# 默认：高性能模式
python -m sglang.launch_server --model-path ...

# 需要负载均衡时
SGLANG_EXPERT_CHOICE=1 python -m sglang.launch_server --model-path ...
```

## 📊 性能对比（预估）

### Qwen3-Next 64层MoE, 64 experts

| 指标 | 标准Routing | Expert Choice | 差异 |
|------|-------------|---------------|------|
| **单层延迟** | 0.1ms | 0.3-0.5ms | ⬆️ 3-5x |
| **总推理延迟** | 100ms | 120-140ms | ⬆️ 20-40% |
| **吞吐量（batch=1)** | 10 tok/s | 7-8 tok/s | ⬇️ 20-30% |
| **吞吐量（batch=256)** | 1000 tok/s | 950-1050 tok/s | ≈ 持平 |
| **负载均衡** | 不均衡 | 完美均衡 | ⬆️ 90% |

## 🔄 如何切换回标准模式

### 如果你已经添加了 use_expert_choice=True

**方法1：直接删除**

```python
# 修改 qwen2_moe.py 第164行
self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    layer_id=layer_id,
    # 删除下面两行
    # use_expert_choice=True,
    # expert_capacity_factor=1.25,
)
```

**方法2：改为False**

```python
self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    layer_id=layer_id,
    use_expert_choice=False,  # 改为False
)
```

**方法3：使用Git回退**

```bash
cd python/sglang/srt/models
git checkout qwen2_moe.py  # 回退到原始版本
```

## 🚀 未来优化方向

Expert Choice的性能问题是可以解决的：

### 短期（1-2周）
- [ ] 优化的PyTorch实现（减少无用计算）
- [ ] 条件分支避免fallback开销

### 中期（1-2月）
- [ ] 定制CUDA kernel（预期5-10x加速）
- [ ] 与sgl-kernel集成

### 长期（3-6月）
- [ ] 自适应策略（小batch禁用，大batch启用）
- [ ] 混合routing（部分层使用expert choice）

到那时，Expert Choice会是性能和负载均衡的双赢方案。

## 💡 总结

**当前状态**：
- Expert Choice Routing **功能正常** ✅
- 但**性能较差** ❌
- 适合**特定场景**，不适合通用场景

**你的场景（离线推理）**：
- **不推荐**启用 Expert Choice
- 建议使用标准routing（高性能）
- 等待未来的CUDA kernel优化版本

**立即行动**：
1. 检查 `qwen2_moe.py` 是否添加了 `use_expert_choice=True`
2. 如果添加了，删除或改为 `False`
3. 重启服务，性能应该恢复正常
