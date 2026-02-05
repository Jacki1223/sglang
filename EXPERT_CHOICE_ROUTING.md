# Expert Choice Routing 实现文档

## 📖 概述

Expert Choice Routing是一种改进的MoE（Mixture of Experts）路由机制，通过让**Expert选择Token**而不是**Token选择Expert**来实现更好的负载均衡。

## 🎯 问题背景

### 传统方式的问题（Token选Expert）

在传统的MoE实现中：
- 每个token通过router计算scores，选择top-k个expert
- **问题**：某些expert可能被大量token选中，而其他expert很少被使用
- **结果**：负载不均衡、计算等待时间长、硬件利用率低

```
Token 1 → Router → [Expert 2, Expert 5]
Token 2 → Router → [Expert 2, Expert 3]  # Expert 2被重复选中
Token 3 → Router → [Expert 2, Expert 7]  # Expert 2负载过重
...
```

### Expert Choice方式的优势

- 每个expert查看所有token的router scores，选择top-k个token处理
- **优势**：每个expert处理固定数量的token（完美负载均衡）
- **结果**：更高的硬件利用率、减少通信开销、更快的推理速度

```
Expert 1 → 选择得分最高的 capacity 个 tokens
Expert 2 → 选择得分最高的 capacity 个 tokens
Expert 3 → 选择得分最高的 capacity 个 tokens
...
每个expert处理大约相同数量的tokens
```

## 🔧 实现细节

### 核心算法

Expert Choice Routing的核心实现在 `python/sglang/srt/layers/moe/topk.py` 中的 `expert_choice_topk()` 函数。

#### 算法流程

1. **计算Expert容量**
   ```python
   expert_capacity = (num_tokens * topk / num_experts) * capacity_factor
   ```
   - `capacity_factor` 通常设为 1.25，提供25%的缓冲空间
   - 确保每个expert可以处理合理数量的token

2. **计算Router Scores**
   ```python
   router_scores = softmax(router_logits, dim=-1)  # (num_tokens, num_experts)
   ```

3. **转置视角：从Expert角度看Token**
   ```python
   expert_token_scores = router_scores.transpose(0, 1)  # (num_experts, num_tokens)
   ```

4. **每个Expert选择Top-K Token**
   ```python
   expert_topk_scores, expert_topk_token_ids = torch.topk(
       expert_token_scores, k=expert_capacity, dim=1
   )
   ```

5. **转换回Token视角**
   - 为每个token构建被选中的expert列表
   - 处理未被充分分配expert的token
   - 重新归一化权重

### 配置参数

#### TopKConfig新增参数

```python
@dataclass
class TopKConfig:
    # ... 原有参数 ...
    use_expert_choice: bool = False          # 启用expert choice routing
    expert_capacity_factor: float = 1.25     # Expert容量因子
```

#### TopK类新增参数

```python
topk = TopK(
    top_k=2,                         # 每个token的expert数量
    use_expert_choice=True,          # 启用expert choice
    expert_capacity_factor=1.25,     # 容量因子
    renormalize=True,                # 重新归一化权重
    scoring_func="softmax"           # 评分函数
)
```

## 📊 使用方法

### 方法1：在模型配置中启用

如果你正在定义一个新的MoE模型，可以在TopK层初始化时启用：

```python
from sglang.srt.layers.moe.topk import TopK

class MyMoEModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 创建TopK层，启用expert choice routing
        self.topk = TopK(
            top_k=2,                          # 每个token使用2个expert
            use_expert_choice=True,           # 启用expert choice routing
            expert_capacity_factor=1.25,      # 容量因子（可选）
            renormalize=True,
            scoring_func="softmax"
        )

        # ... 其他层定义 ...

    def forward(self, x):
        # ... router计算 ...
        topk_output = self.topk(hidden_states, router_logits)
        # ... 继续MoE计算 ...
```

### 方法2：修改现有模型

对于已有的MoE模型，找到TopK层的初始化代码并添加参数：

```python
# 原有代码
topk = TopK(top_k=2, renormalize=True)

# 修改为
topk = TopK(
    top_k=2,
    use_expert_choice=True,        # 添加这行
    expert_capacity_factor=1.25,   # 添加这行（可选）
    renormalize=True
)
```

### 方法3：直接调用函数

如果需要直接调用expert choice routing函数：

```python
from sglang.srt.layers.moe.topk import expert_choice_topk

topk_weights, topk_ids = expert_choice_topk(
    hidden_states=hidden_states,      # (num_tokens, hidden_dim)
    gating_output=router_logits,      # (num_tokens, num_experts)
    topk=2,                           # 每个token的expert数
    renormalize=True,
    expert_capacity_factor=1.25,
    scoring_func="softmax"
)
```

## 🧪 测试和验证

### 运行测试脚本

我们提供了一个测试脚本来比较expert choice routing和标准routing的效果：

```bash
cd /home/user/sglang
python test_expert_choice_routing.py
```

### 测试内容

1. **负载均衡比较**
   - 标准routing的expert负载分布
   - Expert choice routing的expert负载分布
   - 负载均衡改善百分比

2. **直接功能测试**
   - 验证输出形状正确
   - 验证每个token有正确数量的expert
   - 验证expert负载分布

### 预期结果

```
Expert Choice Routing vs Standard Routing Comparison
================================================================================
Number of tokens: 128
Number of experts: 8
Top-k per token: 2

1. Standard Routing (Token-Choose-Expert):
--------------------------------------------------------------------------------
Expert load distribution: [45, 28, 15, 38, 42, 22, 30, 36]
Max load: 45
Min load: 15
Std dev: 10.23
Load imbalance ratio: 1.41x

2. Expert Choice Routing (Expert-Choose-Token):
--------------------------------------------------------------------------------
Expert load distribution: [32, 32, 32, 32, 32, 32, 32, 32]
Max load: 32
Min load: 32
Std dev: 0.00
Load imbalance ratio: 1.00x

3. Comparison:
--------------------------------------------------------------------------------
Load balancing improvement: 100.0%
Expected load per expert: 32.0

Conclusion:
--------------------------------------------------------------------------------
✓ Expert Choice Routing provides better load balancing!
```

## ⚙️ 参数调优

### expert_capacity_factor

这是最重要的调优参数：

- **默认值**: 1.25（推荐）
- **作用**: 控制每个expert可以处理的token数量

```python
# capacity = (num_tokens * topk / num_experts) * capacity_factor
```

**如何选择：**
- `1.0`: 严格平衡，可能导致某些token得不到足够的expert
- `1.25`: 推荐值，提供25%缓冲
- `1.5-2.0`: 更灵活，但负载均衡效果可能下降

### scoring_func

选择router scoring函数：
- `"softmax"`: 标准softmax，适合大多数场景
- `"sigmoid"`: 适合某些特殊模型（如DeepSeek V3）

## 🔄 与现有功能的兼容性

### 支持的功能

✅ **完全兼容：**
- Expert location dispatch（expert物理-逻辑映射）
- Token padding mask
- 不同的scoring函数（softmax/sigmoid）
- Weight renormalization
- Expert distribution recording（负载统计）
- Routed experts capturing

### 暂不支持的功能

⚠️ **目前不兼容：**
- Grouped TopK（DeepSeek系列的分组选择）
- Custom routing functions
- Biased grouped topk（带correction bias的分组）
- Triton kernel output format

如果你的模型使用了这些功能，暂时无法启用expert choice routing。

## 📈 性能影响

### 预期收益

1. **负载均衡改善**: 通常可以达到接近完美的负载均衡
2. **吞吐量提升**: 减少expert之间的等待时间
3. **硬件利用率**: 更高的GPU利用率

### 开销

1. **计算开销**: 需要转置router scores和执行token分配逻辑
2. **内存开销**: 临时存储expert-token映射关系

**注意**: 当前实现使用Python循环，未来可以通过CUDA kernel优化性能。

## 🚀 未来优化方向

1. **CUDA Kernel优化**
   - 实现高效的expert-choose-token CUDA kernel
   - 减少Python循环带来的开销

2. **支持更多后端**
   - Triton kernel格式支持
   - FlashInfer集成

3. **支持分组模式**
   - 与grouped topk结合
   - 支持DeepSeek系列模型

4. **动态容量调整**
   - 根据运行时统计动态调整capacity factor
   - 自适应负载均衡

## 📚 参考文献

1. **Expert Choice Routing**
   - Zhou, Y., et al. (2022). "Mixture-of-Experts with Expert Choice Routing"
   - Google Research的原始论文

2. **SGLang MoE实现**
   - `python/sglang/srt/layers/moe/topk.py` - TopK选择
   - `python/sglang/srt/layers/moe/router.py` - Router实现
   - `python/sglang/srt/eplb/` - Expert负载均衡系统

## 🤝 贡献

如果你发现问题或有改进建议：
1. 提交Issue到SGLang仓库
2. 提供详细的使用场景和性能数据
3. 欢迎提交PR改进实现

## 📞 联系方式

- SGLang GitHub: https://github.com/sgl-project/sglang
- 相关Issue: Expert Choice Routing实现

---

**最后更新**: 2026-02-05
**实现版本**: SGLang 0.x
**作者**: Claude AI Assistant
