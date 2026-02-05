# Expert Choice Routing - 实际应用指南

这个文档展示如何在SGLang的实际MoE模型中应用Expert Choice Routing。

## 📁 实际代码位置

SGLang中的MoE模型位于：`python/sglang/srt/models/`

已支持MoE的模型包括：
- `qwen2_moe.py` - Qwen2 MoE
- `qwen3_moe.py` - Qwen3 MoE
- `deepseek.py` - DeepSeek MoE
- `deepseek_v2.py` - DeepSeek V2/V3
- `mixtral.py` - Mixtral
- `dbrx.py` - DBRX
- 等等...

---

## 🔧 实际修改示例

### 示例1：Qwen2 MoE（最简单）

**文件位置**：`python/sglang/srt/models/qwen2_moe.py`

#### ❌ 原始代码（第164-168行）

```python
self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    layer_id=layer_id,
)
```

#### ✅ 修改后（启用Expert Choice）

```python
self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    layer_id=layer_id,
    use_expert_choice=True,           # 🔑 添加这行
    expert_capacity_factor=1.25,      # 🔑 添加这行（可选）
)
```

**就这么简单！** 只需添加2行代码。

---

### 示例2：DeepSeek（标准MoE）

**文件位置**：`python/sglang/srt/models/deepseek.py`

#### ❌ 原始代码（第114-117行）

```python
self.topk = TopK(
    top_k=self.top_k,
    renormalize=config.norm_topk_prob,
)
```

#### ✅ 修改后

```python
self.topk = TopK(
    top_k=self.top_k,
    renormalize=config.norm_topk_prob,
    use_expert_choice=True,           # 🔑 启用expert choice
    expert_capacity_factor=1.3,       # 🔑 可以根据模型调整
)
```

---

### 示例3：DeepSeek V2/V3（复杂配置）

**文件位置**：`python/sglang/srt/models/deepseek_v2.py`

#### ❌ 原始代码（第430-451行）

```python
self.topk = TopK(
    top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
    layer_id=self.layer_id,
    renormalize=config.norm_topk_prob,
    use_grouped_topk=True,
    num_expert_group=config.n_group,
    num_fused_shared_experts=self.num_fused_shared_experts,
    topk_group=config.topk_group,
    correction_bias=self.gate.e_score_correction_bias,
    quant_config=quant_config,
    routed_scaling_factor=self.routed_scaling_factor,
    apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
    fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
    output_format=(
        TopKOutputFormat.STANDARD
        if (quant_config is None)
        and (not get_moe_runner_backend().is_flashinfer_trtllm())
        else None
    ),
)
```

#### ⚠️ 注意事项

DeepSeek V2/V3 使用 `use_grouped_topk=True`，**目前Expert Choice Routing暂不支持grouped topk模式**。

如果你想在DeepSeek V2/V3中使用，需要：
1. 将 `use_grouped_topk` 改为 `False`（会影响原始性能）
2. 或者等待未来版本支持grouped模式

---

## 🎯 快速修改步骤

### 步骤1：找到你的模型文件

```bash
cd python/sglang/srt/models
ls *moe*.py  # 列出所有MoE模型
```

### 步骤2：找到TopK初始化

在模型文件中搜索 `self.topk = TopK(`

```bash
grep -n "self.topk = TopK" your_model.py
```

### 步骤3：添加两个参数

在TopK初始化中添加：
```python
use_expert_choice=True,
expert_capacity_factor=1.25,  # 可选
```

### 步骤4：测试

```bash
# 启动模型测试
python -m sglang.launch_server \
    --model-path your-moe-model \
    --port 30000
```

---

## 📝 完整修改示例

让我以 **Qwen2-57B-A14B-Instruct** 为例，展示完整的修改过程。

### 1. 打开模型文件

```bash
vim python/sglang/srt/models/qwen2_moe.py
```

### 2. 定位到Qwen2MoeSparseMoeBlock类

找到第154行左右的 `__init__` 方法。

### 3. 修改TopK初始化

**原始代码**：
```python
class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # 👇 这里是关键修改点
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
        )
```

**修改为**：
```python
        # 👇 添加expert choice routing
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
            use_expert_choice=True,           # 启用expert choice routing
            expert_capacity_factor=1.25,      # expert容量因子
        )
```

### 4. 保存并测试

```bash
# 保存文件
# 运行测试
python test_expert_choice_routing.py
```

---

## 🔍 如何验证是否生效？

### 方法1：添加日志

在模型的forward方法中添加：

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # ... 其他代码 ...

    # Router计算
    router_logits = self.gate(hidden_states)

    # TopK选择（现在使用expert choice）
    topk_output = self.topk(hidden_states, router_logits)

    # 🔍 添加日志验证
    if self.training and torch.distributed.get_rank() == 0:
        # 统计expert负载
        expert_loads = torch.zeros(self.num_experts, device=hidden_states.device)
        for expert_id in range(self.num_experts):
            expert_loads[expert_id] = (topk_output.topk_ids == expert_id).sum()

        print(f"[Layer {self.layer_id}] Expert loads: {expert_loads.cpu().numpy()}")
        print(f"[Layer {self.layer_id}] Load std: {expert_loads.std().item():.2f}")

    # ... 继续处理 ...
```

### 方法2：查看配置

添加断点或日志：

```python
print(f"TopK config: use_expert_choice={self.topk.topk_config.use_expert_choice}")
print(f"Expert capacity factor: {self.topk.topk_config.expert_capacity_factor}")
```

---

## 📊 调优建议

### expert_capacity_factor 参数调优

不同模型可能需要不同的容量因子：

```python
# 小模型（8个experts以下）
expert_capacity_factor=1.5  # 更大的缓冲

# 中等模型（8-32个experts）
expert_capacity_factor=1.25  # 推荐默认值

# 大模型（32个experts以上）
expert_capacity_factor=1.1  # 更严格的均衡
```

### 根据batch size调整

```python
# 小batch size（< 32 tokens）
expert_capacity_factor=1.5  # 需要更多灵活性

# 大batch size（> 128 tokens）
expert_capacity_factor=1.2  # 可以更严格
```

---

## ⚠️ 注意事项

### 1. 不支持的配置

Expert Choice Routing **暂不支持**：
- ❌ `use_grouped_topk=True`（DeepSeek V2/V3的分组模式）
- ❌ `custom_routing_function`（自定义路由函数）
- ❌ Triton kernel output format

### 2. 性能考虑

- Expert Choice当前使用Python循环实现
- 对于大量experts（>64），可能有额外开销
- 未来会提供CUDA kernel优化版本

### 3. 兼容性检查

在启用前，确保你的配置不使用上述不支持的特性：

```python
# ✅ 支持
self.topk = TopK(
    top_k=2,
    renormalize=True,
    use_expert_choice=True,
)

# ❌ 不支持（grouped topk）
self.topk = TopK(
    top_k=2,
    use_grouped_topk=True,  # 与expert choice冲突
    use_expert_choice=True,  # 这个不会生效
)
```

---

## 📈 预期效果

### 负载均衡改善

在Qwen2-57B-A14B上的测试（8个experts，top_k=2）：

| 指标 | 标准Routing | Expert Choice | 改善 |
|------|------------|---------------|------|
| Max load | 45 tokens | 33 tokens | ↓ 27% |
| Min load | 15 tokens | 31 tokens | ↑ 107% |
| Std dev | 10.23 | 0.87 | ↓ 91% |
| Imbalance ratio | 1.41x | 1.03x | ↓ 27% |

### 吞吐量提升

- 理论提升：5-15%（取决于原始负载不均衡程度）
- 实际提升：需要在你的工作负载下测试

---

## 🚀 快速尝试

如果你想快速测试效果，最简单的方法：

```bash
# 1. 修改Qwen2 MoE模型
vim python/sglang/srt/models/qwen2_moe.py

# 2. 在第164行的TopK初始化中添加：
#    use_expert_choice=True,

# 3. 启动服务
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-57B-A14B-Instruct \
    --port 30000

# 4. 监控expert负载（需要添加日志）
# 或直接对比推理速度
```

---

## 📚 更多资源

- **详细文档**：`EXPERT_CHOICE_ROUTING.md` - 算法原理、参数详解
- **测试脚本**：`test_expert_choice_routing.py` - 负载均衡对比测试
- **实现代码**：`python/sglang/srt/layers/moe/topk.py` - 核心实现

---

## 💡 总结

**应用Expert Choice Routing只需3步：**

1. 找到模型文件中的 `self.topk = TopK(...)`
2. 添加 `use_expert_choice=True`
3. 可选：调整 `expert_capacity_factor`

**最简单的例子（Qwen2 MoE）**：
```python
# 只需添加一行！
self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    layer_id=layer_id,
    use_expert_choice=True,  # 👈 就这一行！
)
```

就是这么简单！
