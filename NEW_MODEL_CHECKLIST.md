# SGLang 新模型支持快速检查清单

使用此清单跟踪新模型实现进度。基于 MiDashengLM 实际经验总结。

## 📋 阶段 1: 模型分析 (1-2小时)

### 1.1 配置分析
- [ ] 使用 `AutoConfig.from_pretrained()` 加载模型配置
- [ ] 记录 `config.model_type` 和 `config.architectures`
- [ ] 检查 `text_config.model_type` (解码器类型)
- [ ] 检查 `rope_scaling` 配置（如果有）
- [ ] 检查 `vision_config` 或 `audio_config`（如果有）
- [ ] 列出所有特殊配置参数

**输出**: 配置分析文档

```python
# 示例代码
from transformers import AutoConfig
config = AutoConfig.from_pretrained("model-name", trust_remote_code=True)
print(f"Model type: {config.model_type}")
print(f"Decoder: {config.text_config.model_type}")
print(f"RoPE: {getattr(config.text_config, 'rope_scaling', 'None')}")
```

### 1.2 参考实现对比
- [ ] 搜索 vLLM 是否有实现
- [ ] 对比 vLLM 与 HuggingFace 实现差异
- [ ] 记录关键差异点（解码器、RoPE、投影器等）
- [ ] 检查 vLLM 的权重加载逻辑
- [ ] 确认 bias 参数设置

**输出**: 差异对比文档

### 1.3 依赖确认
- [ ] 确认解码器类型（Qwen2, Llama, GPT 等）
- [ ] 检查 SGLang 是否已支持该解码器
- [ ] 确认编码器类型（Whisper, CLIP 等）
- [ ] 列出需要新增的组件

**输出**: 依赖清单

---

## 📐 阶段 2: 架构设计 (2-4小时)

### 2.1 组件设计
- [ ] 设计编码器类（音频/视觉）
  - [ ] 输入格式：`[batch, ...]`
  - [ ] 输出格式：`[batch, seq_len, hidden_dim]`
- [ ] 设计投影器类
  - [ ] 输入维度 = 编码器输出维度
  - [ ] 输出维度 = 语言模型隐藏维度
  - [ ] 下采样因子（如果有）
- [ ] 设计元数据类
  - [ ] 提取多模态输入
  - [ ] 计算偏移位置

**输出**: 类结构图

### 2.2 数据流设计
- [ ] 绘制前向传播流程图
  ```
  输入 -> 编码器 -> 投影器 -> 融合 -> 语言模型 -> 输出
  ```
- [ ] 确定特征融合策略
- [ ] 计算序列长度变化

**输出**: 数据流程图

---

## 💻 阶段 3: 代码实现 (4-8小时)

### 3.1 创建模型文件
- [ ] 复制 `model_template.py` 到 `python/sglang/srt/models/your_model.py`
- [ ] 替换所有 `YOUR_MODEL` 占位符
- [ ] 添加必要的导入

### 3.2 实现编码器
- [ ] 实现 `__init__` 方法
- [ ] 实现 `forward` 方法
- [ ] 测试编码器输出形状正确

**验证**:
```python
encoder = YourEncoder(config)
dummy_input = torch.randn(1, ...)
output = encoder(dummy_input)
print(f"Encoder output shape: {output.shape}")  # 应该是 [1, seq_len, hidden_dim]
```

### 3.3 实现投影器
- [ ] 实现 `__init__` 方法
- [ ] **关键**: 确认 `bias=True/False` 与预训练模型一致
- [ ] 实现 `forward` 方法（包括下采样逻辑）
- [ ] 测试投影器输出形状正确

**验证**:
```python
projector = YourProjector(in_dim, out_dim)
dummy_input = torch.randn(1, seq_len, in_dim)
output = projector(dummy_input)
print(f"Projector output shape: {output.shape}")  # 应该是 [1, new_seq_len, out_dim]
```

### 3.4 实现元数据类
- [ ] 实现 `__init__` 方法
- [ ] 提取多模态输入列表
- [ ] 计算偏移位置
- [ ] 添加调试打印

### 3.5 实现主模型类
- [ ] 实现 `__init__`
  - [ ] 初始化编码器
  - [ ] 初始化投影器
  - [ ] 初始化语言模型
  - [ ] **不修改** rope_scaling 等配置
- [ ] 实现 `forward`
  - [ ] 提取元数据
  - [ ] 编码多模态输入
  - [ ] 获取文本嵌入
  - [ ] 融合特征
  - [ ] 语言模型生成
- [ ] 实现 `_load_and_preprocess`
- [ ] 实现 `_merge_multimodal_embeddings`
- [ ] 实现 `get_mm_input_encoder_grouped_output_size`

**关键检查**:
```python
# ✅ 正确：保留原始配置
text_config = config.text_config
self.language_model = Qwen2ForCausalLM(text_config, ...)

# ❌ 错误：修改配置
if "mrope_section" in text_config.rope_scaling:
    text_config.rope_scaling.pop("mrope_section")  # 不要这样做！
```

---

## 📦 阶段 4: 权重加载 (2-4小时)

### 4.1 实现加载逻辑
- [ ] 实现 `load_weights` 方法
- [ ] 处理编码器权重
- [ ] 处理投影器权重
- [ ] 处理语言模型权重
- [ ] 添加详细的统计打印

### 4.2 调试权重加载
- [ ] 打印所有参数名称
  ```python
  for name, param in model.named_parameters():
      print(f"Model param: {name}, shape: {param.shape}")
  ```
- [ ] 打印所有检查点权重名称
  ```python
  for name, weight in weights:
      print(f"Checkpoint weight: {name}, shape: {weight.shape}")
  ```
- [ ] 对比名称差异，实现映射

### 4.3 验证加载完整性
- [ ] 运行模型，检查加载日志
- [ ] **确保** "Skipped weights: 0"
- [ ] 如果有跳过，诊断原因：
  - [ ] 检查 `bias=True/False` 设置
  - [ ] 检查参数名称映射
  - [ ] 检查是否在缓冲区中

**期望输出**:
```
================================================================================
Weight Loading Summary
================================================================================
✅ Encoder weights loaded: 120
✅ Projector weights loaded: 4
✅ Language model weights loaded: 256

✅ All weights loaded successfully!
================================================================================
```

---

## 🔧 阶段 5: 注册和集成 (1小时)

### 5.1 注册模型
- [ ] 编辑 `python/sglang/srt/models/__init__.py`
- [ ] 在 `_MODELS` 字典中添加条目
- [ ] 在 `_AUDIO_MODELS` 或 `_VISION_MODELS` 中添加（如果适用）
- [ ] 验证导入无错误

```python
# 在 __init__.py 中
_MODELS = {
    "YourModelForConditionalGeneration": ("your_model", "YourModelForConditionalGeneration"),
}

_AUDIO_MODELS = {
    "YourModelForConditionalGeneration",
}
```

### 5.2 测试导入
```bash
python -c "from sglang.srt.models import YourModelForConditionalGeneration; print('Import successful!')"
```

---

## 🧪 阶段 6: 测试验证 (4-8小时)

### 6.1 启动服务器测试
- [ ] 创建启动脚本
  ```bash
  python -m sglang.launch_server \
      --model your-org/your-model \
      --trust-remote-code \
      --enable-multimodal \
      --dtype float16 \
      --port 30000
  ```
- [ ] 检查启动日志无错误
- [ ] 检查权重加载完整

### 6.2 单输入测试
- [ ] 创建 `test_your_model.py`
- [ ] 测试单个音频/图像输入
- [ ] 验证输出合理

```python
# test_your_model.py
import requests

response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "your-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "test.mp3"}},
                {"type": "text", "text": "请转录音频。"}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 512,
    }
)
print(response.json())
```

- [ ] 测试通过
- [ ] 输出质量检查

### 6.3 批量测试
- [ ] 准备测试数据集（5-10个样本）
- [ ] 使用离线推理脚本测试
- [ ] 验证所有样本都能正常处理
- [ ] 检查性能指标（延迟、吞吐量）

### 6.4 对比验证
- [ ] 与 HuggingFace 原始模型对比输出
- [ ] 与 vLLM 实现对比输出（如果有）
- [ ] 确保结果一致性

**对比测试代码**:
```python
# 1. SGLang 输出
sglang_output = sglang_model.generate(...)

# 2. HuggingFace 输出
hf_output = hf_model.generate(...)

# 3. 对比
print(f"SGLang: {sglang_output}")
print(f"HF: {hf_output}")
print(f"Match: {sglang_output == hf_output}")
```

### 6.5 边界情况测试
- [ ] 空输入测试
- [ ] 超长输入测试
- [ ] 多个音频/图像输入测试
- [ ] 并发请求测试
- [ ] 不同 dtype 测试（fp16, bf16, fp32）

---

## 📝 阶段 7: 文档和清理 (2-4小时)

### 7.1 代码文档
- [ ] 添加文档字符串到所有类和方法
- [ ] 添加内联注释解释关键逻辑
- [ ] 添加使用示例

### 7.2 创建示例
- [ ] 创建 README 或使用指南
- [ ] 提供完整的启动命令
- [ ] 提供 API 调用示例
- [ ] 提供离线推理示例

### 7.3 创建测试工具
- [ ] 单元测试脚本
- [ ] 批量测试脚本
- [ ] 性能基准测试脚本

### 7.4 代码审查
- [ ] 检查所有 TODO 都已完成
- [ ] 删除调试打印（或改为日志）
- [ ] 检查代码风格一致性
- [ ] 运行 linter/formatter

---

## ✅ 阶段 8: 提交和发布

### 8.1 Git 提交
- [ ] 审查所有改动
- [ ] 编写清晰的 commit message
- [ ] 提交代码

```bash
git add python/sglang/srt/models/your_model.py
git add python/sglang/srt/models/__init__.py
git commit -m "Add support for YourModel

- Implement YourModelForConditionalGeneration with audio/vision support
- Add encoder, projector, and integration with language model
- Ensure all weights are loaded without skipping
- Add comprehensive tests and examples
"
git push origin your-branch
```

### 8.2 创建 Pull Request (如果贡献到主仓库)
- [ ] 编写 PR 描述
- [ ] 列出关键改动
- [ ] 提供测试结果
- [ ] 链接相关 issue

---

## 🎯 关键成功指标

完成实现后，检查以下指标：

- [ ] ✅ **权重加载**: 所有权重加载，无跳过
- [ ] ✅ **配置保真**: 未修改关键配置（RoPE 等）
- [ ] ✅ **功能正确**: 单输入测试通过
- [ ] ✅ **批量处理**: 批量测试通过，无错误
- [ ] ✅ **输出一致**: 与参考实现输出一致
- [ ] ✅ **性能合理**: 延迟和吞吐量符合预期
- [ ] ✅ **文档完整**: 使用文档和示例齐全

---

## 🐛 常见问题快速诊断

### 问题: Skipped weights
**检查**:
- [ ] `bias=True/False` 是否正确
- [ ] 权重名称映射是否正确
- [ ] 是否检查了缓冲区

### 问题: 输出质量差
**检查**:
- [ ] RoPE 配置是否被修改
- [ ] 多模态特征位置是否对齐
- [ ] 解码器类型是否正确

### 问题: 序列长度错误
**检查**:
- [ ] `get_mm_input_encoder_grouped_output_size` 计算是否正确
- [ ] 下采样因子是否考虑

### 问题: CUDA 错误
**检查**:
- [ ] dtype 是否一致
- [ ] 设备是否正确（CPU vs CUDA）
- [ ] 张量形状是否匹配

---

## 📚 参考资源

- 详细指南: `SGLANG_NEW_MODEL_GUIDE.md`
- 代码模板: `model_template.py`
- MiDashengLM 实现: `python/sglang/srt/models/midashenglm.py`
- 离线推理示例: `test_offline_inference_from_dataset.py`
- 批量测试工具: `test_midashenglm_batch_audio.py`

---

**预计总时间**: 20-40 小时（根据模型复杂度和经验）

**建议**: 按阶段推进，每个阶段完成后进行验证，避免积累问题。
