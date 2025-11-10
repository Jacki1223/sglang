# SGLang 新模型支持方法论

本指南基于 MiDashengLM 模型的实际实现经验，提供了一套完整的方法论来为 SGLang 添加新模型支持。

## 目录
1. [模型分析阶段](#1-模型分析阶段)
2. [架构设计阶段](#2-架构设计阶段)
3. [代码实现阶段](#3-代码实现阶段)
4. [权重加载验证](#4-权重加载验证)
5. [测试验证阶段](#5-测试验证阶段)
6. [常见问题与解决方案](#6-常见问题与解决方案)

---

## 1. 模型分析阶段

### 1.1 获取模型基本信息

**步骤 1：加载模型配置**

```python
from transformers import AutoConfig

# 加载模型配置
config = AutoConfig.from_pretrained("your-model/model-name", trust_remote_code=True)

# 打印配置查看
print(config)
print(f"Model type: {config.model_type}")
print(f"Architecture: {config.architectures}")
```

**实际案例 - MiDashengLM:**
```python
config = AutoConfig.from_pretrained("mispeech/midashenglm-7b", trust_remote_code=True)
# 输出：
# model_type: 'midashenglm'
# architectures: ['MiDashengLMForConditionalGeneration']
# text_config.model_type: 'qwen2_5_omni_text'  # 关键：解码器类型
```

**步骤 2：分析模型组件**

检查模型包含的关键组件：
- **文本编码器/解码器**：是什么架构？(Llama, Qwen, GPT等)
- **多模态编码器**：视觉编码器？音频编码器？
- **投影器/适配器**：如何将多模态特征映射到文本空间
- **特殊配置**：RoPE 类型、注意力机制、分词器特性

```python
# 检查文本配置
if hasattr(config, 'text_config'):
    text_config = config.text_config
    print(f"Text model type: {text_config.model_type}")
    print(f"Hidden size: {text_config.hidden_size}")
    print(f"Num layers: {text_config.num_hidden_layers}")
    print(f"Num attention heads: {text_config.num_attention_heads}")

    # 检查 RoPE 配置（重要！）
    if hasattr(text_config, 'rope_scaling'):
        print(f"RoPE scaling: {text_config.rope_scaling}")

# 检查视觉/音频配置
if hasattr(config, 'vision_config'):
    print(f"Vision config: {config.vision_config}")
if hasattr(config, 'audio_config'):
    print(f"Audio config: {config.audio_config}")
```

**实际案例 - MiDashengLM RoPE 分析:**
```python
# 发现关键问题：text_config 包含 mrope_section
text_config.rope_scaling = {
    'type': 'default',
    'mrope_section': [16, 24, 24]  # M-RoPE 配置，不能删除！
}
```

### 1.2 对比参考实现

**步骤 3：查找现有实现**

```bash
# 检查是否有 vLLM 实现
git clone https://github.com/vllm-project/vllm.git
cd vllm
grep -r "YourModelName" vllm/model_executor/models/

# 检查 HuggingFace 实现
# 查看模型仓库中的 modeling_*.py 文件
```

**实际案例 - MiDashengLM:**
```bash
# 发现 vLLM 有实现
# vllm/model_executor/models/midashenglm.py

# 对比关键差异：
# - vLLM 使用 Qwen2_5OmniForCausalLM 作为解码器
# - vLLM 保留了 mrope_section 配置
# - vLLM 的音频投影器有 bias 参数
```

### 1.3 确定依赖的基础模型

```python
# 检查需要依赖哪些已有的 SGLang 模型
# 例如：解码器用 Qwen2、视觉编码器用 CLIP 等

# 查看 SGLang 已支持的模型
ls python/sglang/srt/models/

# 检查是否需要新增基础模型支持
```

**实际案例 - MiDashengLM 依赖:**
- 解码器：`Qwen2ForCausalLM` (已有) → 但需要支持 M-RoPE
- 音频编码器：`Whisper` (需要新增或复用)
- 投影器：自定义实现

---

## 2. 架构设计阶段

### 2.1 设计模型类结构

**标准多模态模型结构：**

```python
class YourMultimodalModel(nn.Module):
    def __init__(self, config, ...):
        super().__init__()

        # 1. 多模态编码器（视觉/音频等）
        self.vision_encoder = ...
        self.audio_encoder = ...

        # 2. 投影器/适配器
        self.projector = ...

        # 3. 语言模型解码器
        self.language_model = ...

    def forward(self, input_ids, positions, ...):
        # 处理多模态输入
        # 处理文本输入
        # 融合特征
        # 生成输出
        pass

    def load_weights(self, weights):
        # 加载预训练权重
        pass

    def get_mm_input_encoder_grouped_output_size(self, mm_counts):
        # 返回多模态特征尺寸
        pass
```

### 2.2 设计输入处理流程

**多模态输入处理：**

```python
class YourInputMetadata:
    """处理多模态输入的元数据"""

    def __init__(self, forward_batch, ...):
        # 提取图像/音频输入
        self.image_inputs = []
        self.audio_inputs = []

        # 构建位置映射
        self.modality_offset = []

    def process_multimodal_inputs(self, encoder):
        """
        处理多模态输入并返回特征
        """
        # 编码图像/音频
        features = encoder.encode(self.image_inputs)

        # 映射到文本序列位置
        return self._map_to_sequence(features)
```

**实际案例 - MiDashengLM 音频处理:**

```python
class MiDashengLMInputMetadata:
    def __init__(self, forward_batch: ForwardBatch):
        # 提取所有音频输入
        audio_inputs = []
        audio_offsets = []

        for i, rid in enumerate(forward_batch.seq_lens.keys()):
            if rid in forward_batch.reqs.audio_inputs:
                audio_item = forward_batch.reqs.audio_inputs[rid][0]
                audio_inputs.append(audio_item)
                # 计算音频特征在序列中的偏移位置
                audio_offsets.append(
                    forward_batch.extend_prefix_lens[i]
                    + audio_item["offsets"][0]
                )

        self.audio_inputs = audio_inputs
        self.audio_offsets = audio_offsets
```

---

## 3. 代码实现阶段

### 3.1 实现编码器组件

**音频编码器示例：**

```python
class AudioEncoder(nn.Module):
    """音频编码器基类"""

    def __init__(self, config):
        super().__init__()
        # 根据配置初始化编码器
        # 可能是 Whisper, Wav2Vec2 等

    def forward(self, audio_features):
        # 编码音频特征
        return encoded_features
```

**实际案例 - MiDashengLM Whisper 编码器:**

```python
class MiDashengLMWhisperEncoder(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.d_model,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1
        )
        # ... 更多层

    def forward(self, audio_features: torch.Tensor):
        # Conv layers
        hidden_states = F.gelu(self.conv1(audio_features))
        hidden_states = F.gelu(self.conv2(hidden_states))
        # ... transformer blocks
        return hidden_states
```

### 3.2 实现投影器组件

**标准投影器模式：**

```python
class ModalityProjector(nn.Module):
    """将多模态特征投影到语言模型空间"""

    def __init__(self, in_dim, out_dim, ...):
        super().__init__()
        # 简单 MLP
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, features):
        return self.linear2(self.act(self.linear1(features)))
```

**实际案例 - MiDashengLM 带下采样的投影器:**

```python
class AudioProjectorSubsample(nn.Module):
    """音频投影器，包含下采样功能"""

    def __init__(
        self,
        in_dim: int = 1280,
        out_dim: int = 3584,
        k: int = 4,  # 下采样因子
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.k = k

        # 关键：使用 bias=True 确保所有权重都被加载
        self.fc1 = ColumnParallelLinear(
            input_size=in_dim * self.k,
            output_size=out_dim,
            bias=True,  # ⚠️ 重要：必须与预训练模型一致
            quant_config=quant_config,
            prefix=add_prefix("net.0", prefix),
        )
        self.fc2 = RowParallelLinear(
            input_size=out_dim,
            output_size=out_dim,
            bias=True,  # ⚠️ 重要：必须与预训练模型一致
            quant_config=quant_config,
            prefix=add_prefix("net.2", prefix),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, in_dim]
        batch_size, seq_len, dim = x.shape

        # 下采样：将 k 个连续帧合并
        # [batch, seq_len, dim] -> [batch, seq_len//k, dim*k]
        if seq_len % self.k != 0:
            # 填充到 k 的倍数
            padding = self.k - (seq_len % self.k)
            x = F.pad(x, (0, 0, 0, padding))
            seq_len = x.shape[1]

        x = x.reshape(batch_size, seq_len // self.k, dim * self.k)

        # MLP 投影
        x = self.fc1(x)[0]
        x = self.act(x)
        x = self.fc2(x)[0]
        return x
```

### 3.3 实现主模型类

**关键方法实现：**

```python
class YourModelForConditionalGeneration(nn.Module):
    """主模型类"""

    def __init__(self, config, quant_config=None, ...):
        super().__init__()
        self.config = config

        # 1. 初始化编码器
        self.audio_encoder = AudioEncoder(config.audio_config)

        # 2. 初始化投影器
        self.audio_projector = AudioProjector(...)

        # 3. 初始化语言模型
        # ⚠️ 关键：确保解码器配置正确
        self.language_model = LanguageModel(config.text_config, ...)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        ...
    ):
        """前向传播"""

        # 1. 处理多模态输入
        if forward_batch.reqs.audio_inputs:
            metadata = YourInputMetadata(forward_batch)
            audio_features = self._encode_audio(metadata.audio_inputs)
            audio_embeds = self.audio_projector(audio_features)

        # 2. 获取文本嵌入
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 3. 融合多模态特征
        if audio_embeds is not None:
            inputs_embeds = self._merge_features(
                inputs_embeds,
                audio_embeds,
                metadata.audio_offsets
            )

        # 4. 语言模型前向传播
        return self.language_model(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
            ...
        )

    def _merge_features(self, text_embeds, audio_embeds, offsets):
        """将音频特征插入到文本嵌入中"""
        # 根据 offsets 将 audio_embeds 插入到 text_embeds 的正确位置
        # 这是多模态模型的核心逻辑
        pass

    def get_mm_input_encoder_grouped_output_size(self, mm_counts):
        """
        返回多模态输入经过编码器和投影器后的序列长度

        SGLang 需要这个来计算 KV cache 大小
        """
        audio_count = mm_counts.get("audio", 0)

        # 计算音频特征长度
        # 例如：Whisper 输出 1500 帧，下采样 4 倍 -> 375 tokens
        audio_tokens = audio_count * (1500 // self.k)

        return audio_tokens
```

**实际案例 - MiDashengLM 特征融合:**

```python
def forward(self, input_ids, positions, forward_batch, ...):
    # 1. 提取音频输入元数据
    metadata = MiDashengLMInputMetadata(forward_batch)

    # 2. 编码音频
    audio_embeds = None
    if metadata.audio_inputs:
        # 加载和处理音频文件
        audio_features = self._process_audio_files(metadata.audio_inputs)

        # Whisper 编码
        audio_hidden_states = self.audio_tower(audio_features)

        # 投影到语言模型空间
        audio_embeds = self.audio_projector(audio_hidden_states)

    # 3. 获取文本嵌入
    inputs_embeds = self.language_model.model.embed_tokens(input_ids)

    # 4. 融合音频特征
    if audio_embeds is not None:
        # 创建新的嵌入张量
        new_inputs_embeds = inputs_embeds.clone()

        # 在正确位置插入音频特征
        for i, offset in enumerate(metadata.audio_offsets):
            audio_len = audio_embeds[i].shape[0]
            new_inputs_embeds[offset:offset+audio_len] = audio_embeds[i]

        inputs_embeds = new_inputs_embeds

    # 5. 语言模型生成
    return self.language_model(
        input_ids=None,
        positions=positions,
        inputs_embeds=inputs_embeds,
        forward_batch=forward_batch,
        ...
    )
```

### 3.4 处理特殊配置

**RoPE 配置处理 - 重要！**

```python
# ❌ 错误做法：删除必要的配置
if "rope_scaling" in text_config:
    rope_scaling = text_config.rope_scaling
    if "mrope_section" in rope_scaling:
        # 错误：删除 mrope_section 会导致模型行为不一致
        rope_scaling.pop("mrope_section")

# ✅ 正确做法：保持配置完整
# 如果解码器需要 M-RoPE，就保留所有相关配置
# 不要修改从 HuggingFace 加载的配置
```

**实际案例 - MiDashengLM RoPE 修复:**

```python
# 之前的错误代码（已删除）：
# if rope_scaling and "mrope_section" in rope_scaling:
#     rope_scaling.pop("mrope_section")
#     rope_scaling.pop("rope_type", None)

# 修复后：完全保留 rope_scaling 配置
# MiDashengLM 使用 Qwen2.5-Omni-7B Thinker 作为解码器
# 该解码器支持 M-RoPE，需要 mrope_section 配置

text_config = self.config.text_config
# 直接使用，不做任何修改
self.language_model = Qwen2ForCausalLM(text_config, ...)
```

---

## 4. 权重加载验证

### 4.1 实现权重加载方法

**标准权重加载模式：**

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    """加载预训练权重"""

    # 定义权重映射规则
    params_dict = dict(self.named_parameters())

    audio_encoder_weights = []
    audio_projector_weights = []
    language_model_weights = []
    skipped_weights = []

    for name, loaded_weight in weights:
        # 1. 音频编码器权重
        if "audio_tower" in name:
            # 去掉前缀
            param_name = name.replace("audio_tower.", "")
            if param_name in self.audio_tower.state_dict():
                # 加载权重
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                audio_encoder_weights.append(name)
            else:
                skipped_weights.append(name)

        # 2. 音频投影器权重
        elif "audio_projector" in name:
            param_name = name.replace("audio_projector.", "")
            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                audio_projector_weights.append(name)
            else:
                skipped_weights.append(f"{name} (not in model)")

        # 3. 语言模型权重
        elif "language_model" in name:
            param_name = name.replace("language_model.", "")
            # 使用语言模型自己的加载逻辑
            self.language_model.load_weights([(param_name, loaded_weight)])
            language_model_weights.append(name)

    # 打印加载统计
    print(f"✅ Audio encoder weights loaded: {len(audio_encoder_weights)}")
    print(f"✅ Audio projector weights loaded: {len(audio_projector_weights)}")
    print(f"✅ Language model weights loaded: {len(language_model_weights)}")

    if skipped_weights:
        print(f"⚠️  Skipped weights: {len(skipped_weights)}")
        for w in skipped_weights:
            print(f"  - {w}")
```

**实际案例 - MiDashengLM 权重加载调试:**

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    # 收集所有参数名（用于调试）
    params_dict = dict(self.named_parameters())
    buffers_dict = dict(self.named_buffers())

    print("[WEIGHT LOADING] Model parameters:")
    print(f"  Audio tower params: {[k for k in params_dict.keys() if 'audio_tower' in k][:5]}...")
    print(f"  Audio projector params: {[k for k in params_dict.keys() if 'audio_projector' in k]}")

    audio_projector_weights = []
    skipped_audio_projector = []

    for name, loaded_weight in weights:
        if "audio_projector" in name:
            # 去掉 "model.audio_projector." 前缀
            param_name = name.replace("model.audio_projector.", "audio_projector.")

            # 检查是否在参数或缓冲区中
            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                audio_projector_weights.append(name)
            elif param_name in buffers_dict:
                # 缓冲区直接复制
                buffers_dict[param_name].copy_(loaded_weight)
                audio_projector_weights.append(name)
            else:
                # 诊断信息
                reason = "bias not in params/buffers" if "bias" in name else "unknown"
                skipped_audio_projector.append(f"{name} ({reason})")

    # 详细的加载报告
    print(f"\n[WEIGHT LOADING] Audio projector weights loaded: {len(audio_projector_weights)}")
    if skipped_audio_projector:
        print(f"[WEIGHT LOADING] Skipped audio_projector weights:")
        for w in skipped_audio_projector:
            print(f"  - {w}")
```

### 4.2 验证所有权重都被加载

**添加验证逻辑：**

```python
def verify_weights_loaded(model, checkpoint_path):
    """验证所有权重都被正确加载"""

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_keys = set(checkpoint.keys())

    # 获取模型参数
    model_keys = set([name for name, _ in model.named_parameters()])
    model_keys.update([name for name, _ in model.named_buffers()])

    # 检查缺失的权重
    missing_in_model = checkpoint_keys - model_keys
    missing_in_checkpoint = model_keys - checkpoint_keys

    print("=" * 80)
    print("Weight Loading Verification")
    print("=" * 80)
    print(f"Checkpoint weights: {len(checkpoint_keys)}")
    print(f"Model parameters: {len(model_keys)}")

    if missing_in_model:
        print(f"\n⚠️  Weights in checkpoint but not in model ({len(missing_in_model)}):")
        for key in sorted(missing_in_model)[:10]:
            print(f"  - {key}")

    if missing_in_checkpoint:
        print(f"\n⚠️  Parameters in model but not in checkpoint ({len(missing_in_checkpoint)}):")
        for key in sorted(missing_in_checkpoint)[:10]:
            print(f"  - {key}")

    if not missing_in_model and not missing_in_checkpoint:
        print("\n✅ All weights matched!")

    print("=" * 80)
```

**实际问题 - MiDashengLM bias 缺失:**

```
问题：
[WEIGHT LOADING] Audio projector weights loaded: 2
[WEIGHT LOADING] Skipped audio_projector weights:
  - audio_projector.net.0.bias (bias not in params/buffers)
  - audio_projector.net.2.bias (bias not in params/buffers)

原因：
AudioProjectorSubsample 中使用了 bias=False

修复：
将 bias=False 改为 bias=True

结果：
[WEIGHT LOADING] Audio projector weights loaded: 4
[WEIGHT LOADING] Skipped weights: 0
```

---

## 5. 测试验证阶段

### 5.1 注册模型

**在 `python/sglang/srt/models/__init__.py` 中注册：**

```python
# 文件：python/sglang/srt/models/__init__.py

_MODELS = {
    # ... 现有模型
    "YourModelForConditionalGeneration": ("your_model", "YourModelForConditionalGeneration"),
}

_AUDIO_MODELS = {
    "YourModelForConditionalGeneration",
    # ... 其他音频模型
}
```

**实际案例 - MiDashengLM 注册:**

```python
_MODELS = {
    # ...
    "MiDashengLMForConditionalGeneration": (
        "midashenglm",
        "MiDashengLMForConditionalGeneration",
    ),
}

_AUDIO_MODELS = {
    "MiDashengLMForConditionalGeneration",
    "Qwen2AudioForConditionalGeneration",
}
```

### 5.2 单元测试

**创建简单测试脚本：**

```python
#!/usr/bin/env python3
"""test_your_model.py - 快速测试新模型"""

import requests
import json

def test_single_audio():
    """测试单个音频输入"""

    url = "http://localhost:30000/v1/chat/completions"

    payload = {
        "model": "your-model/model-name",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "https://example.com/test.mp3"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请转录这段音频。"
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    response = requests.post(url, json=payload)
    result = response.json()

    print("=" * 80)
    print("测试结果：")
    print("=" * 80)
    print(f"Prompt: {payload['messages'][0]['content'][1]['text']}")
    print(f"Response: {result['choices'][0]['message']['content']}")
    print("=" * 80)

    return result

if __name__ == "__main__":
    # 1. 启动服务器
    print("请先启动服务器：")
    print("python -m sglang.launch_server \\")
    print("    --model your-model/model-name \\")
    print("    --trust-remote-code \\")
    print("    --enable-multimodal \\")
    print("    --port 30000")
    print()

    input("服务器启动后按回车继续...")

    # 2. 运行测试
    test_single_audio()
```

### 5.3 批量测试

**创建数据集测试脚本（参考我们创建的）：**

```python
# test_offline_inference_from_dataset.py
# 支持 API 和 Engine 两种模式
# 详见之前创建的完整脚本
```

### 5.4 性能基准测试

**测试吞吐量和延迟：**

```python
import time
from sglang import Engine

def benchmark_model(
    model_path: str,
    test_cases: list,
    tp_size: int = 1,
):
    """基准测试"""

    print(f"初始化引擎...")
    start_time = time.time()

    engine = Engine(
        model_path=model_path,
        trust_remote_code=True,
        tp_size=tp_size,
    )

    init_time = time.time() - start_time
    print(f"✅ 引擎初始化耗时: {init_time:.2f}s")

    # 预热
    print("预热中...")
    engine.generate(test_cases[0])

    # 测试
    print("开始基准测试...")
    latencies = []

    for i, test_case in enumerate(test_cases):
        start = time.time()
        result = engine.generate(test_case)
        latency = time.time() - start
        latencies.append(latency)

        print(f"  Case {i+1}: {latency:.2f}s, "
              f"{len(result['text'].split())} tokens")

    # 统计
    print(f"\n{'='*80}")
    print(f"基准测试结果")
    print(f"{'='*80}")
    print(f"总样本数: {len(latencies)}")
    print(f"平均延迟: {sum(latencies)/len(latencies):.2f}s")
    print(f"最小延迟: {min(latencies):.2f}s")
    print(f"最大延迟: {max(latencies):.2f}s")
    print(f"{'='*80}")
```

---

## 6. 常见问题与解决方案

### 问题 1: 权重加载不完整

**症状：**
```
[WEIGHT LOADING] Skipped weights: 2
  - model.projector.bias
  - model.encoder.some_param
```

**解决方案：**

1. **检查模型定义中的参数配置**
   ```python
   # 确保 bias 设置正确
   self.linear = nn.Linear(in_dim, out_dim, bias=True)  # 而不是 False
   ```

2. **检查权重名称映射**
   ```python
   # 在 load_weights 中打印权重名称
   print(f"Checkpoint weight: {name}")
   print(f"Model params: {list(params_dict.keys())}")

   # 检查是否需要前缀转换
   param_name = name.replace("model.", "")
   ```

3. **检查是否在缓冲区中**
   ```python
   buffers_dict = dict(self.named_buffers())
   if param_name in buffers_dict:
       buffers_dict[param_name].copy_(loaded_weight)
   ```

### 问题 2: RoPE 配置错误

**症状：**
- 模型输出质量差
- 位置编码不正确
- 与参考实现结果不一致

**解决方案：**

```python
# ❌ 不要随意修改 rope_scaling
# 错误示例：
if "mrope_section" in rope_scaling:
    rope_scaling.pop("mrope_section")

# ✅ 保持原始配置
# 正确做法：直接使用 AutoConfig 加载的配置
text_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True).text_config
# 不做任何修改，直接传给解码器
```

**验证方法：**

```python
# 对比配置
from transformers import AutoConfig

hf_config = AutoConfig.from_pretrained("model-name", trust_remote_code=True)
print("HF Config RoPE:", hf_config.text_config.rope_scaling)

# 检查 SGLang 中的配置
print("SGLang Config RoPE:", self.config.text_config.rope_scaling)

# 应该完全一致！
```

### 问题 3: 多模态特征对齐错误

**症状：**
- 模型输出乱码
- 音频/图像内容理解错误
- 位置偏移问题

**解决方案：**

1. **仔细计算特征长度**
   ```python
   def get_mm_input_encoder_grouped_output_size(self, mm_counts):
       audio_count = mm_counts.get("audio", 0)

       # 计算要准确
       # 音频帧数 -> Whisper 编码 -> 投影器下采样
       whisper_output_len = 1500  # Whisper 输出长度
       downsample_factor = 4       # 投影器下采样因子

       tokens_per_audio = whisper_output_len // downsample_factor

       return audio_count * tokens_per_audio
   ```

2. **验证特征插入位置**
   ```python
   # 打印调试信息
   print(f"Audio offset: {offset}")
   print(f"Audio features shape: {audio_embeds.shape}")
   print(f"Text embeds shape: {inputs_embeds.shape}")

   # 确保不越界
   assert offset + audio_len <= inputs_embeds.shape[0]
   ```

### 问题 4: 解码器不匹配

**症状：**
- 配置显示 `text_config.model_type` 与实际使用的解码器不一致
- vLLM 使用不同的解码器

**解决方案：**

```python
# 检查配置中的真实解码器类型
config = AutoConfig.from_pretrained("model-name", trust_remote_code=True)
print(f"Text model type: {config.text_config.model_type}")

# 根据实际类型选择解码器
if config.text_config.model_type == "qwen2_5_omni_text":
    # 使用 Qwen2（支持 M-RoPE）
    from sglang.srt.models.qwen2 import Qwen2ForCausalLM
    self.language_model = Qwen2ForCausalLM(config.text_config, ...)
elif config.text_config.model_type == "llama":
    from sglang.srt.models.llama import LlamaForCausalLM
    self.language_model = LlamaForCausalLM(config.text_config, ...)
```

### 问题 5: 数据类型不匹配

**症状：**
- CUDA 错误
- 精度损失
- 性能下降

**解决方案：**

```python
# 1. 在启动时指定 dtype
python -m sglang.launch_server \
    --model your-model \
    --dtype float16  # 或 bfloat16

# 2. 在代码中确保一致性
with set_default_torch_dtype(self.config.dtype):
    # 创建模块
    self.projector = Projector(...)
```

---

## 完整实现 Checklist

使用此清单确保实现完整：

### 阶段 1: 分析
- [ ] 使用 `AutoConfig` 加载并检查模型配置
- [ ] 识别文本编码器/解码器类型
- [ ] 识别多模态编码器类型（视觉/音频）
- [ ] 检查 RoPE 配置和特殊注意力机制
- [ ] 查找参考实现（vLLM, HuggingFace）
- [ ] 对比参考实现找出关键差异

### 阶段 2: 设计
- [ ] 设计模型类结构（编码器、投影器、解码器）
- [ ] 设计输入处理流程（元数据类）
- [ ] 设计特征融合逻辑
- [ ] 确定依赖的现有 SGLang 组件

### 阶段 3: 实现
- [ ] 实现多模态编码器
- [ ] 实现投影器（注意 bias 参数）
- [ ] 实现主模型类
- [ ] 实现 `forward` 方法
- [ ] 实现 `get_mm_input_encoder_grouped_output_size`
- [ ] 实现 `load_weights` 方法
- [ ] 保持配置完整性（不修改 RoPE 等关键配置）

### 阶段 4: 权重加载
- [ ] 实现权重名称映射逻辑
- [ ] 添加加载进度和统计信息
- [ ] 验证所有权重都被加载（无跳过）
- [ ] 检查参数和缓冲区

### 阶段 5: 注册和测试
- [ ] 在 `__init__.py` 中注册模型
- [ ] 创建简单测试脚本
- [ ] 测试单个输入
- [ ] 测试批量输入
- [ ] 进行性能基准测试
- [ ] 对比参考实现验证输出一致性

### 阶段 6: 文档和工具
- [ ] 创建使用示例
- [ ] 创建批量测试工具
- [ ] 编写故障排除指南
- [ ] 记录与参考实现的差异

---

## 总结

支持新模型的核心原则：

1. **配置保真**：不随意修改从 HuggingFace 加载的配置，特别是 RoPE 等关键参数
2. **权重完整**：确保所有预训练权重都被正确加载，无跳过
3. **特征对齐**：准确计算多模态特征长度和插入位置
4. **参考对比**：与 vLLM 等参考实现对比，确保一致性
5. **充分测试**：单元测试 + 批量测试 + 性能测试

遵循这套方法论，可以系统地为 SGLang 添加新模型支持，避免常见陷阱。
