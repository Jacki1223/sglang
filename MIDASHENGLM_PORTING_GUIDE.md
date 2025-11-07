# SGLang 多模态模型移植完整指南

## 文档目的
本文档记录了将 MiDashengLM（音频-语言多模态模型）从 vLLM 移植到 SGLang 的完整过程，形成可复用的方法论，适用于其他多模态模型（视觉、音频等）的移植工作。

---

## 目录
1. [移植概述](#移植概述)
2. [架构分析](#架构分析)
3. [核心实现步骤](#核心实现步骤)
4. [组件替换详解](#组件替换详解)
5. [权重加载机制](#权重加载机制)
6. [多模态数据处理](#多模态数据处理)
7. [调试与验证](#调试与验证)
8. [常见问题与解决方案](#常见问题与解决方案)
9. [检查清单](#检查清单)

---

## 移植概述

### 什么是模型移植？
将训练好的模型从一个推理框架（vLLM）迁移到另一个推理框架（SGLang），保持模型功能和输出一致性，同时利用目标框架的优化特性。

### vLLM vs SGLang 的关键差异

| 方面 | vLLM | SGLang |
|------|------|--------|
| **Attention 机制** | PagedAttention | RadixAttention |
| **KV Cache** | Paged KV Cache | Radix Tree KV Cache |
| **多模态处理** | 独立处理器 | 统一的 MultimodalInputs 机制 |
| **Forward 方法** | forward(), forward_batch() | forward(forward_batch: ForwardBatch) |
| **权重加载** | load_weights() | load_weights() + 名称映射 |
| **采样** | 模型内部 Sample | Engine 处理（移除 Sample） |

### MiDashengLM 架构概览
```
Audio Input (Waveform)
    ↓
DashengFrontend (Mel-Spectrogram)
    ↓
DashengAudioTransformer (Encoder with VisionAttention)
    ↓
AudioProjectorSubsample (降采样投影)
    ↓
Token Replacement (替换 audio_token 为 pad_value)
    ↓
Qwen2ForCausalLM (Language Model with RadixAttention)
    ↓
Text Output
```

---

## 架构分析

### 第一步：理解原始模型架构

#### 1.1 分析 vLLM 实现
```bash
# 找到 vLLM 中的原始实现
git clone https://github.com/vllm-project/vllm
cd vllm/vllm/model_executor/models/
# 查看模型文件
cat midashenglm.py
```

#### 1.2 识别关键组件
对于 MiDashengLM，关键组件包括：
- **Audio Frontend**: 音频预处理（波形 → Mel频谱图）
- **Audio Encoder**: 多层 Transformer（带 Attention）
- **Audio Projector**: 特征投影和降采样
- **Language Model**: 文本生成（Qwen2）
- **Multimodal Integration**: 多模态特征融合

#### 1.3 绘制数据流图
```
输入数据流：
1. 音频文件 (.wav) → 处理器
2. 处理器 → 特征提取 (input_values, audio_length)
3. 特征 → 模型编码器 → 嵌入
4. 嵌入 + 文本 token → 语言模型
5. 语言模型 → logits → 采样 → 文本输出
```

---

## 核心实现步骤

### 步骤 1: 创建模型文件

**位置**: `python/sglang/srt/models/midashenglm.py`

**文件结构**:
```python
# 1. 导入语句（只使用 SGLang 组件）
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

# 2. 辅助类和函数
def calculate_mel_frames_dasheng(...): ...

# 3. 音频处理组件
class AudioPatchEmbed(nn.Module): ...
class DashengFrontend(nn.Module): ...  # Mel-spectrogram
class DashengAttention(nn.Module): ...  # VisionAttention wrapper

# 4. 编码器和投影器
class DashengAudioTransformer(nn.Module): ...
class AudioProjectorSubsample(nn.Module): ...

# 5. 主模型类
class MiDashengLMModel(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""): ...
    def pad_input_ids(self, input_ids, mm_inputs): ...
    def get_audio_feature(self, items): ...
    def get_input_embeddings(self): ...
    def forward(self, input_ids, positions, forward_batch, **kwargs): ...
    def load_weights(self, weights): ...

# 6. EntryClass
EntryClass = [MiDashengLMModel]
```

### 步骤 2: 创建处理器

**位置**: `python/sglang/srt/multimodal/processors/midashenglm.py`

**核心功能**:
```python
class MiDashengLMProcessor:
    def __init__(self, hf_processor, config):
        self.hf_processor = hf_processor  # 使用 HuggingFace 处理器

    def __call__(self, prompt, mm_data, modalities):
        # 1. 提取音频特征
        ret = self.hf_processor(
            audios=audio_data,
            return_tensors="pt",
            sampling_rate=16000
        )

        # 2. 创建 MultimodalDataItem
        mm_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=input_values,  # 音频波形
            audio_length=ret["audio_length"],  # Mel 帧数（关键！）
            pad_value=generate_unique_pad_value(),
            hash=get_mm_hash(audio_data)
        )

        # 3. 返回 MultimodalInputs
        return MultimodalInputs(
            prompt_text=prompt,
            mm_items=[mm_item],
            mm_placeholders={Modality.AUDIO: audio_token}
        )
```

**关键点**:
- 使用 HuggingFace 的处理器进行特征提取
- `audio_length` 必须是 **Mel 帧数**，不是波形采样点数
- 每个音频生成唯一的 `pad_value` 用于 token 替换

### 步骤 3: 注册模型和处理器

#### 3.1 注册模型架构
**文件**: `python/sglang/srt/model_executor/model_runner.py`

```python
"MiDashengLMForCausalLM": ("midashenglm", "MiDashengLMModel"),
```

#### 3.2 注册处理器
**文件**: `python/sglang/srt/multimodal/processors/processor_factory.py`

```python
_PROCESSOR_REGISTRY = {
    "MiDashengLMForCausalLM": MiDashengLMProcessor,
}
```

---

## 组件替换详解

### 替换 1: Attention 层

#### Audio Encoder Attention → VisionAttention

**原因**: 音频编码器不需要 KV cache，使用 VisionAttention 更高效

```python
class DashengAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, quant_config=None, prefix=""):
        super().__init__()
        # 使用 VisionAttention（不需要 layer_id）
        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,  # 使用并行 QKV
            proj_bias=True,
            qkv_bias=qkv_bias,
            qkv_backend="sdpa",  # 使用 scaled_dot_product_attention
            softmax_in_single_precision=False,
            flatten_batch=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, x, mask=None):
        # VisionAttention 期望 [B, N, C] 格式
        return self.attn(x, attn_mask=mask)
```

**关键点**:
- VisionAttention 不需要 `layer_id` 参数
- 音频编码器的每一层都独立处理，不共享 KV cache
- 使用 `qkv_backend="sdpa"` 获得最佳性能

#### Language Model Attention → RadixAttention

**实现**: 通过使用 `Qwen2ForCausalLM` 自动获得

```python
# 在 MiDashengLMModel.__init__ 中
self.language_model = Qwen2ForCausalLM(
    config.text_config,
    quant_config=quant_config,
    prefix=add_prefix("decoder", prefix),
)
```

Qwen2ForCausalLM 内部使用 RadixAttention（带 layer_id）处理文本生成。

### 替换 2: LogitsProcessor

```python
# 直接使用 language_model 的 logits_processor
self.logits_processor = self.language_model.logits_processor
```

**说明**: SGLang 的 LogitsProcessor 集成在语言模型中，不需要单独实现。

### 替换 3: 移除 Sample

**vLLM 方式**:
```python
# ❌ 不要这样做
from vllm.model_executor.layers.sampler import Sampler
self.sampler = Sampler()
```

**SGLang 方式**:
```python
# ✅ 移除 Sample，由 Engine 处理
# 模型只返回 logits
```

### 替换 4: Forward 方法

**vLLM 签名**:
```python
def forward(self, input_ids, positions, kv_caches, ...):
    ...

def forward_batch(self, input_ids_list, ...):
    ...
```

**SGLang 签名**:
```python
@torch.no_grad()
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,  # 关键：包含所有批次信息
    **kwargs,
):
    # 使用 general_mm_embed_routine 处理多模态
    return general_mm_embed_routine(
        input_ids=input_ids,
        forward_batch=forward_batch,
        language_model=self.language_model,
        positions=positions,
        data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
    )
```

**关键点**:
- 只需实现 `forward()` 方法，不需要 `forward_batch()`
- `ForwardBatch` 包含所有批次和多模态信息
- 使用 `general_mm_embed_routine` 自动处理多模态集成

---

## 权重加载机制

### 挑战：名称不匹配

vLLM 和 SGLang 使用不同的组件，导致权重名称不匹配：

| Checkpoint (vLLM) | Model (SGLang) | 映射方式 |
|-------------------|----------------|----------|
| `audio_projector.net.0.weight` | `audio_projector.fc1.weight` | 字符串替换 |
| `audio_projector.net.2.weight` | `audio_projector.fc2.weight` | 字符串替换 |
| `audio_encoder.blocks.X.attn.proj.weight` | `audio_encoder.blocks.X.attn.attn.proj.weight` | 字符串替换 |
| `audio_encoder.front_end.0.mel_scale.fb` | `audio_encoder.front_end.0.melscale_fbanks` | Buffer 映射 |
| `decoder.model.layers.X.weight` | `model.layers.X.weight` | Prefix 剥离 |

### load_weights() 实现

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    """Load model weights with name mapping."""
    params_dict = dict(self.named_parameters(remove_duplicate=False))
    buffers_dict = dict(self.named_buffers())  # 关键：加载 buffers

    audio_encoder_loaded = []
    audio_projector_loaded = []
    decoder_weights = []  # 收集 decoder 权重
    skipped_weights = []
    buffer_loaded = []

    for name, loaded_weight in weights:
        # 1. 跳过不需要的权重
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue

        # 2. Decoder 权重单独处理
        if name.startswith("decoder"):
            decoder_weights.append((name, loaded_weight))
            continue

        original_name = name

        # 3. 名称映射：Audio Encoder Frontend Buffers
        if "audio_encoder.front_end" in name:
            if ".mel_scale.fb" in name:
                name = name.replace(".mel_scale.fb", ".melscale_fbanks")
            elif ".spectrogram.window" in name:
                name = name.replace(".spectrogram.window", ".spectrogram_window")

        # 4. 名称映射：Audio Encoder Attention QKV
        if "audio_encoder" in name and ".attn.qkv." in name:
            name = name.replace(".attn.qkv.", ".attn.attn.qkv_proj.")

        # 5. 名称映射：Audio Encoder Attention Output Projection
        if "audio_encoder" in name and ".attn.proj." in name:
            name = name.replace(".attn.proj.", ".attn.attn.proj.")

        # 6. 名称映射：Audio Projector
        if "audio_projector" in name:
            name = name.replace(".net.0.", ".fc1.")
            name = name.replace(".net.2.", ".fc2.")

        # 7. 跳过不存在的 bias
        if name.endswith(".bias") and name not in params_dict and name not in buffers_dict:
            skipped_weights.append(f"{original_name} (bias not in model)")
            continue

        # 8. 加载权重：先尝试 parameter，再尝试 buffer
        if name in params_dict:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        elif name in buffers_dict:
            # Buffer 使用 copy_ 加载
            buffers_dict[name].copy_(loaded_weight)
            buffer_loaded.append(original_name)
        else:
            skipped_weights.append(f"{original_name} -> {name} (not in model)")
            continue

        # 9. 跟踪已加载的权重
        if "audio_encoder" in original_name:
            audio_encoder_loaded.append(original_name)
        elif "audio_projector" in original_name:
            audio_projector_loaded.append(original_name)

    # 10. 传递 decoder 权重给 language_model
    if decoder_weights:
        # 剥离 "decoder." 前缀
        decoder_weights_stripped = [
            (name.replace("decoder.", "", 1), weight)
            for name, weight in decoder_weights
        ]
        self.language_model.load_weights(decoder_weights_stripped)

    # 11. 打印加载摘要
    print(f"Audio encoder weights loaded: {len(audio_encoder_loaded)}")
    print(f"Audio projector weights loaded: {len(audio_projector_loaded)}")
    print(f"Buffers loaded: {len(buffer_loaded)}")
    print(f"Decoder weights passed to language_model: {len(decoder_weights)}")
    print(f"Skipped weights: {len(skipped_weights)}")
```

### Buffer 加载的重要性

**什么是 Buffer？**
- Buffer 是不可训练的张量（如预计算的常量、运行统计数据）
- 使用 `register_buffer()` 注册
- 在推理时必须加载，否则结果错误

**MiDashengLM 的关键 Buffers**:
1. **melscale_fbanks**: Mel 滤波器组（频率到 Mel 尺度的转换矩阵）
2. **spectrogram_window**: Hann 窗函数（用于 STFT）
3. **BatchNorm buffers**: `running_mean`, `running_var`, `num_batches_tracked`

**如何注册 Buffer**:
```python
class DashengFrontend(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 创建 Hann 窗
        spectrogram_window = torch.hann_window(config.win_length)
        self.register_buffer(
            "spectrogram_window",
            spectrogram_window,
            persistent=False,  # 不保存到 state_dict（从 checkpoint 加载）
        )

        # 创建 Mel 滤波器组
        melscale_fbanks = F.melscale_fbanks(
            n_freqs=config.n_fft // 2 + 1,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
        )
        self.register_buffer("melscale_fbanks", melscale_fbanks, persistent=False)

    def forward(self, waveform):
        # 使用 buffer
        spectrogram = F.spectrogram(
            waveform=waveform,
            window=self.spectrogram_window,  # 使用注册的 buffer
            ...
        )
        mel_spectrogram = (spectrogram.mT @ self.melscale_fbanks).mT
        ...
```

---

## 多模态数据处理

### Token 替换机制

SGLang 使用 **pad_value 替换** 来整合多模态特征：

```
原始 input_ids: [1, 2, 3, <audio_token>, 4, 5]
                              ↓
替换后:          [1, 2, 3, <pad_value>, 4, 5]
                              ↓
Embedding查找:   [E1, E2, E3, <pad_emb>, E4, E5]
                              ↓
多模态替换:      [E1, E2, E3, <audio_features>, E4, E5]
```

### pad_input_ids() 方法

```python
def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
    """Pad input IDs with multimodal tokens."""
    pattern = MultiModalityDataPaddingPatternMultimodalTokens()
    return pattern.pad_input_tokens(input_ids, mm_inputs)
```

**工作流程**:
1. 找到 `input_ids` 中的 `<audio_token>` 位置
2. 替换为 `mm_item.pad_value`（唯一标识符）
3. 模型 forward 时，识别 `pad_value` 并替换为音频特征

### get_audio_feature() 方法

```python
def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
    """Process audio inputs and return embeddings."""
    # 1. 提取音频波形
    input_values = torch.cat([item.feature for item in items], dim=0)

    # 2. 获取音频长度（Mel 帧数）
    audio_lengths = []
    for item in items:
        if hasattr(item, 'audio_length') and item.audio_length is not None:
            audio_lengths.append(item.audio_length)
        else:
            # Fallback: 使用波形长度
            audio_lengths.append(item.feature.shape[-1])
    audio_length = torch.tensor(audio_lengths, device=input_values.device)

    # 3. 编码器处理
    encoder_out, encoder_atts = self.audio_encoder(input_values, audio_length)

    # 4. 投影器降采样
    audio_embeddings, _ = self.audio_projector(encoder_out, encoder_atts)

    # 5. 重塑为 [total_tokens, embed_dim]
    batch_size, max_audio_tokens, embed_dim = audio_embeddings.shape
    masked_audio_features = audio_embeddings.reshape(-1, embed_dim)

    return masked_audio_features
```

### 使用 general_mm_embed_routine

```python
def forward(self, input_ids, positions, forward_batch, **kwargs):
    return general_mm_embed_routine(
        input_ids=input_ids,
        forward_batch=forward_batch,
        language_model=self.language_model,
        positions=positions,
        data_embedding_funcs={
            Modality.AUDIO: self.get_audio_feature,  # 注册音频处理函数
        },
    )
```

**general_mm_embed_routine 的工作流程**:
1. 获取文本 embeddings: `embed_tokens(input_ids)`
2. 识别 `input_ids` 中的 `pad_value`
3. 调用 `get_audio_feature()` 获取音频 embeddings
4. 替换 pad_value 对应的位置为音频 embeddings
5. 传递给 language_model

---

## 调试与验证

### 第一阶段：权重加载验证

**添加调试输出**:
```python
def load_weights(self, weights):
    ...
    print(f"\n{'='*80}")
    print(f"Audio encoder weights loaded: {len(audio_encoder_loaded)}")
    print(f"Audio projector weights loaded: {len(audio_projector_loaded)}")
    print(f"Buffers loaded: {len(buffer_loaded)}")
    for buf in buffer_loaded:
        print(f"  - {buf}")
    print(f"Decoder weights: {len(decoder_weights)}")
    print(f"Skipped weights: {len(skipped_weights)}")

    # 详细列出跳过的权重
    if skipped_weights:
        print("\nSkipped weights:")
        for s in skipped_weights[:10]:  # 只显示前 10 个
            print(f"  {s}")
    print(f"{'='*80}\n")
```

**验证清单**:
- [ ] Audio encoder 权重数量正确（约 400+）
- [ ] Audio projector 权重数量正确（约 2-4）
- [ ] Buffers 全部加载（5 个：mel_scale, window, 3x BatchNorm）
- [ ] Decoder 权重传递给 language_model
- [ ] 跳过的权重只有不需要的（rotary_emb, 多余的 bias）

### 第二阶段：特征提取验证

**添加调试输出**:
```python
def get_audio_feature(self, items):
    print(f"\n{'='*80}")
    print(f"get_audio_feature called with {len(items)} items")

    for i, item in enumerate(items):
        print(f"Item {i}:")
        print(f"  feature shape: {item.feature.shape}")
        print(f"  audio_length: {getattr(item, 'audio_length', 'NOT SET')}")
        print(f"  pad_value: {getattr(item, 'pad_value', 'NOT SET')}")

    input_values = torch.cat([item.feature for item in items], dim=0)
    print(f"Concatenated input_values: {input_values.shape}")

    encoder_out, encoder_atts = self.audio_encoder(input_values, audio_length)
    print(f"Encoder output: {encoder_out.shape}")

    audio_embeddings, _ = self.audio_projector(encoder_out, encoder_atts)
    print(f"Projector output: {audio_embeddings.shape}")

    masked_audio_features = audio_embeddings.reshape(-1, embed_dim)
    print(f"Final audio features: {masked_audio_features.shape}")
    print(f"Stats: min={masked_audio_features.min():.4f}, max={masked_audio_features.max():.4f}")
    print(f"{'='*80}\n")

    return masked_audio_features
```

**验证清单**:
- [ ] audio_length 是 Mel 帧数（不是波形采样数）
- [ ] 音频特征不是全零（min != 0 或 max != 0）
- [ ] 特征形状合理（约几百到几千个 token）
- [ ] dtype 和 device 正确

### 第三阶段：Token 替换验证

**添加调试输出**:
```python
def forward(self, input_ids, positions, forward_batch, **kwargs):
    if forward_batch.contains_mm_inputs():
        print(f"\n{'='*80}")
        print(f"input_ids shape: {input_ids.shape}")
        print(f"input_ids first 20: {input_ids[:20].tolist()}")

        if forward_batch.mm_inputs and len(forward_batch.mm_inputs) > 0:
            mm_input = forward_batch.mm_inputs[0]
            if mm_input and len(mm_input.mm_items) > 0:
                pad_value = mm_input.mm_items[0].pad_value
                print(f"Expected pad_value: {pad_value}")
                print(f"Count of pad_value in input_ids: {(input_ids == pad_value).sum().item()}")
        print(f"{'='*80}\n")

    return general_mm_embed_routine(...)
```

**验证清单**:
- [ ] input_ids 中包含 pad_value（不是 audio_token）
- [ ] pad_value 数量 = 音频 token 数量
- [ ] pad_value 是唯一的（不与词表冲突）

### 第四阶段：输出验证

**测试代码**:
```python
from sglang import Engine

llm = Engine(model_path="/path/to/midashenglm-7b", trust_remote_code=True)

outputs = llm.generate(
    prompt="把这段音频的内容转为文字",
    sampling_params={"temperature": 0.8, "top_p": 0.95},
    audio_data="/path/to/audio.wav"
)

print(outputs)
```

**验证清单**:
- [ ] 模型加载成功（无错误）
- [ ] 推理完成（不崩溃）
- [ ] 输出与音频内容匹配（不是随机文本）
- [ ] 输出质量与原始模型一致

---

## 常见问题与解决方案

### 问题 1: 音频特征全零

**症状**:
```
Audio embeddings stats: min=0.0000, max=0.0000
```

**原因**:
- Audio projector 权重未加载

**解决方案**:
```python
# 在 load_weights() 中添加名称映射
if "audio_projector" in name:
    name = name.replace(".net.0.", ".fc1.")
    name = name.replace(".net.2.", ".fc2.")
```

### 问题 2: 大量权重被跳过

**症状**:
```
Skipped weights: 412
```

**原因**:
- 名称不匹配（vLLM 到 SGLang 的组件名称差异）

**解决方案**:
1. 打印跳过的权重名称
2. 识别模式（如 `.attn.proj.` vs `.attn.attn.proj.`）
3. 添加名称映射规则
4. 重新测试

### 问题 3: 输出与音频内容不匹配

**症状**:
```
音频是语音，但模型输出歌词或随机文本
```

**可能原因**:
1. **Buffers 未加载** - Mel 滤波器、窗函数缺失
2. **audio_length 错误** - 使用波形长度而非 Mel 帧数
3. **Token 替换失败** - pad_value 未正确替换

**解决方案**:
```python
# 1. 确保加载 buffers
if name in buffers_dict:
    buffers_dict[name].copy_(loaded_weight)
    buffer_loaded.append(original_name)

# 2. 使用 HuggingFace processor 的 audio_length
if "audio_length" in ret and len(mm_items) > 0:
    audio_length = ret["audio_length"]  # Mel 帧数
    mm_items[0].audio_length = audio_length

# 3. 验证 token 替换
print(f"pad_value count: {(input_ids == pad_value).sum().item()}")
```

### 问题 4: Decoder 权重未加载

**症状**:
```
Skipped weights: 339 (all decoder weights)
```

**原因**:
- 自定义 `load_weights()` 拦截了 decoder 权重

**解决方案**:
```python
def load_weights(self, weights):
    decoder_weights = []

    for name, loaded_weight in weights:
        if name.startswith("decoder"):
            decoder_weights.append((name, loaded_weight))
            continue
        # ... 处理其他权重

    # 传递给 language_model
    if decoder_weights:
        decoder_weights_stripped = [
            (name.replace("decoder.", "", 1), weight)
            for name, weight in decoder_weights
        ]
        self.language_model.load_weights(decoder_weights_stripped)
```

### 问题 5: 内存不足 (OOM)

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 使用量化: `--quantization awq` 或 `--quantization gptq`
2. 减少批次大小: `--max-running-requests 1`
3. 使用更小的模型变体
4. 增加 GPU 数量进行张量并行

### 问题 6: RoPE 配置错误

**症状**:
```
Error: mrope_section not supported
```

**原因**:
- 模型配置包含 M-RoPE（多模态 RoPE），但当前模型使用标准 RoPE

**解决方案**:
```python
def __init__(self, config, quant_config=None, prefix=""):
    # 移除 mrope_section
    if hasattr(config.text_config, 'rope_scaling') and config.text_config.rope_scaling:
        if 'mrope_section' in config.text_config.rope_scaling:
            new_rope_scaling = {
                k: v for k, v in config.text_config.rope_scaling.items()
                if k != 'mrope_section'
            }
            config.text_config.rope_scaling = new_rope_scaling if new_rope_scaling else None
```

---

## 检查清单

### 实现完成度检查

#### ✅ 文件结构
- [ ] `python/sglang/srt/models/midashenglm.py` 已创建
- [ ] `python/sglang/srt/multimodal/processors/midashenglm.py` 已创建
- [ ] 模型已在 `model_runner.py` 注册
- [ ] 处理器已在 `processor_factory.py` 注册

#### ✅ 组件替换
- [ ] Audio Encoder 使用 VisionAttention
- [ ] Language Model 使用 Qwen2ForCausalLM（含 RadixAttention）
- [ ] 使用 SGLang 的 LogitsProcessor
- [ ] 所有 vLLM 导入已移除
- [ ] Sample/Sampler 已移除
- [ ] 使用正确的线性层（ColumnParallelLinear, RowParallelLinear, QKVParallelLinear）

#### ✅ 方法实现
- [ ] `__init__()` 正确初始化所有组件
- [ ] `pad_input_ids()` 实现 token 替换
- [ ] `get_audio_feature()` 处理音频特征提取
- [ ] `get_input_embeddings()` 返回 embed_tokens
- [ ] `forward()` 使用 ForwardBatch 和 general_mm_embed_routine
- [ ] `load_weights()` 实现名称映射和 buffer 加载
- [ ] EntryClass 已定义

#### ✅ 权重加载
- [ ] Audio encoder 权重映射正确
- [ ] Audio projector 权重映射正确
- [ ] Decoder 权重传递给 language_model
- [ ] Buffers 全部加载（mel_scale, window, BatchNorm）
- [ ] 不需要的权重正确跳过（rotary_emb, 多余 bias）

#### ✅ 多模态处理
- [ ] Processor 使用 HuggingFace 处理器提取特征
- [ ] audio_length 使用 Mel 帧数（不是波形长度）
- [ ] MultimodalDataItem 正确创建
- [ ] pad_value 唯一且正确替换
- [ ] MultimodalInputs 正确返回

#### ✅ 测试验证
- [ ] 模型加载无错误
- [ ] 权重加载统计正确
- [ ] 音频特征非零
- [ ] Token 替换成功（pad_value 在 input_ids 中）
- [ ] 推理成功完成
- [ ] 输出与音频内容匹配

---

## 进阶主题

### 性能优化

#### 1. 使用 FlashAttention
```python
# 在 VisionAttention 中
self.attn = VisionAttention(
    ...
    qkv_backend="fa3",  # 使用 FlashAttention 3
)
```

#### 2. 混合精度
```python
# 在模型初始化时
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    encoder_out, encoder_atts = self.audio_encoder(input_values, audio_length)
```

#### 3. 张量并行
```bash
# 启动时指定 TP
python -m sglang.launch_server \
    --model-path /path/to/midashenglm-7b \
    --tp 2  # 2-way tensor parallelism
```

### 支持批处理

确保 `get_audio_feature()` 正确处理多个音频：
```python
def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
    # 支持批处理
    input_values = torch.cat([item.feature for item in items], dim=0)  # [B, T]
    audio_lengths = [item.audio_length for item in items]  # [B]

    # 批量编码
    encoder_out, encoder_atts = self.audio_encoder(input_values, audio_lengths)  # [B, N, C]

    # 批量投影
    audio_embeddings, _ = self.audio_projector(encoder_out, encoder_atts)  # [B, M, C]

    # 展平批次维度
    return audio_embeddings.reshape(-1, embed_dim)  # [B*M, C]
```

### 量化支持

添加量化配置：
```python
class MiDashengLMModel(nn.Module):
    # BitandBytes 量化目标模块
    default_bitsandbytes_target_modules = [
        ".fc1.",
        ".fc2.",
        ".gate_up_proj.",
        ".down_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]

    # 堆叠参数映射
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
```

---

## 总结

本指南展示了从 vLLM 到 SGLang 移植多模态模型的完整方法论：

1. **架构分析** - 理解原始模型和目标框架
2. **组件替换** - 系统性替换 Attention、LogitsProcessor 等
3. **权重加载** - 实现名称映射和 buffer 加载
4. **多模态集成** - 使用 SGLang 的统一机制
5. **调试验证** - 逐阶段验证功能正确性

**关键成功因素**:
- ✅ 完整的名称映射（parameters + buffers）
- ✅ 正确的 audio_length（Mel 帧数）
- ✅ 使用 general_mm_embed_routine
- ✅ Decoder 权重委托给 language_model
- ✅ 系统化的调试和验证

这套方法论可以应用于其他多模态模型（视觉、音频、视频）的移植工作。

---

## 附录

### A. 参考实现
- **Qwen2Audio**: `python/sglang/srt/models/qwen2_audio.py`
- **Qwen2VL**: `python/sglang/srt/models/qwen2_vl.py`
- **LLaVA**: `python/sglang/srt/models/llava.py`

### B. 相关文档
- SGLang Documentation: https://sgl-project.github.io/
- SGLang GitHub: https://github.com/sgl-project/sglang
- vLLM GitHub: https://github.com/vllm-project/vllm

### C. 调试工具
```python
# 打印所有模型参数名称
for name, param in model.named_parameters():
    print(f"Parameter: {name}, shape: {param.shape}")

# 打印所有模型 buffer 名称
for name, buf in model.named_buffers():
    print(f"Buffer: {name}, shape: {buf.shape}")

# 检查权重加载状态
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {missing_keys}")
print(f"Unexpected: {unexpected_keys}")
```

### D. 版本信息
- SGLang: v0.4.0+
- PyTorch: 2.1.0+
- Transformers: 4.40.0+
- CUDA: 11.8+

---

**文档版本**: 1.0
**最后更新**: 2025-01-07
**作者**: SGLang Team
**基于**: MiDashengLM 移植项目
