"""
SGLang 新模型实现模板

这是一个完整的模板文件，展示如何为 SGLang 添加新的多模态模型支持。
基于 MiDashengLM 的实际实现经验整理。

使用方法：
1. 复制此文件到 python/sglang/srt/models/your_model.py
2. 根据你的模型替换所有 YOUR_MODEL 占位符
3. 实现标记为 TODO 的部分
4. 参考 SGLANG_NEW_MODEL_GUIDE.md 了解详细说明
"""

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # TODO: 根据你的解码器类型导入

# TODO: 导入你需要的其他组件


class YourModalityEncoder(nn.Module):
    """
    多模态编码器 (音频/视觉等)

    TODO: 根据你的模型实现编码器
    - 对于音频模型：可能是 Whisper, Wav2Vec2 等
    - 对于视觉模型：可能是 CLIP, SigLIP 等
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # TODO: 实现编码器层
        # 示例：
        # self.conv1 = nn.Conv1d(...)
        # self.transformer = nn.TransformerEncoder(...)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        编码输入特征

        Args:
            inputs: 输入特征，shape 取决于模态类型
                   - 音频: [batch, mel_bins, time_steps]
                   - 图像: [batch, channels, height, width]

        Returns:
            编码后的特征，通常是 [batch, seq_len, hidden_dim]
        """
        # TODO: 实现前向传播
        raise NotImplementedError("请实现编码器的前向传播")


class YourModalityProjector(nn.Module):
    """
    投影器：将多模态特征映射到语言模型空间

    关键点：
    1. 输入维度 = 编码器输出维度
    2. 输出维度 = 语言模型隐藏层维度
    3. 可能包含下采样逻辑
    4. ⚠️ bias 参数必须与预训练模型一致
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        # TODO: 添加其他参数，如 downsample_factor
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # TODO: 实现投影器
        # 简单示例：MLP
        # self.fc1 = ColumnParallelLinear(
        #     input_size=in_dim,
        #     output_size=hidden_dim,
        #     bias=True,  # ⚠️ 关键：必须与预训练模型一致
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.fc1",
        # )
        # self.act = nn.GELU()
        # self.fc2 = RowParallelLinear(
        #     input_size=hidden_dim,
        #     output_size=out_dim,
        #     bias=True,  # ⚠️ 关键：必须与预训练模型一致
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.fc2",
        # )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        投影多模态特征

        Args:
            features: 编码器输出，[batch, seq_len, in_dim]

        Returns:
            投影后的特征，[batch, new_seq_len, out_dim]
            注意：如果有下采样，new_seq_len != seq_len
        """
        # TODO: 实现投影逻辑
        # 示例：
        # x = self.fc1(features)[0]
        # x = self.act(x)
        # x = self.fc2(x)[0]
        # return x
        raise NotImplementedError("请实现投影器的前向传播")


class YourModelInputMetadata:
    """
    输入元数据处理类

    负责从 ForwardBatch 中提取和组织多模态输入信息
    """

    def __init__(self, forward_batch: ForwardBatch):
        """
        从 forward_batch 中提取多模态输入

        Args:
            forward_batch: SGLang 的批次信息
        """
        # TODO: 提取音频/图像输入和偏移位置
        # 示例（音频）：
        # self.audio_inputs = []
        # self.audio_offsets = []
        #
        # for i, rid in enumerate(forward_batch.seq_lens.keys()):
        #     if rid in forward_batch.reqs.audio_inputs:
        #         audio_item = forward_batch.reqs.audio_inputs[rid][0]
        #         self.audio_inputs.append(audio_item)
        #         # 计算在序列中的位置
        #         offset = (
        #             forward_batch.extend_prefix_lens[i]
        #             + audio_item["offsets"][0]
        #         )
        #         self.audio_offsets.append(offset)

        raise NotImplementedError("请实现输入元数据提取逻辑")


class YourModelForConditionalGeneration(nn.Module):
    """
    主模型类

    这是 SGLang 的入口点，需要实现：
    1. __init__: 初始化所有组件
    2. forward: 前向传播
    3. load_weights: 加载预训练权重
    4. get_mm_input_encoder_grouped_output_size: 计算多模态特征长度
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # TODO: 检查配置是否正确
        # 关键：确保解码器类型与配置匹配
        # if hasattr(config, "text_config"):
        #     text_config = config.text_config
        #     print(f"Decoder type: {text_config.model_type}")
        #
        #     # ⚠️ 重要：不要随意修改 rope_scaling 等配置
        #     # 保持从 HuggingFace 加载的原始配置
        #     if hasattr(text_config, "rope_scaling"):
        #         print(f"RoPE config: {text_config.rope_scaling}")

        # 1. 初始化多模态编码器
        # TODO: 根据你的模型类型初始化编码器
        # self.audio_encoder = YourModalityEncoder(
        #     config.audio_config,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.audio_encoder",
        # )
        # 或
        # self.vision_encoder = YourModalityEncoder(
        #     config.vision_config,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.vision_encoder",
        # )

        # 2. 初始化投影器
        # TODO: 初始化投影器，确保维度正确
        # encoder_dim = config.audio_config.hidden_size
        # lm_dim = config.text_config.hidden_size
        # self.projector = YourModalityProjector(
        #     in_dim=encoder_dim,
        #     out_dim=lm_dim,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.projector",
        # )

        # 3. 初始化语言模型
        # TODO: 根据配置选择正确的解码器
        # 示例：
        # if config.text_config.model_type == "qwen2":
        #     from sglang.srt.models.qwen2 import Qwen2ForCausalLM
        #     self.language_model = Qwen2ForCausalLM(
        #         config.text_config,
        #         quant_config=quant_config,
        #         cache_config=cache_config,
        #         prefix=f"{prefix}.language_model",
        #     )
        # elif config.text_config.model_type == "llama":
        #     from sglang.srt.models.llama import LlamaForCausalLM
        #     self.language_model = LlamaForCausalLM(...)

        raise NotImplementedError("请实现模型初始化")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入 token IDs，[total_tokens]
            positions: 位置索引，[total_tokens]
            forward_batch: 批次信息，包含多模态输入
            input_embeds: 可选的输入嵌入
            get_embedding: 是否只返回嵌入

        Returns:
            模型输出或嵌入
        """
        # 1. 提取多模态输入元数据
        # TODO: 处理多模态输入
        # metadata = YourModelInputMetadata(forward_batch)

        # 2. 编码多模态输入
        # TODO: 如果有多模态输入，编码并投影
        # mm_embeds = None
        # if metadata.audio_inputs:
        #     # 2.1 加载和预处理音频/图像
        #     raw_inputs = self._load_and_preprocess(metadata.audio_inputs)
        #
        #     # 2.2 编码
        #     encoded_features = self.audio_encoder(raw_inputs)
        #
        #     # 2.3 投影
        #     mm_embeds = self.projector(encoded_features)

        # 3. 获取文本嵌入
        # if input_embeds is None:
        #     input_embeds = self.language_model.model.embed_tokens(input_ids)

        # 4. 融合多模态特征
        # TODO: 将多模态特征插入到文本嵌入的正确位置
        # if mm_embeds is not None:
        #     input_embeds = self._merge_multimodal_embeddings(
        #         input_embeds,
        #         mm_embeds,
        #         metadata.audio_offsets,  # 或 metadata.image_offsets
        #     )

        # 5. 语言模型生成
        # return self.language_model(
        #     input_ids=None,  # 已经有 input_embeds 了
        #     positions=positions,
        #     forward_batch=forward_batch,
        #     input_embeds=input_embeds,
        #     get_embedding=get_embedding,
        # )

        raise NotImplementedError("请实现前向传播")

    def _load_and_preprocess(self, mm_inputs: List[dict]) -> torch.Tensor:
        """
        加载和预处理多模态输入

        Args:
            mm_inputs: 多模态输入列表，每项包含 URL 或路径

        Returns:
            预处理后的张量
        """
        # TODO: 实现多模态数据加载
        # 示例（音频）：
        # import torchaudio
        # audios = []
        # for item in mm_inputs:
        #     url = item["url"]
        #     # 下载或加载
        #     waveform, sr = torchaudio.load(url)
        #     # 预处理（重采样、提取 mel 谱等）
        #     features = self._preprocess_audio(waveform, sr)
        #     audios.append(features)
        # return torch.stack(audios)

        raise NotImplementedError("请实现多模态数据加载")

    def _merge_multimodal_embeddings(
        self,
        text_embeds: torch.Tensor,
        mm_embeds: torch.Tensor,
        offsets: List[int],
    ) -> torch.Tensor:
        """
        将多模态特征融合到文本嵌入中

        这是多模态模型的核心逻辑！

        Args:
            text_embeds: 文本嵌入，[total_tokens, hidden_dim]
            mm_embeds: 多模态特征，[num_items, feature_len, hidden_dim]
            offsets: 每个多模态特征应该插入的位置

        Returns:
            融合后的嵌入，[total_tokens, hidden_dim]
        """
        # TODO: 实现特征融合逻辑
        # 关键：确保位置对齐正确
        # 示例：
        # new_embeds = text_embeds.clone()
        # for i, offset in enumerate(offsets):
        #     feature_len = mm_embeds[i].shape[0]
        #     # 将多模态特征插入到指定位置
        #     new_embeds[offset:offset+feature_len] = mm_embeds[i]
        # return new_embeds

        raise NotImplementedError("请实现特征融合逻辑")

    def get_mm_input_encoder_grouped_output_size(
        self,
        mm_counts: dict,
    ) -> int:
        """
        计算多模态输入经过编码器和投影器后的序列长度

        SGLang 使用这个来预分配 KV cache 空间

        Args:
            mm_counts: 多模态输入计数，例如 {"audio": 2, "image": 1}

        Returns:
            总的多模态 token 数量
        """
        # TODO: 根据你的模型计算
        # 需要考虑：
        # 1. 编码器输出长度
        # 2. 投影器的下采样因子
        #
        # 示例（音频）：
        # audio_count = mm_counts.get("audio", 0)
        # # Whisper 输出 1500 帧，投影器下采样 4 倍
        # tokens_per_audio = 1500 // 4  # = 375
        # return audio_count * tokens_per_audio
        #
        # 示例（图像）：
        # image_count = mm_counts.get("image", 0)
        # # CLIP 输出 (H/14) * (W/14) 个 patch，例如 224x224 -> 16*16 = 256
        # tokens_per_image = (self.config.vision_config.image_size // 14) ** 2
        # return image_count * tokens_per_image

        raise NotImplementedError("请实现多模态序列长度计算")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        加载预训练权重

        关键点：
        1. 正确映射权重名称
        2. 处理不同组件的权重（编码器、投影器、语言模型）
        3. 验证所有权重都被加载，无跳过
        4. 打印详细的加载统计
        """
        # TODO: 实现权重加载
        # 建议的结构：

        # 1. 获取模型参数和缓冲区
        # params_dict = dict(self.named_parameters())
        # buffers_dict = dict(self.named_buffers())

        # 2. 准备统计信息
        # encoder_weights = []
        # projector_weights = []
        # lm_weights = []
        # skipped_weights = []

        # 3. 遍历检查点权重
        # for name, loaded_weight in weights:
        #     # 3.1 处理编码器权重
        #     if "audio_encoder" in name or "vision_encoder" in name:
        #         # 去掉前缀，例如 "model.audio_encoder." -> "audio_encoder."
        #         param_name = name.replace("model.", "")
        #         if param_name in params_dict:
        #             param = params_dict[param_name]
        #             weight_loader = getattr(
        #                 param, "weight_loader", default_weight_loader
        #             )
        #             weight_loader(param, loaded_weight)
        #             encoder_weights.append(name)
        #         else:
        #             skipped_weights.append(f"{name} (not in model)")
        #
        #     # 3.2 处理投影器权重
        #     elif "projector" in name:
        #         param_name = name.replace("model.", "")
        #         if param_name in params_dict:
        #             param = params_dict[param_name]
        #             weight_loader = getattr(
        #                 param, "weight_loader", default_weight_loader
        #             )
        #             weight_loader(param, loaded_weight)
        #             projector_weights.append(name)
        #         elif param_name in buffers_dict:
        #             # 处理缓冲区（如 running_mean, running_var）
        #             buffers_dict[param_name].copy_(loaded_weight)
        #             projector_weights.append(name)
        #         else:
        #             # ⚠️ 重要：诊断跳过原因
        #             reason = "unknown"
        #             if "bias" in name:
        #                 reason = "bias not in params (check bias=True/False)"
        #             skipped_weights.append(f"{name} ({reason})")
        #
        #     # 3.3 处理语言模型权重
        #     elif "language_model" in name:
        #         param_name = name.replace("language_model.", "")
        #         # 委托给语言模型自己的 load_weights
        #         self.language_model.load_weights([(param_name, loaded_weight)])
        #         lm_weights.append(name)

        # 4. 打印加载统计
        # print(f"{'='*80}")
        # print(f"Weight Loading Summary")
        # print(f"{'='*80}")
        # print(f"✅ Encoder weights loaded: {len(encoder_weights)}")
        # print(f"✅ Projector weights loaded: {len(projector_weights)}")
        # print(f"✅ Language model weights loaded: {len(lm_weights)}")
        #
        # if skipped_weights:
        #     print(f"\n⚠️  Skipped weights: {len(skipped_weights)}")
        #     for w in skipped_weights[:10]:  # 只显示前 10 个
        #         print(f"  - {w}")
        #     if len(skipped_weights) > 10:
        #         print(f"  ... and {len(skipped_weights) - 10} more")
        #     print(f"\n❌ WARNING: Some weights were not loaded!")
        #     print(f"   This may cause poor model performance.")
        #     print(f"   Please check bias settings and parameter names.")
        # else:
        #     print(f"\n✅ All weights loaded successfully!")
        # print(f"{'='*80}")

        raise NotImplementedError("请实现权重加载逻辑")


# ============================================================================
# 注册模型（在 python/sglang/srt/models/__init__.py 中）
# ============================================================================

"""
TODO: 将以下内容添加到 python/sglang/srt/models/__init__.py

1. 在 _MODELS 字典中添加：

_MODELS = {
    # ... 现有模型
    "YourModelForConditionalGeneration": (
        "your_model",  # 你的模型文件名（不含 .py）
        "YourModelForConditionalGeneration",  # 类名
    ),
}

2. 如果是音频模型，在 _AUDIO_MODELS 集合中添加：

_AUDIO_MODELS = {
    "YourModelForConditionalGeneration",
    # ... 其他音频模型
}

3. 如果是视觉模型，在 _VISION_MODELS 集合中添加：

_VISION_MODELS = {
    "YourModelForConditionalGeneration",
    # ... 其他视觉模型
}
"""


# ============================================================================
# 测试代码
# ============================================================================

"""
TODO: 创建测试脚本 test_your_model.py

#!/usr/bin/env python3
import requests

# 1. 启动服务器
# python -m sglang.launch_server \\
#     --model your-org/your-model \\
#     --trust-remote-code \\
#     --enable-multimodal \\
#     --dtype float16 \\
#     --port 30000

# 2. 测试单个输入
def test_single():
    url = "http://localhost:30000/v1/chat/completions"

    payload = {
        "model": "your-org/your-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",  # 或 "image_url"
                        "audio_url": {"url": "path/to/audio.mp3"}
                    },
                    {
                        "type": "text",
                        "text": "请描述这段音频的内容。"
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    response = requests.post(url, json=payload)
    print(response.json())

if __name__ == "__main__":
    test_single()
"""


# ============================================================================
# 常见问题检查清单
# ============================================================================

"""
在实现过程中，请检查以下关键点：

□ 配置保真
  □ 不修改 rope_scaling 配置
  □ 不修改 attention 相关配置
  □ 使用 AutoConfig 加载的原始配置

□ 权重完整性
  □ bias 参数设置正确（bias=True/False 与预训练模型一致）
  □ 所有权重都被加载，无跳过
  □ 权重名称映射正确

□ 多模态对齐
  □ get_mm_input_encoder_grouped_output_size 计算准确
  □ 特征插入位置正确
  □ 编码器输出维度匹配投影器输入维度
  □ 投影器输出维度匹配语言模型隐藏层维度

□ 解码器选择
  □ 根据 text_config.model_type 选择正确的解码器
  □ 解码器支持必要的特性（如 M-RoPE）

□ 测试验证
  □ 单输入测试通过
  □ 批量测试通过
  □ 与参考实现（HuggingFace/vLLM）输出对比一致
  □ 性能符合预期

□ 代码质量
  □ 添加详细注释
  □ 打印有用的调试信息
  □ 错误处理完善
  □ 遵循 SGLang 代码风格
"""
