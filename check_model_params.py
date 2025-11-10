#!/usr/bin/env python3
"""
检查MiDashengLM模型参数命名

这个脚本会显示模型的参数名称，帮助诊断权重加载问题
"""

import sys
sys.path.insert(0, '/home/user/sglang/python')

from transformers import AutoConfig
from sglang.srt.models.midashenglm import MiDashengLMModel

# 加载配置
config = AutoConfig.from_pretrained("mispeech/midashenglm-7b", trust_remote_code=True)

# 创建模型（不加载权重）
model = MiDashengLMModel(config)

print("=" * 80)
print("MiDashengLM 模型参数命名检查")
print("=" * 80)

# 统计不同组件的参数
audio_encoder_params = []
audio_projector_params = []
language_model_params = []

for name, param in model.named_parameters():
    if name.startswith('audio_encoder'):
        audio_encoder_params.append(name)
    elif name.startswith('audio_projector'):
        audio_projector_params.append(name)
    elif name.startswith('decoder'):
        language_model_params.append(name)
    else:
        print(f"⚠️  未分类参数: {name}")

print(f"\n📊 参数统计:")
print(f"  Audio encoder 参数: {len(audio_encoder_params)}")
print(f"  Audio projector 参数: {len(audio_projector_params)}")
print(f"  Language model 参数: {len(language_model_params)}")
print(f"  总计: {len(list(model.named_parameters()))}")

print(f"\n🔍 Language model 参数示例（前10个）:")
for name in language_model_params[:10]:
    print(f"  {name}")

print(f"\n🔍 HuggingFace权重格式应该是:")
print(f"  decoder.model.embed_tokens.weight")
print(f"  decoder.model.layers.0.xxx")
print(f"  decoder.lm_head.weight")

print(f"\n❓ 当前模型期望的权重名称是:")
if language_model_params:
    print(f"  {language_model_params[0]}")
    print(f"  {language_model_params[1] if len(language_model_params) > 1 else ''}")

print("\n" + "=" * 80)
print("问题分析:")
print("=" * 80)

if language_model_params and language_model_params[0].startswith('decoder.'):
    print("✅ 模型参数以 'decoder.' 开头")
    print("✅ 这与HuggingFace权重格式一致")
    print("\n💡 在load_weights中应该:")
    print("   1. 不要剥离 'decoder.' 前缀")
    print("   2. 直接将decoder权重传递给 params_dict")
    print("   3. 或者采用Qwen2Audio的方式处理所有权重")
else:
    print("❌ 模型参数不是以 'decoder.' 开头")
    print("❌ 这会导致权重名称不匹配")
    print(f"   实际参数名称: {language_model_params[0] if language_model_params else 'None'}")

print("=" * 80)
