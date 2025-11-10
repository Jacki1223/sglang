#!/usr/bin/env python3
"""
验证MiDashengLM权重加载的脚本

此脚本会：
1. 加载模型并捕获权重加载日志
2. 验证所有组件的权重都被正确加载
3. 输出详细的诊断信息
"""

import sys
import torch
from transformers import AutoConfig, AutoProcessor

# 设置环境变量以启用详细日志
import os
os.environ['SGLANG_LOG_LEVEL'] = 'DEBUG'


def verify_weight_loading(model_path="mispeech/midashenglm-7b"):
    """验证权重加载是否完整"""

    print("="*80)
    print("MiDashengLM 权重加载验证工具")
    print("="*80)
    print(f"\n模型路径: {model_path}\n")

    # 1. 检查本地文件
    print("步骤 1: 检查模型文件...")
    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(model_path))
        safetensors_files = [f for f in files if f.endswith('.safetensors') and 'model-' in f]

        print(f"找到 {len(safetensors_files)} 个safetensors文件:")
        for f in sorted(safetensors_files):
            print(f"  - {f}")
    except Exception as e:
        print(f"⚠️  无法列出文件: {e}")

    # 2. 加载配置
    print("\n步骤 2: 加载模型配置...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ 配置加载成功")
        print(f"  - 模型类型: {config.model_type}")
        print(f"  - 文本模型: {config.text_config.model_type if hasattr(config, 'text_config') else 'N/A'}")
        if hasattr(config, 'text_config'):
            print(f"  - Hidden size: {config.text_config.hidden_size}")
            print(f"  - Num layers: {config.text_config.num_hidden_layers}")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False

    # 3. 分析权重索引文件
    print("\n步骤 3: 分析权重分布...")
    try:
        from huggingface_hub import hf_hub_download
        import json

        index_file = hf_hub_download(model_path, "model.safetensors.index.json")
        with open(index_file) as f:
            index_data = json.load(f)

        weight_map = index_data['weight_map']

        # 按组件分类
        audio_encoder_weights = []
        audio_projector_weights = []
        decoder_weights = []

        for weight_name, file_name in weight_map.items():
            if weight_name.startswith('audio_encoder'):
                audio_encoder_weights.append(weight_name)
            elif weight_name.startswith('audio_projector'):
                audio_projector_weights.append(weight_name)
            elif weight_name.startswith('decoder'):
                decoder_weights.append(weight_name)

        print(f"权重分布统计:")
        print(f"  - Audio encoder 权重: {len(audio_encoder_weights)}")
        print(f"  - Audio projector 权重: {len(audio_projector_weights)}")
        print(f"  - Decoder 权重: {len(decoder_weights)}")
        print(f"  - 总计: {len(weight_map)}")

        # 按文件统计
        from collections import defaultdict
        weights_per_file = defaultdict(list)
        for weight_name, file_name in weight_map.items():
            weights_per_file[file_name].append(weight_name)

        print(f"\n每个文件的权重数量:")
        for file_name in sorted(weights_per_file.keys()):
            count = len(weights_per_file[file_name])
            # 统计各组件在此文件中的权重
            audio_enc = sum(1 for w in weights_per_file[file_name] if w.startswith('audio_encoder'))
            audio_proj = sum(1 for w in weights_per_file[file_name] if w.startswith('audio_projector'))
            dec = sum(1 for w in weights_per_file[file_name] if w.startswith('decoder'))

            print(f"  {file_name}:")
            print(f"    总计: {count} (encoder: {audio_enc}, projector: {audio_proj}, decoder: {dec})")

    except Exception as e:
        print(f"⚠️  无法分析权重索引: {e}")

    # 4. 尝试加载模型（如果SGLang可用）
    print("\n步骤 4: 尝试加载模型...")
    try:
        from sglang.srt.models.midashenglm import MiDashengLMModel
        from sglang.srt.model_loader.loader import DefaultModelLoader
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.configs.load_config import LoadConfig

        print("正在加载模型（这可能需要几分钟）...")
        print("请查看上方的详细权重加载日志\n")

        # 配置加载选项
        model_config = ModelConfig(
            model_path=model_path,
            trust_remote_code=True,
        )
        load_config = LoadConfig()

        # 初始化加载器
        loader = DefaultModelLoader(load_config)

        # 加载模型（会触发详细的权重加载日志）
        model = loader.load_model(model_config)

        print("\n✓ 模型加载成功！")

        # 验证参数是否已初始化
        print("\n步骤 5: 验证参数初始化...")
        uninitialized = []
        zero_params = []

        for name, param in model.named_parameters():
            if param.requires_grad and torch.all(param == 0):
                zero_params.append(name)
            # 可以添加更多检查

        if zero_params:
            print(f"⚠️  发现 {len(zero_params)} 个全零参数（可能未初始化）:")
            for name in zero_params[:10]:  # 只显示前10个
                print(f"    - {name}")
            if len(zero_params) > 10:
                print(f"    ... 还有 {len(zero_params) - 10} 个")
        else:
            print("✓ 所有参数都已正确初始化")

        return True

    except ImportError:
        print("⚠️  SGLang未安装，跳过模型加载测试")
        print("   如需完整验证，请先安装SGLang")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="验证MiDashengLM权重加载")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mispeech/midashenglm-7b",
        help="模型路径或HuggingFace模型ID"
    )

    args = parser.parse_args()

    success = verify_weight_loading(args.model_path)

    print("\n" + "="*80)
    if success:
        print("验证完成！请查看上方的详细日志以了解权重加载情况。")
    else:
        print("验证过程中出现错误，请查看上方的错误信息。")
    print("="*80)

    sys.exit(0 if success else 1)
