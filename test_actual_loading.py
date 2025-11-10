#!/usr/bin/env python3
"""
验证MiDashengLM实际加载了多少个safetensors文件

通过monkey patching来跟踪实际文件访问
"""

import sys
import os

sys.path.insert(0, '/home/user/sglang/python')

def test_safetensors_loading():
    """测试并跟踪safetensors文件加载"""
    import safetensors.torch

    # 跟踪文件访问
    accessed_files = []
    original_safe_open = safetensors.safe_open
    original_load = safetensors.torch.load

    def tracked_safe_open(filename, *args, **kwargs):
        accessed_files.append(('safe_open', filename))
        print(f"📂 Opening safetensors file: {os.path.basename(filename)}")
        return original_safe_open(filename, *args, **kwargs)

    def tracked_load(data, *args, **kwargs):
        accessed_files.append(('load', 'from_bytes'))
        print(f"📥 Loading safetensors from bytes")
        return original_load(data, *args, **kwargs)

    # Monkey patch
    safetensors.safe_open = tracked_safe_open
    safetensors.torch.load = tracked_load

    print("=" * 80)
    print("开始测试MiDashengLM权重加载")
    print("=" * 80)

    try:
        from sglang.srt.model_loader.weight_utils import prepare_weights

        model_path = "mispeech/midashenglm-7b-0804-fp32"

        print(f"\n模型: {model_path}")
        print("\n🔄 调用prepare_weights()...")

        # 这会触发safetensors文件的迭代加载
        weight_iterator = prepare_weights(
            model_name_or_path=model_path,
            revision=None,
            fall_back_to_pt=False,
        )

        print("\n📊 开始迭代权重...")
        weight_count = 0

        # 迭代所有权重（这会触发实际的文件读取）
        for name, tensor in weight_iterator:
            weight_count += 1
            if weight_count <= 5:
                print(f"  权重 {weight_count}: {name} - shape {tensor.shape}")

        print(f"\n✅ 总共迭代了 {weight_count} 个权重张量")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 恢复原函数
        safetensors.safe_open = original_safe_open
        safetensors.torch.load = original_load

    # 报告结果
    print("\n" + "=" * 80)
    print("📁 文件访问统计")
    print("=" * 80)

    # 统计唯一文件
    unique_files = set()
    for method, filename in accessed_files:
        if method == 'safe_open':
            unique_files.add(os.path.basename(filename))

    print(f"\n访问的唯一safetensors文件数: {len(unique_files)}")

    if unique_files:
        print("\n文件列表:")
        for i, filename in enumerate(sorted(unique_files), 1):
            print(f"  {i}. {filename}")

    print(f"\n总访问次数: {len(accessed_files)}")
    print(f"  - safe_open调用: {sum(1 for m, _ in accessed_files if m == 'safe_open')}")
    print(f"  - load调用: {sum(1 for m, _ in accessed_files if m == 'load')}")

    return len(unique_files), weight_count

def compare_with_transformers():
    """对比使用HuggingFace transformers直接加载"""
    print("\n" + "=" * 80)
    print("对比：使用HuggingFace Transformers加载")
    print("=" * 80)

    try:
        from transformers import AutoModel
        import safetensors.torch

        accessed_files = []
        original_safe_open = safetensors.safe_open

        def tracked_safe_open(filename, *args, **kwargs):
            accessed_files.append(filename)
            print(f"📂 HF loading: {os.path.basename(filename)}")
            return original_safe_open(filename, *args, **kwargs)

        safetensors.safe_open = tracked_safe_open

        print("\n🔄 使用AutoModel.from_pretrained()...")

        model = AutoModel.from_pretrained(
            "mispeech/midashenglm-7b-0804-fp32",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        safetensors.safe_open = original_safe_open

        unique_files = set(os.path.basename(f) for f in accessed_files)
        print(f"\n✅ HuggingFace加载了 {len(unique_files)} 个文件")

        return len(unique_files)

    except Exception as e:
        print(f"\n⚠️  HuggingFace加载失败: {e}")
        return None

def main():
    print("\n🔍 MiDashengLM文件加载验证工具\n")

    # 测试SGLang加载
    sglang_files, sglang_weights = test_safetensors_loading()

    # 测试HuggingFace加载（如果可用）
    # hf_files = compare_with_transformers()

    print("\n" + "=" * 80)
    print("🎯 结论")
    print("=" * 80)

    print(f"""
SGLang加载结果:
  - 访问的safetensors文件数: {sglang_files}
  - 加载的权重张量数: {sglang_weights}

预期值:
  - safetensors文件: 7个 (model-00001 到 model-00007)
  - 权重张量: ~740个

状态: {"✅ 正常" if sglang_files == 7 else f"⚠️  只加载了{sglang_files}个文件！"}
""")

    if sglang_files < 7:
        print("""
⚠️  警告：加载的文件数少于预期！

可能原因:
1. 模型目录中缺少某些文件
2. 权重加载逻辑有问题
3. 某些文件被跳过了

建议:
- 检查模型缓存目录是否完整
- 验证model.safetensors.index.json内容
- 检查是否有错误日志
""")

if __name__ == "__main__":
    main()
