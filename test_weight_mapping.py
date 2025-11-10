#!/usr/bin/env python3
"""
测试和验证MiDashengLM权重名称映射

这个脚本会：
1. 检查代码版本是否正确
2. 测试权重名称映射逻辑
3. 提供修复建议
"""

import sys
import os

# 添加python目录到路径
sys.path.insert(0, '/home/user/sglang/python')

def check_code_version():
    """检查load_weights代码是否包含修复"""
    print("=" * 80)
    print("步骤 1: 检查代码版本")
    print("=" * 80)

    model_file = '/home/user/sglang/python/sglang/srt/models/midashenglm.py'

    with open(model_file, 'r') as f:
        content = f.read()

    # 检查关键修复代码
    if 'name.replace("front_end.0.", "front_end.")' in content:
        print("✅ 修复代码已存在于源文件中")
        return True
    else:
        print("❌ 修复代码不存在！需要更新代码")
        return False

def test_name_mapping():
    """测试权重名称映射逻辑"""
    print("\n" + "=" * 80)
    print("步骤 2: 测试权重名称映射")
    print("=" * 80)

    # HuggingFace权重名称（实际情况）
    test_cases = [
        ("audio_encoder.front_end.0.mel_scale.fb",
         "audio_encoder.front_end.melscale_fbanks"),
        ("audio_encoder.front_end.0.spectrogram.window",
         "audio_encoder.front_end.spectrogram_window"),
    ]

    for hf_name, expected_name in test_cases:
        # 模拟映射逻辑（修复后）
        name = hf_name

        if "audio_encoder.front_end" in name:
            # 第1步：移除 .0.
            name = name.replace("front_end.0.", "front_end.")

            # 第2步：重命名buffer
            if ".mel_scale.fb" in name:
                name = name.replace(".mel_scale.fb", ".melscale_fbanks")
            elif ".spectrogram.window" in name:
                name = name.replace(".spectrogram.window", ".spectrogram_window")

        status = "✅" if name == expected_name else "❌"
        print(f"{status} {hf_name}")
        print(f"   → {name}")
        if name != expected_name:
            print(f"   Expected: {expected_name}")
        print()

def check_imported_module():
    """检查当前导入的模块版本"""
    print("=" * 80)
    print("步骤 3: 检查已导入的模块")
    print("=" * 80)

    try:
        # 尝试导入
        from sglang.srt.models.midashenglm import MiDashengLMModel
        import inspect

        # 获取load_weights源代码
        source = inspect.getsource(MiDashengLMModel.load_weights)

        if 'front_end.0.' in source:
            print("✅ 导入的模块包含修复代码")

            # 检查是否有新的调试输出
            if 'Starting weight loading for MiDashengLM' in source:
                print("✅ 包含增强的调试输出")
            else:
                print("⚠️  缺少增强的调试输出（可能是旧版本）")

            return True
        else:
            print("❌ 导入的模块不包含修复代码！")
            print("   说明: 您导入的是旧版本的代码")
            return False

    except ImportError as e:
        print(f"⚠️  无法导入模块: {e}")
        return False

def show_fix_instructions():
    """显示修复说明"""
    print("\n" + "=" * 80)
    print("修复方案")
    print("=" * 80)

    print("""
要使修复生效，您需要重新加载Python模块：

方法1: 重启Python进程（推荐）
-------------------------------
如果您在运行 sglang server:

    # 停止服务
    pkill -f sglang

    # 重新启动
    python -m sglang.launch_server \\
        --model mispeech/midashenglm-7b \\
        --dtype bfloat16

方法2: 清除Python缓存
----------------------
    cd /home/user/sglang

    # 清除所有.pyc文件
    find python -type f -name "*.pyc" -delete
    find python -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

    # 然后重启您的进程

方法3: 在Python中强制重新导入
-----------------------------
    import sys
    import importlib

    # 移除旧模块
    modules_to_remove = [k for k in sys.modules if k.startswith('sglang')]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # 重新导入
    from sglang.srt.models.midashenglm import MiDashengLMModel

验证修复
--------
重新加载后，您应该看到：

1. 新的日志格式：
   [WEIGHT LOADING] Starting weight loading for MiDashengLM
   [WEIGHT LOADING] Total weights received from iterator: 736

2. Skipped weights减少：
   之前: Skipped weights: 4, Skipped audio_encoder weights: 2
   之后: Skipped weights: 2, Skipped audio_encoder weights: 0

3. 只剩bias的跳过（正常）：
   audio_projector.net.0.bias (bias not in params/buffers)
   audio_projector.net.2.bias (bias not in params/buffers)
""")

def main():
    print("\n🔍 MiDashengLM 权重加载诊断工具\n")

    # 检查代码版本
    code_ok = check_code_version()

    # 测试映射逻辑
    test_name_mapping()

    # 检查导入的模块
    module_ok = check_imported_module()

    # 显示修复说明
    if code_ok and not module_ok:
        print("\n⚠️  源代码已修复，但导入的模块是旧版本")
        print("   您需要重新加载Python模块")
        show_fix_instructions()
    elif not code_ok:
        print("\n❌ 源代码未修复，请先更新代码")
        print("   git pull origin claude/analyze-sglang-vllm-midasheng-011CUyN6ah3d1Sp5KxeN2zUK")
    else:
        print("\n✅ 所有检查通过！")
        print("   如果仍看到旧日志，请重启您的进程")
        show_fix_instructions()

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
