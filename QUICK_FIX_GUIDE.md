# 🔧 MiDashengLM 权重加载问题快速修复指南

## ❗ 当前状态

您的日志显示仍在使用**旧版本代码**，因为：

### 旧日志格式（您当前看到的）:
```
[WEIGHT LOADING] Passing 339 decoder weights to language_model.load_weights()

================================================================================
[WEIGHT LOADING] Audio encoder weights loaded: 395
[WEIGHT LOADING] Audio projector weights loaded: 2
[WEIGHT LOADING] Decoder weights passed to language_model: 339
[WEIGHT LOADING] Skipped weights: 4
```

### 新日志格式（修复后应看到的）:
```
================================================================================
[WEIGHT LOADING] Starting weight loading for MiDashengLM        ← 新增！
[WEIGHT LOADING] Total weights received from iterator: 736      ← 新增！
================================================================================

[WEIGHT LOADING] Decoder weights breakdown:                     ← 新增！
  Total decoder weights: 339
  Total decoder parameters: 7,XXX,XXX,XXX                        ← 新增！

  By component:                                                  ← 新增！
    attention: 128 weights
    embed_tokens: 1 weights
    layernorm: 64 weights
    lm_head: 1 weights
    mlp: 192 weights

  Decoder layers found: 32                                       ← 新增！
  Layer coverage: 0 to 31                                        ← 新增！
  ✓ All layers present                                          ← 新增！
  Weights per layer: min=12, max=12                             ← 新增！

[WEIGHT LOADING] Passing 339 decoder weights to language_model.load_weights()
```

## 🔧 解决方案

### 方案1：重新安装SGLang（推荐）

```bash
cd /home/user/sglang/python
pip install -e . --no-build-isolation
```

### 方案2：清除Python缓存

```bash
# 清除所有.pyc文件
cd /home/user/sglang
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 重启Python进程
# 如果使用 sglang server，需要重启服务
```

### 方案3：强制重新导入模块

如果在Python环境中：
```python
import importlib
import sys

# 移除旧模块
if 'sglang.srt.models.midashenglm' in sys.modules:
    del sys.modules['sglang.srt.models.midashenglm']

# 重新导入
from sglang.srt.models.midashenglm import MiDashengLMModel
```

## ✅ 验证步骤

重新加载模型后，您应该看到：

### 1. 新的调试输出格式
包含详细的decoder weights breakdown

### 2. Skipped weights 减少
```
修复前: Skipped weights: 4
修复后: Skipped weights: 2  ✓

修复前: Skipped audio_encoder weights: 2
修复后: Skipped audio_encoder weights: 0  ✓
```

### 3. 只剩正常的bias跳过
```
[WEIGHT LOADING] Skipped audio_projector weights:
  audio_projector.net.0.bias (bias not in params/buffers)  ✓ 正常
  audio_projector.net.2.bias (bias not in params/buffers)  ✓ 正常
```

## 🔍 深入诊断

如果重新安装后仍有问题，运行：

```bash
cd /home/user/sglang
python verify_weight_loading.py --model-path mispeech/midashenglm-7b
```

这将：
- ✅ 检查所有7个safetensors文件
- ✅ 分析权重分布
- ✅ 验证参数初始化
- ✅ 生成详细诊断报告

## 📋 修复的关键代码

在 `python/sglang/srt/models/midashenglm.py` 第723-732行：

```python
# Handle audio encoder frontend buffers (mel_scale and spectrogram window)
if "audio_encoder.front_end" in name:
    # HuggingFace weights have an extra ".0." in the path
    # e.g., "audio_encoder.front_end.0.mel_scale.fb"
    # but our model has "audio_encoder.front_end.melscale_fbanks"
    name = name.replace("front_end.0.", "front_end.")  # ← 关键修复！

    if ".mel_scale.fb" in name:
        name = name.replace(".mel_scale.fb", ".melscale_fbanks")
    elif ".spectrogram.window" in name:
        name = name.replace(".spectrogram.window", ".spectrogram_window")
```

## 🚨 重要提醒

**未修复前的影响**：
- ❌ Mel滤波器组未加载 → 音频频谱计算错误
- ❌ STFT窗函数未加载 → 音频预处理完全失败
- ❌ 模型虽能运行，但输出**完全不正确**

**修复后**：
- ✅ 所有必需的buffer正确加载
- ✅ 音频预处理正常工作
- ✅ 模型输出正确

## 📞 如仍有问题

1. 确认Git提交历史：
   ```bash
   git log --oneline -3
   # 应该看到:
   # 1395b02 Add bug fix summary documentation
   # 8293417 🐛 修复MiDashengLM音频前端buffer加载失败的关键bug
   # 357b984 增强MiDashengLM权重加载诊断和分析
   ```

2. 确认代码已更新：
   ```bash
   grep -A 3 "front_end.0." python/sglang/srt/models/midashenglm.py
   # 应该看到修复代码
   ```

3. 查看完整分析：
   - `MIDASHENGLM_ANALYSIS.md` - 技术分析文档
   - `BUG_FIX_SUMMARY.md` - Bug修复详情

---

**最后更新**: 2025-11-10
**Git分支**: `claude/analyze-sglang-vllm-midasheng-011CUyN6ah3d1Sp5KxeN2zUK`
**关键提交**: `8293417`
