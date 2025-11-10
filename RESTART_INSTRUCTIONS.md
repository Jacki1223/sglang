# 🔄 重启SGLang服务以应用修复

## 📋 当前状态

✅ 修复代码已在源文件中 (`python/sglang/srt/models/midashenglm.py:723-732`)
✅ Python缓存已清除
⏳ 等待pip安装完成（可选）

## 🚨 关键问题

您当前看到的日志：
```
[WEIGHT LOADING] Skipped audio_encoder weights: 2
  audio_encoder.front_end.0.mel_scale.fb (not in model)
  audio_encoder.front_end.0.spectrogram.window (not in model)
```

**原因**: Python进程正在使用旧的缓存模块

**解决**: 重启SGLang服务以加载新代码

---

## 🔧 操作步骤

### 步骤1: 停止当前的SGLang服务

```bash
# 查找并停止sglang进程
pkill -f sglang

# 或者如果知道进程ID
kill <PID>
```

### 步骤2: 等待pip安装完成（如果还在运行）

```bash
# 检查pip是否还在运行
ps aux | grep pip

# 等待完成，或者按Ctrl+C停止（已清除缓存就够了）
```

### 步骤3: 重新启动SGLang服务

```bash
cd /home/user/sglang

# 使用您原来的启动命令，例如：
python -m sglang.launch_server \
    --model mispeech/midashenglm-7b \
    --dtype bfloat16 \
    --port 30000
```

---

## ✅ 验证修复成功

重启后，您应该看到：

### 修复前的日志（当前）:
```
[WEIGHT LOADING] Skipped weights: 4
[WEIGHT LOADING] Skipped audio_encoder weights: 2
  First 10 non-bias skipped:
    audio_encoder.front_end.0.mel_scale.fb (not in model)
    audio_encoder.front_end.0.spectrogram.window (not in model)
```

### 修复后的日志（预期）:
```
[WEIGHT LOADING] Skipped weights: 2
[WEIGHT LOADING] Skipped audio_encoder weights: 0  ← 变成0！

[WEIGHT LOADING] Skipped audio_projector weights:
  audio_projector.net.0.bias (bias not in params/buffers)  ← 只剩这2个（正常）
  audio_projector.net.2.bias (bias not in params/buffers)
```

**关键变化**:
- ✅ `Skipped weights: 4` → `2`
- ✅ `Skipped audio_encoder weights: 2` → `0`
- ✅ 不再显示 `front_end.0.mel_scale.fb` 和 `front_end.0.spectrogram.window`

---

## 🔍 修复的技术细节

### 修复代码位置
`python/sglang/srt/models/midashenglm.py` 第723-732行:

```python
# Handle audio encoder frontend buffers (mel_scale and spectrogram window)
if "audio_encoder.front_end" in name:
    # HuggingFace weights have an extra ".0." in the path
    # e.g., "audio_encoder.front_end.0.mel_scale.fb"
    # but our model has "audio_encoder.front_end.melscale_fbanks"
    name = name.replace("front_end.0.", "front_end.")  # ← 关键修复！

    # Map buffer names
    if ".mel_scale.fb" in name:
        name = name.replace(".mel_scale.fb", ".melscale_fbanks")
    elif ".spectrogram.window" in name:
        name = name.replace(".spectrogram.window", ".spectrogram_window")
```

### 为什么需要这个修复

| HuggingFace权重名称 | 模型Buffer名称 | 需要的映射 |
|---------------------|----------------|-----------|
| `audio_encoder.front_end.0.mel_scale.fb` | `audio_encoder.front_end.melscale_fbanks` | 移除`.0.` + 重命名 |
| `audio_encoder.front_end.0.spectrogram.window` | `audio_encoder.front_end.spectrogram_window` | 移除`.0.` + 重命名 |

**修复前**: 这2个buffer未加载 → 音频预处理失败
**修复后**: 所有buffer正确加载 → 音频处理正常

---

## 🆘 如果仍有问题

### 1. 确认修复代码存在
```bash
grep -n "front_end.0." /home/user/sglang/python/sglang/srt/models/midashenglm.py
# 应该看到第727行: name = name.replace("front_end.0.", "front_end.")
```

### 2. 确认缓存已清除
```bash
find /home/user/sglang/python -name "*.pyc" | wc -l
# 应该是0或很小的数字
```

### 3. 强制重新导入
在Python中：
```python
import sys
# 清除所有sglang模块
for key in list(sys.modules.keys()):
    if key.startswith('sglang'):
        del sys.modules[key]

# 重新导入
from sglang.srt.models.midashenglm import MiDashengLMModel
```

### 4. 查看提交历史
```bash
git log --oneline -3
# 应该看到:
# 4da47be 📚 Add comprehensive checkpoint loading analysis and documentation
# a9f8adb Revert decoder weight loading changes - restore working version
# ...
# 8293417 🐛 修复MiDashengLM音频前端buffer加载失败的关键bug
```

---

## 📚 相关文档

- `CHECKPOINT_LOADING_EXPLAINED.md` - 完整的checkpoint加载机制说明
- `BUG_FIX_SUMMARY.md` - Bug修复详情
- `QUICK_FIX_GUIDE.md` - 快速修复指南
- `MIDASHENGLM_ANALYSIS.md` - 技术实现分析

---

## 🎯 预期结果

重启后，MiDashengLM应该：
- ✅ 正确加载所有音频前端buffer
- ✅ 音频预处理正常工作
- ✅ 模型输出正确
- ✅ 只有2个bias被跳过（这是正常的）

---

**最后更新**: 2025-11-10
**修复提交**: 8293417
**当前分支**: `claude/analyze-sglang-vllm-midasheng-011CUyN6ah3d1Sp5KxeN2zUK`
