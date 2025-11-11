# 调试音频重复问题使用指南

## 问题现象

使用真实语音数据集运行 bench_serving时，看到多个重复的输出结果。

## 已添加的调试功能

我在 bench_serving.py 中添加了详细的调试日志来追踪问题：

### 1. 数据集加载日志

位置：`sample_audio_requests` 函数末尾

输出格式：
```
[DEBUG_AUDIO_DATASET] Loaded audio samples:
  Sample 1: prompt='转写这段音频...' audio_data_len=1 audio_hash=a1b2c3d4
  Sample 2: prompt='描述这段音频...' audio_data_len=1 audio_hash=e5f6g7h8
  Sample 3: prompt='分析这段音频...' audio_data_len=1 audio_hash=i9j0k1l2
```

**关键信息**：
- `audio_data_len`: 每个样本的audio_data数组长度（应该是1）
- `audio_hash`: 音频base64编码的前200字符的hash（用于检测是否相同）
- 如果多个样本的hash相同 → 使用了相同的音频文件

### 2. 请求发送日志

位置：`async_request_openai_chat_completions` 函数中

输出格式：
```
[DEBUG_AUDIO_REQUEST] {"prompt_preview": "转写这段音频...", "audio_data_count": 1, "audio_hashes": ["0:a1b2c3d4"], "content_items": {"audio": 1, "text": 1, "total": 2}}
```

**关键信息**：
- `audio_data_count`: 这个请求包含多少个音频（应该是1）
- `audio_hashes`: 每个音频的hash值
- `content_items`: 请求content中的item数量
  - `audio`: 音频item数量（应该是1）
  - `text`: 文本item数量（应该是1）
  - `total`: 总item数量（应该是2）

## 使用步骤

### 步骤1: 运行带调试日志的bench_serving

```bash
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/your_dataset.jsonl \
    --num-prompts 3 \
    --port 30000 \
    --output-file debug_results.jsonl \
    --output-details 2>&1 | tee bench_debug.log
```

### 步骤2: 分析调试日志

从 `bench_debug.log` 中查找调试信息：

```bash
# 查看数据集加载信息
grep "DEBUG_AUDIO_DATASET" bench_debug.log

# 查看请求发送信息
grep "DEBUG_AUDIO_REQUEST" bench_debug.log
```

### 步骤3: 检查关键指标

#### A. 检查数据集加载

```bash
grep "DEBUG_AUDIO_DATASET" bench_debug.log -A 10
```

**期望输出**：
- 样本数量 = num_prompts
- 每个样本的 audio_data_len = 1
- 每个样本的 audio_hash 不同（如果相同说明重复了）

**异常情况**：
- ❌ 样本数量 > num_prompts → 数据集采样有问题
- ❌ audio_data_len > 1 → 同一个请求包含多个音频
- ❌ 多个样本有相同的 audio_hash → 使用了相同的音频文件

#### B. 检查请求发送

```bash
grep "DEBUG_AUDIO_REQUEST" bench_debug.log
```

**期望输出**：
- 请求数量 = num_prompts
- 每个请求的 audio_data_count = 1
- 每个请求的 content_items.audio = 1
- 每个请求的 content_items.total = 2

**异常情况**：
- ❌ 请求数量 > num_prompts → 发送了额外的请求
- ❌ audio_data_count > 1 → 单个请求包含多个音频
- ❌ content_items.audio > 1 → content中有多个audio item
- ❌ content_items.total > 2 → content结构异常

### 步骤4: 分析结果文件

```bash
# 使用诊断脚本
/tmp/diagnose_request_count.py debug_results.jsonl

# 检查生成的文本
python3 /tmp/view_results.py debug_results.jsonl --texts
```

## 诊断决策树

```
开始
│
├─ [DEBUG_AUDIO_DATASET] 显示N个样本，N = num_prompts？
│   ├─ NO → 数据集采样逻辑有问题
│   └─ YES → 继续
│
├─ 所有样本的 audio_hash 都不同？
│   ├─ NO → **找到原因**：数据集中有重复的音频文件
│   └─ YES → 继续
│
├─ 所有样本的 audio_data_len = 1？
│   ├─ NO → **找到原因**：audio_data数组被意外修改
│   └─ YES → 继续
│
├─ [DEBUG_AUDIO_REQUEST] 有N条记录，N = num_prompts？
│   ├─ NO → **找到原因**：请求发送数量异常
│   └─ YES → 继续
│
├─ 所有请求的 audio_data_count = 1？
│   ├─ NO → **找到原因**：请求包含多个音频
│   └─ YES → 继续
│
├─ 所有请求的 content_items.audio = 1？
│   ├─ NO → **找到原因**：content中有多个audio item
│   └─ YES → 继续
│
└─ 问题不在bench_serving代码中
    └─ 检查：
        1. 模型推理参数（temperature, repetition_penalty）
        2. 音频内容本身（是否真的有重复的语音）
        3. 模型行为（是否对某类音频有固定响应模式）
```

## 辅助调试工具

### 工具1: 检查数据集

```bash
/tmp/check_audio_dataset.sh /path/to/your_dataset.jsonl
```

输出：
- 数据集中有多少条记录
- 每条记录的详细信息
- 是否有重复的audio_path

### 工具2: 模拟数据处理流程

```bash
python3 /tmp/debug_bench_audio.py /path/to/your_dataset.jsonl 3
```

输出：
- 模拟采样过程
- 模拟请求构建过程
- 每个步骤的详细信息

### 工具3: 测试单条请求

创建测试脚本 `/tmp/test_single_audio.py`：

```python
#!/usr/bin/env python3
import requests
import json
import librosa
import pybase64
import wave
import io
import numpy as np

# 读取数据集第一条
with open('/path/to/your_dataset.jsonl', 'r') as f:
    item = json.loads(f.readline())

print(f"测试音频: {item['audio_path']}")
print(f"提示词: {item['prompt']}")

# 加载音频
audio_array, _ = librosa.load(item['audio_path'], sr=16000, mono=True)

# 编码音频
buffer = io.BytesIO()
with wave.open(buffer, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(16000)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    wav_file.writeframes(audio_int16.tobytes())

buffer.seek(0)
encoded = pybase64.b64encode(buffer.read()).decode('utf-8')
audio_uri = f"data:audio/wav;base64,{encoded}"

# 发送请求（使用与bench_serving不同的参数）
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "your-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_uri}},
                {"type": "text", "text": item['prompt']}
            ]
        }],
        "temperature": 0.7,
        "repetition_penalty": 1.05,
        "max_completion_tokens": 256
    }
)

result = response.json()
print("\n单条请求结果:")
print(result['choices'][0]['message']['content'])
```

## 预期发现

基于调试日志，我们可以找到以下问题之一：

### 发现1: 数据集重复
```
[DEBUG_AUDIO_DATASET] Loaded audio samples:
  Sample 1: ... audio_hash=a1b2c3d4
  Sample 2: ... audio_hash=a1b2c3d4  ← 相同！
  Sample 3: ... audio_hash=a1b2c3d4  ← 相同！
```
**解决方案**: 修改数据集，使用不同的音频文件

### 发现2: audio_data数组异常
```
[DEBUG_AUDIO_DATASET] Loaded audio samples:
  Sample 1: ... audio_data_len=3  ← 应该是1！
```
**解决方案**: 代码bug，需要修复

### 发现3: 请求数量异常
```
[DEBUG_AUDIO_REQUEST] ... (出现4次，但num_prompts=3)
```
**解决方案**: 检查warmup请求或其他重复发送的原因

### 发现4: content结构异常
```
[DEBUG_AUDIO_REQUEST] {"content_items": {"audio": 2, ...}}  ← 应该是1！
```
**解决方案**: 请求构建有bug，需要修复

## 下一步

1. **运行带调试日志的bench_serving**（按照步骤1）
2. **收集调试日志**
3. **分析日志并识别异常**
4. **报告发现的问题**

请将调试日志的关键部分（`[DEBUG_AUDIO_DATASET]` 和 `[DEBUG_AUDIO_REQUEST]` 相关内容）提供给我，我可以帮您准确定位问题。
