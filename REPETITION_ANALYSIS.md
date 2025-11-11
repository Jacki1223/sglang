# 音频基准测试重复输出深度分析

## 问题现象

用户报告使用真实语音数据集运行 bench_serving 时，看到4个结果，每个都输出重复内容：
```
结果 1: 。\n这段音频的内容是：'我跟你说，我跟你说，我跟你说，...'
结果 2: 。\n这段音频的内容是：'我跟你说，我跟你说，我跟你说，...'
结果 3: 。\n这段音频的内容是：'我跟你说，我跟你说，我跟你说，...'
结果 4: 。\n这段音频的内容是：'我跟你说，我跟你说，我跟你说，...'
```

## 代码检查结果

已完成以下代码路径的检查：

### ✅ 1. 数据采样逻辑 (sample_audio_requests)
- **位置**: bench_serving.py:1538-1623
- **检查结果**: 正常
  - 使用 `random.sample()` 无放回采样
  - 每个音频只被加载和编码一次
  - 不会重复处理同一音频

### ✅ 2. 请求构建 (async_request_openai_chat_completions)
- **位置**: bench_serving.py:318-361
- **检查结果**: 正常
  - 音频正确添加到 content_items
  - 每个请求独立构建
  - payload 结构正确

### ✅ 3. 请求生成和发送 (benchmark函数)
- **位置**: bench_serving.py:2045-2079
- **检查结果**: 正常
  - warmup_tasks 和 tasks 是分开的列表
  - warmup_outputs 不会混入 outputs
  - 每个 input_request 对应一个 task

### ✅ 4. 结果收集
- **位置**: bench_serving.py:2274-2291
- **检查结果**: 正常
  - generated_texts 从 outputs 收集
  - 数组长度应该与请求数一致

## 可能的原因分析

### 可能性1: 数据集中的音频文件重复

**假设**: 数据集JSONL文件中有4条记录，但它们指向相同的音频文件

**示例**:
```jsonl
{"prompt": "转写这段音频", "audio_path": "/path/same_audio.wav", "output_len": 256}
{"prompt": "描述这段音频", "audio_path": "/path/same_audio.wav", "output_len": 256}
{"prompt": "分析这段音频", "audio_path": "/path/same_audio.wav", "output_len": 256}
{"prompt": "识别这段音频", "audio_path": "/path/same_audio.wav", "output_len": 256}
```

**验证方法**:
```bash
/tmp/check_audio_dataset.sh /path/to/your_dataset.jsonl
```

### 可能性2: 音频内容确实是重复的语音

**假设**: 真实音频文件的内容就是有人重复说"我跟你说"

**特征**:
- 如果音频是测试录音或特定场景录音
- 说话人可能真的在重复同一句话

**验证方法**:
```bash
# 播放音频文件听一下
ffplay /path/to/audio.wav

# 或使用其他音频播放器
```

### 可能性3: 模型采样参数导致确定性重复

**问题根源**: bench_serving 硬编码了不利于避免重复的参数

**当前设置** (bench_serving.py:356-359):
```python
"temperature": 0.0,              # 贪婪采样
"ignore_eos": True,              # 忽略结束标记
# 没有设置 repetition_penalty
```

**影响**:
- temperature=0.0 → 每次选择概率最高的token
- 如果模型对某类音频倾向于输出固定模式
- 加上 ignore_eos=True，会强制生成到max_tokens
- 结果：可能陷入循环重复

### 可能性4: 结果数量异常 (4个而非3个)

**num_prompts=3 但有4个结果**

**可能原因**:
1. **warmup 请求被计入** (需要验证)
2. **文件是多次运行累积的** (JSONL追加模式)
3. **数据集实际有4条记录**

**验证方法**:
```bash
# 检查结果文件
/tmp/diagnose_request_count.py results.jsonl

# 检查数据集
wc -l /path/to/dataset.jsonl
cat /path/to/dataset.jsonl | jq .
```

## 诊断步骤

### 步骤1: 检查数据集
```bash
/tmp/check_audio_dataset.sh /path/to/your_dataset.jsonl
```

**关键问题**:
- 数据集有多少条记录？
- 是否有重复的 audio_path？
- 音频文件是否都存在？

### 步骤2: 检查结果文件
```bash
/tmp/diagnose_request_count.py results.jsonl
```

**关键问题**:
- 文件有几行（运行了几次）？
- 每次运行有多少个 generated_texts？
- 是否与 num_prompts 一致？

### 步骤3: 播放音频文件
```bash
# 听一下实际的音频内容
ffplay /path/to/audio1.wav
ffplay /path/to/audio2.wav
```

**关键问题**:
- 音频内容是什么？
- 是否真的有人在重复说话？

### 步骤4: 测试不同的采样参数
```bash
# 测试1: 添加 repetition_penalty
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/dataset.jsonl \
    --num-prompts 3 \
    --extra-request-body '{"repetition_penalty": 1.1}' \
    --output-file test1.jsonl \
    --output-details

# 测试2: 启用 EOS 检测
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/dataset.jsonl \
    --num-prompts 3 \
    --disable-ignore-eos \
    --output-file test2.jsonl \
    --output-details

# 测试3: 调整 temperature
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/dataset.jsonl \
    --num-prompts 3 \
    --extra-request-body '{"temperature": 0.7, "repetition_penalty": 1.05}' \
    --output-file test3.jsonl \
    --output-details
```

### 步骤5: 对比单条 generate
```python
import requests
import json

# 使用相同的音频和prompt测试单条请求
with open('/path/to/dataset.jsonl', 'r') as f:
    first_item = json.loads(f.readline())

# 加载和编码音频（与bench_serving相同的方式）
import librosa
import pybase64
import wave
import io

audio_array, _ = librosa.load(first_item['audio_path'], sr=16000, mono=True)
buffer = io.BytesIO()
with wave.open(buffer, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(16000)
    audio_int16 = (audio_array * 32767).astype('int16')
    wav_file.writeframes(audio_int16.tobytes())
buffer.seek(0)
encoded = pybase64.b64encode(buffer.read()).decode('utf-8')
audio_uri = f"data:audio/wav;base64,{encoded}"

# 发送请求
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "your-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_uri}},
                {"type": "text", "text": first_item['prompt']}
            ]
        }],
        "temperature": 0.7,         # 不同于bench_serving的0.0
        "repetition_penalty": 1.05,
        "max_completion_tokens": 256
    }
)

print("单条generate结果:")
print(response.json()['choices'][0]['message']['content'])
```

## 需要用户提供的信息

为了准确定位问题，请提供：

1. **数据集文件内容**
   ```bash
   cat /path/to/your_dataset.jsonl
   ```

2. **运行命令**
   ```bash
   # 您实际使用的完整命令
   ```

3. **音频文件信息**
   ```bash
   # 音频文件路径、大小、时长
   ls -lh /path/to/audio_files/
   ```

4. **单条generate的结果**
   - 如果单条generate正常，输出是什么？
   - 使用的参数是什么？

5. **结果文件分析**
   ```bash
   /tmp/diagnose_request_count.py results.jsonl
   ```

## 下一步行动

根据诊断结果，可能的解决方案：

### 如果是数据集重复
- 修改数据集，使用不同的音频文件

### 如果是采样参数问题
- 添加 `repetition_penalty`
- 调整 `temperature`
- 使用 `--disable-ignore-eos`

### 如果是音频内容问题
- 使用更多样化的真实语音数据
- 检查音频质量和内容

### 如果是代码bug
- 需要用户提供具体的复现信息
- 可能需要添加调试日志
- 修复发现的问题
