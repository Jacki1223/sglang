# MiDashengLM 批量音频测试指南

## 📋 目录
1. [启动服务器](#启动服务器)
2. [基础测试](#基础测试)
3. [批量测试](#批量测试)
4. [自定义测试](#自定义测试)
5. [故障排查](#故障排查)

---

## 🚀 启动服务器

### 方法 1: 基本启动

```bash
python -m sglang.launch_server \
    --model mispeech/midashenglm-7b \
    --trust-remote-code \
    --enable-multimodal \
    --port 30000
```

### 方法 2: 使用更多GPU

```bash
python -m sglang.launch_server \
    --model mispeech/midashenglm-7b \
    --trust-remote-code \
    --enable-multimodal \
    --tp 2 \
    --port 30000
```

### 方法 3: 调试模式

```bash
SGLANG_LOGGING=DEBUG python -m sglang.launch_server \
    --model mispeech/midashenglm-7b \
    --trust-remote-code \
    --enable-multimodal \
    --port 30000
```

---

## 🎯 基础测试

### 1. 最简单的单个音频测试

```python
import openai

client = openai.Client(
    api_key="sk-123456",
    base_url="http://localhost:30000/v1"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": "/path/to/audio.mp3"},
            },
            {
                "type": "text",
                "text": "Please transcribe this audio.",
            },
        ],
    }
]

response = client.chat.completions.create(
    model="default",
    messages=messages,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

### 2. 使用 URL 的音频

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "https://example.com/audio.mp3"
                },
            },
            {
                "type": "text",
                "text": "Describe this audio.",
            },
        ],
    }
]
```

---

## 📦 批量测试

### 使用提供的批量测试脚本

1. **保存脚本**
   ```bash
   # 将前面生成的 test_midashenglm_batch_audio.py 保存到本地
   ```

2. **运行测试**
   ```bash
   python test_midashenglm_batch_audio.py
   ```

3. **查看结果**
   脚本会自动：
   - 下载测试音频文件
   - 测试多个提示词
   - 显示每个测试的响应时间
   - 打印结果摘要

### 批量测试示例输出

```
================================================================================
🎤 MiDashengLM Batch Audio Testing
================================================================================
Server: http://localhost:30000/v1
Audio files: 2
Prompts: 3
Total tests: 6
================================================================================

📥 Downloading: Trump_WEF_2018_10s.mp3
✅ Downloaded: /home/user/.cache/audio_test/Trump_WEF_2018_10s.mp3

================================================================================
🎵 Testing: trump_speech with prompt: transcribe
================================================================================
✅ Success (2.34s)
Response: Thank you very much, it's a privilege to be here at this forum...

...

📈 Sequential Testing Complete
Total time: 15.67s
Tests: 6
Success: 6
Failed: 0
```

---

## 🛠️ 自定义测试

### 添加自己的音频文件

```python
# 在 test_midashenglm_batch_audio.py 中修改

TEST_AUDIO_URLS = {
    "my_audio_1": "file:///path/to/my_audio1.mp3",
    "my_audio_2": "https://example.com/audio2.mp3",
    "my_audio_3": "/absolute/path/to/audio3.wav",
}
```

### 添加自定义提示词

```python
TEST_PROMPTS = {
    "transcribe_chinese": "请将这段音频转录为中文。",
    "identify_speaker": "Who is speaking in this audio?",
    "count_speakers": "How many different speakers are in this audio?",
    "extract_keywords": "Extract the key words from this audio.",
}
```

### 调整测试参数

```python
response = client.chat.completions.create(
    model="default",
    messages=messages,
    temperature=0.7,        # 提高创造性
    max_tokens=512,         # 允许更长的响应
    top_p=0.9,              # 核采样
    frequency_penalty=0.5,  # 减少重复
)
```

---

## 🔧 故障排查

### 问题 1: 连接失败

**症状**: `Connection refused` 或 `Connection error`

**解决方案**:
1. 确认服务器正在运行
   ```bash
   curl http://localhost:30000/v1/models
   ```

2. 检查端口是否正确
   ```bash
   netstat -an | grep 30000
   ```

### 问题 2: 音频文件不支持

**症状**: `Unsupported audio format` 或 `Failed to process audio`

**解决方案**:
- 支持的格式: MP3, WAV, FLAC, OGG
- 转换格式:
  ```bash
  ffmpeg -i input.m4a -ar 16000 output.wav
  ```

### 问题 3: 内存不足

**症状**: `CUDA out of memory` 或 `RuntimeError: out of memory`

**解决方案**:
1. 减少批次大小
2. 使用张量并行:
   ```bash
   --tp 2
   ```
3. 使用量化:
   ```bash
   --quantization fp8
   ```

### 问题 4: 响应速度慢

**优化建议**:
1. 使用 continuous batching (SGLang 默认开启)
2. 调整 max_tokens 限制:
   ```python
   max_tokens=128  # 减少生成长度
   ```
3. 启用流式输出:
   ```python
   stream=True
   ```

---

## 📊 性能基准

### 预期性能（单张 A100）

| 音频长度 | 处理时间 | 首token时间 | 吞吐量 |
|---------|---------|------------|--------|
| 5s      | ~1.5s   | ~0.5s      | 50 tok/s |
| 10s     | ~2.5s   | ~0.8s      | 45 tok/s |
| 30s     | ~5.0s   | ~1.5s      | 40 tok/s |

### 批量处理性能

- **顺序处理**: 2-3 个请求/秒
- **并行处理**: 5-8 个请求/秒（取决于 GPU 和 max_batch_size）

---

## 💡 最佳实践

### 1. 音频预处理

```python
import librosa

# 重采样到 16kHz
audio, sr = librosa.load("input.mp3", sr=16000)
librosa.output.write_wav("output.wav", audio, sr)
```

### 2. 提示词优化

**好的提示词**:
- ✅ "Please listen to this audio and transcribe the speech in English."
- ✅ "Describe the main content of this audio in 2-3 sentences."

**不好的提示词**:
- ❌ "What?" (太简短)
- ❌ "Tell me everything about this audio in extreme detail..." (太长)

### 3. 错误处理

```python
import time

def test_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="default",
                messages=messages,
                timeout=60,
            )
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

---

## 📚 更多资源

- [SGLang 文档](https://docs.sglang.ai)
- [MiDashengLM 论文](https://arxiv.org/abs/...)
- [问题反馈](https://github.com/sgl-project/sglang/issues)

---

## ✅ 快速检查清单

在运行批量测试前，确保：

- [ ] 服务器正在运行 (`curl http://localhost:30000/health`)
- [ ] 模型已加载完成
- [ ] 音频文件格式正确 (MP3/WAV/FLAC)
- [ ] 音频文件可访问（路径正确）
- [ ] Python 环境安装了 `openai` 包
- [ ] 有足够的 GPU 内存

---

**Happy Testing! 🎵**
