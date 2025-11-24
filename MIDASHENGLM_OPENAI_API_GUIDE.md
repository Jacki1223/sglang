# MiDashengLM OpenAI Chat API 调用指南

> **版本**: 1.0
> **最后更新**: 2025-11-10
> **适用模型**: MiDashengLM-7B
> **SGLang版本**: Latest (with MiDashengLM support)

---

## 目录

1. [快速开始](#1-快速开始)
2. [OpenAI API格式详解](#2-openai-api格式详解)
3. [音频数据传递方式](#3-音频数据传递方式)
4. [完整代码示例](#4-完整代码示例)
5. [流式响应](#5-流式响应)
6. [错误处理](#6-错误处理)
7. [性能优化](#7-性能优化)
8. [与标准OpenAI API的差异](#8-与标准openai-api的差异)

---

## 1. 快速开始

### 1.1 启动SGLang服务

```bash
# 启动服务，启用OpenAI兼容API
python -m sglang.launch_server \
    --model mispeech/midashenglm-7b-0804-fp32 \
    --port 30000 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --api-key your-secret-key
```

**端点**:
- Base URL: `http://localhost:30000`
- Chat Completions: `http://localhost:30000/v1/chat/completions`
- Models: `http://localhost:30000/v1/models`

### 1.2 最简单的调用示例

```python
from openai import OpenAI

# 创建客户端
client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:30000/v1"
)

# 发送请求（文本 + 音频）
response = client.chat.completions.create(
    model="mispeech/midashenglm-7b-0804-fp32",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "file:///path/to/audio.wav"
                    }
                },
                {
                    "type": "text",
                    "text": "请描述这段音频的内容"
                }
            ]
        }
    ],
    max_tokens=256,
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## 2. OpenAI API格式详解

### 2.1 请求结构

**完整的ChatCompletionRequest结构**:

```python
{
    "model": "mispeech/midashenglm-7b-0804-fp32",
    "messages": [
        {
            "role": "user" | "assistant" | "system",
            "content": str | List[ContentPart]
        }
    ],

    # 生成参数
    "max_tokens": int,              # 最大生成token数
    "temperature": float,           # 温度 (0-2)
    "top_p": float,                 # nucleus sampling
    "top_k": int,                   # top-k sampling
    "frequency_penalty": float,     # 频率惩罚
    "presence_penalty": float,      # 存在惩罚
    "stop": List[str] | str,        # 停止序列

    # 流式
    "stream": bool,                 # 是否流式返回

    # 其他
    "n": int,                       # 生成数量
    "logprobs": bool,               # 返回logprobs
    "echo": bool,                   # 回显输入
}
```

### 2.2 Messages格式

#### 纯文本消息

```python
messages = [
    {
        "role": "system",
        "content": "你是一个有用的助手。"
    },
    {
        "role": "user",
        "content": "你好！"
    }
]
```

#### 多模态消息（文本 + 音频）

```python
messages = [
    {
        "role": "user",
        "content": [
            # 音频部分
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "file:///path/to/audio.wav"
                }
            },
            # 文本部分
            {
                "type": "text",
                "text": "请转录这段音频"
            }
        ]
    }
]
```

#### 多轮对话

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": "file:///audio1.wav"}
            },
            {
                "type": "text",
                "text": "这段音频说了什么？"
            }
        ]
    },
    {
        "role": "assistant",
        "content": "这段音频讨论了人工智能的发展..."
    },
    {
        "role": "user",
        "content": "请详细解释一下"
    }
]
```

### 2.3 ContentPart类型

SGLang支持以下content part类型：

```python
# 1. 文本
{
    "type": "text",
    "text": "你的文本内容"
}

# 2. 音频URL
{
    "type": "audio_url",
    "audio_url": {
        "url": "file:///path/to/audio.wav"  # 本地文件
        # 或
        "url": "http://example.com/audio.wav"  # HTTP URL
        # 或
        "url": "data:audio/wav;base64,UklGRiQAAABXQVZF..."  # Base64
    }
}

# 3. 图像URL（如果模型支持）
{
    "type": "image_url",
    "image_url": {
        "url": "file:///path/to/image.jpg"
    }
}

# 4. 视频URL（如果模型支持）
{
    "type": "video_url",
    "video_url": {
        "url": "file:///path/to/video.mp4"
    }
}
```

---

## 3. 音频数据传递方式

### 3.1 方式1: 本地文件路径

**最简单，推荐用于本地部署**

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "file:///absolute/path/to/audio.wav"
                }
            },
            {
                "type": "text",
                "text": "请分析这段音频"
            }
        ]
    }
]
```

**要求**:
- ✅ 使用绝对路径
- ✅ 文件必须在服务器可访问的位置
- ✅ 支持格式: WAV, MP3, FLAC, OGG等

**示例**:
```python
import os

# 获取绝对路径
audio_path = os.path.abspath("./my_audio.wav")

content = [
    {
        "type": "audio_url",
        "audio_url": {"url": f"file://{audio_path}"}
    },
    {
        "type": "text",
        "text": "转录这段音频"
    }
]
```

### 3.2 方式2: HTTP URL

**适合远程文件**

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "https://example.com/audio/sample.wav"
                }
            },
            {
                "type": "text",
                "text": "这段音频是什么内容？"
            }
        ]
    }
]
```

**要求**:
- ✅ URL必须可公开访问
- ✅ 服务器需要能访问该URL
- ⚠️ 注意网络延迟

### 3.3 方式3: Base64编码

**适合小文件或需要嵌入的场景**

```python
import base64

# 读取音频文件并编码
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

# 构建data URL
data_url = f"data:audio/wav;base64,{audio_base64}"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": data_url}
            },
            {
                "type": "text",
                "text": "分析这段音频"
            }
        ]
    }
]
```

**优缺点**:
- ✅ 不需要额外的文件服务器
- ✅ 可以在请求中直接传输
- ❌ Base64编码会增加33%的大小
- ❌ 不适合大文件

---

## 4. 完整代码示例

### 4.1 使用OpenAI Python SDK

#### 安装

```bash
pip install openai
```

#### 基础示例

```python
from openai import OpenAI
import os

# 创建客户端
client = OpenAI(
    api_key="your-secret-key",  # 如果启动时设置了--api-key
    base_url="http://localhost:30000/v1"
)

# 音频文件路径
audio_path = os.path.abspath("./sample.wav")

# 发送请求
response = client.chat.completions.create(
    model="mispeech/midashenglm-7b-0804-fp32",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{audio_path}"}
                },
                {
                    "type": "text",
                    "text": "请转录并总结这段音频的主要内容"
                }
            ]
        }
    ],
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# 输出结果
print("Response:", response.choices[0].message.content)
print("\nUsage:", response.usage)
```

#### 多轮对话示例

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:30000/v1"
)

# 对话历史
conversation = []

# 第一轮：音频 + 问题
audio_path = os.path.abspath("./lecture.wav")
conversation.append({
    "role": "user",
    "content": [
        {
            "type": "audio_url",
            "audio_url": {"url": f"file://{audio_path}"}
        },
        {
            "type": "text",
            "text": "这个讲座主要讨论了什么主题？"
        }
    ]
})

response1 = client.chat.completions.create(
    model="mispeech/midashenglm-7b-0804-fp32",
    messages=conversation,
    max_tokens=256,
)

# 保存助手回复
conversation.append({
    "role": "assistant",
    "content": response1.choices[0].message.content
})

print("第一轮回复:", response1.choices[0].message.content)

# 第二轮：追问（不需要再传音频）
conversation.append({
    "role": "user",
    "content": "能详细解释一下其中的关键观点吗？"
})

response2 = client.chat.completions.create(
    model="mispeech/midashenglm-7b-0804-fp32",
    messages=conversation,
    max_tokens=512,
)

print("\n第二轮回复:", response2.choices[0].message.content)
```

### 4.2 使用requests库

#### 基础请求

```python
import requests
import json
import os

# 配置
API_URL = "http://localhost:30000/v1/chat/completions"
API_KEY = "your-secret-key"

# 音频路径
audio_path = os.path.abspath("./audio.wav")

# 构建请求
payload = {
    "model": "mispeech/midashenglm-7b-0804-fp32",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"file://{audio_path}"
                    }
                },
                {
                    "type": "text",
                    "text": "请转录这段音频"
                }
            ]
        }
    ],
    "max_tokens": 256,
    "temperature": 0.7
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 发送请求
response = requests.post(
    API_URL,
    headers=headers,
    json=payload
)

# 解析响应
if response.status_code == 200:
    result = response.json()
    print("回复:", result["choices"][0]["message"]["content"])
    print("\nToken使用:", result["usage"])
else:
    print(f"错误: {response.status_code}")
    print(response.text)
```

#### Base64编码示例

```python
import requests
import base64
import json

# 读取并编码音频
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# 构建请求
payload = {
    "model": "mispeech/midashenglm-7b-0804-fp32",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "分析这段音频"
                }
            ]
        }
    ],
    "max_tokens": 512
}

response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-secret-key"
    },
    json=payload
)

print(response.json())
```

### 4.3 批量处理示例

```python
from openai import OpenAI
import os
from pathlib import Path

client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:30000/v1"
)

# 批量处理多个音频文件
audio_dir = Path("./audio_samples")
results = []

for audio_file in audio_dir.glob("*.wav"):
    print(f"处理: {audio_file.name}")

    response = client.chat.completions.create(
        model="mispeech/midashenglm-7b-0804-fp32",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"file://{audio_file.absolute()}"}
                    },
                    {
                        "type": "text",
                        "text": "请转录这段音频并总结要点"
                    }
                ]
            }
        ],
        max_tokens=512,
    )

    results.append({
        "file": audio_file.name,
        "transcription": response.choices[0].message.content,
        "tokens": response.usage.total_tokens
    })

# 保存结果
import json
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n处理完成！共处理 {len(results)} 个文件")
```

---

## 5. 流式响应

### 5.1 使用OpenAI SDK的流式API

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:30000/v1"
)

audio_path = os.path.abspath("./audio.wav")

# 启用流式响应
stream = client.chat.completions.create(
    model="mispeech/midashenglm-7b-0804-fp32",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{audio_path}"}
                },
                {
                    "type": "text",
                    "text": "请详细描述这段音频"
                }
            ]
        }
    ],
    max_tokens=512,
    stream=True,  # 启用流式
)

# 逐个处理chunk
print("回复: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # 换行
```

### 5.2 使用requests的流式处理

```python
import requests
import json
import os

audio_path = os.path.abspath("./audio.wav")

payload = {
    "model": "mispeech/midashenglm-7b-0804-fp32",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{audio_path}"}
                },
                {
                    "type": "text",
                    "text": "转录这段音频"
                }
            ]
        }
    ],
    "max_tokens": 256,
    "stream": True  # 启用流式
}

response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-secret-key"
    },
    json=payload,
    stream=True  # 流式接收
)

print("回复: ", end="", flush=True)
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith("data: "):
            data_str = line[6:]  # 移除"data: "前缀
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if data["choices"][0]["delta"].get("content"):
                    print(data["choices"][0]["delta"]["content"], end="", flush=True)
            except json.JSONDecodeError:
                continue
print()  # 换行
```

---

## 6. 错误处理

### 6.1 常见错误

#### 错误1: 文件未找到

```python
# 错误响应
{
    "error": {
        "message": "Audio file not found: /path/to/audio.wav",
        "type": "invalid_request_error",
        "code": "file_not_found"
    }
}
```

**解决方案**:
```python
import os

# 检查文件是否存在
audio_path = "/path/to/audio.wav"
if not os.path.exists(audio_path):
    raise FileNotFoundError(f"音频文件不存在: {audio_path}")

# 使用绝对路径
audio_path = os.path.abspath(audio_path)
```

#### 错误2: 不支持的音频格式

```python
# 错误响应
{
    "error": {
        "message": "Unsupported audio format",
        "type": "invalid_request_error"
    }
}
```

**解决方案**:
```python
import torchaudio

# 转换音频格式
def convert_to_wav(input_path, output_path):
    """转换音频为WAV格式"""
    waveform, sample_rate = torchaudio.load(input_path)
    torchaudio.save(output_path, waveform, sample_rate)

# 使用
convert_to_wav("audio.mp3", "audio.wav")
```

#### 错误3: 音频文件过大

```python
# 错误响应
{
    "error": {
        "message": "Audio file too large. Maximum size: 25MB",
        "type": "invalid_request_error"
    }
}
```

**解决方案**:
```python
import torchaudio

def trim_audio(input_path, output_path, max_duration=30):
    """裁剪音频到指定时长（秒）"""
    waveform, sample_rate = torchaudio.load(input_path)

    # 计算最大样本数
    max_samples = sample_rate * max_duration

    # 裁剪
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    torchaudio.save(output_path, waveform, sample_rate)

# 使用
trim_audio("long_audio.wav", "trimmed_audio.wav", max_duration=30)
```

### 6.2 完整的错误处理示例

```python
from openai import OpenAI
import os
import time

client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:30000/v1"
)

def transcribe_audio_with_retry(
    audio_path: str,
    prompt: str,
    max_retries: int = 3
):
    """带重试机制的音频转录"""

    # 检查文件
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 转换为绝对路径
    audio_path = os.path.abspath(audio_path)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="mispeech/midashenglm-7b-0804-fp32",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": f"file://{audio_path}"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=512,
                timeout=60,  # 60秒超时
            )

            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)

            # 特定错误处理
            if "file not found" in error_msg.lower():
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")

            if "unsupported format" in error_msg.lower():
                raise ValueError(f"不支持的音频格式: {audio_path}")

            # 重试逻辑
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"请求失败，{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise Exception(f"请求失败，已重试{max_retries}次: {error_msg}")

# 使用
try:
    result = transcribe_audio_with_retry(
        "audio.wav",
        "请转录这段音频"
    )
    print("转录结果:", result)
except Exception as e:
    print(f"错误: {e}")
```

---

## 7. 性能优化

### 7.1 批处理优化

```python
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:30000/v1"
)

def process_single_audio(audio_file: str, prompt: str):
    """处理单个音频文件"""
    try:
        response = client.chat.completions.create(
            model="mispeech/midashenglm-7b-0804-fp32",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"file://{os.path.abspath(audio_file)}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=512,
        )
        return {
            "file": audio_file,
            "result": response.choices[0].message.content,
            "success": True
        }
    except Exception as e:
        return {
            "file": audio_file,
            "error": str(e),
            "success": False
        }

def batch_process_audios(audio_files: list, prompt: str, max_workers: int = 4):
    """并行批量处理音频文件"""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_audio, audio_file, prompt): audio_file
            for audio_file in audio_files
        }

        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result["success"]:
                print(f"✓ {result['file']}")
            else:
                print(f"✗ {result['file']}: {result['error']}")

    return results

# 使用
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = batch_process_audios(
    audio_files,
    prompt="请转录这段音频",
    max_workers=4  # 并行数
)
```

### 7.2 音频预处理

```python
import torchaudio
import torch

def preprocess_audio(
    audio_path: str,
    target_sample_rate: int = 16000,
    max_duration: float = 30.0
):
    """预处理音频：重采样、裁剪、归一化"""

    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_path)

    # 重采样到16kHz
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    # 转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 裁剪到最大时长
    max_samples = int(target_sample_rate * max_duration)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # 归一化
    waveform = waveform / torch.max(torch.abs(waveform))

    # 保存预处理后的音频
    output_path = audio_path.replace(".wav", "_processed.wav")
    torchaudio.save(output_path, waveform, target_sample_rate)

    return output_path

# 使用
processed_audio = preprocess_audio("raw_audio.wav")
# 然后使用processed_audio调用API
```

### 7.3 缓存和复用

```python
from functools import lru_cache
import hashlib
import os

@lru_cache(maxsize=100)
def get_audio_transcription(audio_hash: str, prompt: str):
    """缓存音频转录结果"""
    # 实际的API调用
    # 注意：这里使用hash作为key，实际音频路径需要从其他地方获取
    pass

def transcribe_with_cache(audio_path: str, prompt: str):
    """带缓存的音频转录"""

    # 计算音频文件的hash
    with open(audio_path, 'rb') as f:
        audio_hash = hashlib.md5(f.read()).hexdigest()

    # 检查缓存
    cache_key = f"{audio_hash}:{prompt}"

    # 这里可以使用Redis、文件系统等持久化缓存
    # 简化示例，实际应用中应该实现真正的缓存逻辑

    # 调用API
    response = client.chat.completions.create(
        model="mispeech/midashenglm-7b-0804-fp32",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"file://{os.path.abspath(audio_path)}"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=512,
    )

    return response.choices[0].message.content
```

---

## 8. 与标准OpenAI API的差异

### 8.1 主要差异

| 特性 | OpenAI官方 | SGLang MiDashengLM |
|------|------------|-------------------|
| **音频输入** | ✅ 支持（GPT-4 Audio） | ✅ 支持 |
| **音频格式** | `audio` field in message | `audio_url` content part |
| **Base64编码** | ✅ 支持 | ✅ 支持 |
| **本地文件** | ❌ 不支持 | ✅ 支持 `file://` |
| **流式输出** | ✅ 完全支持 | ✅ 完全支持 |
| **Function calling** | ✅ 支持 | ✅ 支持 |
| **Vision** | ✅ 支持 | ⚠️ 取决于模型 |

### 8.2 音频格式对比

#### OpenAI官方格式（GPT-4 Audio）

```python
# OpenAI官方的音频输入格式
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "<base64_encoded_audio>",
                    "format": "wav"
                }
            }
        ]
    }
]
```

#### SGLang MiDashengLM格式

```python
# SGLang使用audio_url格式
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "file:///path/to/audio.wav"
                    # 或 "data:audio/wav;base64,..."
                }
            }
        ]
    }
]
```

### 8.3 兼容性说明

**与OpenAI SDK的兼容性**:
- ✅ 可以使用官方的`openai` Python包
- ✅ 只需修改`base_url`参数
- ✅ 大部分参数完全兼容
- ⚠️ 音频格式使用`audio_url`而不是`input_audio`

**迁移指南**:
```python
# 从OpenAI官方API迁移到SGLang

# 1. 修改client初始化
# 原来：
# client = OpenAI(api_key="sk-...")

# 现在：
client = OpenAI(
    api_key="your-sglang-key",
    base_url="http://localhost:30000/v1"  # 指向SGLang服务
)

# 2. 修改音频输入格式
# 原来：
# {"type": "input_audio", "input_audio": {...}}

# 现在：
# {"type": "audio_url", "audio_url": {"url": "file://..."}}

# 3. 其他参数保持不变
response = client.chat.completions.create(
    model="mispeech/midashenglm-7b-0804-fp32",  # 修改模型名
    messages=messages,  # 修改音频格式
    max_tokens=512,     # 其他参数不变
    temperature=0.7,
    # ... 其他参数
)
```

---

## 9. 完整应用示例

### 9.1 音频转录服务

```python
from fastapi import FastAPI, File, UploadFile, Form
from openai import OpenAI
import tempfile
import os

app = FastAPI()

client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:30000/v1"
)

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    prompt: str = Form(default="请转录这段音频")
):
    """音频转录API"""

    # 保存上传的文件到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 调用SGLang服务
        response = client.chat.completions.create(
            model="mispeech/midashenglm-7b-0804-fp32",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"file://{tmp_path}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=512,
        )

        return {
            "transcription": response.choices[0].message.content,
            "usage": {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# 运行服务
# uvicorn app:app --host 0.0.0.0 --port 8000
```

### 9.2 命令行工具

```python
#!/usr/bin/env python3
"""音频转录命令行工具"""

import argparse
from openai import OpenAI
import os

def main():
    parser = argparse.ArgumentParser(description="MiDashengLM音频转录工具")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--prompt", default="请转录这段音频", help="提示词")
    parser.add_argument("--url", default="http://localhost:30000/v1", help="SGLang服务URL")
    parser.add_argument("--api-key", default="your-secret-key", help="API密钥")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--stream", action="store_true", help="流式输出")

    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.audio):
        print(f"错误: 文件不存在: {args.audio}")
        return 1

    # 创建客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.url
    )

    # 构建请求
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{os.path.abspath(args.audio)}"}
                },
                {
                    "type": "text",
                    "text": args.prompt
                }
            ]
        }
    ]

    if args.stream:
        # 流式输出
        print("转录结果: ", end="", flush=True)
        stream = client.chat.completions.create(
            model="mispeech/midashenglm-7b-0804-fp32",
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    else:
        # 普通输出
        response = client.chat.completions.create(
            model="mispeech/midashenglm-7b-0804-fp32",
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("转录结果:")
        print(response.choices[0].message.content)
        print(f"\nToken使用: {response.usage.total_tokens}")

    return 0

if __name__ == "__main__":
    exit(main())
```

**使用示例**:
```bash
# 基础使用
python transcribe.py audio.wav

# 自定义提示词
python transcribe.py audio.wav --prompt "请转录并总结这段音频"

# 流式输出
python transcribe.py audio.wav --stream

# 指定服务器
python transcribe.py audio.wav --url http://192.168.1.100:30000/v1
```

---

## 10. 总结

### 关键要点

1. **启动服务**: 使用`--api-key`启用认证
2. **音频格式**: 使用`audio_url` content part
3. **文件路径**: 本地文件使用`file://`绝对路径
4. **兼容性**: 使用OpenAI Python SDK，修改`base_url`
5. **流式响应**: 设置`stream=True`
6. **错误处理**: 实现重试和异常捕获

### 最佳实践

- ✅ 使用绝对路径访问本地文件
- ✅ 预处理音频（重采样到16kHz、单声道）
- ✅ 实现错误处理和重试机制
- ✅ 对于大批量处理，使用并行处理
- ✅ 缓存常用音频的转录结果
- ✅ 使用流式响应提升用户体验

### 下一步

- 查看 `MIDASHENGLM_TECHNICAL_DOCUMENTATION.md` 了解模型架构
- 查看 `WEIGHT_LOADING_VERIFICATION.md` 确认模型正确加载
- 实现自己的应用集成

---

**文档版本**: 1.0
**最后更新**: 2025-11-10
**相关文档**: MIDASHENGLM_TECHNICAL_DOCUMENTATION.md
**问题反馈**: https://github.com/sgl-project/sglang
