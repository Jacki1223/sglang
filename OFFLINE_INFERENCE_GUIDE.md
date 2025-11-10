# MiDashengLM 离线推理指南

## 📋 数据集格式

你的数据集应该是这样的格式：

### 文本文件 (.txt)
```
音频URL1,中文描述1
音频URL2,中文描述2
音频URL3,中文描述3
```

### CSV文件 (.csv)
```csv
音频URL,描述
https://example.com/audio1.mp3,这是一段演讲
https://example.com/audio2.wav,这是一段音乐
/local/path/audio3.mp3,这是环境音
```

### 示例数据集
```
https://example.com/speech.mp3,商业会议的录音
https://example.com/music.mp3,古典音乐演奏
/path/to/local/audio.wav,电话对话记录
```

**注意**：
- 每行一个样本
- 音频URL和描述用**英文逗号**分隔
- 支持 HTTP/HTTPS URL 和本地文件路径
- 描述可以包含中文
- 空行和 # 开头的行会被忽略

---

## 🚀 使用方法

脚本提供两种推理模式：

### 模式 1: API 模式（推荐）

**适用场景**：已经启动了 SGLang 服务器

#### 步骤 1: 启动服务器
```bash
python -m sglang.launch_server \
    --model mispeech/midashenglm-7b \
    --trust-remote-code \
    --enable-multimodal \
    --port 30000
```

#### 步骤 2: 运行推理
```bash
python test_offline_inference_from_dataset.py \
    --dataset your_dataset.txt \
    --mode api \
    --prompt "请转录这段音频的内容。" \
    --output results.jsonl
```

### 模式 2: Engine 模式（真正的离线）

**适用场景**：不想启动服务器，直接加载模型推理

```bash
python test_offline_inference_from_dataset.py \
    --dataset your_dataset.txt \
    --mode engine \
    --model mispeech/midashenglm-7b \
    --prompt "请转录这段音频的内容。" \
    --output results.jsonl \
    --tp-size 1
```

---

## 📖 命令行参数详解

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset` | 数据集文件路径 | `--dataset data.txt` |

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--limit` | None | 限制处理的样本数量 |
| `--prompt` | "请转录这段音频的内容。" | 推理提示词 |
| `--output` | inference_results.jsonl | 输出文件路径 |
| `--max-tokens` | 512 | 最大生成token数 |
| `--temperature` | 0.0 | 采样温度 (0.0=贪婪) |
| `--mode` | api | 推理模式: api 或 engine |

### API 模式专用

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url` | http://localhost:30000/v1 | 服务器地址 |
| `--api-key` | sk-123456 | API密钥 |

### Engine 模式专用

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | mispeech/midashenglm-7b | 模型路径 |
| `--tp-size` | 1 | 张量并行大小 |

---

## 💡 使用示例

### 示例 1: 处理前10个样本（API模式）

```bash
python test_offline_inference_from_dataset.py \
    --dataset my_audio_data.txt \
    --limit 10 \
    --mode api \
    --output test_results.jsonl
```

### 示例 2: 使用自定义提示词

```bash
python test_offline_inference_from_dataset.py \
    --dataset my_audio_data.txt \
    --mode api \
    --prompt "请详细描述这段音频的内容，包括说话人、语气和主题。" \
    --max-tokens 1024
```

### 示例 3: Engine模式 + 多GPU

```bash
python test_offline_inference_from_dataset.py \
    --dataset my_audio_data.txt \
    --mode engine \
    --model mispeech/midashenglm-7b \
    --tp-size 2 \
    --output results.jsonl
```

### 示例 4: 使用本地模型路径

```bash
python test_offline_inference_from_dataset.py \
    --dataset my_audio_data.txt \
    --mode engine \
    --model /path/to/local/midashenglm-7b \
    --output results.jsonl
```

---

## 📊 输出格式

结果保存为 **JSONL 格式**（每行一个JSON对象）：

```jsonl
{"index": 1, "audio_url": "https://...", "ground_truth": "原始描述", "transcription": "转录结果", "elapsed_time": 2.34, "success": true}
{"index": 2, "audio_url": "https://...", "ground_truth": "原始描述", "transcription": "转录结果", "elapsed_time": 1.98, "success": true}
{"index": 3, "audio_url": "https://...", "ground_truth": "原始描述", "error": "错误信息", "success": false}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `index` | 样本序号 |
| `audio_url` | 音频URL |
| `ground_truth` | 数据集中的原始描述 |
| `transcription` | 模型生成的转录结果 |
| `elapsed_time` | 推理耗时（秒） |
| `success` | 是否成功 |
| `error` | 错误信息（失败时） |

---

## 📈 读取和分析结果

### Python 读取结果

```python
import json

# 读取结果
results = []
with open('inference_results.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        results.append(json.loads(line))

# 统计
success_count = sum(1 for r in results if r['success'])
total_time = sum(r.get('elapsed_time', 0) for r in results if r['success'])
avg_time = total_time / success_count if success_count > 0 else 0

print(f"总样本: {len(results)}")
print(f"成功: {success_count}")
print(f"失败: {len(results) - success_count}")
print(f"平均耗时: {avg_time:.2f}秒")

# 查看具体结果
for r in results[:5]:
    print(f"\n样本 {r['index']}:")
    print(f"  Ground truth: {r['ground_truth']}")
    if r['success']:
        print(f"  Transcription: {r['transcription']}")
    else:
        print(f"  Error: {r.get('error', 'Unknown')}")
```

### 转换为CSV

```python
import json
import csv

# 读取JSONL
results = []
with open('inference_results.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        results.append(json.loads(line))

# 写入CSV
with open('results.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'URL', 'Ground Truth', 'Transcription', 'Time', 'Success'])
    
    for r in results:
        writer.writerow([
            r['index'],
            r['audio_url'],
            r['ground_truth'],
            r.get('transcription', r.get('error', '')),
            r.get('elapsed_time', 0),
            r['success']
        ])

print("已保存到 results.csv")
```

---

## 🔧 常见问题

### Q1: 数据集格式错误怎么办？

**检查清单**：
- [ ] 每行只有一个逗号分隔URL和描述
- [ ] 使用英文逗号 `,` 而不是中文逗号 `，`
- [ ] 没有多余的空格
- [ ] 文件编码是 UTF-8

**示例对比**：

✅ 正确：
```
https://example.com/audio.mp3,这是一段演讲
```

❌ 错误：
```
https://example.com/audio.mp3，这是一段演讲  # 中文逗号
https://example.com/audio.mp3 , 这是一段演讲  # 多余空格
https://example.com/audio.mp3  # 缺少描述
```

### Q2: 音频下载失败怎么办？

脚本会自动缓存已下载的音频到 `~/.cache/audio_inference/`

如果下载失败：
1. 检查网络连接
2. 检查URL是否有效
3. 对于本地文件，使用绝对路径

### Q3: 内存不足怎么办？

**方法 1**：分批处理
```bash
# 每次处理100个
python test_offline_inference_from_dataset.py \
    --dataset data.txt \
    --limit 100 \
    --output batch1.jsonl

# 可以修改脚本支持offset参数来处理第101-200个
```

**方法 2**：使用API模式（服务器可以更好地管理内存）

**方法 3**：Engine模式使用更多GPU
```bash
--tp-size 2  # 使用2个GPU
```

### Q4: 如何修改提示词？

```bash
# 中文转录
--prompt "请转录这段音频的内容。"

# 英文转录
--prompt "Please transcribe this audio in English."

# 详细描述
--prompt "请详细描述这段音频，包括说话人、背景音、语气和主要内容。"

# 摘要
--prompt "请总结这段音频的主要内容。"
```

### Q5: 推理速度慢怎么办？

**优化建议**：
1. 使用 API 模式（支持 continuous batching）
2. 减少 `--max-tokens`
3. 使用多GPU (`--tp-size 2`)
4. 预先下载所有音频文件

---

## 📝 完整工作流程示例

### 1. 准备数据集

```bash
# 创建数据集文件
cat > my_dataset.txt << EOF
https://example.com/audio1.mp3,商务会议录音
https://example.com/audio2.mp3,客服电话记录
/local/audio3.wav,培训课程录音
EOF
```

### 2. 启动服务器（API模式）

```bash
python -m sglang.launch_server \
    --model mispeech/midashenglm-7b \
    --trust-remote-code \
    --enable-multimodal \
    --port 30000
```

### 3. 运行推理

```bash
python test_offline_inference_from_dataset.py \
    --dataset my_dataset.txt \
    --mode api \
    --output results.jsonl
```

### 4. 分析结果

```python
import json

with open('results.jsonl', 'r', encoding='utf-8') as f:
    results = [json.loads(line) for line in f]

print(f"处理了 {len(results)} 个样本")
for r in results:
    print(f"\n{r['index']}. {r['audio_url']}")
    print(f"   转录: {r.get('transcription', 'FAILED')[:100]}...")
```

---

## 🎯 最佳实践

1. **先用小数据集测试**
   ```bash
   --limit 10  # 先测试10个样本
   ```

2. **使用合适的提示词**
   - 转录任务：简单明确
   - 理解任务：详细说明需要提取的信息

3. **实时保存结果**
   - 脚本会实时写入 JSONL 文件
   - 即使中断也不会丢失已处理的结果

4. **检查失败样本**
   ```bash
   grep '"success": false' results.jsonl
   ```

5. **监控进度**
   - 脚本使用 tqdm 显示进度条
   - 失败的样本会实时打印错误信息

---

**Happy Inferencing! 🎵**
