# 音频基准测试使用指南

## 修复说明

我已经修复了 `bench_serving.py` 中JSONL解析的问题，现在可以正确处理包含空行的数据集文件。

### 修改内容

**文件**: `python/sglang/bench_serving.py`
- **修复行**: 1540
- **原代码**: `dataset_json = [json.loads(line) for line in f]`
- **新代码**: `dataset_json = [json.loads(line) for line in f if line.strip()]`

这个修改会自动跳过空行和只包含空白字符的行。

## 使用方法

### 方法1：直接使用源代码（推荐）

使用提供的包装脚本，它会自动设置正确的Python路径：

```bash
# 基础运行（只显示性能指标）
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /tmp/audio_benchmark_test/audio_dataset.jsonl \
    --num-prompts 3 \
    --port 30000

# 保存推理结果到文件
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /tmp/audio_benchmark_test/audio_dataset.jsonl \
    --num-prompts 3 \
    --port 30000 \
    --output-file results.jsonl \
    --output-details
```

### 方法2：手动设置PYTHONPATH

```bash
# 设置环境变量
export PYTHONPATH="/home/user/sglang/python:$PYTHONPATH"

# 运行bench_serving
python3 /home/user/sglang/python/sglang/bench_serving.py \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/your/audio_dataset.jsonl \
    --num-prompts 10 \
    --port 30000
```

### 方法3：诊断您的数据集文件

如果您仍然遇到错误，可以使用诊断脚本检查您的数据集文件：

```bash
python3 /tmp/diagnose_dataset.py /path/to/your/audio_dataset.jsonl
```

这会告诉您：
- 文件中有多少行
- 哪些行有问题（空行或无效JSON）
- 如何修复问题

## 数据集格式

音频数据集应该是JSONL格式（每行一个JSON对象）：

```jsonl
{"prompt": "描述这段音频", "audio_path": "/path/to/audio1.wav", "output_len": 128}
{"prompt": "音频中说了什么？", "audio_path": "/path/to/audio2.wav", "output_len": 256}
```

### 字段说明

- `prompt` (必需): 文本提示词
- `audio_path` (必需): 音频文件的绝对路径
- `output_len` (可选): 期望的输出token长度，默认256

### 支持的音频格式

- WAV, MP3, FLAC, OGG
- 所有librosa支持的格式

### 注意事项

1. 不要在JSONL文件中添加空行
2. 如果有空行，使用修复后的代码会自动跳过
3. 音频文件路径必须是绝对路径或相对于运行目录的相对路径

## 示例测试

使用提供的测试数据集：

```bash
# 测试数据集位于
/tmp/audio_benchmark_test/audio_dataset.jsonl

# 测试音频文件
/tmp/audio_benchmark_test/test_audio_1.wav
/tmp/audio_benchmark_test/test_audio_2.wav
/tmp/audio_benchmark_test/test_audio_3.wav
```

运行测试：

```bash
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /tmp/audio_benchmark_test/audio_dataset.jsonl \
    --num-prompts 3
```

## 查看推理结果 ⭐

bench_serving默认只显示性能指标，但可以保存完整的推理结果。

### 保存推理结果

添加 `--output-file` 和 `--output-details` 参数：

```bash
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /tmp/audio_benchmark_test/audio_dataset.jsonl \
    --num-prompts 3 \
    --output-file results.jsonl \
    --output-details
```

### 查看结果文件

结果文件包含以下信息：
- `generated_texts`: 模型生成的文本（推理结果） ✨
- `input_lens`: 输入token长度
- `output_lens`: 输出token长度
- `ttfts`: 首token延迟（毫秒）
- `itls`: token间延迟列表
- `errors`: 错误信息（如果有）

#### 方法1：使用便捷查看脚本

```bash
# 查看所有信息（推荐）
/tmp/view_benchmark_results.sh results.jsonl

# 只查看生成的文本
/tmp/view_benchmark_results.sh results.jsonl --texts

# 只查看性能指标
/tmp/view_benchmark_results.sh results.jsonl --metrics

# 只查看错误
/tmp/view_benchmark_results.sh results.jsonl --errors
```

#### 方法2：使用Python查看（推荐，正确显示中文）

```bash
# 查看所有信息
python3 /tmp/view_results.py results.jsonl

# 只查看生成的文本
python3 /tmp/view_results.py results.jsonl --texts

# 只查看性能指标
python3 /tmp/view_results.py results.jsonl --metrics
```

#### 方法3：使用jq查看（必须使用 -r 参数显示中文）

```bash
# ⚠️ 重要：必须使用 -r 参数，否则中文显示为 \uXXXX 转义序列

# 查看所有生成的文本
cat results.jsonl | jq -r '.generated_texts[]'

# 查看第一个生成的文本
cat results.jsonl | jq -r '.generated_texts[0]'

# 查看性能摘要
cat results.jsonl | jq '{
  requests: .total_requests,
  throughput: .request_throughput
}'

# 查看第一个结果的文本（使用 -r）
cat results.jsonl | jq -r '.generated_texts[0]'

# 格式化显示所有文本
cat results.jsonl | jq -r '.generated_texts[] | "========================================\n" + . + "\n"'
```

#### 方法4：自定义Python脚本

```python
import json

with open('results.jsonl', 'r', encoding='utf-8') as f:
    results = json.loads(f.read())

# 查看所有生成的文本
for i, text in enumerate(results['generated_texts'], 1):
    print(f"=== 结果 {i} ===")
    print(text)
    print()

# 查看指标
print(f"吞吐量: {results['request_throughput']:.2f} req/s")
print(f"平均TTFT: {results['mean_ttft_ms']:.2f} ms")
```

### 完整演示

使用自动化演示脚本：

```bash
# 运行完整的测试并自动显示结果
/tmp/audio_benchmark_demo.sh
```

这个脚本会：
1. 运行基准测试
2. 保存结果到带时间戳的文件
3. 自动显示性能指标和生成的文本
4. 提供查看详细结果的命令

## 故障排除

### 错误: JSONDecodeError

如果仍然遇到此错误：

1. 运行诊断脚本检查数据集：
   ```bash
   python3 /tmp/diagnose_dataset.py your_dataset.jsonl
   ```

2. 清理数据集文件中的空行：
   ```bash
   grep -v '^$' your_dataset.jsonl > your_dataset_clean.jsonl
   ```

3. 确保JSON格式正确：
   ```bash
   cat your_dataset.jsonl | while read line; do echo "$line" | python3 -m json.tool > /dev/null || echo "Invalid line: $line"; done
   ```

### 错误: ModuleNotFoundError

如果缺少依赖包，需要安装：
```bash
pip install librosa numpy pybase64
```

### 错误: Audio file not found

确保音频文件路径正确，使用绝对路径或检查相对路径。

## 性能指标

运行后会输出以下指标：

- **Request throughput**: 请求吞吐量 (req/s)
- **Token throughput**: Token吞吐量
  - Input tokens/s (包括文本和音频tokens)
  - Output tokens/s
- **Latency metrics**:
  - TTFT: Time to first token (首token延迟)
  - ITL: Inter-token latency (token间延迟)
  - E2E: End-to-end latency (端到端延迟)
- **Token counts**:
  - Text tokens
  - Audio tokens (存储在vision_prompt_len字段)

## 更多信息

详细文档请查看：
- `/tmp/audio_benchmark_test/README.md` - 完整的技术文档
- `/home/user/sglang/python/sglang/bench_serving.py` - 源代码

## Git提交信息

所有修改已提交到分支：
- Branch: `claude/analyze-benchserving-code-011CV1RycUict3jNwpX9v4zD`
- Commits:
  1. "Add audio dataset support to benchserving for MiDashengLM model"
  2. "Fix JSONL parsing to skip empty lines in sample_audio_requests"
