# 如何使用追踪日志定位问题

## 我添加了什么

我在 bench_serving.py 的每个关键步骤都添加了 `[TRACE]` 日志，精确追踪请求数量的变化。

## 运行追踪

```bash
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/your_dataset.jsonl \
    --num-prompts 3 \
    --port 30000 \
    --output-file trace_results.jsonl \
    --output-details 2>&1 | tee trace.log
```

## 提取追踪信息

```bash
# 提取所有追踪日志
grep "\[TRACE\]" trace.log

# 或者提取关键的数量信息
grep -E "\[TRACE\]|DEBUG_AUDIO" trace.log
```

## 期望的正常输出

如果 `--num-prompts 3`，您应该看到：

```
[TRACE] sample_audio_requests returning 3 samples (requested: 3)
[DEBUG_AUDIO_DATASET] Loaded audio samples:
  Sample 1: prompt='...' audio_data_len=1 audio_hash=abc12345
  Sample 2: prompt='...' audio_data_len=1 audio_hash=def67890
  Sample 3: prompt='...' audio_data_len=1 audio_hash=ghi24680

[TRACE] benchmark() received 3 input_requests

[DEBUG_AUDIO_REQUEST] {"prompt_preview": "...", "audio_data_count": 1, ...}
[DEBUG_AUDIO_REQUEST] {"prompt_preview": "...", "audio_data_count": 1, ...}
[DEBUG_AUDIO_REQUEST] {"prompt_preview": "...", "audio_data_count": 1, ...}

[TRACE] Created 3 tasks, now gathering results...
[TRACE] Gathered 3 outputs
[TRACE] result_details created with 3 generated_texts
```

## 如何识别问题

### 问题1: 数据集采样数量错误

```
[TRACE] sample_audio_requests returning 4 samples (requested: 3)  ← 错误！
```

**说明**: sample_audio_requests 函数返回了错误的数量
**原因**: random.sample 或数据集处理有bug

### 问题2: benchmark接收数量错误

```
[TRACE] sample_audio_requests returning 3 samples (requested: 3)
[TRACE] benchmark() received 4 input_requests  ← 错误！
```

**说明**: 从采样到benchmark之间，请求数量被修改了
**原因**: 调用 benchmark 时传入了错误的数量

### 问题3: tasks创建数量错误

```
[TRACE] benchmark() received 3 input_requests
[TRACE] Created 4 tasks, now gathering results...  ← 错误！
```

**说明**: get_request 生成器或 tasks 创建循环有问题
**原因**: 生成器重复生成了某个请求，或者循环有bug

### 问题4: outputs收集数量错误

```
[TRACE] Created 3 tasks, now gathering results...
[TRACE] Gathered 4 outputs  ← 错误！
```

**说明**: asyncio.gather 返回了错误的数量
**原因**: 这种情况几乎不可能，可能是内存错误

### 问题5: result_details数量错误

```
[TRACE] Gathered 3 outputs
[TRACE] result_details created with 4 generated_texts  ← 错误！
```

**说明**: 列表推导式或 outputs 被修改了
**原因**: result_details 构建逻辑有bug

### 问题6: audio_hash重复

```
[DEBUG_AUDIO_DATASET] Loaded audio samples:
  Sample 1: ... audio_hash=abc12345
  Sample 2: ... audio_hash=abc12345  ← 与Sample1相同！
  Sample 3: ... audio_hash=abc12345  ← 与Sample1相同！
```

**说明**: 所有样本使用了相同的音频文件
**原因**: 数据集中的 audio_path 重复

### 问题7: audio_data_count > 1

```
[DEBUG_AUDIO_REQUEST] {"audio_data_count": 2, ...}  ← 应该是1！
```

**说明**: 单个请求包含多个音频
**原因**: audio_data 数组被意外修改或重复添加

## 下一步

**请运行追踪版本的bench_serving，然后提供**：

```bash
# 提取追踪信息
grep -E "\[TRACE\]|DEBUG_AUDIO" trace.log

# 或者直接将完整的trace.log内容提供给我
```

有了这些信息，我就能**精确定位**是在哪一步出现了问题！

## 快速检查命令

```bash
# 运行并立即查看追踪
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/your_dataset.jsonl \
    --num-prompts 3 \
    --port 30000 2>&1 | grep -E "\[TRACE\]|DEBUG_AUDIO"
```

这会只显示追踪和调试信息，忽略其他输出。
