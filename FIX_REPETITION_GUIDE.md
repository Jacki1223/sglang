# 修复模型输出重复问题指南

## 问题现象

模型输出重复内容：
```
我跟你说，我跟你说，我跟你说，我跟你说...（重复到max_tokens）
```

## 根本原因

1. **测试音频是纯音调**（440Hz正弦波），不是真实语音
2. **采样参数导致循环**：
   - `temperature: 0.0` - 贪婪采样，每次选择最高概率token
   - `ignore_eos: True` - 忽略结束标记，强制生成到max_tokens
   - 没有设置 `repetition_penalty`

## 解决方案

### 方案1: 使用真实语音音频（推荐）

**当前测试音频**（纯音调）:
```python
# /tmp/audio_benchmark_test/generate_test_audio.py
# 生成440Hz正弦波 - 没有语音内容
create_test_audio(filename, frequency=440)
```

**需要替换为真实语音**，例如：
- 录制的人声
- TTS生成的语音
- 真实音频数据集

### 方案2: 调整bench_serving采样参数

运行时添加参数来减少重复：

```bash
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/dataset.jsonl \
    --num-prompts 3 \
    --extra-request-body '{"temperature": 0.7, "repetition_penalty": 1.05}'
```

**参数说明**:
- `temperature: 0.7` - 增加随机性，避免固定模式
- `repetition_penalty: 1.05` - 惩罚重复token

### 方案3: 不忽略EOS标记

```bash
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/dataset.jsonl \
    --num-prompts 3 \
    --disable-ignore-eos  # 允许模型在EOS处停止
```

## 对比单条generate

如果您的单条generate命令没有重复问题，请检查它使用的参数：

```python
# 单条generate示例
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "your-model",
        "messages": [...],
        "temperature": 0.7,        # 可能不是0.0
        "repetition_penalty": 1.0, # 可能设置了
        # "ignore_eos": False,     # 可能没有忽略EOS
    }
)
```

## 快速测试

### 测试1: 使用真实音频

如果您有真实的语音文件：

```bash
# 创建新的数据集
cat > /tmp/real_audio_dataset.jsonl << 'EOF'
{"prompt": "请转写这段音频", "audio_path": "/path/to/real_speech.wav", "output_len": 128}
EOF

# 运行测试
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /tmp/real_audio_dataset.jsonl \
    --num-prompts 1 \
    --output-file test_real_audio.jsonl \
    --output-details
```

### 测试2: 调整参数

使用当前纯音调测试，但调整参数：

```bash
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /tmp/audio_benchmark_test/audio_dataset.jsonl \
    --num-prompts 1 \
    --extra-request-body '{"temperature": 0.8, "repetition_penalty": 1.1, "top_p": 0.9}' \
    --output-file test_adjusted.jsonl \
    --output-details

# 查看结果
python3 /tmp/view_results.py test_adjusted.jsonl --texts
```

## 诊断步骤

1. **确认是否真的是纯音调测试**
   ```bash
   cat /tmp/audio_benchmark_test/audio_dataset.jsonl
   # 如果audio_path指向generate_test_audio.py生成的文件，就是纯音调
   ```

2. **比较单条generate的参数**
   - 检查您成功的单条generate命令
   - 对比temperature、repetition_penalty等参数

3. **尝试真实音频**
   - 使用任何真实语音文件测试
   - 看输出是否还重复

## 预期结果

- ✅ **使用真实语音**: 模型应该能正确转写/描述音频内容
- ✅ **调整参数**: 即使是纯音调，也应该减少重复程度
- ⚠️ **纯音调+temperature=0.0**: 很可能重复（这是当前情况）

## 总结

这不是bench_serving的bug，而是：
1. 测试数据（纯音调）不适合语音模型
2. 采样参数（temperature=0.0 + ignore_eos）导致确定性重复

**最佳解决方案**: 使用包含真实语音的音频文件进行测试。
