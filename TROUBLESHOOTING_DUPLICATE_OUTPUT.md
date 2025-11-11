# 重复字段问题排查指南

## 问题描述
用户反馈："大量输出重复字段"，且"如果采用generate的方式推理单条是没问题的"

## 需要收集的信息

### 1. 具体的重复内容是什么？

请提供以下信息之一：

**选项A - 控制台输出重复**
```bash
# 运行bench_serving并保存完整输出
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/dataset.jsonl \
    --num-prompts 3 2>&1 | tee bench_output.log

# 然后检查 bench_output.log 中哪些内容重复了
```

**选项B - 输出文件中的重复**
```bash
# 检查输出文件
cat results.jsonl | jq '.' > results_formatted.json

# 查看字段列表
cat results.jsonl | jq 'keys' | sort

# 检查是否有重复的字段名
cat results.jsonl | jq 'keys' | sort | uniq -d
```

### 2. 重复的模式

请说明重复的具体表现：

- [ ] 同一个字段名出现多次（如：`"prompt": "xxx", "prompt": "yyy"`）
- [ ] 每个请求的结果都单独显示（如：显示了3次完整的结果）
- [ ] Token统计信息显示多次（如：多次打印 "Input tokens: XXX"）
- [ ] 性能指标重复显示（如：多次显示吞吐量）
- [ ] 其他（请描述）

### 3. 对比单条generate的输出

**单条generate命令示例**：
```python
# 使用Python直接调用
import requests

response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "your-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}},
                    {"type": "text", "text": "描述这段音频"}
                ]
            }
        ],
        "max_completion_tokens": 256
    }
)
print(response.json())
```

**期望**: 单条请求返回一个干净的JSON响应，无重复字段

## 可能的原因和解决方案

### 原因1: JSONL文件追加模式导致累积

**症状**: 输出文件包含多次运行的结果

**检查方法**:
```bash
# 查看文件有多少行（每行一次运行）
cat results.jsonl | wc -l

# 如果大于1，说明文件累积了多次运行
```

**解决方案**:
```bash
# 每次运行前删除旧文件
rm -f results.jsonl

# 或使用带时间戳的文件名
--output-file results_$(date +%Y%m%d_%H%M%S).jsonl
```

### 原因2: 控制台显示详细token分解

**症状**: 看到类似输出
```
Total input tokens:           1000
Total input text tokens:      200
Total input vision tokens:    797
```

**说明**: 这不是重复，而是多模态模型的详细分解
- Total input tokens = text + vision + special tokens
- 这是正常的信息性输出

### 原因3: 每个请求的详细输出

**症状**: 每个请求都显示完整的统计信息

**检查**: 查看是否有类似输出重复N次（N=请求数）

**可能的代码问题**:
- 在请求循环中有打印语句
- 异步请求的回调中有日志输出

### 原因4: 数据集采样的重复

**症状**: 相同的音频文件被处理多次

**检查方法**:
```python
# 检查数据集文件
cat /path/to/dataset.jsonl

# 确保没有重复行
cat /path/to/dataset.jsonl | sort | uniq -c
```

## 临时诊断脚本

### 检查输出文件结构
```bash
cat > /tmp/check_output_structure.py << 'EOF'
#!/usr/bin/env python3
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python3 check_output_structure.py results.jsonl")
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

print(f"文件包含 {len(lines)} 行（运行次数）\n")

for i, line in enumerate(lines, 1):
    result = json.loads(line)
    keys = list(result.keys())
    print(f"第 {i} 次运行:")
    print(f"  字段数量: {len(keys)}")
    print(f"  字段列表: {', '.join(keys[:10])}...")

    # 检查重复字段
    if len(keys) != len(set(keys)):
        duplicates = [k for k in keys if keys.count(k) > 1]
        print(f"  ⚠️  发现重复字段: {set(duplicates)}")

    # 检查数组长度
    if 'generated_texts' in result:
        print(f"  生成文本数量: {len(result['generated_texts'])}")
    if 'input_lens' in result:
        print(f"  请求数量: {len(result['input_lens'])}")
    print()
EOF

chmod +x /tmp/check_output_structure.py
```

运行检查：
```bash
python3 /tmp/check_output_structure.py results.jsonl
```

## 下一步

请提供以下信息以便进一步诊断：

1. 运行 bench_serving 的完整命令
2. 控制台输出的截图或文本（至少包含开始和结束部分）
3. 如果有输出文件，运行上述检查脚本的结果
4. 具体哪些字段/信息重复了（截图或文本）

有了这些信息，我可以：
- 准确定位问题
- 提供针对性的修复
- 如果是设计特性，解释为什么这样设计
