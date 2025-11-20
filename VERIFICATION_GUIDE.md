# Mamba State Recomputation 验证指南

本指南提供多种方法验证 Mamba state recomputation 的正确性和效果。

---

## 方法 1: 自动化验证脚本（推荐）

### 快速验证

```bash
# 启动服务器（开启重计算）
python -m sglang.launch_server \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --enable-mamba-state-recomputation \
    --mamba-recompute-max-tokens 20 \
    --log-level info

# 运行验证脚本（另一个终端）
python verify_mamba_recomputation.py --url http://localhost:30000
```

### 预期输出

```
✅ Passed: 7
⚠️  Warned: 0
❌ Failed: 0

🎉 All tests passed! Mamba recomputation appears to be working correctly.
```

---

## 方法 2: 日志分析验证

### 2.1 检查启动日志

**正确配置的标志：**

```bash
# 在启动日志中查找
grep "Recomputation ENABLED" server.log

# 应该看到：
# MambaRadixCache: Recomputation ENABLED (max_tokens=20, prioritize_mamba=True, eviction_threshold=0.9)
```

**如果没看到：**
- 检查 `--enable-mamba-state-recomputation` 是否设置
- 检查是否是 hybrid GDN 模型（Qwen3-Next）

### 2.2 监控运行日志

**A. 重计算触发日志**

```bash
# 查找重计算日志
grep "Tombstone detected" server.log | head -20

# 正常输出示例：
# Tombstone detected: recompute_len=2, max_tokens=20, last_valid_node=missing, total_value_len=2
# No valid starting mamba state found. Will attempt recomputation from zero-initialized state for 2 tokens...
# ✓ Mamba state recomputed successfully: 2 tokens, total hits: 1
```

**B. 重计算统计**

```bash
# 统计重计算成功次数
grep "✓ Mamba state recomputed successfully" server.log | wc -l

# 统计重计算失败次数
grep "✗ Mamba state recomputation failed" server.log | wc -l

# 计算成功率
# success_rate = success_count / (success_count + failure_count)
```

**C. Cache Hit 日志**

```bash
# 查找 prefill batch 日志
grep "Prefill batch" server.log | tail -20

# 对比 cached-token 数量
# 之前（无重计算）: #cached-token: 0
# 之后（有重计算）: #cached-token: 50-200+
```

### 2.3 异常检查

**检查错误日志：**

```bash
# 检查内存泄漏
grep "memory leak detected" server.log
# 应该没有输出（如果有，说明仍有问题）

# 检查崩溃
grep "AssertionError\|Traceback" server.log
# 应该没有输出

# 检查警告
grep "WARNING.*mamba" server.log | tail -20
# 适量的 "Target node already has mamba_value" 是正常的（并发场景）
```

---

## 方法 3: Benchmark 对比验证

### 3.1 准备对比测试

**测试场景 A: 无重计算（基线）**

```bash
# 启动服务器（禁用重计算）
python -m sglang.launch_server \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --port 30000

# 运行 benchmark
python -m sglang.bench_serving \
    --backend sglang \
    --port 30000 \
    --dataset-name sharegpt \
    --num-prompts 100 \
    --request-rate 1 \
    > baseline_results.txt
```

**测试场景 B: 有重计算（优化）**

```bash
# 启动服务器（启用重计算）
python -m sglang.launch_server \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --enable-mamba-state-recomputation \
    --mamba-recompute-max-tokens 20 \
    --port 30000

# 运行相同 benchmark
python -m sglang.bench_serving \
    --backend sglang \
    --port 30000 \
    --dataset-name sharegpt \
    --num-prompts 100 \
    --request-rate 1 \
    > optimized_results.txt
```

### 3.2 对比关键指标

```bash
# 提取关键指标
echo "=== Baseline (No Recomputation) ==="
grep -E "Throughput|Latency|cached" baseline_results.txt

echo "=== Optimized (With Recomputation) ==="
grep -E "Throughput|Latency|cached" optimized_results.txt
```

**预期改进：**

| 指标 | 基线 | 优化后 | 改进 |
|------|------|--------|------|
| Avg cached tokens | 0-10 | 50-200 | ✅ +500% |
| Request throughput | X req/s | 1.2-1.5X req/s | ✅ +20-50% |
| Avg latency | Y ms | 0.7-0.9Y ms | ✅ -10-30% |
| P99 latency | Z ms | 0.8-0.95Z ms | ✅ -5-20% |

---

## 方法 4: 质量验证

### 4.1 确定性测试（Temperature = 0）

```python
import requests

def test_determinism(prompt, num_runs=3):
    """测试相同输入是否产生相同输出"""
    url = "http://localhost:30000/v1/completions"
    outputs = []

    for i in range(num_runs):
        response = requests.post(url, json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.0,  # 确定性采样
        })
        output = response.json()["choices"][0]["text"]
        outputs.append(output)
        print(f"Run {i+1}: {output[:80]}...")

    # 检查一致性
    all_same = all(out == outputs[0] for out in outputs)
    print(f"\nDeterministic: {all_same}")
    return all_same

# 测试
test_determinism("The capital of France is")
test_determinism("def quicksort(arr):\n    ")
```

**预期结果：**
- ✅ Temperature=0 时，多次运行输出应该**完全一致**
- 如果不一致，可能说明近似影响了确定性（需要调查）

### 4.2 质量主观评估

```bash
# 生成样本对比
python -c "
import requests

prompts = [
    'Write a poem about spring',
    'Translate to Chinese: Hello world',
    'Explain quantum computing in simple terms',
]

for prompt in prompts:
    resp = requests.post('http://localhost:30000/v1/completions', json={
        'model': 'default',
        'prompt': prompt,
        'max_tokens': 100,
        'temperature': 0.7,
    })
    print(f'Prompt: {prompt}')
    print(f'Output: {resp.json()[\"choices\"][0][\"text\"]}')
    print('-' * 80)
"
```

**评估标准：**
- ✅ 输出流畅、连贯
- ✅ 符合 prompt 意图
- ✅ 没有明显的重复、错误、乱码
- ✅ 与禁用重计算时的质量相当

### 4.3 Perplexity 测试（如果可用）

```python
# 使用标准测试集计算 perplexity
# 对比开启/关闭重计算的 perplexity 差异

# 预期：perplexity 差异 < 5%
```

---

## 方法 5: 压力测试

### 5.1 长时间稳定性测试

```bash
# 运行 1 小时持续负载
python -m sglang.bench_serving \
    --backend sglang \
    --port 30000 \
    --dataset-name sharegpt \
    --num-prompts 1000 \
    --request-rate 2 \
    --duration 3600

# 监控过程中的异常
tail -f server.log | grep -E "ERROR|Traceback|memory leak"
```

**检查项：**
- ✅ 无崩溃
- ✅ 无内存泄漏
- ✅ 性能稳定（无明显下降）

### 5.2 并发压力测试

```bash
# 高并发请求
python -m sglang.bench_serving \
    --backend sglang \
    --port 30000 \
    --dataset-name sharegpt \
    --num-prompts 500 \
    --request-rate 10  # 高并发

# 检查日志中的并发问题
grep "already has mamba_value" server.log | wc -l
# 少量出现是正常的（并发检测机制）
# 大量出现可能说明并发控制有问题
```

---

## 方法 6: 内存和性能分析

### 6.1 内存使用监控

```bash
# 监控 GPU 内存
watch -n 1 nvidia-smi

# 或使用脚本
while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    sleep 1
done > gpu_memory.log
```

**检查：**
- ✅ 内存使用稳定（无持续增长）
- ✅ 与禁用重计算时内存使用相当

### 6.2 计算开销分析

```bash
# 使用 nsys 或 nvprof 进行性能分析（高级）
nsys profile -o recomputation_profile python -m sglang.launch_server ...

# 分析热点
# 预期：recompute_mamba_state 的开销应该很小（主要是 memory copy）
```

---

## 方法 7: 代码逻辑验证

### 7.1 单元测试（如果有）

```bash
# 运行相关单元测试
pytest test/srt/test_mamba_recompute.py -v
```

### 7.2 手动代码审查检查清单

- [ ] `enable_mamba_state_recomputation` 正确传递到 MambaRadixCache
- [ ] `model_runner` 引用正确设置（不是 None）
- [ ] `_try_rebuild_mamba_state` 正确分配和释放 mamba states
- [ ] 内存会计正确（available + evictable + protected = total）
- [ ] 并发情况下无重复分配
- [ ] 零初始化逻辑正确执行
- [ ] 状态复制逻辑正确执行

---

## 快速验证清单（5分钟）

快速验证重计算是否工作：

```bash
# 1. 检查配置
grep "Recomputation ENABLED" server.log
# ✅ 应该有输出

# 2. 检查重计算触发
grep "Mamba state recomputed successfully" server.log | head -5
# ✅ 应该有输出（说明重计算被触发）

# 3. 检查 cache hit
grep "Prefill batch" server.log | tail -5 | grep cached-token
# ✅ 应该看到非零的 cached-token

# 4. 检查无错误
grep -E "ERROR.*mamba|memory leak|AssertionError" server.log
# ✅ 应该无输出（或只有少量警告）

# 5. 运行快速测试
python verify_mamba_recomputation.py --url http://localhost:30000
# ✅ 应该显示 "All tests passed"
```

---

## 常见问题诊断

### Q: cached_token 仍然是 0

**检查：**
1. 确认 `--enable-mamba-state-recomputation` 已设置
2. 检查日志是否有 "Recomputation ENABLED"
3. 检查是否是支持的模型（hybrid GDN）
4. 查看是否有 "Tombstone detected" 但没有 "recomputed successfully"

### Q: 有重计算日志但 cached_token 很少

**可能原因：**
1. Benchmark 场景没有共享 prefix（正常）
2. `mamba_recompute_max_tokens` 设置太小
3. Mamba states 被频繁驱逐（增加 mamba pool size）

### Q: 生成质量下降

**检查：**
1. 重计算距离是否过长（查看 recompute_len）
2. 降低 `mamba_recompute_max_tokens`（如 10-20）
3. 启用 `--prioritize-mamba-retention`

### Q: 出现内存泄漏

**检查：**
1. 查看 "Target node already has mamba_value" 警告频率
2. 确认所有分配的 mamba states 都被正确释放
3. 报告 bug 并附上日志

---

## 总结

**验证正确性的关键指标：**

| 指标 | 检查方法 | 预期值 |
|------|---------|--------|
| **功能性** | 启动日志 | "Recomputation ENABLED" |
| **触发率** | grep "recomputed successfully" | > 0 |
| **成功率** | 成功数 / (成功数 + 失败数) | > 95% |
| **Cache Hit** | cached-token in logs | 增加 50-500% |
| **吞吐量** | Benchmark | 提升 20-50% |
| **质量** | 主观评估 + 确定性测试 | 无明显下降 |
| **稳定性** | 长时间测试 | 无崩溃/泄漏 |

**如果所有指标都达标 → 重计算工作正常！✅**
