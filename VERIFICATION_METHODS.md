# 如何验证正确性和误差

本文档提供多种方法来验证 Mamba State Recomputation 的正确性和质量影响。

---

## 📋 目录

1. [验证 cached token 是否提升](#1-验证-cached-token-是否提升)
2. [验证生成质量和误差](#2-验证生成质量和误差)
3. [验证系统稳定性](#3-验证系统稳定性)
4. [性能指标对比](#4-性能指标对比)

---

## 1. 验证 cached token 是否提升

### 方法 1.1: 查看服务器日志

**启动服务时添加详细日志**：

```bash
python -m sglang.launch_server \
    --model-path /path/to/Qwen3-Next \
    --enable-mamba-state-recomputation \
    --log-level debug \
    2>&1 | tee server.log
```

**关键日志标记**：

```bash
# 查找缓存命中日志
grep "cached.*token" server.log

# 应该看到类似：
# [INFO] Cached tokens: 45/50 (90.0% hit rate)
# [INFO] Tombstone detected: recompute_len=10
# [INFO] ✓ Mamba state recomputed successfully: 10 tokens
```

**对比实验**：

```bash
# 实验 1: 不启用重计算
python -m sglang.launch_server --model-path /path/to/model \
    2>&1 | grep "cached.*token" > baseline.log

# 实验 2: 启用重计算
python -m sglang.launch_server --model-path /path/to/model \
    --enable-mamba-state-recomputation \
    2>&1 | grep "cached.*token" > recomputation.log

# 对比
echo "=== Baseline ==="
cat baseline.log
echo "=== With Recomputation ==="
cat recomputation.log
```

---

### 方法 1.2: 使用 benchmark 脚本

创建测试脚本 `test_cache_hits.py`：

```python
import requests
import json
import time

# 测试场景：重复的 prefix
test_cases = [
    {
        "prompt": "今天天气怎么样？",
        "description": "第一次请求，无缓存"
    },
    {
        "prompt": "今天天气怎么样？明天呢？",
        "description": "第二次请求，共享前缀"
    },
    {
        "prompt": "今天天气怎么样？后天呢？",
        "description": "第三次请求，共享前缀"
    }
]

def test_cache_hits(url="http://localhost:30000/generate"):
    results = []

    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {case['description']}")
        print(f"Prompt: {case['prompt']}")

        start_time = time.time()

        response = requests.post(url, json={
            "text": case["prompt"],
            "sampling_params": {
                "max_new_tokens": 20,
                "temperature": 0
            }
        })

        latency = time.time() - start_time

        result = response.json()

        # 从响应中提取指标（需要在 server 中返回）
        meta = result.get("meta_info", {})

        results.append({
            "test": i+1,
            "prompt_tokens": len(case["prompt"].split()),
            "cached_tokens": meta.get("cached_tokens", 0),
            "latency_ms": latency * 1000,
            "output": result.get("text", "")[:50]
        })

        print(f"Cached tokens: {meta.get('cached_tokens', 0)}")
        print(f"Latency: {latency*1000:.2f}ms")

    # 打印汇总
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'Test':<6} {'Prompt Tokens':<15} {'Cached Tokens':<15} {'Cache Rate':<12} {'Latency (ms)'}")
    print("-" * 60)

    for r in results:
        cache_rate = r['cached_tokens'] / max(r['prompt_tokens'], 1) * 100
        print(f"{r['test']:<6} {r['prompt_tokens']:<15} {r['cached_tokens']:<15} {cache_rate:<11.1f}% {r['latency_ms']:.2f}")

    return results

if __name__ == "__main__":
    print("Testing cache hits with Mamba State Recomputation...")
    results = test_cache_hits()

    # 保存结果
    with open("cache_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to cache_test_results.json")
```

**运行测试**：

```bash
# 启动服务（启用重计算）
python -m sglang.launch_server --model-path /path/to/model \
    --enable-mamba-state-recomputation &

# 等待服务启动
sleep 10

# 运行测试
python test_cache_hits.py
```

**预期结果**：

```
Test 1: 第一次请求，无缓存
Cached tokens: 0
Latency: 150.00ms

Test 2: 第二次请求，共享前缀
Cached tokens: 15  ← 应该看到提升！
Latency: 45.00ms   ← 延迟降低

Test 3: 第三次请求，共享前缀
Cached tokens: 15  ← 持续命中
Latency: 43.00ms
```

---

## 2. 验证生成质量和误差

### 方法 2.1: 对比输出一致性

创建测试脚本 `test_quality.py`：

```python
import requests
import json
from difflib import SequenceMatcher

def generate(prompt, url="http://localhost:30000/generate", temp=0):
    """生成文本"""
    response = requests.post(url, json={
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": 100,
            "temperature": temp
        }
    })
    return response.json()["text"]

def similarity(text1, text2):
    """计算文本相似度"""
    return SequenceMatcher(None, text1, text2).ratio()

def test_quality():
    """测试生成质量"""

    test_prompts = [
        "请用100字介绍一下人工智能。",
        "Write a short story about a robot.",
        "解释一下量子计算的基本原理。",
        "What are the benefits of exercise?",
    ]

    results = []

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")

        # 生成3次（temperature=0 应该一致）
        outputs = []
        for i in range(3):
            output = generate(prompt)
            outputs.append(output)
            print(f"\nRun {i+1}:")
            print(output[:100] + "...")

        # 计算一致性
        sim_01 = similarity(outputs[0], outputs[1])
        sim_02 = similarity(outputs[0], outputs[2])
        sim_12 = similarity(outputs[1], outputs[2])
        avg_sim = (sim_01 + sim_02 + sim_12) / 3

        print(f"\nSimilarity scores:")
        print(f"  Run 1 vs Run 2: {sim_01:.2%}")
        print(f"  Run 1 vs Run 3: {sim_02:.2%}")
        print(f"  Run 2 vs Run 3: {sim_12:.2%}")
        print(f"  Average: {avg_sim:.2%}")

        results.append({
            "prompt": prompt,
            "outputs": outputs,
            "similarity": avg_sim
        })

    # 汇总
    print(f"\n{'='*60}")
    print("Quality Summary:")
    avg_quality = sum(r["similarity"] for r in results) / len(results)
    print(f"Average consistency: {avg_quality:.2%}")

    if avg_quality > 0.95:
        print("✅ Quality is excellent (>95%)")
    elif avg_quality > 0.90:
        print("✅ Quality is good (>90%)")
    elif avg_quality > 0.85:
        print("⚠️  Quality is acceptable (>85%)")
    else:
        print("❌ Quality needs improvement (<85%)")

    return results

if __name__ == "__main__":
    print("Testing output quality with Mamba State Recomputation...")
    results = test_quality()

    with open("quality_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to quality_test_results.json")
```

**运行测试**：

```bash
python test_quality.py
```

**预期结果**：

```
Average consistency: 98.5%
✅ Quality is excellent (>95%)
```

---

### 方法 2.2: 标准 Benchmark 测试

使用标准评测工具测试质量：

```bash
# 安装 lm-evaluation-harness
pip install lm-eval

# 测试多个任务
lm_eval --model sglang \
    --model_args "endpoint=http://localhost:30000/generate" \
    --tasks mmlu,hellaswag,winogrande \
    --batch_size 8 \
    --output_path eval_results.json
```

**对比测试**：

```bash
# 1. Baseline (不启用重计算)
python -m sglang.launch_server --model-path /path/to/model &
sleep 10
lm_eval --model sglang --tasks mmlu --output_path baseline_mmlu.json

# 2. With Recomputation (启用重计算)
pkill -f sglang.launch_server
python -m sglang.launch_server --model-path /path/to/model \
    --enable-mamba-state-recomputation &
sleep 10
lm_eval --model sglang --tasks mmlu --output_path recomputation_mmlu.json

# 3. 对比结果
python -c "
import json
with open('baseline_mmlu.json') as f:
    baseline = json.load(f)
with open('recomputation_mmlu.json') as f:
    recomp = json.load(f)

baseline_acc = baseline['results']['mmlu']['acc']
recomp_acc = recomp['results']['mmlu']['acc']
diff = (recomp_acc - baseline_acc) / baseline_acc * 100

print(f'Baseline accuracy: {baseline_acc:.4f}')
print(f'Recomputation accuracy: {recomp_acc:.4f}')
print(f'Difference: {diff:+.2f}%')

if abs(diff) < 2:
    print('✅ Quality impact is minimal (<2%)')
else:
    print('⚠️  Quality impact is significant (>2%)')
"
```

---

### 方法 2.3: 人工评估

创建评估脚本 `human_eval.py`：

```python
import requests
import json

def generate_with_method(prompt, method):
    """使用指定方法生成"""
    if method == "baseline":
        url = "http://localhost:30000/generate"  # 不启用重计算
    else:
        url = "http://localhost:30001/generate"  # 启用重计算

    response = requests.post(url, json={
        "text": prompt,
        "sampling_params": {"max_new_tokens": 100, "temperature": 0.7}
    })
    return response.json()["text"]

def create_evaluation_set():
    """创建人工评估集"""
    prompts = [
        "写一首关于春天的诗。",
        "Explain the theory of relativity in simple terms.",
        "编写一个 Python 函数来计算斐波那契数列。",
        "Describe the process of photosynthesis.",
        "给我讲一个有趣的故事。",
    ]

    eval_data = []

    for i, prompt in enumerate(prompts):
        baseline = generate_with_method(prompt, "baseline")
        recomputation = generate_with_method(prompt, "recomputation")

        eval_data.append({
            "id": i + 1,
            "prompt": prompt,
            "output_a": baseline,
            "output_b": recomputation,
            "rating": None  # 待评分
        })

    # 保存为评估文件
    with open("human_eval_set.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"Created evaluation set with {len(eval_data)} examples")
    print("Please rate each output (A vs B) on a scale of 1-5:")
    print("  5 = A much better")
    print("  4 = A slightly better")
    print("  3 = Equal quality")
    print("  2 = B slightly better")
    print("  1 = B much better")

    return eval_data

def collect_ratings():
    """收集人工评分"""
    with open("human_eval_set.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    for item in eval_data:
        print(f"\n{'='*60}")
        print(f"Example {item['id']}")
        print(f"Prompt: {item['prompt']}\n")
        print(f"Output A (Baseline):\n{item['output_a']}\n")
        print(f"Output B (Recomputation):\n{item['output_b']}\n")

        rating = int(input("Rating (1-5): "))
        item["rating"] = rating

    # 保存评分
    with open("human_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    # 统计
    avg_rating = sum(item["rating"] for item in eval_data) / len(eval_data)

    print(f"\n{'='*60}")
    print("Human Evaluation Results:")
    print(f"Average rating: {avg_rating:.2f}")

    if avg_rating > 2.8 and avg_rating < 3.2:
        print("✅ Quality is comparable (rating ≈ 3)")
    elif avg_rating >= 2.5:
        print("✅ Quality is acceptable (rating >= 2.5)")
    else:
        print("⚠️  Quality degradation detected (rating < 2.5)")

if __name__ == "__main__":
    print("Creating human evaluation set...")
    create_evaluation_set()
    print("\nTo collect ratings, run:")
    print("python human_eval.py --collect")
```

---

## 3. 验证系统稳定性

### 方法 3.1: 长时间压力测试

```bash
# 创建压力测试脚本 stress_test.sh
cat > stress_test.sh << 'EOF'
#!/bin/bash

URL="http://localhost:30000/generate"
DURATION=3600  # 1小时
CONCURRENT=10

echo "Starting stress test..."
echo "Duration: ${DURATION}s"
echo "Concurrent requests: ${CONCURRENT}"

start_time=$(date +%s)
request_count=0
error_count=0

while [ $(($(date +%s) - start_time)) -lt $DURATION ]; do
    for i in $(seq 1 $CONCURRENT); do
        (
            response=$(curl -s -X POST "$URL" \
                -H "Content-Type: application/json" \
                -d '{
                    "text": "Tell me about artificial intelligence.",
                    "sampling_params": {"max_new_tokens": 50}
                }')

            if echo "$response" | grep -q "text"; then
                echo "Request $((++request_count)) succeeded"
            else
                echo "Request $((++request_count)) failed"
                ((error_count++))
            fi
        ) &
    done
    wait
    sleep 1
done

echo ""
echo "Stress test completed"
echo "Total requests: $request_count"
echo "Errors: $error_count"
echo "Error rate: $(echo "scale=2; $error_count * 100 / $request_count" | bc)%"
EOF

chmod +x stress_test.sh

# 运行测试
./stress_test.sh
```

**监控内存泄漏**：

```bash
# 在另一个终端监控内存
watch -n 1 'ps aux | grep sglang | grep -v grep'

# 或使用更详细的监控
while true; do
    echo "$(date): $(ps aux | grep "sglang.launch_server" | grep -v grep | awk '{print "CPU: "$3"% MEM: "$4"%"}')"
    sleep 5
done | tee memory_monitor.log
```

---

### 方法 3.2: 检查内存一致性

在服务器代码中添加定期检查（已在实现中包含）：

```python
# 在 mamba_radix_cache.py 中已有检查
def _check_memory_consistency(self):
    """检查内存池是否一致"""
    mamba_pool = self.req_to_token_pool.mamba_pool

    total = (mamba_pool.available_size() +
             self.mamba_evictable_size_ +
             self.mamba_protected_size_)

    if total != mamba_pool.size:
        logger.error(
            f"Memory inconsistency detected! "
            f"available={mamba_pool.available_size()}, "
            f"evictable={self.mamba_evictable_size_}, "
            f"protected={self.mamba_protected_size_}, "
            f"total={total}, pool_size={mamba_pool.size}, "
            f"leaked={mamba_pool.size - total}"
        )
        return False
    return True
```

**查看日志中的内存检查**：

```bash
grep "Memory inconsistency" server.log
# 不应该有任何输出

grep "mamba_available_size" server.log | tail -20
# 应该看到数字加起来等于 pool_size
```

---

## 4. 性能指标对比

### 方法 4.1: Benchmark 对比测试

创建完整的性能测试脚本 `benchmark.py`：

```python
import requests
import time
import json
import numpy as np
from typing import List, Dict

def benchmark(
    url: str,
    prompts: List[str],
    num_runs: int = 10
) -> Dict:
    """运行 benchmark 测试"""

    results = {
        "latencies": [],
        "throughputs": [],
        "cached_tokens": [],
        "total_tokens": []
    }

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        for prompt in prompts:
            start = time.time()

            response = requests.post(url, json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 50,
                    "temperature": 0
                }
            })

            latency = time.time() - start

            result = response.json()
            meta = result.get("meta_info", {})

            results["latencies"].append(latency * 1000)  # ms

            # 计算 throughput
            total_tokens = len(prompt.split()) + 50  # 近似
            throughput = total_tokens / latency
            results["throughputs"].append(throughput)

            results["cached_tokens"].append(meta.get("cached_tokens", 0))
            results["total_tokens"].append(total_tokens)

    # 计算统计数据
    stats = {
        "latency_ms": {
            "mean": np.mean(results["latencies"]),
            "std": np.std(results["latencies"]),
            "p50": np.percentile(results["latencies"], 50),
            "p95": np.percentile(results["latencies"], 95),
            "p99": np.percentile(results["latencies"], 99),
        },
        "throughput_tokens_per_sec": {
            "mean": np.mean(results["throughputs"]),
            "std": np.std(results["throughputs"]),
        },
        "cache_hit_rate": {
            "mean": np.mean([c/t for c, t in zip(results["cached_tokens"], results["total_tokens"])]),
        }
    }

    return stats

def main():
    """主函数"""

    # 测试 prompts（有重复前缀）
    prompts = [
        "今天天气怎么样？",
        "今天天气怎么样？明天呢？",
        "今天天气怎么样？后天呢？",
        "今天天气怎么样？大后天呢？",
    ]

    print("="*60)
    print("Baseline (without recomputation)")
    print("="*60)
    baseline_stats = benchmark("http://localhost:30000/generate", prompts, num_runs=5)

    print("\n" + "="*60)
    print("With Recomputation")
    print("="*60)
    recomp_stats = benchmark("http://localhost:30001/generate", prompts, num_runs=5)

    # 对比报告
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)

    print("\nLatency (ms):")
    print(f"  Baseline:       {baseline_stats['latency_ms']['mean']:.2f} ± {baseline_stats['latency_ms']['std']:.2f}")
    print(f"  Recomputation:  {recomp_stats['latency_ms']['mean']:.2f} ± {recomp_stats['latency_ms']['std']:.2f}")
    latency_improvement = (baseline_stats['latency_ms']['mean'] - recomp_stats['latency_ms']['mean']) / baseline_stats['latency_ms']['mean'] * 100
    print(f"  Improvement:    {latency_improvement:+.1f}%")

    print("\nThroughput (tokens/sec):")
    print(f"  Baseline:       {baseline_stats['throughput_tokens_per_sec']['mean']:.2f}")
    print(f"  Recomputation:  {recomp_stats['throughput_tokens_per_sec']['mean']:.2f}")
    throughput_improvement = (recomp_stats['throughput_tokens_per_sec']['mean'] - baseline_stats['throughput_tokens_per_sec']['mean']) / baseline_stats['throughput_tokens_per_sec']['mean'] * 100
    print(f"  Improvement:    {throughput_improvement:+.1f}%")

    print("\nCache Hit Rate:")
    print(f"  Baseline:       {baseline_stats['cache_hit_rate']['mean']:.1%}")
    print(f"  Recomputation:  {recomp_stats['cache_hit_rate']['mean']:.1%}")

    # 保存结果
    report = {
        "baseline": baseline_stats,
        "recomputation": recomp_stats,
        "improvements": {
            "latency_percent": latency_improvement,
            "throughput_percent": throughput_improvement,
        }
    }

    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存到 benchmark_report.json")

if __name__ == "__main__":
    main()
```

**运行完整测试**：

```bash
# 1. 启动 baseline 服务
python -m sglang.launch_server --model-path /path/to/model \
    --port 30000 &

# 2. 启动 recomputation 服务
python -m sglang.launch_server --model-path /path/to/model \
    --enable-mamba-state-recomputation \
    --port 30001 &

# 3. 等待服务启动
sleep 20

# 4. 运行 benchmark
python benchmark.py
```

---

## 5. 快速验证清单

### ✅ 最小验证步骤

```bash
# 1. 检查服务启动
curl http://localhost:30000/health
# 应返回: {"status": "ok"}

# 2. 发送测试请求
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "sampling_params": {"max_new_tokens": 10}
  }'

# 3. 检查日志中的缓存信息
tail -100 server.log | grep -E "(cached|tombstone|recompute)"

# 4. 查找重计算成功的标记
grep "✓ Mamba state recomputed" server.log | wc -l
# 应该 > 0

# 5. 检查是否有错误
grep -E "(ERROR|WARNING)" server.log | grep -v "Expected"
# 应该没有严重错误
```

---

## 6. 预期结果总结

### ✅ 成功的标志

1. **Cached tokens 提升**:
   - Baseline: 0-20% cache hit rate
   - With recomputation: 40-80% cache hit rate

2. **性能提升**:
   - Latency: -20% to -40%
   - Throughput: +20% to +50%

3. **质量保持**:
   - Benchmark accuracy: -2% or better
   - Human evaluation: rating ≥ 2.8/5.0
   - Output consistency: ≥ 95%

4. **系统稳定**:
   - 无内存泄漏
   - 无 crash
   - 长时间运行稳定

### ⚠️ 需要注意的问题

1. **Cache hit rate 没有提升**:
   - 检查是否真的有 tombstone 节点
   - 检查 `--enable-mamba-state-recomputation` 是否生效
   - 查看日志确认重计算是否被调用

2. **质量明显下降**:
   - 可能是模型特定问题
   - 尝试调整 `--mamba-recompute-max-tokens`
   - 考虑实现真正的状态计算而不是近似

3. **内存泄漏**:
   - 检查日志中的 "Memory inconsistency" 错误
   - 确保所有分配都有对应的释放
   - 使用 valgrind 或类似工具检查

---

## 7. 故障排除

### 问题：看不到缓存提升

**检查步骤**：

```bash
# 1. 确认功能启用
ps aux | grep sglang | grep "enable-mamba-state-recomputation"

# 2. 确认有 tombstone 节点
grep "tombstone" server.log

# 3. 确认重计算被调用
grep "recompute_mamba_state" server.log

# 4. 检查是否有错误
grep "recomputation failed" server.log
```

### 问题：质量下降太多

**可能原因**：
1. Mamba state 占比较高的模型
2. 任务对历史依赖较强
3. 近似方法不适用

**解决方案**：
```bash
# 减少重计算的 token 数量
--mamba-recompute-max-tokens 128  # 降低阈值

# 或禁用重计算，只依赖自然缓存
--disable-mamba-state-recomputation
```

---

## 8. 总结

**推荐的验证流程**：

1. **快速验证** (5分钟):
   ```bash
   # 查看日志中的缓存信息
   grep "cached.*token" server.log
   ```

2. **性能验证** (30分钟):
   ```bash
   # 运行 benchmark 脚本
   python benchmark.py
   ```

3. **质量验证** (2小时):
   ```bash
   # 运行标准评测
   lm_eval --model sglang --tasks mmlu
   ```

4. **稳定性验证** (overnight):
   ```bash
   # 运行压力测试
   ./stress_test.sh
   ```

**关键指标**：
- ✅ Cache hit rate: +50% or more
- ✅ Latency: -20% or more
- ✅ Quality: within 2% of baseline
- ✅ No memory leaks or crashes

完成这些验证后，您就可以确信实现是正确且有效的！
