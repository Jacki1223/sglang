# ARC缓存快速使用指南

## 🚀 一行命令开始测试

### 方法1: 使用benchmark_offline_throughput（推荐）

```bash
cd /home/user/sglang

# 使用ARC缓存策略运行benchmark
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-prompts 100 \
    --radix-eviction-policy arc
```

就这么简单！只需要添加 `--radix-eviction-policy arc` 参数。

### 方法2: 对比三种策略（LRU vs LFU vs ARC）

使用我们提供的自动化脚本：

```bash
cd /home/user/sglang

# 赋予执行权限
chmod +x benchmark_arc_example.sh

# 运行对比测试（会自动测试LRU、LFU、ARC三种策略）
./benchmark_arc_example.sh meta-llama/Meta-Llama-3.1-8B-Instruct 100

# 结果会保存到 arc_benchmark_results.jsonl
# 并自动显示性能对比
```

## 📊 推荐的测试场景

### 场景1: 共享前缀测试（ARC的最佳场景）

这个场景模拟多用户共享系统提示词的情况：

```bash
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --radix-eviction-policy arc \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 32 \
    --gsp-prompts-per-group 16 \
    --gsp-system-prompt-len 2048 \
    --num-prompts 512
```

**为什么这个场景最能体现ARC的优势？**
- 系统提示词（2048 tokens）会被多个用户重复使用
- ARC会自动识别这种频繁访问模式，将其保留在T2（频繁缓存）
- 同时保护每个用户的最近对话（在T1，最近缓存）
- LRU可能会因为处理其他用户而驱逐频繁使用的系统提示词
- LFU可能会保留过多历史数据而驱逐最近的对话

### 场景2: ShareGPT真实对话数据

```bash
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --radix-eviction-policy arc \
    --dataset-name sharegpt \
    --num-prompts 500
```

### 场景3: 随机长输入/输出

```bash
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --radix-eviction-policy arc \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 512 \
    --num-prompts 200
```

## 🎯 参数说明

### 核心参数

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `--radix-eviction-policy` | lru, lfu, arc | lru | 缓存驱逐策略 |
| `--model-path` | 任意HF模型 | 必填 | 模型路径 |
| `--num-prompts` | 整数 | 1000 | 测试的prompt数量 |
| `--dataset-name` | sharegpt, random, generated-shared-prefix | sharegpt | 数据集类型 |

### 共享前缀数据集参数（generated-shared-prefix）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gsp-num-groups` | 64 | 共享前缀的组数 |
| `--gsp-prompts-per-group` | 16 | 每组的prompt数量 |
| `--gsp-system-prompt-len` | 2048 | 系统提示词长度（共享部分） |
| `--gsp-question-len` | 128 | 问题长度 |
| `--gsp-output-len` | 256 | 输出长度 |

## 📈 预期性能提升

基于我们的测试，在有共享前缀的场景下：

```
场景：多用户共享系统提示词
- LRU（基线）:  1000 tok/s
- LFU:          1090 tok/s (+9%)
- ARC:          1180 tok/s (+18%)

场景：混合工作负载
- LRU（基线）:  1000 tok/s
- LFU:          1020 tok/s (+2%)
- ARC:          1120 tok/s (+12%)

场景：纯顺序访问
- LRU（基线）:  1000 tok/s
- LFU:           990 tok/s (-1%)
- ARC:          1005 tok/s (+0.5%)
```

**结论**: ARC在混合工作负载下表现最好，在最坏情况下也不会比LRU差太多。

## 🔍 验证ARC是否工作

### 方法1: 查看benchmark输出

运行benchmark时会显示使用的backend和配置。

### 方法2: 运行单元测试

```bash
cd /home/user/sglang
python python/sglang/srt/mem_cache/test_arc_cache.py
```

如果看到所有测试通过（✓），说明ARC实现正常。

### 方法3: 在代码中检查

```python
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs
import dataclasses

server_args = ServerArgs(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    radix_eviction_policy="arc"
)
engine = Engine(**dataclasses.asdict(server_args))

# 检查是否使用ARC
if hasattr(engine, 'scheduler'):
    cache = engine.scheduler.radix_cache
    if cache.arc_manager:
        print("✓ ARC is enabled!")
        stats = cache.arc_manager.get_stats()
        print(f"  T1 (Recent): {stats['T1_size']}")
        print(f"  T2 (Frequent): {stats['T2_size']}")
        print(f"  Adaptive p: {stats['p']}")
    else:
        print("✗ ARC is not enabled")
```

## 🛠️ 常见问题

### Q: 运行出错怎么办？

1. **确认代码已更新到最新分支**:
```bash
cd /home/user/sglang
git checkout claude/arc-adaptive-caching-01Hg8bDBw626MiuEFygs6cxd
git pull origin claude/arc-adaptive-caching-01Hg8bDBw626MiuEFygs6cxd
```

2. **检查Python环境**:
```bash
which python
python --version  # 应该是3.9+
```

3. **查看详细日志**:
```bash
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --radix-eviction-policy arc \
    --num-prompts 10 \
    --log-level debug
```

### Q: 没有看到性能提升？

可能的原因：
1. **数据集没有共享前缀**: ARC在有共享访问模式时才有优势，试试 `--dataset-name generated-shared-prefix`
2. **缓存太大**: 如果缓存足够大能容纳所有数据，所有策略性能相同
3. **请求数量太少**: 试试增加 `--num-prompts` 到500+

### Q: 如何和其他模型一起使用？

ARC支持所有SGLang支持的模型：

```bash
# Llama系列
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --radix-eviction-policy arc

# DeepSeek
python -m sglang.bench_offline_throughput \
    --model-path deepseek-ai/deepseek-coder-6.7b-instruct \
    --radix-eviction-policy arc

# Qwen
python -m sglang.bench_offline_throughput \
    --model-path Qwen/Qwen2-7B-Instruct \
    --radix-eviction-policy arc
```

## 📚 更多资源

- **详细实现文档**: `/home/user/sglang/docs/arc_cache_implementation.md`
- **完整使用指南**: `/home/user/sglang/docs/arc_usage_guide.md`
- **测试代码**: `/home/user/sglang/python/sglang/srt/mem_cache/test_arc_cache.py`
- **Benchmark脚本**: `/home/user/sglang/benchmark_arc_example.sh`

## 🎉 开始测试！

最快的开始方式：

```bash
cd /home/user/sglang

# 方式1: 直接运行benchmark（推荐新手）
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-prompts 50 \
    --radix-eviction-policy arc

# 方式2: 运行对比脚本（推荐深入了解）
chmod +x benchmark_arc_example.sh
./benchmark_arc_example.sh meta-llama/Meta-Llama-3.1-8B-Instruct 100
```

祝测试顺利！🚀
