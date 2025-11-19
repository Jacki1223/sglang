# KV Cache预分配池集成和使用指南

本指南详细说明如何在SGLang推理中使用KV Cache预分配池功能。

## 目录
1. [快速开始](#快速开始)
2. [方式一：代码级别集成](#方式一代码级别集成)
3. [方式二：命令行参数集成](#方式二命令行参数集成)
4. [配置选项详解](#配置选项详解)
5. [性能调优](#性能调优)
6. [监控和调试](#监控和调试)

---

## 快速开始

### 直接使用（无需修改配置）

如果你想在现有代码中快速测试预分配池，可以直接修改 `model_runner.py`：

```python
# 在 python/sglang/srt/model_executor/model_runner.py 中
# 找到 PagedTokenToKVPoolAllocator 的初始化位置（大约第1905行）

# 原来的代码:
self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
    self.max_total_num_tokens,
    page_size=self.page_size,
    dtype=self.kv_cache_dtype,
    device=self.device,
    kvcache=self.token_to_kv_pool,
    need_sort=need_sort,
)

# 替换为:
from sglang.srt.mem_cache.allocator import PreallocatedPagedTokenToKVPoolAllocator

self.token_to_kv_pool_allocator = PreallocatedPagedTokenToKVPoolAllocator(
    self.max_total_num_tokens,
    page_size=self.page_size,
    dtype=self.kv_cache_dtype,
    device=self.device,
    kvcache=self.token_to_kv_pool,
    need_sort=need_sort,
    enable_prealloc=True,                    # 启用预分配
    prealloc_bucket_sizes=[1, 2, 4, 8, 16, 32, 64, 128],  # 桶大小
    prealloc_ratio=0.8,                      # 80%用于预分配
)
```

---

## 方式一：代码级别集成

### 1. 修改 model_runner.py

编辑 `python/sglang/srt/model_executor/model_runner.py`：

```python
# 在文件开头添加导入
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    PreallocatedPagedTokenToKVPoolAllocator,  # 新增
    TokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)

# 在 init_token_to_kv_pool_allocator 方法中（大约第1871行）
def init_token_to_kv_pool_allocator(self, need_sort: bool = False):
    if self.token_to_kv_pool_allocator is None:
        # ... 其他条件判断 ...

        else:
            if self.page_size == 1:
                # ... TokenToKVPoolAllocator 逻辑 ...
            else:
                assert not self.is_hybrid

                # 检查是否启用预分配
                enable_prealloc = getattr(self.server_args, 'enable_kv_prealloc', False)

                if enable_prealloc:
                    # 使用预分配池版本
                    prealloc_ratio = getattr(self.server_args, 'kv_prealloc_ratio', 0.8)
                    prealloc_buckets_str = getattr(self.server_args, 'kv_prealloc_buckets', None)

                    # 解析桶大小
                    if prealloc_buckets_str:
                        prealloc_buckets = [int(x) for x in prealloc_buckets_str.split(',')]
                    else:
                        prealloc_buckets = None  # 使用默认值

                    self.token_to_kv_pool_allocator = PreallocatedPagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                        enable_prealloc=True,
                        prealloc_bucket_sizes=prealloc_buckets,
                        prealloc_ratio=prealloc_ratio,
                    )
                else:
                    # 使用原版本
                    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
```

### 2. 添加 ServerArgs 配置选项

编辑 `python/sglang/srt/server_args.py`：

```python
@dataclasses.dataclass
class ServerArgs:
    # ... 现有字段 ...

    # KV Cache preallocation (添加到 Memory and scheduling 组)
    enable_kv_prealloc: bool = False
    kv_prealloc_ratio: float = 0.8
    kv_prealloc_buckets: Optional[str] = None  # 格式: "1,2,4,8,16,32,64,128"
```

### 3. 添加命令行参数

在 `server_args.py` 的 `add_cli_args` 方法中添加：

```python
@staticmethod
def add_cli_args(parser: argparse.ArgumentParser):
    # ... 现有参数 ...

    # KV Cache preallocation
    parser.add_argument(
        "--enable-kv-prealloc",
        action="store_true",
        help="Enable KV cache preallocation pool for better memory efficiency",
    )
    parser.add_argument(
        "--kv-prealloc-ratio",
        type=float,
        default=ServerArgs.kv_prealloc_ratio,
        help="Ratio of pages to use for preallocation pool (0.0-1.0, default: 0.8)",
    )
    parser.add_argument(
        "--kv-prealloc-buckets",
        type=str,
        default=None,
        help="Comma-separated list of bucket sizes in pages (e.g., '1,2,4,8,16,32,64,128')",
    )
```

---

## 方式二：命令行参数集成

完成上述代码集成后，可以通过命令行参数启用预分配池。

### Python API 使用

```python
import sglang as sgl

# 启动服务器，启用KV预分配池
llm = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    enable_kv_prealloc=True,           # 启用预分配
    kv_prealloc_ratio=0.8,             # 80%页面用于预分配
    kv_prealloc_buckets="1,2,4,8,16,32,64",  # 自定义桶大小
)

# 正常推理
response = llm.generate(
    "What is the capital of France?",
    sampling_params={"temperature": 0.7, "max_new_tokens": 100}
)
print(response)
```

### 命令行启动服务器

```bash
# 基本启用
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-kv-prealloc

# 完整配置
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-kv-prealloc \
    --kv-prealloc-ratio 0.75 \
    --kv-prealloc-buckets "1,2,4,8,16,32,64,128" \
    --port 30000

# 大模型优化配置
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --enable-kv-prealloc \
    --kv-prealloc-ratio 0.85 \
    --kv-prealloc-buckets "2,4,8,16,32,64,128,256" \
    --mem-fraction-static 0.9 \
    --tp-size 4
```

### OpenAI兼容API

```bash
# 启动服务器
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-kv-prealloc \
    --port 30000

# 使用OpenAI客户端
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

---

## 配置选项详解

### enable_kv_prealloc

- **类型**: bool
- **默认值**: False
- **说明**: 是否启用KV Cache预分配池
- **推荐**: 对于page_size > 1的配置，建议启用

### kv_prealloc_ratio

- **类型**: float (0.0 - 1.0)
- **默认值**: 0.8
- **说明**: 用于预分配池的页面比例
  - 0.8 表示80%页面用于预分配池，20%作为回退
  - 较高的比例提供更好的性能，但降低灵活性
  - 较低的比例提供更多灵活性，但可能增加分配开销

**推荐配置**:
- 通用场景: 0.8
- 高吞吐场景: 0.85-0.9
- 变化较大的工作负载: 0.7-0.75

### kv_prealloc_buckets

- **类型**: str (逗号分隔的整数)
- **默认值**: None (使用默认 [1,2,4,8,16,32,64,128])
- **说明**: 预分配桶的大小（以页为单位）
- **格式**: "1,2,4,8,16,32,64,128"

**推荐配置**:

```bash
# 小模型 (< 13B)
--kv-prealloc-buckets "1,2,4,8,16,32,64"

# 中等模型 (13B - 70B)
--kv-prealloc-buckets "1,2,4,8,16,32,64,128"

# 大模型 (> 70B)
--kv-prealloc-buckets "2,4,8,16,32,64,128,256"

# 短文本场景
--kv-prealloc-buckets "1,2,4,8,16,32"

# 长文本场景
--kv-prealloc-buckets "4,8,16,32,64,128,256,512"
```

---

## 性能调优

### 1. 根据工作负载调整桶大小

```python
# 分析你的工作负载
# 启用调试模式查看分配模式
export SGLANG_DEBUG_MEMORY_POOL=1

# 运行推理
python your_inference_script.py

# 查看日志中的分配统计
# 根据最常见的分配大小调整桶配置
```

### 2. 优化预分配比例

```bash
# 监控场景
--kv-prealloc-ratio 0.7   # 更多回退空间

# 生产场景
--kv-prealloc-ratio 0.8   # 平衡性能和灵活性

# 高性能场景
--kv-prealloc-ratio 0.9   # 最大化预分配
```

### 3. 针对不同page_size优化

```bash
# page_size=16 (默认)
--kv-prealloc-buckets "1,2,4,8,16,32,64"

# page_size=32
--kv-prealloc-buckets "1,2,4,8,16,32"

# page_size=64
--kv-prealloc-buckets "1,2,4,8,16"
```

---

## 监控和调试

### 1. 启用调试模式

```bash
# 环境变量
export SGLANG_DEBUG_MEMORY_POOL=1

# 运行推理
python -m sglang.launch_server \
    --model-path your-model \
    --enable-kv-prealloc
```

### 2. 获取运行时统计

在代码中访问统计信息：

```python
# 在 model_runner 中
stats = self.token_to_kv_pool_allocator.get_statistics()

print(f"Total pages: {stats['total_pages']}")
print(f"Available size: {stats['total_available_size']}")

if 'prealloc' in stats:
    prealloc_stats = stats['prealloc']
    print(f"Prealloc utilization: {prealloc_stats['utilization']:.2%}")
    print(f"Total allocations: {prealloc_stats['total_allocations']}")
    print(f"Split operations: {prealloc_stats['split_operations']}")

    # Per-bucket 统计
    for bucket_size, bucket_stats in prealloc_stats['buckets'].items():
        if bucket_stats['allocations'] > 0:
            print(f"Bucket {bucket_size}: {bucket_stats['allocations']} allocs")
```

### 3. 性能基准测试

```python
import time
import sglang as sgl

# 不启用预分配
llm_baseline = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    enable_kv_prealloc=False
)

start = time.time()
for _ in range(100):
    llm_baseline.generate("Test prompt", {"max_new_tokens": 50})
baseline_time = time.time() - start

# 启用预分配
llm_prealloc = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    enable_kv_prealloc=True,
    kv_prealloc_ratio=0.8
)

start = time.time()
for _ in range(100):
    llm_prealloc.generate("Test prompt", {"max_new_tokens": 50})
prealloc_time = time.time() - start

print(f"Baseline: {baseline_time:.2f}s")
print(f"With prealloc: {prealloc_time:.2f}s")
print(f"Speedup: {baseline_time/prealloc_time:.2f}x")
```

---

## 常见问题

### Q1: 何时应该启用预分配池？

**A**: 推荐在以下情况启用：
- 使用 page_size > 1 的配置
- 有稳定的工作负载模式
- 追求最优性能
- GPU内存充足

**不推荐**:
- page_size = 1（使用 TokenToKVPoolAllocator）
- 内存极度受限
- 工作负载高度动态变化

### Q2: 预分配池会增加内存使用吗？

**A**: 不会。预分配池只是改变内存的组织方式，不增加总内存使用。实际上通过减少碎片可能略微降低内存使用。

### Q3: 如何选择最优的 prealloc_ratio？

**A**: 建议：
1. 从默认值 0.8 开始
2. 启用调试模式运行你的工作负载
3. 观察 fallback 分配的频率
4. 如果很少回退，可以增加比例到 0.85-0.9
5. 如果经常回退失败，降低到 0.7-0.75

### Q4: 桶大小如何影响性能？

**A**:
- **更多桶**: 减少块分割，提高内存利用率，但增加内存管理开销
- **更少桶**: 简化管理，但可能增加块分割次数
- **推荐**: 使用默认的8个桶，覆盖2的幂次方大小

### Q5: 可以动态调整配置吗？

**A**: 当前实现不支持运行时动态调整。需要重启服务器以应用新配置。

---

## 示例场景

### 场景1: 聊天应用

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-kv-prealloc \
    --kv-prealloc-ratio 0.8 \
    --kv-prealloc-buckets "1,2,4,8,16,32" \
    --max-running-requests 128
```

### 场景2: 长文本生成

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --enable-kv-prealloc \
    --kv-prealloc-ratio 0.85 \
    --kv-prealloc-buckets "8,16,32,64,128,256" \
    --context-length 32768 \
    --tp-size 4
```

### 场景3: 批量推理

```bash
python -m sglang.launch_server \
    --model-path your-model \
    --enable-kv-prealloc \
    --kv-prealloc-ratio 0.9 \
    --kv-prealloc-buckets "2,4,8,16,32,64,128" \
    --max-running-requests 256
```

---

## 总结

KV Cache预分配池通过以下方式提升性能：

1. **减少分配开销**: O(1)时间复杂度的分配
2. **降低内存碎片**: 基于大小的池化策略
3. **提高缓存命中率**: 预分配常用大小
4. **可预测的性能**: 避免运行时分配失败

根据你的具体需求选择合适的配置，并通过监控和调优获得最佳性能。
