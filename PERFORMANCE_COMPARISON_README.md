# KV Cache分配器性能对比指南

这个文档说明如何使用 `compare_allocators.py` 脚本对比不同KV Cache分配器的性能。

## 快速开始

### 运行完整对比测试

```bash
cd /home/user/sglang
python compare_allocators.py
```

这将自动对比三种实现：
1. **PagedTokenToKVPoolAllocator** - 原始实现 (Baseline)
2. **PreallocatedPagedTokenToKVPoolAllocator (标准)** - 标准预分配池
3. **PreallocatedPagedTokenToKVPoolAllocator (优化)** - 优化预分配池

## 测试场景

脚本会自动测试3种真实工作负载：

### 1. 聊天场景
- **特点**: 小到中等大小的分配 (16-256 tokens)
- **用例**: 对话系统，问答应用
- **操作数**: 500次
- **预期**: 优化版本应该有 **20-35%** 的性能提升

### 2. 长文本场景
- **特点**: 大块分配 (512-4096 tokens)
- **用例**: 文档分析，长文本生成
- **操作数**: 200次
- **预期**: 优化版本应该有 **15-25%** 的性能提升

### 3. 混合场景
- **特点**: Prefill (大块) + Decode (小块) 混合
- **用例**: 生产环境的典型工作负载
- **操作数**: 1000次
- **预期**: 优化版本应该有 **20-30%** 的性能提升

## 输出解读

### 示例输出

```
======================================================================
工作负载: 聊天场景 (小到中等分配)
总操作数: 500
内存池大小: 50000 tokens, 页大小: 16
设备: cuda
======================================================================

测试 1/3: 原始PagedTokenToKVPoolAllocator (Baseline)...
测试 2/3: PreallocatedPagedTokenToKVPoolAllocator (标准)...
测试 3/3: PreallocatedPagedTokenToKVPoolAllocator (优化)...

指标                      Baseline        Standard        Optimized       Std vs Base     Opt vs Base
----------------------------------------------------------------------------------------------------
总时间 (秒)               0.156           0.132           0.118           1.18x ✓         1.32x ✅
吞吐量 (ops/s)            3205            3788            4237            1.18x ✓         1.32x ✅
平均分配时间 (ms)         0.2850          0.2150          0.1820          1.33x ✅        1.57x ✅
P95分配时间 (ms)          0.4120          0.2980          0.2450          1.38x ✅        1.68x ✅
平均释放时间 (ms)         0.0580          0.0450          0.0320          1.29x ✅        1.81x ✅
成功率                    100.00%         100.00%         100.00%         =               =

📊 总结:
  标准预分配池: 1.18x 吞吐量 (+18.2% ✓)
  优化预分配池: 1.32x 吞吐量 (+32.2% ✅)
  ✅ 优化版本性能提升显著 (32.2% 更快)
```

### 关键指标说明

| 指标 | 说明 | 目标 |
|------|------|------|
| **总时间** | 完成所有操作的总时间 | 越低越好 |
| **吞吐量** | 每秒操作数 (ops/s) | 越高越好 |
| **平均分配时间** | 单次分配的平均延迟 | < 0.2ms 为优秀 |
| **P95分配时间** | 95%分位数延迟 | < 0.3ms 为优秀 |
| **平均释放时间** | 单次释放的平均延迟 | < 0.05ms 为优秀 |
| **成功率** | 成功分配的比例 | 应该 = 100% |

### 加速比标记

- ✅ **> 1.1x**: 显著提升
- ✓ **1.0-1.1x**: 轻微提升
- ≈ **~1.0x**: 性能相近
- ❌ **< 0.9x**: 性能下降

## 自定义测试

### 修改测试参数

编辑 `compare_allocators.py` 的 `main()` 函数：

```python
# 修改聊天场景的请求数
chat_workload = generator.generate_chat_workload(num_requests=1000)  # 默认500

# 修改内存池大小
benchmark.compare_allocators(
    chat_workload,
    "聊天场景",
    size=100000,  # 默认50000
    page_size=32   # 默认16
)
```

### 添加自定义工作负载

```python
# 在 main() 函数中添加
custom_workload = [
    ('alloc', 64),   # 分配64 tokens
    ('alloc', 128),  # 分配128 tokens
    ('free', 64),    # 释放64 tokens
    # ... 更多操作
]

benchmark.compare_allocators(
    custom_workload,
    "自定义场景",
    size=50000,
    page_size=16
)
```

## 性能分析

### 查看详细的per-allocator统计

在测试后，可以获取详细统计：

```python
# 在脚本中添加
if hasattr(allocator3, 'get_statistics'):
    stats = allocator3.get_statistics()
    print(f"快速路径命中率: {stats['prealloc'].get('fast_path_hit_rate', 0):.2%}")
    print(f"块分割率: {stats['prealloc']['split_operations'] / stats['prealloc']['total_allocations']:.2%}")
```

### 使用性能诊断工具

```python
from sglang.srt.mem_cache.performance_diagnostics import diagnose_allocation_pattern

# 运行测试后
diagnose_allocation_pattern(allocator3)
```

## 性能问题排查

### 如果优化版本性能没有提升

#### 1. 检查快速路径命中率

```python
stats = allocator.get_statistics()
fast_path_rate = stats['prealloc'].get('fast_path_hit_rate', 0)

if fast_path_rate < 0.6:
    print("⚠️ 快速路径命中率过低，需要调整桶大小")
```

**解决方案:**
- 分析实际分配大小分布
- 调整 `prealloc_bucket_sizes` 以匹配常用大小
- 参考 `QUICK_FIX_PERFORMANCE.md`

#### 2. 检查块分割率

```python
split_rate = stats['prealloc']['split_operations'] / stats['prealloc']['total_allocations']

if split_rate > 0.5:
    print("⚠️ 块分割过多，内存效率低")
```

**解决方案:**
- 增加桶大小的种类
- 调整桶大小以覆盖更多常用大小

#### 3. 检查设备

```python
print(f"使用设备: {benchmark.device}")

# CUDA vs CPU性能差异可能很大
```

**注意:**
- CUDA上通常有更明显的性能提升
- CPU上由于内存操作较慢，提升可能较小

## 对比不同配置

### 测试不同的预分配比例

```python
for ratio in [0.7, 0.8, 0.9]:
    allocator = PreallocatedPagedTokenToKVPoolAllocator(
        # ... 其他参数 ...
        prealloc_ratio=ratio,
    )
    result = benchmark.benchmark_allocator(allocator, workload, f"Ratio-{ratio}")
    print(f"Ratio {ratio}: {result['ops_per_sec']:.0f} ops/s")
```

### 测试不同的桶配置

```python
configs = {
    "小桶": [1, 2, 4, 8, 16, 32],
    "大桶": [16, 32, 64, 128, 256],
    "均衡": [1, 2, 4, 8, 16, 32, 64, 128],
}

for name, buckets in configs.items():
    allocator = PreallocatedPagedTokenToKVPoolAllocator(
        # ... 其他参数 ...
        prealloc_bucket_sizes=buckets,
    )
    result = benchmark.benchmark_allocator(allocator, workload, name)
    print(f"{name}: {result['ops_per_sec']:.0f} ops/s")
```

## 保存测试结果

### 导出为JSON

```python
import json

# 在 main() 末尾添加
results = benchmark.results
with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 导出为CSV

```python
import csv

with open('benchmark_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Workload', 'Implementation', 'Ops/s', 'Avg Alloc (ms)', 'Avg Free (ms)'])

    for workload, results in benchmark.results.items():
        for impl, data in results.items():
            writer.writerow([
                workload,
                impl,
                f"{data['ops_per_sec']:.0f}",
                f"{data['avg_alloc_time']:.4f}",
                f"{data['avg_free_time']:.4f}"
            ])
```

## 持续集成

### 在CI中运行

```bash
# 在CI脚本中
python compare_allocators.py > benchmark_results.txt

# 检查是否有性能退化
python -c "
import json
with open('benchmark_results.json') as f:
    results = json.load(f)

for workload, data in results.items():
    speedup = data['optimized']['ops_per_sec'] / data['baseline']['ops_per_sec']
    if speedup < 1.1:
        print(f'⚠️ Performance regression in {workload}: {speedup:.2f}x')
        exit(1)
"
```

## 环境要求

### 最小要求

```bash
# PyTorch
pip install torch

# SGLang源码
# 确保在SGLang根目录运行
```

### 推荐环境

```bash
# CUDA环境以获得最佳性能
# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 完整SGLang依赖
pip install -e .
```

## 常见问题

### Q: 为什么我的结果和文档中的不一样？

**A:** 性能受多种因素影响：
- 硬件配置 (GPU型号, CPU, 内存)
- 工作负载特点
- 系统负载
- CUDA版本和驱动

### Q: 如何确保测试公平？

**A:** 脚本已包含以下措施：
- 预热阶段 (warmup)
- 相同的工作负载
- CUDA同步
- 多次运行取平均

### Q: 测试时间太长怎么办？

**A:** 减少操作数：

```python
# 减少请求数
chat_workload = generator.generate_chat_workload(num_requests=100)  # 默认500
```

### Q: 想测试特定的分配模式怎么办？

**A:** 直接创建工作负载：

```python
# 例如：只测试64 token的分配
workload = [('alloc', 64) for _ in range(1000)]
```

## 相关文档

- **QUICK_FIX_PERFORMANCE.md** - 快速性能修复指南
- **PERFORMANCE_OPTIMIZATION_GUIDE.md** - 完整性能优化指南
- **KV_CACHE_PREALLOCATION_README.md** - 技术文档
- **INTEGRATION_GUIDE.md** - 集成指南

## 总结

使用此对比脚本，你可以：

✅ 客观对比三种实现的性能
✅ 验证优化效果
✅ 找到最适合你工作负载的配置
✅ 持续监控性能退化

**推荐做法:**
1. 运行基准测试
2. 分析结果
3. 根据建议调整配置
4. 在生产环境验证
5. 定期重新测试
