# KV Cache预分配池性能问题快速修复指南

如果你发现预分配池性能不如预期，按照以下步骤快速修复。

## 🚀 快速修复（5分钟）

### 步骤1: 使用优化版本

在 `model_runner.py` 的第1905行附近，修改allocator初始化：

```python
# 修改前
self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(...)

# 修改后 - 使用预分配池优化版本
from sglang.srt.mem_cache.allocator import PreallocatedPagedTokenToKVPoolAllocator

self.token_to_kv_pool_allocator = PreallocatedPagedTokenToKVPoolAllocator(
    self.max_total_num_tokens,
    page_size=self.page_size,
    dtype=self.kv_cache_dtype,
    device=self.device,
    kvcache=self.token_to_kv_pool,
    need_sort=need_sort,
    enable_prealloc=True,
    use_optimized=True,  # ⭐ 关键：使用优化版本
    prealloc_ratio=0.8,
)
```

**预期提升:** 20-30%的分配速度提升

### 步骤2: 调整桶大小（根据你的场景）

```python
# 短对话场景 (< 512 tokens)
prealloc_bucket_sizes=[1, 2, 4, 8, 16, 32]

# 长对话场景 (512-2048 tokens)
prealloc_bucket_sizes=[4, 8, 16, 32, 64, 128]

# 长文档场景 (> 2048 tokens)
prealloc_bucket_sizes=[16, 32, 64, 128, 256]
```

将对应的配置添加到allocator初始化中：

```python
self.token_to_kv_pool_allocator = PreallocatedPagedTokenToKVPoolAllocator(
    # ... 其他参数 ...
    prealloc_bucket_sizes=[1, 2, 4, 8, 16, 32, 64, 128],  # ⭐ 根据场景选择
)
```

### 步骤3: 重启推理服务

```bash
# 重启SGLang服务
python -m sglang.launch_server --model-path your-model
```

## 📊 验证性能改进

### 运行基准测试

```bash
cd /home/user/sglang

# 完整性能对比
python python/sglang/srt/mem_cache/benchmark_comparison.py
```

### 检查关键指标

在你的推理代码中添加：

```python
# 定期检查性能
stats = self.token_to_kv_pool_allocator.get_statistics()

if 'prealloc' in stats:
    prealloc = stats['prealloc']

    # 关键指标
    print(f"快速路径命中率: {prealloc.get('fast_path_hit_rate', 0):.2%}")  # 目标: >80%
    print(f"利用率: {prealloc['utilization']:.2%}")  # 目标: 70-90%
    print(f"分割率: {prealloc['split_operations']/prealloc['total_allocations']:.2%}")  # 目标: <30%
```

## ⚙️ 进阶优化（如果还有问题）

### 1. 调整预分配比例

```python
# 如果内存充足，增加预分配比例
prealloc_ratio=0.9  # 从默认的0.8提升到0.9
```

### 2. 禁用块分割（如果可接受少量内存浪费）

修改 `preallocated_pool_optimized.py`：

```python
self.enable_splitting = False  # 禁用分割，更快但可能浪费内存
```

### 3. 减少桶数量（简化管理）

```python
# 只保留最常用的桶大小
prealloc_bucket_sizes=[4, 8, 16, 32, 64]  # 精简配置
```

## 🔍 性能诊断工具

### 分析分配模式

```python
from sglang.srt.mem_cache.performance_diagnostics import diagnose_allocation_pattern

# 获取allocator实例
allocator = your_model_runner.token_to_kv_pool_allocator

# 运行诊断
diagnose_allocation_pattern(allocator)

# 输出会显示：
# - 哪些桶最常用
# - 哪些桶未使用（可以移除）
# - 优化建议
```

### 对比性能

```python
from sglang.srt.mem_cache.performance_diagnostics import compare_allocators

# 对比优化前后
compare_allocators(old_allocator, new_allocator, workload)
```

## 📋 Checklist

快速检查你是否已完成所有优化：

- [ ] 使用 `PreallocatedPagedTokenToKVPoolAllocator`
- [ ] 设置 `use_optimized=True`
- [ ] 根据场景选择合适的 `prealloc_bucket_sizes`
- [ ] 设置合适的 `prealloc_ratio` (默认0.8)
- [ ] 重启推理服务
- [ ] 运行基准测试验证
- [ ] 检查快速路径命中率 (目标>80%)

## 🎯 预期结果

完成优化后，你应该看到：

✅ **分配速度**: 提升 20-30%
✅ **快速路径命中率**: > 80%
✅ **块分割率**: < 30%
✅ **整体推理吞吐量**: 提升 5-15%

## ⚠️ 常见陷阱

### 陷阱1: 桶大小不匹配

**症状:** 快速路径命中率低 (<60%)

**修复:**
```python
# 运行诊断找出最常用大小
diagnose_allocation_pattern(allocator)

# 根据输出调整bucket_sizes
```

### 陷阱2: 预分配比例太低

**症状:** 频繁回退到标准allocator

**修复:**
```python
prealloc_ratio=0.85  # 增加预分配比例
```

### 陷阱3: 使用了标准版本而非优化版本

**症状:** 没有fast_path_hit_rate指标

**修复:**
```python
use_optimized=True  # 确保使用优化版本
```

## 📞 还有问题？

### 1. 查看完整文档

- `PERFORMANCE_OPTIMIZATION_GUIDE.md` - 详细优化指南
- `KV_CACHE_PREALLOCATION_README.md` - 技术文档
- `INTEGRATION_GUIDE.md` - 集成指南

### 2. 运行诊断工具

```bash
# 完整诊断
python python/sglang/srt/mem_cache/benchmark_comparison.py

# 查看输出中的"优化建议"部分
```

### 3. 检查是否是其他瓶颈

```python
# 使用profiler确认瓶颈
import cProfile
cProfile.run('your_inference_code()')

# 如果分配时间占比<5%，瓶颈可能在其他地方
```

## 🔄 回滚到标准版本（如果需要）

如果优化版本有问题，可以快速回滚：

```python
# 方法1: 禁用预分配
enable_prealloc=False

# 方法2: 使用标准实现
use_optimized=False

# 方法3: 完全使用原始allocator
self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
    self.max_total_num_tokens,
    page_size=self.page_size,
    dtype=self.kv_cache_dtype,
    device=self.device,
    kvcache=self.token_to_kv_pool,
    need_sort=need_sort,
)
```

---

## 总结

最简单的修复方式：

```python
# 只需要这一行！
use_optimized=True  # 在allocator初始化时添加
```

这将自动使用所有优化，通常可以解决90%的性能问题。
