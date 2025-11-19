# KV Cache预分配池扩展

## 概述

本实现为SGLang添加了KV Cache预分配池功能，参考了vLLM的BlockManager设计。通过预分配常用大小的KV块，减少分配开销并提高内存效率。

## 实现架构

### 核心组件

1. **PreallocatedKVBlockPool** (`python/sglang/srt/mem_cache/preallocated_pool.py`)
   - 管理多个大小的KV块池
   - 支持快速O(1)分配和释放
   - 支持块分割和合并
   - 提供详细的统计信息

2. **PreallocatedPagedTokenToKVPoolAllocator** (`python/sglang/srt/mem_cache/allocator.py`)
   - 扩展了PagedTokenToKVPoolAllocator
   - 集成预分配池和回退机制
   - 保持与现有API兼容

### 关键特性

#### 1. 多级桶(Bucket)策略

预分配池将页面组织成多个桶，每个桶包含固定大小的块：

```python
# 默认桶大小（页数）
bucket_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
```

这些桶涵盖了从小型decode操作（1页）到大型prefill操作（128页）的常见分配模式。

#### 2. 智能分配策略

分配算法：
1. 尝试从精确大小的桶分配
2. 如果没有精确匹配，从更大的桶分配
3. 启用块分割时，将较大的块分割成需要的大小和剩余部分
4. 剩余部分返回到对应大小的桶中

```python
# 示例：请求5页
# 1. 检查5页桶 -> 空
# 2. 检查8页桶 -> 有空闲块
# 3. 分割8页块 -> 返回5页，剩余3页
# 4. 将3页返回到3页桶
```

#### 3. 双池管理

PreallocatedPagedTokenToKVPoolAllocator维护两个池：

- **预分配池**: 管理80%的页面（可配置），用于快速分配
- **回退池**: 管理20%的页面，用于预分配池无法满足的请求

```python
allocator = PreallocatedPagedTokenToKVPoolAllocator(
    size=10000,
    page_size=16,
    prealloc_ratio=0.8,  # 80%用于预分配池
    enable_prealloc=True
)
```

#### 4. 统计和监控

提供详细的统计信息用于性能分析：

```python
stats = allocator.get_statistics()
# 包含:
# - total_pages: 总页数
# - available_pages: 可用页数
# - utilization: 利用率
# - bucket_allocations: 每个桶的分配次数
# - bucket_frees: 每个桶的释放次数
# - split_operations: 块分割次数
# - fallback_allocations: 回退分配次数
```

## 使用方法

### 基本使用

```python
from sglang.srt.mem_cache.preallocated_pool import PreallocatedKVBlockPool

# 创建预分配池
pool = PreallocatedKVBlockPool(
    total_pages=1000,
    page_size=16,
    device="cuda",
    bucket_sizes=[1, 2, 4, 8, 16, 32],
    enable_splitting=True,
    debug_mode=False
)

# 分配4页
pages = pool.allocate(4)

# 释放页面
pool.free(pages)

# 获取统计信息
stats = pool.get_statistics()
print(f"Utilization: {stats['utilization']:.2%}")
```

### 集成到现有系统

```python
from sglang.srt.mem_cache.allocator import PreallocatedPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

# 创建KV Cache
kvcache = MHATokenToKVPool(
    size=10000,
    page_size=16,
    dtype=torch.float16,
    head_num=32,
    head_dim=128,
    layer_num=32,
    device="cuda",
    enable_memory_saver=False
)

# 创建带预分配的allocator
allocator = PreallocatedPagedTokenToKVPoolAllocator(
    size=10000,
    page_size=16,
    dtype=torch.int64,
    device="cuda",
    kvcache=kvcache,
    need_sort=True,
    enable_prealloc=True,
    prealloc_bucket_sizes=[1, 2, 4, 8, 16, 32, 64],
    prealloc_ratio=0.8
)

# 使用allocator进行分配
indices = allocator.alloc(need_size=64)  # 分配64个token (4页)
allocator.free(indices)
```

### 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `total_pages` | int | - | 总页数 |
| `page_size` | int | - | 每页token数 |
| `device` | str | - | 设备("cuda"或"cpu") |
| `bucket_sizes` | List[int] | [1,2,4,8,16,32,64,128] | 桶大小列表 |
| `enable_splitting` | bool | True | 是否启用块分割 |
| `prealloc_ratio` | float | 0.8 | 预分配池比例(0.0-1.0) |
| `debug_mode` | bool | False | 是否启用调试模式 |

## 性能优势

### 1. 减少分配开销
- 预分配常用大小的块，避免频繁的页面查找和排序
- O(1)时间复杂度的分配操作

### 2. 降低内存碎片
- 按大小组织块，减少内存碎片
- 块分割机制确保内存高效利用

### 3. 提高缓存命中率
- 常用大小的块预先分配，提高缓存命中率
- 减少与内核态的交互

### 4. 可预测的性能
- 预分配确保内存可用性
- 避免运行时分配失败

## 与vLLM BlockManager的对比

| 特性 | vLLM BlockManager | SGLang PreallocatedPool |
|------|-------------------|------------------------|
| 块组织 | 固定大小块 | 多级桶策略 |
| 分配策略 | 单一大小 | 多大小支持+分割 |
| 内存管理 | 全局哈希表 | 分级池+回退机制 |
| 前缀缓存 | 自动哈希匹配 | 由RadixCache处理 |
| 统计信息 | 基础统计 | 详细的按桶统计 |

## 测试

运行测试套件：

```bash
cd /home/user/sglang
python python/sglang/srt/mem_cache/test_preallocated_pool.py
```

测试包括：
1. 基本分配和释放
2. 块分割功能
3. 压力测试
4. 清空功能

## 调试

启用调试模式可以获得详细的日志输出：

```python
pool = PreallocatedKVBlockPool(
    total_pages=1000,
    page_size=16,
    device="cuda",
    debug_mode=True  # 启用调试
)
```

或者通过环境变量：

```bash
export SGLANG_DEBUG_MEMORY_POOL=1
```

调试模式将输出：
- 每次分配和释放的详细信息
- 块分割操作
- 桶状态变化
- 断言检查

## 未来改进方向

1. **块合并**: 实现相邻块的自动合并，进一步减少碎片
2. **自适应桶大小**: 根据实际使用模式动态调整桶大小
3. **层级缓存**: 支持GPU-CPU-SSD多级缓存
4. **NUMA感知**: 在多GPU系统中优化NUMA访问
5. **预热机制**: 启动时预热常用大小的块

## 贡献者

- 实现参考了vLLM项目的BlockManager设计
- 基于SGLang现有的PagedTokenToKVPoolAllocator扩展

## 许可证

Apache License 2.0

## 相关文件

- `python/sglang/srt/mem_cache/preallocated_pool.py` - 预分配池实现
- `python/sglang/srt/mem_cache/allocator.py` - 集成的allocator
- `python/sglang/srt/mem_cache/test_preallocated_pool.py` - 测试套件
- `python/sglang/srt/mem_cache/memory_pool.py` - KV Cache池实现
- `python/sglang/srt/mem_cache/radix_cache.py` - Radix Cache实现
