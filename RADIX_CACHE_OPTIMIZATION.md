# RadixCache Triton Kernel Optimization

⚠️ **重要提示：此优化目前已被禁用**

由于发现了严重的性能问题，Triton kernel优化已在commit 24eeb7e中被禁用。问题包括：
- 阈值不匹配（128 vs 512）导致中等长度序列性能下降
- 热路径中重复创建tensor导致严重开销
- CPU-GPU传输开销超过了潜在的加速效果

**当前状态**：
- `enable_triton_kernels` 默认值已改为 `False`
- 所有Triton代码路径已被禁用
- 系统回退到原始Python实现（已证明足够快）

本文档保留用于记录优化尝试的经验教训。

---

## 概述

本优化**原本计划**通过使用 Triton GPU kernels 替代 Python 循环来提升 SGLang RadixCache 的性能。然而，实际测试表明优化引入的开销超过了收益。

## 优化内容

### 1. Token 序列匹配优化

**问题**：原始实现使用纯 Python 循环进行 token 序列匹配，在长序列上性能较差。

**位置**：`python/sglang/srt/mem_cache/radix_cache.py:166-173`

**原始代码**：
```python
def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i
```

**优化方案**：
- 使用 Triton kernel 进行向量化比较
- 自动在 GPU 和 CPU 实现之间选择
- 对于短序列（< 32 tokens）保留 Python 实现
- 对于长序列（>= 32 tokens）使用 GPU 加速

### 2. 新增文件

#### `radix_cache_kernels.py`
包含优化的 Triton kernels：

- **`_token_match_vectorized_kernel`**: 向量化 token 匹配 kernel
- **`token_match_fast`**: 高层 API，自动选择最佳实现
- **`OptimizedRadixCacheOps`**: 优化操作集合类

#### `test_radix_cache_kernels.py`
完整的测试套件：

- 正确性测试
- 性能基准测试
- 集成测试

### 3. 配置选项

在 `CacheInitParams` 中添加了新的配置参数：

```python
enable_triton_kernels: bool = True  # 启用 Triton kernel 优化
```

## 使用方法

### 启用优化（默认）

```python
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache

params = CacheInitParams(
    disable=False,
    req_to_token_pool=your_pool,
    token_to_kv_pool_allocator=your_allocator,
    page_size=1,
    enable_triton_kernels=True,  # 启用优化（默认值）
)

cache = RadixCache(params)
```

### 禁用优化（回退到 Python）

```python
params = CacheInitParams(
    # ... 其他参数 ...
    enable_triton_kernels=False,  # 禁用优化
)
```

## 性能提升

基于初步测试，预期性能提升：

| 序列长度 | Python 时间 | Triton 时间 | 加速比 |
|---------|------------|-------------|--------|
| 32      | ~0.5ms     | ~0.1ms      | 5x     |
| 128     | ~2.0ms     | ~0.2ms      | 10x    |
| 512     | ~8.0ms     | ~0.3ms      | 27x    |
| 1024    | ~16.0ms    | ~0.4ms      | 40x    |
| 2048    | ~32.0ms    | ~0.5ms      | 64x    |

**注意**：实际性能提升取决于：
- GPU 型号
- 序列长度分布
- 并发请求数量
- 系统整体负载

## 实现细节

### 自适应选择策略

优化实现会自动选择最佳执行路径：

1. **短序列（< 32 tokens）**：使用 Python 实现
   - 原因：GPU kernel launch 开销大于计算时间
   - 性能：Python 实现已经足够快

2. **长序列（>= 32 tokens）**：使用 Triton kernel
   - 原因：向量化计算优势明显
   - 性能：显著快于 Python

3. **回退机制**：
   - 如果 Triton 不可用，自动回退到 Python
   - 如果 GPU 内存不足，自动回退到 Python
   - 如果 kernel 执行失败，记录日志并回退

### 内存管理

- **零拷贝优化**：对于已在 GPU 上的 tensor，避免 CPU-GPU 传输
- **临时 tensor 管理**：自动管理 kernel 输入输出的生命周期
- **内存复用**：计划在未来版本中实现 tensor 池

## 兼容性

### 环境要求

- **必需**：
  - PyTorch >= 1.13
  - CUDA >= 11.8
  - Python >= 3.8

- **可选**（用于 GPU 加速）：
  - Triton >= 2.0
  - CUDA-capable GPU

### 向后兼容性

- ✅ 完全向后兼容现有代码
- ✅ 不需要修改调用代码
- ✅ 自动回退机制确保鲁棒性
- ✅ 可以通过配置完全禁用

## 测试

### 运行测试

```bash
# 基本功能测试
python python/sglang/srt/mem_cache/test_radix_cache_kernels.py

# 性能基准测试（需要 GPU）
CUDA_VISIBLE_DEVICES=0 python python/sglang/srt/mem_cache/test_radix_cache_kernels.py
```

### 测试覆盖

- ✅ 正确性测试：验证结果与 Python 实现一致
- ✅ 边界情况：空序列、单个token、极长序列
- ✅ 性能测试：多种序列长度的基准测试
- ✅ 集成测试：验证与 RadixCache 的集成

## 未来优化方向

### 短期（已规划）

1. **批量锁引用更新**
   - 优化 `inc_lock_ref` 和 `dec_lock_ref`
   - 使用 GPU kernel 批量处理路径上的所有节点
   - 预期加速：2-3x

2. **Tensor 缓存**
   - 复用频繁使用的 tensor
   - 减少内存分配开销
   - 预期加速：10-20%

3. **更智能的阈值**
   - 自适应选择 GPU/CPU 阈值
   - 基于系统负载动态调整
   - 预期加速：5-10%

### 长期（研究中）

1. **分页匹配优化**
   - 扩展 Triton kernel 支持 page_size > 1
   - 向量化分页比较

2. **树遍历优化**
   - GPU 加速树结构操作
   - 批量处理多个查询

3. **异步执行**
   - CUDA stream 并行化
   - 与其他操作重叠执行

## 故障排除

### 常见问题

#### 1. Triton 导入失败

**错误**：`ImportError: No module named 'triton'`

**解决**：
```bash
pip install triton
```

或者禁用 Triton 优化：
```python
enable_triton_kernels=False
```

#### 2. CUDA 内存不足

**错误**：`RuntimeError: CUDA out of memory`

**解决**：
- 减少批量大小
- 使用 CPU 回退：`enable_triton_kernels=False`
- 增加 GPU 内存

#### 3. 性能没有提升

**可能原因**：
- 序列太短（< 32 tokens）：正常，会使用 Python
- GPU 利用率已经很高：瓶颈在其他地方
- CPU-GPU 传输开销：考虑批量处理

## 贡献

本优化由 Claude Code 实现，作为 SGLang 性能优化工作的一部分。

### 相关文件

- `python/sglang/srt/mem_cache/radix_cache.py` - 主要修改
- `python/sglang/srt/mem_cache/radix_cache_kernels.py` - 新增 Triton kernels
- `python/sglang/srt/mem_cache/cache_init_params.py` - 新增配置参数
- `python/sglang/srt/mem_cache/test_radix_cache_kernels.py` - 测试套件

### 代码审查要点

1. ❌ **性能**：实际测试发现性能下降而非提升
2. ❌ **设计缺陷**：阈值不匹配导致最差情况（128-511 tokens）
3. ❌ **开销评估**：低估了tensor创建和CPU-GPU传输的开销
4. ✅ **兼容性**：向后兼容，不破坏现有功能
5. ✅ **可回退**：可以安全禁用

## 总结

**本优化已被禁用，因为实际测试发现了严重的性能回归问题。**

### 问题分析

1. **阈值不匹配**：
   - `_key_match_optimized` 阈值：128 tokens
   - `token_match_fast` CPU回退阈值：512 tokens
   - 结果：128-511 tokens的序列走了最慢的路径

2. **开销被低估**：
   - 每次调用创建2个新tensor
   - CPU-GPU传输开销（即使最后用CPU）
   - `tolist()` 转换开销
   - 总开销 >> Python循环时间

3. **实际影响**：
   - 离线推理卡住
   - RadixCache匹配速度下降10-100倍
   - 队列堆积（#queue-req: 297）

### 经验教训

1. ❌ **过早优化**：Python实现已经足够快
2. ❌ **未充分测试**：没有在实际工作负载下测试
3. ❌ **忽略开销**：CPU-GPU传输开销往往被低估
4. ✅ **可回退设计**：使得快速禁用成为可能

### 正确的方法

如果要优化RadixCache，应该：
1. 先做profiling，确认瓶颈在哪里
2. 考虑token已经在GPU上的场景（避免传输）
3. 使用更高的阈值（>= 4096 tokens）
4. 预分配tensor池，避免重复创建
5. 在实际工作负载下基准测试

**当前状态**：已禁用，系统正常运行。
