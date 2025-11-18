# SGLang 推理性能优化分析报告

## 执行摘要

本报告深入分析了SGLang的代码架构，对比了vLLM、TensorRT-LLM等业界领先推理框架的优化技术，识别出**8个高收益、低侵入性**的优化机会。这些优化预计可带来**15-40%**的性能提升，且大部分可以独立实施。

---

## 1. 核心架构分析

### 1.1 SGLang当前架构

SGLang采用三进程架构：
- **TokenizerManager**: 处理tokenization
- **Scheduler**: 请求调度、批次管理、模型执行
- **DetokenizerManager**: 处理detokenization

**关键发现**:
- Scheduler承担了调度和执行双重职责
- CPU密集型任务（调度、策略计算）与GPU执行在同一进程
- 通过ZMQ进行进程间通信

### 1.2 现有优化特性

SGLang已实现的优化：
- ✅ **RadixAttention** (Prefix Caching)
- ✅ **Continuous Batching**
- ✅ **Chunked Prefill**
- ✅ **CUDA Graph** (包括分段CUDA Graph)
- ✅ **Speculative Decoding** (EAGLE, N-gram)
- ✅ **Hierarchical Cache** (HiCache)
- ✅ **DP Attention** (用于DeepSeek等大模型)
- ✅ **Expert Parallelism**

---

## 2. 业界最佳实践对比

### 2.1 vLLM V1引擎的关键创新

| 优化技术 | 性能提升 | SGLang现状 |
|---------|---------|-----------|
| **进程分离** (Scheduler独立进程) | 1.7x | ❌ 未实现 |
| **Persistent Batch** (缓存输入tensor) | 15-20% | ❌ 未实现 |
| **零开销Prefix Caching** | 0% overhead | ⚠️ 有优化空间 |
| **Numpy操作替代Python** | 10-15% CPU | ⚠️ 部分实现 |
| **多模态异步预处理** | 非阻塞 | ⚠️ 待验证 |

### 2.2 TensorRT-LLM的优化

| 优化技术 | 适用场景 | SGLang现状 |
|---------|---------|-----------|
| **Kernel Fusion** | 所有阶段 | ⚠️ 部分kernel可融合 |
| **Chunked Prefill** | 长上下文 | ✅ 已实现 |
| **自适应Chunk Size** | 动态负载 | ❌ 固定chunk size |

### 2.3 POD-Attention (ASPLOS'25)

- **Prefill-Decode Overlap**: 在单个forward pass中同时处理prefill和decode
- **性能提升**: Attention计算快59%，端到端吞吐量提升22%
- **SGLang现状**: ❌ 未实现（有`enable_mixed_chunk`但不是完整的POD）

---

## 3. 性能瓶颈识别

### 3.1 CPU同步开销

**位置**: `schedule_batch.py:1737`
```python
self.seq_lens_sum = self.seq_lens.sum().item()  # GPU->CPU同步
```

**影响**:
- 每个batch都调用`.item()`导致GPU-CPU同步
- 阻塞GPU执行直到tensor计算完成

**测量方法**: 在`schedule_batch.py`中添加
```python
# 当前代码
self.seq_lens_sum = self.seq_lens.sum().item()
```

### 3.2 调度器CPU开销

**位置**: `schedule_policy.py:145-147`
```python
def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
    if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
        # Turn off the expensive prefix matching and sorting when the #queue is large.
        return CacheAgnosticPolicy.FCFS
```

**问题**:
- LPM策略在队列>128时自动降级为FCFS
- Prefix matching在Python中实现，CPU开销大
- 排序算法复杂度O(n log n)

### 3.3 批处理数据准备开销

**位置**: `schedule_batch.py:1806-1874 (get_model_worker_batch)`

**问题**:
- 每次forward都创建新的`ModelWorkerBatch`对象
- 大量Python对象创建和字段复制
- List comprehension: `lora_ids=[req.lora_id for req in self.reqs]`

### 3.4 Python对象创建开销

**位置**: `schedule_batch.py:1876-1895 (copy方法)`

**问题**:
- 在overlap模式下，每个batch都要复制
- 创建新的dataclass实例，Python对象创建成本高

### 3.5 Tensor拼接开销

**分析**: 多处使用`torch.cat()`和`torch.stack()`
- 每次调用都分配新内存
- 可能导致内存碎片

---

## 4. 优化建议（按优先级排序）

### 🔥 优先级1: 减少GPU-CPU同步 (预计收益: 8-15%)

#### 优化4.1: 延迟同步和批量处理

**当前问题**:
```python
# schedule_batch.py:1737
self.seq_lens_sum = self.seq_lens.sum().item()  # 立即同步
```

**优化方案**:
```python
# 方案1: 保持GPU tensor，延迟到真正需要时再同步
self.seq_lens_sum_tensor = self.seq_lens.sum()  # 保留在GPU
# 只在必要时（如logging）才调用 .item()

# 方案2: 使用CUDA Stream异步复制
self.seq_lens_sum_cpu = torch.empty(1, dtype=torch.int64, pin_memory=True)
self.seq_lens.sum().to(self.seq_lens_sum_cpu, non_blocking=True)
```

**实施难度**: 🟢 低
- 代码改动小（约10-20行）
- 主要修改`schedule_batch.py`中的几处`.item()`调用
- 需要识别哪些地方真正需要CPU值

**兼容性**: 🟢 高
- 不改变外部API
- 向后兼容

**相关代码位置**:
- `python/sglang/srt/managers/schedule_batch.py:1737`
- 搜索所有`.item()`和`.tolist()`调用

---

### 🔥 优先级2: Persistent Batch优化 (预计收益: 15-20%)

#### 优化4.2: 缓存和复用Batch Tensors

**灵感来源**: vLLM V1的Persistent Batch技术

**当前问题**:
```python
# 每次forward都创建新的ModelWorkerBatch对象
def get_model_worker_batch(self) -> ModelWorkerBatch:
    return ModelWorkerBatch(
        forward_mode=self.forward_mode,
        input_ids=self.input_ids,
        req_pool_indices=self.req_pool_indices,
        # ... 20多个字段
    )
```

**优化方案**:
```python
class ScheduleBatch:
    def __init__(self, ...):
        # 预分配persistent batch对象
        self._persistent_model_worker_batch = None
        self._max_batch_size = 512  # 可配置

        # 预分配tensor缓冲区
        self._input_ids_buffer = torch.empty(
            self._max_batch_size, dtype=torch.int64, device=device
        )
        self._req_pool_indices_buffer = torch.empty(
            self._max_batch_size, dtype=torch.int32, device=device
        )
        # ... 其他tensor缓冲区

    def get_model_worker_batch(self) -> ModelWorkerBatch:
        batch_size = len(self.reqs)

        # 复用缓冲区，只更新变化的部分
        if batch_size <= self._input_ids_buffer.size(0):
            # In-place更新
            self._input_ids_buffer[:batch_size].copy_(self.input_ids)
        else:
            # 扩容（罕见）
            self._resize_buffers(batch_size)

        # 复用ModelWorkerBatch对象，只更新字段
        if self._persistent_model_worker_batch is None:
            self._persistent_model_worker_batch = ModelWorkerBatch(...)
        else:
            # 只更新变化的字段
            self._persistent_model_worker_batch.forward_mode = self.forward_mode
            self._persistent_model_worker_batch.input_ids = self._input_ids_buffer[:batch_size]
            # ...

        return self._persistent_model_worker_batch
```

**实施难度**: 🟡 中
- 需要重构`get_model_worker_batch()`方法
- 约100-150行代码
- 需要仔细管理对象生命周期

**兼容性**: 🟢 高
- API保持不变
- 内部实现优化

**相关代码位置**:
- `python/sglang/srt/managers/schedule_batch.py:1806-1874`

---

### 🔥 优先级3: 优化调度器CPU开销 (预计收益: 10-15%)

#### 优化4.3: 使用Numpy/Tensor操作替代Python循环

**灵感来源**: vLLM V1大量使用Numpy操作减少Python开销

**当前问题**: 调度策略中大量Python操作
```python
# schedule_policy.py
# 当前：纯Python实现
for req in waiting_queue:
    prefix_len = self.tree_cache.match_prefix(req.input_ids)
    req.prefix_len = prefix_len
```

**优化方案**:
```python
# 方案1: 向量化prefix matching（在可行的情况下）
# 将多个请求的prefix matching批量处理

# 方案2: C++/Triton kernel实现热点路径
# 特别是对于LPM策略的prefix matching

# 方案3: 缓存计算结果
class SchedulePolicy:
    def __init__(self):
        self._prefix_len_cache = {}  # 缓存最近计算的prefix length
        self._cache_hits = 0
        self._cache_misses = 0

    def calc_priority(self, waiting_queue: List[Req]):
        # 检查缓存
        for req in waiting_queue:
            cache_key = hash(tuple(req.input_ids))  # 或使用更高效的key
            if cache_key in self._prefix_len_cache:
                req.prefix_len = self._prefix_len_cache[cache_key]
                self._cache_hits += 1
            else:
                req.prefix_len = self.tree_cache.match_prefix(req.input_ids)
                self._prefix_len_cache[cache_key] = req.prefix_len
                self._cache_misses += 1
```

**实施难度**: 🟡 中
- 需要profiling识别热点
- 向量化可能受限于算法特性
- 缓存方案实现相对简单

**兼容性**: 🟢 高
- 不改变调度行为
- 只优化实现效率

**相关代码位置**:
- `python/sglang/srt/managers/schedule_policy.py:105-143`
- `python/sglang/srt/mem_cache/radix_cache.py` (match_prefix方法)

---

### 🔥 优先级4: Kernel Fusion (预计收益: 8-12%)

#### 优化4.4: 融合Sampler中的操作

**灵感来源**: TensorRT-LLM的kernel fusion

**当前问题**:
```python
# sampler.py
# 多个独立的kernel调用
logits = self._preprocess_logits(logits, sampling_info)  # Kernel 1
logprobs = torch.nn.functional.log_softmax(logits, dim=-1)  # Kernel 2
batch_next_token_ids = torch.argmax(logits, -1)  # Kernel 3
```

**优化方案**:
```python
# 融合为单个Triton kernel
# sgl-kernel/csrc/sampling/fused_sample_kernel.cu (新文件)

# 融合以下操作：
# 1. Logits预处理（bias, temperature）
# 2. Softmax计算
# 3. Top-k/top-p采样
# 4. Logprob计算

@triton.jit
def fused_sample_kernel(
    logits_ptr, bias_ptr, output_ptr, logprob_ptr,
    temperature, top_k, top_p,
    BLOCK_SIZE: tl.constexpr
):
    # 1. 加载logits + bias
    # 2. 应用temperature
    # 3. 计算softmax (online算法)
    # 4. 执行top-k/top-p
    # 5. 采样
    # 6. 计算logprob
    # 全部在一次kernel调用中完成
```

**实施难度**: 🔴 高
- 需要编写CUDA/Triton kernel
- 约300-500行kernel代码
- 需要处理各种采样策略

**兼容性**: 🟢 高
- 通过flag控制启用/禁用
- 可与现有实现并存

**替代方案**: 🟡 中等难度
- 先融合贪婪采样场景（最常见）
- 逐步扩展到其他采样策略

**相关代码位置**:
- `python/sglang/srt/layers/sampler.py:64-180`
- 新建: `sgl-kernel/csrc/sampling/fused_sample_kernel.cu`

---

### 🟡 优先级5: 改进Chunked Prefill (预计收益: 5-10%)

#### 优化4.5: 自适应Chunk Size

**灵感来源**: TensorRT-LLM的自适应chunk size

**当前问题**:
- Chunk size固定（`--chunked-prefill-size=8192`）
- 不考虑当前GPU负载和内存状态
- 可能导致：
  - Chunk过大 → OOM
  - Chunk过小 → 效率低下

**优化方案**:
```python
class AdaptiveChunkSizer:
    def __init__(self, base_chunk_size=8192):
        self.base_chunk_size = base_chunk_size
        self.min_chunk_size = 2048
        self.max_chunk_size = 16384

        # 自适应因子
        self.memory_pressure_threshold = 0.85
        self.utilization_target = 0.90

    def compute_chunk_size(
        self,
        prefill_length: int,
        current_memory_usage: float,
        num_running_requests: int,
        available_kv_slots: int
    ) -> int:
        chunk_size = self.base_chunk_size

        # 1. 根据内存压力调整
        if current_memory_usage > self.memory_pressure_threshold:
            chunk_size = int(chunk_size * 0.7)  # 减小30%

        # 2. 根据当前负载调整
        if num_running_requests > 100:
            chunk_size = int(chunk_size * 0.8)  # 更多小chunk以提高响应性

        # 3. 根据输入长度优化
        if prefill_length < chunk_size * 1.5:
            # 避免将短prefill分成多个chunk
            chunk_size = prefill_length

        # 4. 确保chunk size是2的幂（有利于GPU效率）
        chunk_size = next_power_of_2(chunk_size)

        return max(self.min_chunk_size, min(chunk_size, self.max_chunk_size))
```

**实施难度**: 🟡 中
- 约50-80行代码
- 需要集成到scheduler中
- 需要实验调优启发式规则

**兼容性**: 🟢 高
- 通过flag控制: `--enable-adaptive-chunking`
- 默认使用固定chunk size

**相关代码位置**:
- `python/sglang/srt/managers/scheduler.py` (添加自适应逻辑)
- `python/sglang/srt/server_args.py` (新增配置参数)

---

### 🟡 优先级6: 优化Prefix Cache数据结构 (预计收益: 5-8%)

#### 优化4.6: Zero-overhead Cache Eviction

**灵感来源**: vLLM V1的constant-time cache eviction

**当前问题**: `radix_cache.py`
```python
class RadixCache:
    def evict(self):
        # 当前实现可能需要遍历树找到最佳驱逐候选
        # O(n)复杂度，在cache满时会成为瓶颈
```

**优化方案**:
```python
class FastRadixCache(RadixCache):
    def __init__(self, ...):
        super().__init__(...)

        # 添加双向链表用于O(1) LRU
        from collections import OrderedDict
        self._lru_list = OrderedDict()  # key -> TreeNode

        # 添加优先队列用于LFU
        import heapq
        self._lfu_heap = []  # [(hit_count, timestamp, node_id)]

    def evict_lru_fast(self) -> TreeNode:
        # O(1) LRU驱逐
        if not self._lru_list:
            return None
        _, node = self._lru_list.popitem(last=False)  # FIFO
        return node

    def evict_lfu_fast(self) -> TreeNode:
        # O(log n) LFU驱逐
        if not self._lfu_heap:
            return None
        _, _, node_id = heapq.heappop(self._lfu_heap)
        return self._node_map[node_id]

    def access_node(self, node: TreeNode):
        # 更新LRU
        if node.key in self._lru_list:
            self._lru_list.move_to_end(node.key)
        else:
            self._lru_list[node.key] = node

        # 更新LFU（延迟更新策略以减少heap操作）
        node.hit_count += 1
```

**实施难度**: 🟡 中
- 需要重构`RadixCache`类
- 约100-150行代码
- 需要维护额外的数据结构

**兼容性**: 🟢 高
- 可作为新的cache实现选项
- 不影响现有代码

**相关代码位置**:
- `python/sglang/srt/mem_cache/radix_cache.py:100-300`

---

### 🟡 优先级7: 异步KV Cache预取 (预计收益: 10-15%, 长上下文场景)

#### 优化4.7: L2 Cache导向的预取

**灵感来源**: 最新研究 - "Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching"

**原理**:
- 在计算attention时，预取下一步需要的KV cache到L2 cache
- 使用CUDA Stream并行预取和计算
- 特别适合长上下文场景（> 8K tokens）

**优化方案**:
```python
# 在flashinfer_backend.py中添加
class PrefetchingFlashInferBackend(FlashInferAttnBackend):
    def __init__(self, ...):
        super().__init__(...)
        self.prefetch_stream = torch.cuda.Stream()
        self.prefetch_distance = 2  # 提前预取2步

    def forward_decode(
        self, q, k_cache, v_cache, forward_batch, ...
    ):
        current_layer = forward_batch.current_layer

        # 异步预取下一层的KV cache
        if current_layer + self.prefetch_distance < self.num_layers:
            with torch.cuda.stream(self.prefetch_stream):
                next_k = k_cache[current_layer + self.prefetch_distance]
                next_v = v_cache[current_layer + self.prefetch_distance]
                # 触发预取到L2 cache
                _ = next_k[0, 0]  # 访问触发prefetch
                _ = next_v[0, 0]

        # 当前层的attention计算
        output = super().forward_decode(q, k_cache, v_cache, ...)

        # 同步预取流（如果需要）
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)

        return output
```

**实施难度**: 🔴 高
- 需要深入理解CUDA Stream和内存层次
- 需要careful tuning预取距离
- 约200-300行代码

**兼容性**: 🟢 高
- 作为可选backend
- `--attention-backend flashinfer_prefetch`

**适用场景**:
- ✅ 长上下文推理（> 8K tokens）
- ✅ 大batch size（> 32）
- ❌ 小模型、短上下文（收益有限）

**相关代码位置**:
- `python/sglang/srt/layers/attention/flashinfer_backend.py`
- 新建: `flashinfer_prefetch_backend.py`

---

### 🟠 优先级8: POD-Attention实现 (预计收益: 15-22%, 高难度)

#### 优化4.8: Prefill-Decode Overlap

**灵感来源**: ASPLOS'25论文 "POD-Attention"

**原理**:
- 在单个attention kernel中同时处理prefill和decode请求
- 消除prefill/decode切换的overhead
- 优化GPU资源利用率

**挑战**:
- 需要实现复杂的混合attention kernel
- Prefill和decode的计算模式差异大
- 需要careful的内存管理

**实施建议**:
- 这是一个**长期项目**（2-3个月）
- 建议先实施优先级1-6的优化
- 需要专门的kernel开发团队

**简化版方案** (更容易实施):
```python
# 改进现有的enable_mixed_chunk
# 当前：简单地在同一batch中混合prefill和decode
# 改进：优化kernel调用模式，减少切换overhead

class MixedBatchOptimizer:
    def optimize_batch_order(self, prefill_reqs, decode_reqs):
        # 智能排序，最小化kernel切换
        # 1. 按序列长度分组
        # 2. Prefill请求按长度排序（长->短）
        # 3. 交织prefill和decode以保持GPU饱和
        pass
```

**相关代码位置**:
- `python/sglang/srt/model_executor/model_runner.py:2179-2268`
- 新建: `sgl-kernel/csrc/attention/pod_attention_kernel.cu`

---

## 5. 实施路线图

### 阶段1: 快速胜利 (1-2周)
- ✅ 优化4.1: 减少GPU-CPU同步
- ✅ 优化4.3: 缓存版本的调度优化
- **预期收益**: 10-18%

### 阶段2: 中等改进 (3-4周)
- ✅ 优化4.2: Persistent Batch
- ✅ 优化4.5: 自适应Chunk Size
- ✅ 优化4.6: 快速Cache Eviction
- **预期收益**: 额外15-20%

### 阶段3: 高级优化 (2-3个月)
- ✅ 优化4.4: Kernel Fusion（从greedy sampling开始）
- ✅ 优化4.7: 异步KV Cache预取
- ✅ 优化4.8: POD-Attention（可选，长期项目）
- **预期收益**: 额外10-15%

### 总计预期收益
- **保守估计**: 25-35% (阶段1+2)
- **乐观估计**: 35-50% (阶段1+2+3)

---

## 6. 具体实施建议

### 6.1 开发流程

1. **建立Baseline**
   ```bash
   # 运行benchmark获取baseline
   python3 bench_serving.py --dataset-name random \
     --num-prompts 1000 --request-rate 10 \
     --backend sglang --port 30000

   # 记录关键指标
   # - Throughput (token/s)
   # - TTFT (ms)
   # - ITL (ms)
   # - P50/P90/P99 latency
   ```

2. **逐个实施优化**
   - 每个优化独立分支
   - 实施前后都运行相同的benchmark
   - 记录性能变化

3. **Profiling验证**
   ```bash
   # 使用PyTorch Profiler
   SGLANG_TORCH_PROFILER_DIR=/tmp/profile python3 ...

   # 使用Nsight Systems
   nsys profile -o profile.qdrep \
     python3 -m sglang.launch_server ...
   ```

### 6.2 性能测试矩阵

| 场景 | Batch Size | Context Length | QPS | 优化重点 |
|------|-----------|----------------|-----|---------|
| 短上下文高吞吐 | 64-256 | 512-2K | 高 | 优化1,2,3 |
| 长上下文 | 16-64 | 8K-32K | 中 | 优化5,7 |
| 混合负载 | 32-128 | 混合 | 中 | 优化4,8 |
| 低延迟 | 1-16 | 1K-4K | 低 | 优化1,6 |

### 6.3 回滚策略

每个优化都应该有Feature Flag：
```python
# server_args.py
@dataclass
class ServerArgs:
    # 优化开关
    enable_persistent_batch: bool = True
    enable_adaptive_chunking: bool = False  # 默认关闭新特性
    enable_fused_sampling: bool = False
    enable_async_kv_prefetch: bool = False
```

---

## 7. 风险评估

### 低风险优化
- ✅ 优化4.1 (减少同步)
- ✅ 优化4.5 (自适应chunking)
- ✅ 优化4.6 (cache数据结构)

### 中等风险优化
- ⚠️ 优化4.2 (Persistent Batch) - 需要careful的内存管理
- ⚠️ 优化4.3 (调度优化) - 可能影响调度公平性

### 高风险优化
- ⚠️ 优化4.4 (Kernel Fusion) - 新kernel可能有bug
- ⚠️ 优化4.7 (异步预取) - Stream同步问题
- ⚠️ 优化4.8 (POD-Attention) - 复杂度高，测试充分性

---

## 8. 额外优化机会（需进一步验证）

### 8.1 调度器进程分离

**灵感**: vLLM V1将Scheduler分离到独立进程

**优势**:
- CPU密集型调度不阻塞GPU执行
- 更好的并行性

**挑战**:
- 需要重构进程架构
- 增加IPC开销
- 可能不适合SGLang的设计哲学（简洁性）

**建议**:
- 先实施优先级1-6的优化
- 如果CPU仍是瓶颈，再考虑此优化

### 8.2 动态Batching改进

**当前**: Continuous batching已经很好
**可能改进**:
- 智能的batch组装策略
- 考虑请求的相似性（shared prefix）
- 预测性调度（根据历史数据）

### 8.3 更激进的CUDA Graph

**当前**: 支持常见batch size的CUDA Graph
**改进**:
- 支持更大的batch size (>512)
- 动态CUDA Graph（根据实际batch size）

---

## 9. 结论

SGLang已经实现了许多先进的优化技术，但与vLLM V1和TensorRT-LLM相比，仍有**明显的优化空间**，特别是在：

1. **CPU-GPU同步开销** - 最容易实施，收益明显
2. **Python对象创建开销** - Persistent Batch技术
3. **Kernel Fusion** - 需要kernel开发经验

**推荐的实施顺序**:
```
阶段1 (快速胜利) → 阶段2 (中等改进) → 根据需求决定是否进入阶段3
```

**关键成功因素**:
- 建立完善的benchmark和profiling流程
- 每个优化独立实施和验证
- 通过Feature Flag控制风险
- 充分的测试覆盖

---

## 附录A: 代码位置索引

| 优化 | 主要文件 | 行号 |
|-----|---------|-----|
| 优化4.1 | `schedule_batch.py` | 1737 |
| 优化4.2 | `schedule_batch.py` | 1806-1874 |
| 优化4.3 | `schedule_policy.py` | 105-143 |
| 优化4.4 | `sampler.py` | 64-180 |
| 优化4.5 | `scheduler.py` | 调度逻辑 |
| 优化4.6 | `radix_cache.py` | 100-300 |
| 优化4.7 | `flashinfer_backend.py` | 全文件 |
| 优化4.8 | `model_runner.py` | 2179-2268 |

## 附录B: 性能监控命令

```bash
# 1. 基础性能测试
python3 bench_serving.py \
  --dataset-name random \
  --num-prompts 2000 \
  --request-rate 10 \
  --backend sglang

# 2. PyTorch Profiler
SGLANG_TORCH_PROFILER_DIR=/tmp/profile \
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf

# 3. Nsight Systems
nsys profile -o sglang_profile.qdrep \
  --trace=cuda,nvtx \
  python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf

# 4. 监控GPU利用率
nvidia-smi dmon -s pucvmet -d 1

# 5. 查看调度器日志
# 在运行时查找类似输出:
# "Decode batch. #running-req: 233, #token: 370959, token usage: 0.82"
```

## 附录C: 参考资源

1. **vLLM V1 Blog**: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
2. **TensorRT-LLM Chunked Prefill**: https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/
3. **POD-Attention Paper**: ASPLOS'25 - "POD-Attention: Unlocking Full Prefill-Decode Overlap"
4. **FlashAttention**: https://github.com/Dao-AILab/flash-attention
5. **SGLang Documentation**: https://docs.sglang.ai/

---

**报告生成时间**: 2025-11-18
**分析代码版本**: SGLang main branch (commit: eb67410)
