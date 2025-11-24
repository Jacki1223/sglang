# AllReduce优化测试指南

## 背景

优化后的kernel虽然单次耗时降低，但AllReduce延迟大幅增加，导致端到端性能没有提升甚至下降。

**根本原因**：优化后的kernel（BV=64, num_warps=4/8）占用过多GPU资源（SM、registers、shared memory），导致AllReduce kernel被阻塞，无法获得足够的执行资源。

## 测试版本说明

我们创建了两个新的测试版本来解决这个问题：

### 版本1：BV=32 保守版本
**文件**：`fused_sigmoid_gating_recurrent_bv32_conservative.py`

**配置**：
```python
BV = min(triton.next_power_of_2(V), 32)  # 降低到32

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["BK", "BV", "K", "V"],
)
```

**特点**：
- ✅ 最低的SM资源占用
- ✅ 为AllReduce留出最多的GPU资源
- ⚠️ 单次kernel性能可能略低于BV=64
- ✅ 但端到端性能可能最优

**预期**：
- Kernel耗时：相比原始版本降低10-15%
- AllReduce耗时：接近原始版本（+10-20%）
- **端到端吞吐：提升10-15%** ✅

### 版本2：BV=64 平衡版本
**文件**：`fused_sigmoid_gating_recurrent_bv64_warps2_balanced.py`

**配置**：
```python
BV = min(triton.next_power_of_2(V), 64)  # 保持64

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=3, num_stages=2),
    ],
    key=["BK", "BV", "K", "V"],
)
```

**特点**：
- ✅ 保留BV=64的高性能
- ✅ 通过降低num_warps（2-3）减少资源占用
- ✅ 在性能和资源占用之间取得平衡

**预期**：
- Kernel耗时：相比原始版本降低15-18%
- AllReduce耗时：略有增加（+30-50%）
- **端到端吞吐：提升10-12%** ✅

## 测试方法

### Step 1：准备测试环境

```python
# test_allreduce_optimization.py
import torch
import torch.distributed as dist
import time

# 导入不同版本
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as baseline_version
)

from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent_bv32_conservative import (
    fused_sigmoid_gating_delta_rule_update as bv32_conservative
)

from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent_bv64_warps2_balanced import (
    fused_sigmoid_gating_delta_rule_update as bv64_balanced
)
```

### Step 2：创建Benchmark函数

```python
def benchmark_with_allreduce(
    kernel_func,
    inputs,
    num_warmup=20,
    num_runs=100,
    use_allreduce=True
):
    """
    Benchmark kernel with AllReduce to measure end-to-end performance
    """
    # Warmup
    print(f"Warmup: {num_warmup} iterations...")
    for _ in range(num_warmup):
        output = kernel_func(**inputs)
        if use_allreduce:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Measure
    print(f"Measuring: {num_runs} iterations...")

    # Time kernel only
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_runs):
        output = kernel_func(**inputs)
    end.record()
    torch.cuda.synchronize()

    kernel_time = start.elapsed_time(end) / num_runs

    # Time kernel + allreduce
    if use_allreduce:
        start_total = torch.cuda.Event(enable_timing=True)
        end_total = torch.cuda.Event(enable_timing=True)

        start_total.record()
        for _ in range(num_runs):
            output = kernel_func(**inputs)
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        end_total.record()
        torch.cuda.synchronize()

        total_time = start_total.elapsed_time(end_total) / num_runs
        allreduce_time = total_time - kernel_time
    else:
        total_time = kernel_time
        allreduce_time = 0.0

    return {
        'kernel_time': kernel_time,
        'allreduce_time': allreduce_time,
        'total_time': total_time,
        'throughput': 1000.0 / total_time  # ops/sec
    }
```

### Step 3：运行测试

```python
def run_comparison():
    # 准备测试数据
    B, T, H, K, V = 2, 128, 32, 64, 128
    HV = 32

    A_log = torch.randn(HV, device='cuda', dtype=torch.float32)
    a = torch.randn(B * T, HV, device='cuda', dtype=torch.float32)
    dt_bias = torch.randn(HV, device='cuda', dtype=torch.float32)
    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.float16)
    k = torch.randn(B, T, H, K, device='cuda', dtype=torch.float16)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=torch.float16)
    b = torch.randn(B * T, HV, device='cuda', dtype=torch.float32)

    initial_state_source = torch.zeros(1, HV, K, V, device='cuda', dtype=torch.float32)
    initial_state_indices = torch.zeros(B, device='cuda', dtype=torch.int64)

    inputs = {
        'A_log': A_log,
        'a': a,
        'dt_bias': dt_bias,
        'q': q,
        'k': k,
        'v': v,
        'b': b,
        'initial_state_source': initial_state_source,
        'initial_state_indices': initial_state_indices,
        'softplus_beta': 1.0,
        'softplus_threshold': 20.0,
        'use_qk_l2norm_in_kernel': True,
    }

    # 测试所有版本
    versions = {
        'Baseline (Original)': baseline_version,
        'BV=32 Conservative': bv32_conservative,
        'BV=64 Balanced': bv64_balanced,
    }

    results = {}

    print("=" * 80)
    print("Testing with AllReduce (Multi-GPU scenario)")
    print("=" * 80)

    for name, func in versions.items():
        print(f"\n{name}:")
        result = benchmark_with_allreduce(func, inputs, use_allreduce=True)
        results[name] = result

        print(f"  Kernel time:    {result['kernel_time']:.3f} ms")
        print(f"  AllReduce time: {result['allreduce_time']:.3f} ms")
        print(f"  Total time:     {result['total_time']:.3f} ms")
        print(f"  Throughput:     {result['throughput']:.2f} ops/sec")

    # 打印对比表格
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print(f"{'Version':<25} {'Kernel (ms)':<15} {'AllReduce (ms)':<15} {'Total (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline_total = results['Baseline (Original)']['total_time']

    for name, result in results.items():
        speedup = baseline_total / result['total_time']
        print(f"{name:<25} {result['kernel_time']:>12.3f}   {result['allreduce_time']:>12.3f}   {result['total_time']:>12.3f}   {speedup:>8.2f}×")

if __name__ == '__main__':
    # 初始化分布式环境（如果需要）
    # dist.init_process_group(backend='nccl')

    run_comparison()
```

### Step 4：使用nsys分析

```bash
# 测试BV=32保守版本
nsys profile -o bv32_conservative \
    --trace=cuda,nvtx,osrt \
    python test_allreduce_optimization.py --version bv32_conservative

# 测试BV=64平衡版本
nsys profile -o bv64_balanced \
    --trace=cuda,nvtx,osrt \
    python test_allreduce_optimization.py --version bv64_balanced

# 打开nsys-ui对比timeline
nsys-ui bv32_conservative.nsys-rep &
nsys-ui bv64_balanced.nsys-rep &
```

**查看重点**：
1. Compute kernel和AllReduce的重叠情况
2. AllReduce的实际执行时间（不包括等待时间）
3. SM占用率（是否接近100%）
4. 不同GPU之间的同步gap

### Step 5：使用NCU分析资源占用

```bash
# 分析BV=32版本的资源占用
ncu --set full \
    --metrics sm__warps_active.avg.pct_of_peak,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum \
    python test_allreduce_optimization.py --version bv32_conservative

# 分析BV=64版本的资源占用
ncu --set full \
    --metrics sm__warps_active.avg.pct_of_peak,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum \
    python test_allreduce_optimization.py --version bv64_balanced
```

**关键指标**：
- `sm__warps_active.avg.pct_of_peak`: 应该在60-80%（不是100%，给AllReduce留空间）
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`: 计算吞吐率
- L1 cache hit rate: 应该仍然保持高位

## 预期结果分析

### Scenario 1：BV=32 保守版本表现最好

```
Kernel:    100ms → 85ms  (↓15%)
AllReduce:  10ms → 12ms  (↑20%)
Total:     110ms → 97ms  (↓11.8% ✅)
```

**说明**：
- 系统瓶颈在GPU资源竞争
- 降低资源占用比提升单个kernel性能更重要
- **推荐在生产环境使用这个版本**

### Scenario 2：BV=64 平衡版本表现最好

```
Kernel:    100ms → 82ms  (↓18%)
AllReduce:  10ms → 15ms  (↑50%)
Total:     110ms → 97ms  (↓11.8% ✅)
```

**说明**：
- 系统有一定的资源余量
- 可以通过更高的kernel性能来补偿AllReduce增加的延迟
- **如果计算密集度高，推荐使用这个版本**

### Scenario 3：两个版本性能接近

**说明**：
- 系统处于资源竞争的临界点
- 可以根据具体workload选择：
  - 大batch size：选择BV=32（更稳定）
  - 小batch size：选择BV=64（更高性能）

## 集成到生产代码

确定最优版本后，替换主文件：

```bash
# 假设BV=32保守版本表现最好
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_bv32_conservative.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

# 或者如果BV=64平衡版本表现最好
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_bv64_warps2_balanced.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
```

## 进一步优化建议

### 如果两个版本都不理想

1. **调整num_warps范围**：
```python
# 尝试更细粒度的num_warps配置
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3),  # 最小资源占用
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=3, num_stages=2),
    ],
    key=["BK", "BV", "K", "V"],
)
```

2. **调整BV范围**：
```python
# 尝试BV=16或BV=48
BV = min(triton.next_power_of_2(V), 16)  # 或 48
```

3. **使用混合策略**：
```python
# 根据V的大小动态选择BV
if V <= 128:
    BV = min(triton.next_power_of_2(V), 32)
else:
    BV = min(triton.next_power_of_2(V), 64)
```

4. **优化AllReduce本身**：
```python
# 使用异步AllReduce
handle = dist.all_reduce(output, async_op=True)
# 继续其他计算...
handle.wait()
```

## 总结

| 版本 | BV | num_warps范围 | 资源占用 | 预期收益 | 适用场景 |
|------|----|--------------|---------|---------|---------
| 原始 | 8 | 1 | 低 | 基线 | - |
| 激进优化 | 64 | 4-8 | 很高 | 0% (AllReduce阻塞) | ❌ 不推荐 |
| 保守优化 | 32 | 2-4 | 中 | 10-15% | ✅ 资源受限环境 |
| 平衡优化 | 64 | 2-3 | 中高 | 10-12% | ✅ 计算密集型 |

**关键要点**：
1. ✅ 端到端性能 > 单个kernel性能
2. ✅ 必须考虑GPU资源竞争
3. ✅ AllReduce延迟同样重要
4. ✅ 测试时必须包含AllReduce操作
5. ⚠️ 过度优化计算可能损害通信
