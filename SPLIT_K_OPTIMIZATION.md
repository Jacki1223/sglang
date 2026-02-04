# Split-K Optimization for Fused MoE

## 简介

本次提交为 SGLang 的 fused_moe kernel 添加了 **Split-K 优化**，通过在 K 维度上增加并行性来提高 GPU 利用率。

## 什么是 Split-K?

在矩阵乘法 `C = A @ B` 中：
- A 的形状: (M, K)
- B 的形状: (K, N)
- C 的形状: (M, N)

传统方法在 M 和 N 维度上并行化。当 K 很大但 M 和 N 相对较小时，GPU 利用率不足。

**Split-K** 将 K 维度分成多个部分，每个部分由不同的线程块并行计算，最后通过原子操作累加结果。

## 实现方式

### 核心修改（仅修改 `fused_moe_triton_kernels.py`）

1. **添加 SPLIT_K 参数到 kernel**
   ```python
   @triton.jit
   def fused_moe_kernel(..., SPLIT_K: tl.constexpr):
   ```

2. **根据 SPLIT_K 分配工作**
   - 当 `SPLIT_K > 1` 时，使用 `tl.program_id(axis=1)` 获取 split 索引
   - 每个线程块只处理 K 维度的一个子集: `[k_start, k_end)`

3. **使用原子操作累加结果**
   - 当 `SPLIT_K > 1` 时，使用 `tl.atomic_add` 累加部分结果
   - 当 `SPLIT_K == 1` 时，使用普通的 `tl.store`（保持原有行为）

4. **修改 grid 定义**
   - `SPLIT_K == 1`: 1D grid (M维 × N维)
   - `SPLIT_K > 1`: 2D grid (M维 × N维, SPLIT_K)

## 使用方法

### 自动使用（推荐）

在配置中添加 `SPLIT_K` 参数：

```python
config = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "SPLIT_K": 4,  # 将 K 维度分成 4 份
}
```

### 示例

```python
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe

# 使用 split-k=4 的配置
config["SPLIT_K"] = 4

output = fused_moe(
    hidden_states=hidden_states,
    w1=w1,
    w2=w2,
    topk_output=topk_output,
    moe_runner_config=moe_runner_config,
)
```

## 性能考虑

### 适用场景
✅ K 维度较大 (K >= 4096)
✅ M 和 N 相对较小
✅ 专家利用率不均衡

### 不适用场景
❌ K 维度很小（原子操作开销大于收益）
❌ M × N 已经很大（并行性已足够）

### 推荐配置

- **小批次 + 大隐藏维度**: SPLIT_K = 4~8
- **正常工作负载**: SPLIT_K = 1（默认）
- **超大 K 维度**: SPLIT_K = 8~16

## 限制

1. **原子操作开销**: 当 SPLIT_K > 1 时使用原子加法，可能在某些情况下降低性能
2. **不支持 bias 和 routing weights**: 当 SPLIT_K > 1 时，这些操作被跳过（需要在外部处理）
3. **需要初始化输出**: 输出张量会被自动清零

## 向后兼容

- 默认 `SPLIT_K = 1`，行为与原来完全一致
- 不修改任何其他文件
- 完全向后兼容现有代码

## 技术细节

修改文件：
- `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`

关键改动：
1. Kernel 签名添加 `SPLIT_K: tl.constexpr`
2. 程序 ID 计算中添加 split-k 支持
3. K 循环范围限制为 `[k_start, k_end)`
4. 输出使用原子操作（当 SPLIT_K > 1）
5. Grid 定义支持 2D（当 SPLIT_K > 1）

## 未来改进

1. **避免原子操作**: 使用两阶段方法（split + reduce kernel）
2. **自动调优**: 根据工作负载自动选择最优 SPLIT_K
3. **量化支持**: 扩展到所有量化模式

## 参考

- [Triton Split-K GEMM](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [CUTLASS Split-K](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
