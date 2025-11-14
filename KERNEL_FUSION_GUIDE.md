# SGLang Kernel融合优化实施指南

**目标**: 通过kernel融合减少内存访问和kernel启动开销，提升8-15%的端到端性能

---

## 目录

1. [当前状态分析](#1-当前状态分析)
2. [融合机会识别](#2-融合机会识别)
3. [实施方案](#3-实施方案)
4. [性能优化示例](#4-性能优化示例)
5. [测试验证](#5-测试验证)

---

## 1. 当前状态分析

### 1.1 已有的融合Kernel

SGLang已经实现了部分kernel融合：

| Kernel | 文件位置 | 类型 | 性能提升 |
|--------|---------|------|---------|
| `silu_and_mul` | `layers/activation.py:62` | CUDA/Triton | Activation融合 |
| `gelu_and_mul` | `layers/activation.py:98` | CUDA/Triton | Activation融合 |
| `fused_dual_residual_rmsnorm` | `layers/elementwise.py:188` | Triton | Norm+Residual |
| `fused_rmsnorm` | `layers/elementwise.py:254` | Triton | RMSNorm |
| `fused_moe_*` | `layers/moe/fused_moe_triton/` | Triton/CUTLASS | MOE融合 |

**现状**:
- ✅ **Activation融合**: SiLU/GELU与Mul已融合
- ✅ **Norm融合**: RMSNorm已优化
- ❌ **Linear+Activation**: 未融合 (关键优化点!)
- ❌ **Norm+Linear**: 未融合
- ⚠️ **MOE**: 部分融合，可进一步优化

### 1.2 典型MLP Forward Pass分析

当前Llama MLP实现 (`models/llama.py:94-106`):

```python
def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)      # Kernel 1: GEMM
    x = self.act_fn(gate_up)                # Kernel 2: SiLU+Mul (已融合)
    x, _ = self.down_proj(x)                # Kernel 3: GEMM
    return x
```

**当前问题**:
- `gate_up_proj` (GEMM) 和 `act_fn` (SiLU) 之间有**中间结果写回内存**
- 延迟 = GEMM延迟 + 内存写入 + 内存读取 + SiLU延迟
- 内存带宽浪费严重 (尤其是大模型如7B/13B/70B)

**优化潜力**:
- 融合GEMM+Activation: 减少1次内存往返
- 预期提升: **8-12% MLP吞吐量**, 5-8% 端到端

---

## 2. 融合机会识别

### 2.1 高优先级融合Pattern

#### Pattern 1: Linear + Activation (关键!)

**出现位置**: 所有模型的MLP层

```python
# 当前 (未融合)
y = Linear(x)           # Write to memory
z = Activation(y)       # Read from memory

# 优化后 (融合)
z = FusedLinearActivation(x)  # Direct register-to-register
```

**收益**:
- 减少内存往返: **2x** 中间结果大小
- 减少kernel启动: 2个kernel → 1个kernel
- 提升吞吐量: **8-15%**

---

#### Pattern 2: LayerNorm/RMSNorm + Linear

**出现位置**: Attention前的QKV projection

```python
# 当前 (未融合)
y = RMSNorm(x)
z = Linear(y)

# 优化后 (融合)
z = FusedNormLinear(x)
```

**收益**:
- 减少内存往返: 1次
- 提升吞吐量: **5-8%** (针对attention-bound模型)

---

#### Pattern 3: Residual + Norm + Linear

**出现位置**: Transformer block

```python
# 当前 (未融合)
y = x + residual
z = RMSNorm(y)
w = Linear(z)

# 优化后 (融合)
w = FusedResidualNormLinear(x, residual)
```

**收益**:
- 减少内存往返: 2次
- 提升吞吐量: **10-15%**

---

### 2.2 融合优先级矩阵

| Pattern | 优先级 | 预期提升 | 实施难度 | 建议阶段 |
|---------|--------|---------|---------|---------|
| Linear+Activation | 🔴 最高 | 8-15% | 中 | Phase 1 (立即) |
| Norm+Linear | 🟡 高 | 5-8% | 中 | Phase 1 |
| Residual+Norm+Linear | 🟡 高 | 10-15% | 高 | Phase 2 |
| MOE深度融合 | 🟢 中 | 5-10% (MOE模型) | 高 | Phase 2 |
| Attention融合 | 🟢 中 | 已有FlashInfer | 低 | 维护 |

---

## 3. 实施方案

### 3.1 方案概览

我们提供**三种实施路径**,从易到难:

1. **torch.compile融合** (最快, 中等性能)
2. **Triton kernel融合** (中速, 良好性能)
3. **CUDA kernel融合** (最慢, 最佳性能)

建议: **先torch.compile快速验证 → 再Triton优化 → 最后CUDA极致优化**

---

### 3.2 方案1: torch.compile融合 (Phase 1)

**优势**:
- 实施最快 (1-2天)
- 无需手写kernel
- PyTorch 2.0+自动优化

**实施步骤**:

#### Step 1: 创建融合模块

创建文件: `python/sglang/srt/layers/fused_layers.py`

```python
"""Fused layers using torch.compile for automatic kernel fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLinearSiLU(nn.Module):
    """Fused Linear + SiLU activation.

    Replaces:
        y = linear(x)
        z = F.silu(y)

    With single fused operation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if dtype is None:
            dtype = torch.get_default_dtype()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    @torch.compile(mode="max-autotune", fullgraph=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused forward pass.

        torch.compile will automatically fuse:
        - F.linear (GEMM)
        - F.silu (activation)
        into a single kernel.
        """
        # Linear projection
        y = F.linear(x, self.weight, self.bias)
        # SiLU activation (fused by torch.compile)
        z = F.silu(y)
        return z


class FusedLinearGELU(nn.Module):
    """Fused Linear + GELU activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        approximate: str = "none",
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.approximate = approximate

        if dtype is None:
            dtype = torch.get_default_dtype()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    @torch.compile(mode="max-autotune", fullgraph=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        z = F.gelu(y, approximate=self.approximate)
        return z


class FusedRMSNormLinear(nn.Module):
    """Fused RMSNorm + Linear projection.

    Common in:
    - Attention QKV projection: RMSNorm(x) → Linear
    - FFN input: RMSNorm(x) → Linear
    """

    def __init__(
        self,
        normalized_shape: int,
        out_features: int,
        eps: float = 1e-6,
        bias: bool = False,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.out_features = out_features
        self.eps = eps

        if dtype is None:
            dtype = torch.get_default_dtype()

        # RMSNorm weight
        self.norm_weight = nn.Parameter(
            torch.ones(normalized_shape, dtype=dtype)
        )

        # Linear weight
        self.linear_weight = nn.Parameter(
            torch.empty(out_features, normalized_shape, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    @torch.compile(mode="max-autotune", fullgraph=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused RMSNorm + Linear.

        torch.compile will fuse:
        1. RMSNorm computation
        2. Linear projection
        """
        # RMSNorm
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        normed = self.norm_weight * x

        # Linear (fused with above)
        out = F.linear(normed, self.linear_weight, self.bias)
        return out


class FusedGateUpProjection(nn.Module):
    """Fused gate_proj + up_proj for MLP.

    Replaces:
        gate = gate_proj(x)
        up = up_proj(x)
        y = torch.cat([gate, up], dim=-1)

    With single fused operation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if dtype is None:
            dtype = torch.get_default_dtype()

        # Fused weight: [2*intermediate_size, hidden_size]
        self.weight = nn.Parameter(
            torch.empty(2 * intermediate_size, hidden_size, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(2 * intermediate_size, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)

    @torch.compile(mode="max-autotune", fullgraph=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single GEMM for both gate and up projections."""
        return F.linear(x, self.weight, self.bias)
```

---

#### Step 2: 优化MLP层

修改文件: `python/sglang/srt/models/llama.py`

```python
# 在文件顶部添加import
from sglang.srt.layers.fused_layers import (
    FusedLinearSiLU,
    FusedGateUpProjection,
)

class OptimizedLlamaMLP(nn.Module):
    """Optimized MLP with kernel fusion."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()

        # Option 1: 使用FusedGateUpProjection (推荐)
        if quant_config is None and hidden_act == "silu":
            self.gate_up_proj = FusedGateUpProjection(
                hidden_size,
                intermediate_size,
                bias=False,
            )
            self.act_fn = SiluAndMul()  # 保持兼容
            self.use_fused = True
        else:
            # Fallback to original implementation
            self.gate_up_proj = MergedColumnParallelLinear(...)
            self.act_fn = SiluAndMul()
            self.use_fused = False

        self.down_proj = RowParallelLinear(...)

    def forward(self, x):
        if self.use_fused:
            # 融合路径
            gate_up = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
        else:
            # 原始路径
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
        return x
```

---

#### Step 3: torch.compile配置

创建文件: `python/sglang/srt/compile_config.py`

```python
"""torch.compile configuration for SGLang."""

import torch
import logging

logger = logging.getLogger(__name__)


def configure_torch_compile():
    """Configure torch.compile for optimal kernel fusion."""

    # 启用inductor优化
    torch._inductor.config.optimize_for_inference = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_gemm_backends = "TRITON,CUDA"

    # 启用kernel融合
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.triton.cudagraphs = True

    # FP16/BF16优化
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    logger.info("torch.compile configured for kernel fusion")


def compile_model(model, mode="max-autotune"):
    """Compile model with torch.compile.

    Args:
        model: PyTorch model
        mode: Compilation mode
            - "default": Balanced
            - "reduce-overhead": Reduce overhead
            - "max-autotune": Maximum optimization (recommended)

    Returns:
        Compiled model
    """
    configure_torch_compile()

    compiled = torch.compile(
        model,
        mode=mode,
        fullgraph=True,  # 尝试编译整个图
        dynamic=False,    # 静态shape优化
        backend="inductor",
    )

    logger.info(f"Model compiled with mode={mode}")
    return compiled
```

---

#### Step 4: 在ModelRunner中集成

修改文件: `python/sglang/srt/model_executor/model_runner.py`

```python
# 在__init__中添加
from sglang.srt.compile_config import configure_torch_compile

class ModelRunner:
    def __init__(self, ...):
        # ...existing code...

        # 配置torch.compile
        if self.server_args.enable_kernel_fusion:
            configure_torch_compile()
            logger.info("Kernel fusion enabled via torch.compile")

    def load_model(self):
        # ...existing model loading...

        # 可选: 编译整个模型
        if self.server_args.compile_model:
            from sglang.srt.compile_config import compile_model
            self.model = compile_model(
                self.model,
                mode="max-autotune"
            )
```

---

#### Step 5: 添加Server Args

修改文件: `python/sglang/srt/server_args.py`

```python
class ServerArgs:
    # ...existing args...

    # Kernel fusion options
    enable_kernel_fusion: bool = False
    compile_model: bool = False
    fusion_mode: str = "max-autotune"  # default, reduce-overhead, max-autotune
```

---

### 3.3 方案2: Triton Kernel融合 (Phase 1-2)

**优势**:
- 更好的性能控制
- 可定制优化
- 跨硬件兼容

**实施步骤**:

创建文件: `python/sglang/srt/layers/triton_fused_kernels.py`

```python
"""Triton fused kernels for common patterns."""

import torch
import triton
import triton.language as tl


# ============================================================================
# Linear + SiLU Fusion
# ============================================================================

@triton.jit
def fused_linear_silu_kernel(
    # Input
    input_ptr,
    weight_ptr,
    bias_ptr,
    # Output
    output_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Fused Linear + SiLU kernel.

    Computes: output = SiLU(input @ weight^T + bias)

    This kernel fuses:
    1. Matrix multiplication (GEMM)
    2. Bias addition (if HAS_BIAS)
    3. SiLU activation

    into a single Triton kernel, avoiding intermediate memory writes.
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    weight_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Matrix multiplication loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Load input and weight tiles
        input_tile = tl.load(
            input_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        weight_tile = tl.load(
            weight_ptrs,
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )

        # Accumulate
        accumulator += tl.dot(input_tile, weight_tile)

        # Advance pointers
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    # Add bias (if present)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    # Apply SiLU activation: x * sigmoid(x)
    # SiLU(x) = x / (1 + exp(-x))
    sigmoid_acc = 1.0 / (1.0 + tl.exp(-accumulator))
    output = accumulator * sigmoid_acc

    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output, mask=mask)


def fused_linear_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Fused Linear + SiLU operation.

    Args:
        input: Input tensor [M, K]
        weight: Weight matrix [N, K] (transposed)
        bias: Optional bias [N]

    Returns:
        Output tensor [M, N] with SiLU applied
    """
    assert input.dim() == 2, "Input must be 2D"
    assert weight.dim() == 2, "Weight must be 2D"

    M, K = input.shape
    N, K2 = weight.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"

    # Allocate output
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Grid configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    # Launch kernel
    fused_linear_silu_kernel[grid](
        input,
        weight,
        bias if bias is not None else input,  # Dummy pointer if no bias
        output,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=bias is not None,
    )

    return output


# ============================================================================
# RMSNorm + Linear Fusion
# ============================================================================

@triton.jit
def fused_rmsnorm_linear_kernel(
    # Input
    input_ptr,
    norm_weight_ptr,
    linear_weight_ptr,
    bias_ptr,
    # Output
    output_ptr,
    # Dimensions
    M, N, K,
    # Config
    eps,
    # Strides
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_om, stride_on,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Fused RMSNorm + Linear kernel.

    Computes: output = Linear(RMSNorm(input))

    Fuses:
    1. RMSNorm computation
    2. Linear projection
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Step 1: Compute RMSNorm
    # Load input for normalization
    input_norm_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik

    # Compute RMS (root mean square)
    rms_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(
            input_norm_ptrs + k * stride_ik,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        rms_acc += tl.sum(x * x, axis=1)

    rms = tl.sqrt(rms_acc / K + eps)

    # Step 2: Normalize and apply norm_weight, then Linear projection
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    linear_weight_ptrs = linear_weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    for k in range(0, K, BLOCK_SIZE_K):
        # Load and normalize input
        x = tl.load(
            input_ptrs + k * stride_ik,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )

        # Apply RMSNorm
        norm_w = tl.load(
            norm_weight_ptr + offs_k + k,
            mask=offs_k + k < K,
            other=1.0
        )
        x_normed = (x / rms[:, None]) * norm_w[None, :]

        # Load linear weight
        w = tl.load(
            linear_weight_ptrs + k * stride_wk,
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )

        # Accumulate
        accumulator += tl.dot(x_normed.to(tl.float16), w)

    # Add bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    # Store
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=mask)


def fused_rmsnorm_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-6,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Fused RMSNorm + Linear.

    Args:
        input: [M, K]
        norm_weight: [K]
        linear_weight: [N, K]
        eps: RMSNorm epsilon
        bias: Optional [N]

    Returns:
        Output [M, N]
    """
    M, K = input.shape
    N, K2 = linear_weight.shape
    assert K == K2
    assert norm_weight.shape[0] == K

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    fused_rmsnorm_linear_kernel[grid](
        input,
        norm_weight,
        linear_weight,
        bias if bias is not None else input,
        output,
        M, N, K,
        eps,
        input.stride(0), input.stride(1),
        linear_weight.stride(0), linear_weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=bias is not None,
    )

    return output


# ============================================================================
# Fused Linear Layer (Triton)
# ============================================================================

class TritonFusedLinearSiLU(torch.nn.Module):
    """Triton-based Fused Linear + SiLU."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return fused_linear_silu(x, self.weight, self.bias)


class TritonFusedRMSNormLinear(torch.nn.Module):
    """Triton-based Fused RMSNorm + Linear."""

    def __init__(self, normalized_shape, out_features, eps=1e-6, bias=False):
        super().__init__()
        self.norm_weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.linear_weight = torch.nn.Parameter(
            torch.empty(out_features, normalized_shape)
        )
        self.eps = eps
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return fused_rmsnorm_linear(
            x, self.norm_weight, self.linear_weight, self.eps, self.bias
        )
```

---

### 3.4 方案3: CUDA Kernel融合 (Phase 2)

**最佳性能方案** - 使用sgl-kernel扩展

创建文件: `sgl-kernel/csrc/fused_kernels/fused_linear_silu.cu`

```cuda
// Fused Linear + SiLU CUDA kernel
// Optimized for NVIDIA GPUs (SM80+)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

// ============================================================================
// Fused Linear + SiLU using Tensor Cores
// ============================================================================

template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void fused_linear_silu_kernel_tensor_core(
    const half* __restrict__ input,      // [M, K]
    const half* __restrict__ weight,     // [N, K]
    const half* __restrict__ bias,       // [N] or nullptr
    half* __restrict__ output,           // [M, N]
    int M, int N, int K
) {
    // Tensor Core fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    // Block and thread indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Matrix multiply accumulate
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices
            wmma::load_matrix_sync(a_frag, input + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, weight + bRow * N + bCol, N);

            // Multiply-accumulate
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Add bias and apply SiLU activation
    if (bias != nullptr) {
        for (int i = 0; i < acc_frag.num_elements; i++) {
            int col = warpN * WMMA_N + (i % WMMA_N);
            if (col < N) {
                float val = __half2float(acc_frag.x[i]) + __half2float(bias[col]);
                // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                float sigmoid_val = 1.0f / (1.0f + expf(-val));
                acc_frag.x[i] = __float2half(val * sigmoid_val);
            }
        }
    } else {
        for (int i = 0; i < acc_frag.num_elements; i++) {
            float val = __half2float(acc_frag.x[i]);
            float sigmoid_val = 1.0f / (1.0f + expf(-val));
            acc_frag.x[i] = __float2half(val * sigmoid_val);
        }
    }

    // Store output
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(output + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}

// Python binding
torch::Tensor fused_linear_silu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    auto output = torch::empty({M, N}, input.options());

    const half* bias_ptr = bias.has_value() ? bias.value().data_ptr<half>() : nullptr;

    // Launch configuration
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    dim3 grid((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 block(32, 1);  // 1 warp per block

    fused_linear_silu_kernel_tensor_core<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
        input.data_ptr<half>(),
        weight.data_ptr<half>(),
        bias_ptr,
        output.data_ptr<half>(),
        M, N, K
    );

    return output;
}
```

---

## 4. 性能优化示例

### 4.1 Benchmark测试

创建文件: `benchmark/kernel_fusion_benchmark.py`

```python
"""Benchmark kernel fusion performance."""

import torch
import time
from typing import Callable

from sglang.srt.layers.fused_layers import FusedLinearSiLU
from sglang.srt.layers.activation import SiluAndMul


def benchmark_kernel(
    fn: Callable,
    *args,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs
) -> float:
    """Benchmark a kernel function.

    Returns:
        Average latency in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_latency_ms = (end - start) / iterations * 1000
    return avg_latency_ms


def test_linear_silu_fusion():
    """Test Linear + SiLU fusion performance."""

    # Configuration
    batch_size = 32
    seq_len = 2048
    hidden_size = 4096
    intermediate_size = 11008

    device = torch.device("cuda")
    dtype = torch.float16

    # Input
    x = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)

    # Unfused baseline
    linear = torch.nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
    silu = SiluAndMul()

    def unfused_forward(x):
        y = linear(x)
        z = silu(y)
        return z

    # Fused version
    fused_linear_silu = FusedLinearSiLU(hidden_size, intermediate_size, bias=False).to(device).to(dtype)
    fused_linear_silu.weight.data.copy_(linear.weight.data)

    # Benchmark
    print("=" * 80)
    print("Linear + SiLU Fusion Benchmark")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden size: {hidden_size}")
    print(f"Intermediate size: {intermediate_size}")
    print(f"Total tokens: {batch_size * seq_len}")
    print("-" * 80)

    unfused_latency = benchmark_kernel(unfused_forward, x)
    fused_latency = benchmark_kernel(fused_linear_silu, x)

    speedup = unfused_latency / fused_latency

    print(f"Unfused latency: {unfused_latency:.4f} ms")
    print(f"Fused latency:   {fused_latency:.4f} ms")
    print(f"Speedup:         {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print("=" * 80)

    return speedup


def test_rmsnorm_linear_fusion():
    """Test RMSNorm + Linear fusion."""

    batch_size = 32
    seq_len = 2048
    hidden_size = 4096

    device = torch.device("cuda")
    dtype = torch.float16

    x = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)

    # Unfused
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    rmsnorm = LlamaRMSNorm(hidden_size).to(device).to(dtype)
    linear = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)

    def unfused_forward(x):
        y = rmsnorm(x)
        z = linear(y)
        return z

    # Fused
    from sglang.srt.layers.fused_layers import FusedRMSNormLinear
    fused = FusedRMSNormLinear(hidden_size, hidden_size, bias=False).to(device).to(dtype)

    # Benchmark
    print("\n" + "=" * 80)
    print("RMSNorm + Linear Fusion Benchmark")
    print("=" * 80)

    unfused_latency = benchmark_kernel(unfused_forward, x)
    fused_latency = benchmark_kernel(fused, x)

    speedup = unfused_latency / fused_latency

    print(f"Unfused latency: {unfused_latency:.4f} ms")
    print(f"Fused latency:   {fused_latency:.4f} ms")
    print(f"Speedup:         {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print("=" * 80)


if __name__ == "__main__":
    print("Kernel Fusion Benchmarks\n")

    # Test 1: Linear + SiLU
    speedup1 = test_linear_silu_fusion()

    # Test 2: RMSNorm + Linear
    test_rmsnorm_linear_fusion()

    print("\n✅ Benchmarks completed!")
```

---

## 5. 测试验证

### 5.1 单元测试

创建文件: `test/test_kernel_fusion.py`

```python
"""Unit tests for kernel fusion."""

import torch
import pytest

from sglang.srt.layers.fused_layers import (
    FusedLinearSiLU,
    FusedRMSNormLinear,
)


@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("seq_len", [128, 512, 2048])
@pytest.mark.parametrize("hidden_size", [768, 4096])
def test_fused_linear_silu_correctness(batch_size, seq_len, hidden_size):
    """Test fused Linear+SiLU correctness."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Input
    x = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)

    # Reference implementation
    linear_ref = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
    def reference(x):
        y = linear_ref(x)
        return torch.nn.functional.silu(y)

    # Fused implementation
    fused = FusedLinearSiLU(hidden_size, hidden_size, bias=False).to(device).to(dtype)
    fused.weight.data.copy_(linear_ref.weight.data)

    # Compare outputs
    with torch.no_grad():
        ref_output = reference(x)
        fused_output = fused(x)

    # Check correctness
    torch.testing.assert_close(
        fused_output,
        ref_output,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Fused output mismatch (bs={batch_size}, seq={seq_len}, hidden={hidden_size})"
    )

    print(f"✅ Test passed: bs={batch_size}, seq={seq_len}, hidden={hidden_size}")


def test_fused_rmsnorm_linear_correctness():
    """Test fused RMSNorm+Linear correctness."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    batch_size, seq_len = 16, 512
    hidden_size = 4096

    x = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)

    # Reference
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    rmsnorm_ref = LlamaRMSNorm(hidden_size).to(device).to(dtype)
    linear_ref = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)

    def reference(x):
        y = rmsnorm_ref(x)
        return linear_ref(y)

    # Fused
    fused = FusedRMSNormLinear(hidden_size, hidden_size, bias=False).to(device).to(dtype)
    fused.norm_weight.data.copy_(rmsnorm_ref.weight.data)
    fused.linear_weight.data.copy_(linear_ref.weight.data)

    with torch.no_grad():
        ref_output = reference(x)
        fused_output = fused(x)

    torch.testing.assert_close(
        fused_output,
        ref_output,
        rtol=1e-2,
        atol=1e-2,
        msg="Fused RMSNorm+Linear output mismatch"
    )

    print("✅ RMSNorm+Linear fusion test passed")


if __name__ == "__main__":
    print("Running kernel fusion tests...\n")

    # Test 1: Linear + SiLU
    for bs in [1, 16]:
        for seq in [128, 512]:
            for hidden in [768, 4096]:
                test_fused_linear_silu_correctness(bs, seq, hidden)

    # Test 2: RMSNorm + Linear
    test_fused_rmsnorm_linear_correctness()

    print("\n✅ All tests passed!")
```

---

### 5.2 端到端性能测试

创建文件: `benchmark/e2e_fusion_benchmark.py`

```python
"""End-to-end benchmarking with/without kernel fusion."""

import torch
from sglang import Engine
from sglang.srt.server_args import ServerArgs


def benchmark_e2e():
    """Benchmark end-to-end performance."""

    model_path = "meta-llama/Llama-2-7b-hf"

    # Baseline (no fusion)
    print("=" * 80)
    print("Baseline (No Fusion)")
    print("=" * 80)

    engine_baseline = Engine(
        model_path=model_path,
        enable_kernel_fusion=False,
    )

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "In a hole in the ground there lived a hobbit.",
    ] * 10

    # Warmup
    _ = engine_baseline.generate(prompts[:3], max_new_tokens=128)

    # Benchmark
    import time
    start = time.time()
    outputs_baseline = engine_baseline.generate(prompts, max_new_tokens=128)
    baseline_time = time.time() - start

    print(f"Time: {baseline_time:.2f}s")
    print(f"Throughput: {len(prompts) / baseline_time:.2f} req/s")

    # With fusion
    print("\n" + "=" * 80)
    print("With Kernel Fusion")
    print("=" * 80)

    engine_fused = Engine(
        model_path=model_path,
        enable_kernel_fusion=True,
        compile_model=True,
    )

    # Warmup
    _ = engine_fused.generate(prompts[:3], max_new_tokens=128)

    # Benchmark
    start = time.time()
    outputs_fused = engine_fused.generate(prompts, max_new_tokens=128)
    fused_time = time.time() - start

    print(f"Time: {fused_time:.2f}s")
    print(f"Throughput: {len(prompts) / fused_time:.2f} req/s")

    # Summary
    speedup = baseline_time / fused_time
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Baseline:  {baseline_time:.2f}s")
    print(f"Fused:     {fused_time:.2f}s")
    print(f"Speedup:   {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_e2e()
```

---

## 6. 总结与建议

### 6.1 实施优先级

**Phase 1 (立即, 1-2周)**:
1. ✅ 实现torch.compile融合 (方案1)
2. ✅ 优化MLP层 (Linear+SiLU)
3. ✅ 添加benchmark测试
4. ✅ 验证正确性

**预期收益**: 8-12% 端到端性能提升

---

**Phase 2 (短期, 1个月)**:
1. ✅ 实现Triton融合kernel (方案2)
2. ✅ 优化QKV projection (Norm+Linear)
3. ✅ MOE kernel深度融合
4. ✅ 性能调优

**预期收益**: 额外5-8% 性能提升

---

**Phase 3 (长期, 2-3个月)**:
1. ✅ CUDA kernel优化 (方案3)
2. ✅ Tensor Core优化
3. ✅ 多硬件适配 (H100, A100)
4. ✅ 产品化

**预期收益**: 额外5-10% 性能提升

---

### 6.2 关键要点

1. **torch.compile是快速入门的最佳选择** - 最小代码改动，合理性能提升
2. **Triton适合定制优化** - 更好的性能控制，跨硬件兼容
3. **CUDA适合极致性能** - 最佳性能，但开发成本高
4. **务必做正确性测试** - 融合kernel容易引入bug
5. **benchmark驱动优化** - 用数据证明优化效果

---

### 6.3 预期性能提升

| 优化项 | 预期提升 | 实施难度 | 时间投入 |
|--------|---------|---------|---------|
| torch.compile融合 | 8-12% | 低 | 1-2周 |
| Triton kernel融合 | 5-8% | 中 | 2-4周 |
| CUDA kernel优化 | 5-10% | 高 | 1-2个月 |
| **总计** | **18-30%** | - | **2-3个月** |

---

### 6.4 下一步行动

1. **立即开始**: 实施方案1 (torch.compile)
2. **测试验证**: 运行benchmark和单元测试
3. **性能分析**: 使用Nsight或profiler确认瓶颈
4. **迭代优化**: 逐步推进Phase 2和Phase 3

---

**生成时间**: 2025-11-14
**作者**: SGLang Performance Team
