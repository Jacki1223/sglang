# 循环不变量优化单独测试

## 版本信息

**文件名**: `fused_sigmoid_gating_recurrent_loop_invariant_only.py`

**版本号**: TEST VERSION 8

## 优化内容

### ✅ 唯一的优化：循环不变量提升 (Loop-Invariant Code Motion)

将不随时间变化的变量移到循环外，减少重复加载和计算：

```python
# ❌ 原始代码 - 在循环内重复执行
for _ in range(0, T):
    b_A_log = tl.load(p_A_log).to(tl.float32)      # 每次都 load
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)  # 每次都 load
    b_g = -tl.exp(b_A_log) * softplus_x            # 每次都计算 exp

# ✅ 优化后 - 移到循环外
b_A_log = tl.load(p_A_log).to(tl.float32)    # 只 load 一次
b_dt_bias = tl.load(p_dt_bias).to(tl.float32) # 只 load 一次
neg_exp_A = -tl.exp(b_A_log)                  # 预计算

for _ in range(0, T):
    b_g = neg_exp_A * softplus_x  # 直接使用预计算值
```

### 优化的3个具体改动

1. **移动 `b_A_log` 加载**：从循环内移到循环外
   - 原因：`A_log` 是时间不变的参数
   - 节省：T 次内存加载操作

2. **移动 `b_dt_bias` 加载**：从循环内移到循环外
   - 原因：`dt_bias` 是时间不变的参数
   - 节省：T 次内存加载操作

3. **预计算 `neg_exp_A`**：在循环外计算 `-exp(b_A_log)`
   - 原因：结果不随时间变化
   - 节省：T 次 `exp()` 和 T 次取负操作

## 保持不变（与原始代码完全一致）

- ❌ **BV = 8** (未增大到 64)
- ❌ **num_warps = 1** (未调优并行度)
- ❌ **num_stages = 3** (未调优流水线)
- ❌ **无 Autotune** (无自动配置搜索)
- ❌ **手动 Sigmoid**：`1.0 / (1.0 + exp(-b_b))` (未使用 `tl.sigmoid`)
- ❌ **手动 rsqrt**：`1 / sqrt(...)` (未使用 `tl.rsqrt`)
- ❌ **无显式赋值优化**

## 预期性能

**保守估计**：+5-10% vs 基准

**理由**：
- 节省了 2×T 次内存加载 (但内存带宽可能不是瓶颈)
- 节省了 T 次 `exp()` 计算 (但计算可能被隐藏)
- 没有改变并行度和工作粒度 (主要瓶颈未解决)

**可能结果**：
1. **如果提升明显 (>10%)**：说明循环不变量优化本身很有价值
2. **如果提升很小 (<5%)**：说明循环不变量必须配合其他优化才有效
3. **如果无提升或下降**：说明编译器可能已经自动优化了

## 测试步骤

### 1. 替换文件

```bash
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_loop_invariant_only.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
```

### 2. 运行测试

```bash
python bench_sglang.py --num-questions 200 --port 12312
```

### 3. 记录结果

```
基准 (原始):         _____ tok/s (100%)
循环不变量优化:      _____ tok/s (____%)

提升：____%
```

## 与其他版本对比

| 版本 | BV | warps | Autotune | 循环不变 | 快速函数 | 预期性能 |
|------|----|----|----|----|----|----|
| 原始 | 8 | 1 | ❌ | ❌ | ❌ | 100% |
| **V8 (本版本)** | **8** | **1** | ❌ | **✅** | ❌ | **105-110%** |
| V1 (bv64_only) | 64 | 1 | ❌ | ❌ | ❌ | 80% ⚠️ |
| V2 (bv64_auto) | 64 | 自动 | ✅ | ❌ | ❌ | 120% ✅ |
| V7 (完整) | 64 | 自动 | ✅ | ✅ | ✅ | 130-140% |

## 分析价值

这个测试版本能回答：

### 问题 1：循环不变量优化单独有价值吗？
- 如果有明显提升 → 是的，值得应用
- 如果提升很小 → 需要配合其他优化

### 问题 2：为什么 4-优化版本性能变差？
- 如果本版本提升小 → 说明循环不变量不是主要贡献者
- 结合已知：BV=64 单独会降低 20%
- **结论**：BV + num_warps 协同才是关键

### 问题 3：完整版的 30-40% 提升来自哪里？
- 循环不变量：~5-10% (本版本测试)
- BV + Autotune：~20% (V2 已测试)
- 其他优化（快速函数等）：~5-10% (V7 - V2 - 本版本)

## 代码改动总结

```diff
  b_h = tl.zeros([BK, BV], dtype=tl.float32)
  if USE_INITIAL_STATE:
      # ... 初始化 ...

+ # ✅ Loop-invariant optimization
+ b_A_log = tl.load(p_A_log).to(tl.float32)
+ b_dt_bias = tl.load(p_dt_bias).to(tl.float32)
+ neg_exp_A = -tl.exp(b_A_log)

  for _ in range(0, T):
      b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
      b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
      b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
      b_b = tl.load(p_b).to(tl.float32)

-     # Compute sigmoid gating
-     # Load gating parameters
-     b_A_log = tl.load(p_A_log).to(tl.float32)
      b_a = tl.load(p_a).to(tl.float32)  # 保留，因为是时间变化的
-     b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

      x = b_a + b_dt_bias
      # ... softplus 计算 ...
-     b_g = -tl.exp(b_A_log) * softplus_x
+     b_g = neg_exp_A * softplus_x
```

**改动行数**：~10 行
**改动性质**：纯优化，逻辑完全等价

## 正确性验证

循环不变量优化是**语义保持变换** (semantics-preserving transformation)：

- ✅ 不改变计算结果
- ✅ 不改变数据流
- ✅ 只改变计算时机

**验证方法**：
- Accuracy 应该保持 ≈ 1.000
- 如果 Accuracy 下降 → 说明实现有误

## 理论分析

### 循环执行次数
假设 T = 序列长度 (通常几百到几千)

**原始代码每次循环**：
- 2 次内存加载 (`b_A_log`, `b_dt_bias`)
- 1 次 `exp()` 计算
- 1 次取负操作

**总开销**：T × (2 loads + 1 exp + 1 neg)

**优化后**：
- 循环外：2 loads + 1 exp + 1 neg
- 循环内：0 额外开销

**节省**：(T-1) × (2 loads + 1 exp + 1 neg)

### 为什么可能提升不大？

1. **内存带宽充足**：
   - 如果内存访问不是瓶颈，节省 2 次 load 影响小

2. **指令延迟隐藏**：
   - GPU 可能通过线程切换隐藏了 `exp()` 延迟

3. **寄存器压力**：
   - 增加循环外变量可能增加寄存器使用
   - 但 BV=8 时寄存器压力应该不大

4. **编译器优化**：
   - Triton 编译器可能已经做了类似优化 (但通常不会)

## 后续测试建议

### 如果本版本提升明显 (>10%)
→ 说明循环不变量优化很有价值
→ 建议测试：循环不变量 + BV=64 + Autotune (应该比 V2 更好)

### 如果本版本提升很小 (<5%)
→ 说明循环不变量需要配合其他优化
→ 确认：主要性能来自 BV + num_warps 协同 (V2: +20%)

### 如果本版本无提升或下降
→ 检查正确性 (Accuracy)
→ 可能编译器已经优化，或引入了寄存器压力

## 总结

这是一个**最小化**的优化版本：
- ✅ 只改动必要的代码
- ✅ 易于理解和验证
- ✅ 可以隔离循环不变量优化的贡献

测试这个版本能帮助我们理解：
1. 循环不变量优化的单独价值
2. 完整版性能提升的来源分解
3. 为什么 4-优化版本反而变慢 (循环不变量不是主因)

**下一步**：运行测试并记录结果！
