# 性能测试指南

## 概述

本指南提供详细的步骤来测试各个优化版本，以确定哪些修改真正带来了性能提升。

## 测试版本总览

| 版本号 | 文件名 | BV | num_warps | Autotune | 内核优化 | 目的 |
|-------|--------|----|-----------|---------|---------|----- |
| 0 (基准) | original | 8 | 1 | ❌ | ❌ | 性能基准 |
| 1 | bv64_only | 64 | 1 | ❌ | ❌ | 隔离 BV 影响 |
| 2 | bv64_autotune | 64 | 自动 | ✅ | ❌ | BV + 并行度协同 |
| 3 | bv64_warps4 | 64 | 4 | ❌ | ❌ | 验证 warps=4 |
| 4 | bv64_warps8 | 64 | 8 | ❌ | ❌ | 验证 warps=8 |
| 5 | bv32_autotune | 32 | 自动 | ✅ | ❌ | 探索最优 BV |
| 6 | bv16_autotune | 16 | 自动 | ✅ | ❌ | 探索最优 BV |
| 7 (完整) | full_optimized | 64 | 自动 | ✅ | ✅ | 全部优化 |

**内核优化**包括：
- 循环不变量提升 (Loop-invariant code motion)
- 快速 Sigmoid (`tl.sigmoid`)
- 快速 rsqrt (`tl.rsqrt`)
- 显式赋值优化

## 测试步骤

### 准备工作

1. **备份原始文件**
```bash
cd /home/user/sglang
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_backup.py
```

2. **准备测试脚本**（如果还没有）
```bash
# 确保你的测试脚本存在
ls bench_sglang.py
```

### 测试矩阵

#### 阶段 1：基准测试（版本 0）

**目的**：建立性能基准

```bash
# 使用原始版本
cp /tmp/original.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

# 运行测试
python bench_sglang.py --num-questions 200 --port 12312

# 记录结果：
# Accuracy: _______
# Latency:  _______ s
# Throughput: _______ token/s
```

#### 阶段 2：隔离 BV 影响（版本 1）

**目的**：验证 BV=64 单独是否有帮助（预期：**性能下降 20%**）

```bash
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_bv64_only.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

python bench_sglang.py --num-questions 200 --port 12312

# 记录结果并与基准比较
```

**预期结果**：
- 性能下降 15-25%
- 原因：BV=64 但 num_warps=1 导致线程过载

#### 阶段 3：BV + Autotune 协同（版本 2）

**目的**：验证 BV=64 + autotune 的协同效应（预期：**性能提升 20%**）

```bash
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_bv64_autotune.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

python bench_sglang.py --num-questions 200 --port 12312

# 记录结果并与基准比较
```

**预期结果**：
- 性能提升 15-25%
- 原因：Autotune 选择合适的 num_warps (4 或 8) 来平衡工作负载

#### 阶段 4：验证 Autotune 选择（版本 3-4）

**目的**：确定 autotune 选择了哪个 num_warps 配置

**测试 num_warps=4**：
```bash
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_bv64_warps4.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

python bench_sglang.py --num-questions 200 --port 12312
```

**测试 num_warps=8**：
```bash
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_bv64_warps8.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

python bench_sglang.py --num-questions 200 --port 12312
```

**分析**：
- 如果 warps=4 接近 bv64_autotune 性能 → autotune 选择了 4
- 如果 warps=8 接近 bv64_autotune 性能 → autotune 选择了 8
- 如果都不接近 → autotune 在不同输入下选择不同配置

#### 阶段 5：探索最优 BV（版本 5-6）

**目的**：确定 BV=64 是否是最优值，或者更小的 BV 更好

**测试 BV=32**：
```bash
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_bv32_autotune.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

python bench_sglang.py --num-questions 200 --port 12312
```

**测试 BV=16**：
```bash
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_bv16_autotune.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

python bench_sglang.py --num-questions 200 --port 12312
```

**预期结果**：
- BV=16: 可能比原始稍好（5-10%），但不如 BV=64
- BV=32: 可能是中间值（10-15%），但仍不如 BV=64
- BV=64: 应该是最优或接近最优

#### 阶段 6：完整优化版本（版本 7）

**目的**：量化内核代码优化的额外贡献

```bash
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_full_optimized.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

python bench_sglang.py --num-questions 200 --port 12312
```

**预期结果**：
- 相比 bv64_autotune，额外提升 5-15%
- 总提升：25-40%（相比基准）

**分析**：
- 额外提升 = 完整版性能 - bv64_autotune 性能
- 这个额外提升来自：循环不变量、快速函数、显式赋值

## 结果记录表

创建一个表格记录所有测试结果：

```
┌──────────────┬──────────┬────────────┬──────────────┬─────────┬──────────┐
│ 版本         │ Accuracy │ Latency(s) │ Throughput   │ vs基准  │ 备注     │
│              │          │            │ (token/s)    │         │          │
├──────────────┼──────────┼────────────┼──────────────┼─────────┼──────────┤
│ 0. 基准      │          │            │              │ 100%    │          │
│ 1. bv64_only │          │            │              │         │ 预期-20% │
│ 2. bv64_auto │          │            │              │         │ 预期+20% │
│ 3. bv64_w4   │          │            │              │         │          │
│ 4. bv64_w8   │          │            │              │         │          │
│ 5. bv32_auto │          │            │              │         │          │
│ 6. bv16_auto │          │            │              │         │          │
│ 7. 完整优化  │          │            │              │         │ 预期+30% │
└──────────────┴──────────┴────────────┴──────────────┴─────────┴──────────┘
```

## 性能分析框架

### 计算相对性能
```python
# 基准吞吐量
baseline_throughput = <记录的基准值>

# 计算相对性能
for version in versions:
    speedup = version_throughput / baseline_throughput
    print(f"{version}: {speedup:.2f}x ({(speedup-1)*100:.1f}% 提升)")
```

### 分解性能贡献

基于测试结果，填写下表：

```
优化项目                          | 单独贡献 | 组合贡献 | 备注
----------------------------------|---------|---------|------
BV: 8→64 (单独)                   | _____%  | N/A     | 来自版本 1
BV: 8→64 + Autotune              | N/A     | _____%  | 来自版本 2
内核代码优化 (循环不变量等)        | ~5-15%  | _____%  | 版本7 - 版本2
```

## 关键假设验证

### 假设 1：BV 和 num_warps 必须协同优化

**验证方法**：比较版本 1 和版本 2
- 如果版本 1 性能下降，版本 2 性能提升 → **假设成立 ✅**
- 如果两者都提升 → 假设不成立 ❌

**当前状态**：根据你的测试
- 版本 1 (bv64_only): **-20%** ❌
- 版本 2 (bv64_autotune): **+20%** ✅
- **结论：假设成立 ✅**

### 假设 2：Autotune 会选择 num_warps=4 或 8

**验证方法**：比较版本 3、4 与版本 2
- 如果版本 3 或 4 的性能接近版本 2 → 假设成立
- 如果都不接近 → autotune 可能选择其他配置或根据输入动态选择

### 假设 3：BV=64 是最优值

**验证方法**：比较版本 2、5、6
- 如果版本 2 (BV=64) 性能最好 → 假设成立
- 如果版本 5 (BV=32) 或 版本 6 (BV=16) 更好 → 需要重新评估

### 假设 4：内核代码优化是次要贡献

**验证方法**：比较版本 7 和版本 2
- 如果额外提升 < 10% → 假设成立（次要）
- 如果额外提升 > 20% → 假设不成立（主要）
- 如果额外提升 10-20% → 中等贡献

## 常见问题

### Q1: 测试结果波动较大怎么办？

**解决方案**：
1. 多次运行取平均值（建议至少 3 次）
2. 确保测试环境一致（GPU 温度、其他进程等）
3. 使用更长的测试序列（增加 `--num-questions`）

### Q2: Accuracy 不是 1.000 怎么办？

**解决方案**：
1. 如果 Accuracy < 0.95，说明有正确性问题，不要使用该版本
2. 如果 0.95 < Accuracy < 0.99，需要进一步调查
3. 只有 Accuracy ≈ 1.000 的版本才能用于性能比较

### Q3: 如何快速测试？

**快速测试方案**（牺牲准确性）：
```bash
# 减少测试问题数量
python bench_sglang.py --num-questions 50 --port 12312
```

但最终结果**必须**使用完整测试（200+ questions）。

## 测试后清理

完成所有测试后，恢复你想要的版本：

```bash
# 恢复完整优化版本（推荐）
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_full_optimized.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

# 或恢复备份
cp python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent_backup.py \
   python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
```

## 预期测试时间

- 每个版本测试时间：~5-10 分钟（取决于测试规模）
- 总测试版本：8 个
- 预计总时间：~1-2 小时

## 结果解读

完成所有测试后，你应该能够回答：

1. ✅ **BV=64 单独会提升性能吗？** → 不会，反而下降 20%
2. ✅ **BV=64 + Autotune 一起会提升性能吗？** → 会，提升约 20%
3. ❓ **Autotune 选择了哪个 num_warps？** → 需要测试版本 3-4
4. ❓ **BV=64 是最优值吗？** → 需要测试版本 5-6
5. ❓ **循环不变量等优化贡献多少？** → 需要测试版本 7

## 报告模板

测试完成后，可以使用以下模板总结结果：

```markdown
# 性能优化测试报告

## 测试环境
- GPU: [型号]
- 测试规模: [num_questions]
- 日期: [日期]

## 主要发现

1. **BV 和 num_warps 协同优化**
   - BV=64 单独：-20% (版本 1)
   - BV=64 + Autotune：+20% (版本 2)
   - **结论**：必须协同优化

2. **Autotune 选择**
   - 版本 3 (warps=4): ____%
   - 版本 4 (warps=8): ____%
   - **结论**：Autotune 选择了 _____

3. **最优 BV 值**
   - BV=16: ____%
   - BV=32: ____%
   - BV=64: ____%
   - **结论**：最优 BV = _____

4. **内核代码优化贡献**
   - bv64_autotune: ____%
   - 完整优化: ____%
   - **额外贡献**：____%

## 最终性能提升

- **总提升**：____% (相比基准)
- **主要贡献**：BV + num_warps 协同 (____%)
- **次要贡献**：内核代码优化 (____%)

## 建议

[基于测试结果的优化建议]
```

祝测试顺利！如有问题，请参考 `PERFORMANCE_ROOT_CAUSE.md` 了解理论分析。
