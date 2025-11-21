# 测试版本快速参考

## 📋 所有测试版本一览

| # | 文件名 | BV | warps | Auto | 内核优化 | 预期性能 | 说明 |
|---|--------|-------|-------|------|---------|---------|------|
| 0 | (original) | 8 | 1 | ❌ | ❌ | 100% | 基准 |
| 1 | `bv64_only.py` | **64** | 1 | ❌ | ❌ | **80%** ⚠️ | BV 单独影响 |
| 2 | `bv64_autotune.py` | **64** | 自动 | ✅ | ❌ | **120%** ✅ | BV + 并行度 |
| 3 | `bv64_warps4.py` | **64** | **4** | ❌ | ❌ | ~115% | 验证 warps=4 |
| 4 | `bv64_warps8.py` | **64** | **8** | ❌ | ❌ | ~120% | 验证 warps=8 |
| 5 | `bv32_autotune.py` | **32** | 自动 | ✅ | ❌ | ~110% | 探索最优 BV |
| 6 | `bv16_autotune.py` | **16** | 自动 | ✅ | ❌ | ~105% | 探索最优 BV |
| 7 | `full_optimized.py` | **64** | 自动 | ✅ | ✅ | **130-140%** | 完整优化 |

## 🎯 测试目标

### 阶段 1：验证协同效应
- ✅ **版本 1 vs 版本 2**：证明 BV 和 num_warps 必须协同优化
  - 版本 1 (BV=64, warps=1): **-20%** ❌
  - 版本 2 (BV=64, autotune): **+20%** ✅
  - **结论**：协同效应得到证实

### 阶段 2：确定 Autotune 选择
- ❓ **版本 3 vs 版本 4**：Autotune 选择了哪个配置？
  - 如果版本 3 ≈ 版本 2 → 选择 warps=4
  - 如果版本 4 ≈ 版本 2 → 选择 warps=8

### 阶段 3：探索最优 BV
- ❓ **版本 2 vs 版本 5 vs 版本 6**：BV=64 是最优吗？
  - 预期：BV=64 > BV=32 > BV=16 > BV=8

### 阶段 4：量化内核优化
- ❓ **版本 7 vs 版本 2**：循环不变量等优化贡献多少？
  - 预期额外提升：5-15%

## 🚀 快速测试命令

### 替换测试版本模板
```bash
# 变量定义
TEST_DIR="python/sglang/srt/layers/attention/fla"
TARGET="$TEST_DIR/fused_sigmoid_gating_recurrent.py"

# 测试某个版本（替换 VERSION）
cp $TEST_DIR/fused_sigmoid_gating_recurrent_VERSION.py $TARGET
python bench_sglang.py --num-questions 200 --port 12312
```

### 批量测试脚本
```bash
#!/bin/bash
# test_all_versions.sh

TEST_DIR="python/sglang/srt/layers/attention/fla"
TARGET="$TEST_DIR/fused_sigmoid_gating_recurrent.py"

VERSIONS=(
  "original"
  "bv64_only"
  "bv64_autotune"
  "bv64_warps4"
  "bv64_warps8"
  "bv32_autotune"
  "bv16_autotune"
  "full_optimized"
)

echo "开始批量测试..."
for VERSION in "${VERSIONS[@]}"; do
  echo "======================================"
  echo "测试版本: $VERSION"
  echo "======================================"

  if [ "$VERSION" = "original" ]; then
    cp /tmp/original.py $TARGET
  else
    cp "$TEST_DIR/fused_sigmoid_gating_recurrent_$VERSION.py" $TARGET
  fi

  python bench_sglang.py --num-questions 200 --port 12312
  echo ""
  sleep 5  # 冷却时间
done

echo "所有测试完成！"
```

## 📊 已知结果（基于你的测试）

| 版本 | 性能变化 | 状态 |
|------|---------|------|
| 0. 基准 | 100% | ✅ |
| 1. bv64_only | **-20%** | ✅ 已测试 |
| 2. bv64_autotune | **+20%** | ✅ 已测试 |
| 3. bv64_warps4 | ❓ | 待测试 |
| 4. bv64_warps8 | ❓ | 待测试 |
| 5. bv32_autotune | ❓ | 待测试 |
| 6. bv16_autotune | ❓ | 待测试 |
| 7. full_optimized | ❓ | 待测试 |

## 🔬 测试优先级

**高优先级**（必须测试）：
1. ✅ 版本 1 (bv64_only) - 已完成
2. ✅ 版本 2 (bv64_autotune) - 已完成
3. ⭐ **版本 7 (full_optimized)** - 确定总提升和内核优化贡献

**中优先级**（建议测试）：
4. 版本 3 和 4 (bv64_warps4/8) - 确定 autotune 选择

**低优先级**（可选）：
5. 版本 5 和 6 (bv16/32_autotune) - 探索最优 BV

## 📈 预期性能曲线

```
性能
 ^
140%|                                    ●7
    |                                   /
120%|                    ●2  ●4       /
    |                   /   /        /
110%|              ●5  /   /        /
    |             /   /   /        /
105%|        ●6  /   /   /        /
    |       /   /   /   /        /
100%|●0────────────────────────────────>
    |       \
 80%|        ●1
    |
    +────────────────────────────────> BV
         8   16  32  64  64  64  64  64
                         (w4)(w8)(at)(full)

●0 = 原始 (BV=8, w=1)
●1 = BV=64, w=1 (过载)
●6 = BV=16 + autotune
●5 = BV=32 + autotune
●2 = BV=64 + autotune
●4 = BV=64, w=8
●7 = 完整优化
```

## 🧪 验证检查清单

完成测试后，确认：

- [ ] 版本 1 性能比基准**更低**
- [ ] 版本 2 性能比基准**更高**
- [ ] 版本 3 或 4 中至少一个接近版本 2
- [ ] 版本 2/5/6 中 BV=64 性能最好
- [ ] 版本 7 性能最高，是最终版本
- [ ] 所有版本 Accuracy > 0.99

## 💡 关键洞察

1. **BV 单独增大会降低性能**（版本 1: -20%）
   - 原因：线程过载 (128 elements/thread)
   - 解决：增加 num_warps

2. **BV + num_warps 协同优化带来提升**（版本 2: +20%）
   - 原因：平衡工作负载 (32 elements/thread)
   - 机制：Autotune 自动选择

3. **内核代码优化是锦上添花**（版本 7 - 版本 2）
   - 预期：额外 5-15% 提升
   - 包括：循环不变量、快速函数等

## 📝 结果记录

在 `TESTING_GUIDE.md` 中有详细的结果记录表格。

快速记录模板：
```
V0: _____ tok/s (基准)
V1: _____ tok/s (____%)
V2: _____ tok/s (____%)
V7: _____ tok/s (____%)
```

## 🎓 教训

这个测试系列展示了 GPU 优化的重要原则：

1. **协同原则**：多个参数必须一起优化
2. **平衡原则**：工作量与处理能力要匹配
3. **测量原则**：直觉可能错误，必须实测
4. **整体原则**：关注系统整体，非局部

完整指南请参考 `TESTING_GUIDE.md`！
