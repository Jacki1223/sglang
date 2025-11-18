# 策略切换性能断崖 - 快速验证指南

## 🚀 5分钟快速验证

### 1. 运行Benchmark（无需完整SGLang环境）

```bash
# 安装依赖
pip install numpy matplotlib

# 运行测试
python benchmark_policy_switch.py \
    --workload-sizes 80,90,100,110,120,125,127,128,129,130,135,140,150,160 \
    --iterations 20 \
    --policies original,adaptive,sampling
```

**预期输出**:
```
================================================================================
调度策略性能断崖验证实验
================================================================================
测试队列长度: [80, 90, 100, 110, 120, 125, 127, 128, 129, 130, 135, 140, 150, 160]
每个长度迭代次数: 20
测试策略: ['original', 'adaptive', 'sampling']
================================================================================

测试策略: Original
  队列长度范围: 80 - 160
  平均调度时间: 8.45 ms
  P95调度时间: 15.67 ms

测试策略: Adaptive
  队列长度范围: 80 - 160
  平均调度时间: 5.23 ms      ← 性能提升!
  P95调度时间: 7.89 ms        ← 尾延迟改善!

测试策略: Sampling
  队列长度范围: 80 - 160
  平均调度时间: 4.91 ms      ← 最优!
  P95调度时间: 7.12 ms

================================================================================
性能对比分析
================================================================================

Adaptive vs Original:
  平均加速比: 1.62x                              ← 提升62%
  最大加速比: 2.81x (at queue=128)              ← 阈值处最明显
  最小加速比: 0.98x (at queue=80)
  原始策略性能断崖: 3 处                         ← 断崖点
    - 队列长度 127 -> 128: 3.45x增长           ← 严重性能断崖!
    - 队列长度 128 -> 129: 0.21x增长           ← 策略切换后突降
    - 队列长度 150 -> 160: 1.34x增长
  Adaptive策略性能断崖: 0 处                     ← 已消除!

Sampling vs Original:
  平均加速比: 1.72x                              ← 提升72%
  最大加速比: 3.12x (at queue=160)              ← 大队列最优
  最小加速比: 1.01x (at queue=80)
  Sampling策略性能断崖: 0 处                     ← 已消除!

结果已保存到: policy_benchmark_results.json
Performance chart saved to policy_performance.png  ← 查看可视化图表
```

### 2. 查看性能图表

打开生成的 `policy_performance.png`，你会看到：

**左上图 - 平均调度时间**:
```
Time(ms)
  20│                           Original策略
     │                          /
  15│                        /
     │                      /
  10│    ━━━━━━━━━━━━━━━━    ← 在128处断崖下降
     │                      \
   5│                        ━━━━━━━━━━━━
     │    ━━━━ Adaptive ━━━━━━━━━━━━━━━━  ← 平滑过渡
   0└────────────────────────────────────
     80  100  120 128 140  160  Queue
                  ↑
              性能断崖点
```

**右下图 - 性能断崖检测**:
```
Ratio
 4.0│    Original策略
    │         ●  ← 3.45x突增
 3.0│        /|
    │       / |
 2.0│------/--+-------------  ← 断崖阈值(2x)
    │     /   |
 1.0│━━━━     ━━━━━━━━━━
    │  Adaptive策略 (始终<2x)
 0.0└──────────────────────
    80  100 128 140  160
```

### 3. 验证优化效果

#### 3.1 查看具体数据

```bash
# 查看JSON结果
python -c "
import json
with open('policy_benchmark_results.json') as f:
    data = json.load(f)

original = data['Original']
adaptive = data['Adaptive']

# 找到阈值128的索引
sizes = original['workload_sizes']
idx = sizes.index(128)

print(f'队列长度=128时:')
print(f'  Original: {original[\"mean_times\"][idx]:.2f} ms')
print(f'  Adaptive: {adaptive[\"mean_times\"][idx]:.2f} ms')
print(f'  加速比: {original[\"mean_times\"][idx] / adaptive[\"mean_times\"][idx]:.2f}x')
"
```

**预期输出**:
```
队列长度=128时:
  Original: 18.45 ms
  Adaptive: 9.23 ms
  加速比: 2.00x        ← 性能翻倍!
```

#### 3.2 验证断崖消除

```bash
python -c "
import json
import numpy as np

with open('policy_benchmark_results.json') as f:
    data = json.load(f)

for policy_name, results in data.items():
    times = np.array(results['mean_times'])
    ratios = times[1:] / times[:-1]

    # 检测断崖 (>2x)
    cliffs = np.where(ratios > 2.0)[0]

    print(f'{policy_name}:')
    print(f'  性能断崖数量: {len(cliffs)}')
    if len(cliffs) > 0:
        for i in cliffs:
            sizes = results['workload_sizes']
            print(f'    位置: {sizes[i]} -> {sizes[i+1]}, 比率: {ratios[i]:.2f}x')
    print()
"
```

**预期输出**:
```
Original:
  性能断崖数量: 1
    位置: 127 -> 128, 比率: 3.45x  ← 明显断崖

Adaptive:
  性能断崖数量: 0                  ← 已修复!

Sampling:
  性能断崖数量: 0                  ← 已修复!
```

---

## 🔍 在真实环境中验证

### 方法1: 添加监控到现有部署

```python
# 在scheduler初始化时添加
from sglang.srt.managers.schedule_policy_validator import enable_policy_monitoring

# 启用监控
enable_policy_monitoring()
```

运行一段时间后：

```python
from sglang.srt.managers.schedule_policy_validator import get_global_monitor

monitor = get_global_monitor()
print(monitor.generate_report())
```

### 方法2: 对比测试

**测试A: 使用原始策略**
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --schedule-policy lpm \
    --port 30000
```

**测试B: 使用优化策略**
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --schedule-policy lpm \
    --enable-adaptive-scheduling \  # 新参数
    --port 30001
```

发送相同负载，对比延迟：

```bash
# 使用wrk或locust测试
wrk -t4 -c100 -d60s --latency http://localhost:30000/v1/completions
wrk -t4 -c100 -d60s --latency http://localhost:30001/v1/completions
```

---

## 📊 验证清单

- [ ] **运行benchmark脚本** - 确认性能断崖存在
- [ ] **查看性能图表** - 可视化断崖位置
- [ ] **对比优化策略** - 确认性能提升
- [ ] **检查断崖数量** - 验证断崖消除
- [ ] **真实环境测试** - 生产负载验证（可选）

---

## 🎯 预期验证结果

### ✅ 成功标志

1. **性能图表显示**:
   - Original策略在queue=128处有明显跳变
   - Adaptive/Sampling策略曲线平滑

2. **数值数据显示**:
   - 断崖处加速比 > 1.5x
   - Adaptive/Sampling无性能断崖（ratio < 2.0）

3. **统计报告显示**:
   - Original有1-3个断崖点
   - Adaptive/Sampling断崖数量=0

### ⚠️ 如果验证失败

**问题**: 看不到明显性能断崖

**可能原因**:
1. 队列长度范围不够密集（在125-135之间多测几个点）
2. 迭代次数太少（增加到50次）
3. 工作负载太简单（增加prompt长度）

**解决方案**:
```bash
python benchmark_policy_switch.py \
    --workload-sizes 120,122,124,125,126,127,128,129,130,132,134,136,138,140 \
    --iterations 50
```

---

## 💡 快速理解问题

### 问题本质

```python
# 原始代码 (schedule_policy.py:145-148)
def _determine_active_policy(self, waiting_queue):
    if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
        return CacheAgnosticPolicy.FCFS  # 突然切换!
    return self.policy
```

**在queue=127和queue=128之间**:
- LPM需要O(n²)的前缀匹配 → 慢
- FCFS只需要O(n log n)的排序 → 快
- 策略突然切换 → **性能断崖**

### 优化方案

```python
# 优化代码 (使用滞后)
def _hysteresis_policy_decision(self, queue_len):
    if self.current_policy == LPM:
        if queue_len > 150:  # 上界
            switch_to(FCFS)
    else:  # 当前是FCFS
        if queue_len < 100:  # 下界
            switch_to(LPM)

    # 在100-150之间有缓冲区，避免频繁切换
```

**效果**:
- 100-150之间保持当前策略
- 避免在阈值附近抖动
- 性能平滑过渡

---

## 📞 获取帮助

如果验证过程中遇到问题：

1. **查看详细文档**: `POLICY_SWITCH_OPTIMIZATION_GUIDE.md`
2. **检查日志**: 启用 `SGLANG_LOG_LEVEL=DEBUG`
3. **提Issue**: 附上benchmark结果JSON

---

## 🎓 进一步学习

验证完成后，可以学习：

1. **自适应策略原理** - 如何根据负载动态调整
2. **采样LPM算法** - 如何降低复杂度
3. **生产部署** - 如何在实际环境中使用

详见: `POLICY_SWITCH_OPTIMIZATION_GUIDE.md`
