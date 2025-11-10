# 🔧 tqdm进度条更新频率修复

## 问题描述

**用户反馈**: "还是只有三个进度条"

实际日志显示：
```
Loading safetensors: 0% | 0/7
Loading safetensors: 14% | 1/7
Loading safetensors: 100% | 7/7
```

只看到3行更新，而期望看到7行（每个文件一行）。

## 根本原因

### tqdm的默认更新策略

tqdm有两个控制更新频率的参数：

1. **mininterval** (默认0.1秒)
   - 两次显示更新之间的最小时间间隔
   - 如果上次更新后时间 < mininterval，跳过此次更新

2. **miniters** (默认自动)
   - 两次显示更新之间的最小迭代次数
   - 如果上次更新后迭代次数 < miniters，跳过此次更新

### 为什么只看到3行？

从日志可以看出：
```
Loading safetensors: 100% | 7/7 [00:00<00:00, 14.11it/s]
                                              ^^^^^^^^^^^^
                                         14个文件/秒
```

**时间计算**：
- 总速度: 14.11文件/秒
- 每个文件: 1 ÷ 14.11 ≈ 0.071秒
- 7个文件总计: 7 ÷ 14.11 ≈ 0.5秒

**更新逻辑**：
```
文件1: t=0.071s  → 距离上次(t=0) 0.071s < 0.1s → 可能显示（第一个例外）
文件2: t=0.142s  → 距离上次 0.071s < 0.1s → 跳过
文件3: t=0.213s  → 距离上次 0.071s < 0.1s → 跳过
文件4: t=0.284s  → 距离上次 0.071s < 0.1s → 跳过
文件5: t=0.355s  → 距离上次 0.071s < 0.1s → 跳过
文件6: t=0.426s  → 距离上次 0.071s < 0.1s → 跳过
文件7: t=0.497s  → 最后一个，强制显示
```

结果：只显示开始(0/7)、第1个(1/7)、结束(7/7) = 3行

## 修复方案

### 修改tqdm配置

```python
# 修复前（使用默认值）
for st_file in tqdm(
    hf_weights_files,
    desc="Loading safetensors checkpoint shards",
    bar_format=_BAR_FORMAT,
):

# 修复后（强制每次都更新）
for st_file in tqdm(
    hf_weights_files,
    desc="Loading safetensors checkpoint shards",
    bar_format=_BAR_FORMAT,
    mininterval=0,  # 不限制时间间隔，立即更新
    miniters=1,     # 每次迭代都更新
):
```

### 修改位置

修改了两处safetensors加载函数：

1. **safetensors_weights_iterator** (第630行)
   - 标准的safetensors加载
   - 最常用的路径

2. **runai_safetensors_weights_iterator** (第894行)
   - 使用Runai Model Streamer的加载
   - 特定环境使用

## 修复效果

### 修复前
```
Loading safetensors checkpoint shards: 0% | 0/7
Loading safetensors checkpoint shards: 14% | 1/7
Loading safetensors checkpoint shards: 100% | 7/7
```
只看到3行更新

### 修复后（预期）
```
Loading safetensors checkpoint shards: 0% | 0/7
Loading safetensors checkpoint shards: 14% | 1/7
Loading safetensors checkpoint shards: 28% | 2/7
Loading safetensors checkpoint shards: 42% | 3/7
Loading safetensors checkpoint shards: 57% | 4/7
Loading safetensors checkpoint shards: 71% | 5/7
Loading safetensors checkpoint shards: 85% | 6/7
Loading safetensors checkpoint shards: 100% | 7/7
```
看到全部7行更新（每个文件一行）

## 应用修复

### 1. 重启SGLang服务

```bash
# 停止当前服务
pkill -f sglang

# 清除Python缓存（可选但推荐）
cd /home/user/sglang
find python -name "*.pyc" -delete
find python -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 重新启动服务
python -m sglang.launch_server \
    --model mispeech/midashenglm-7b-0804-fp32 \
    --dtype bfloat16
```

### 2. 验证修复

启动后观察日志，应该看到：
- ✅ 8行进度输出（0/7 + 7个文件 + 7/7）
- ✅ 每个文件加载都有对应的进度行
- ✅ 百分比递增：0% → 14% → 28% → 42% → 57% → 71% → 85% → 100%

## 技术细节

### tqdm更新算法

```python
class tqdm:
    def update(self, n=1):
        self.n += n  # 更新计数

        # 检查是否应该显示
        if (time.time() - self.last_print_t) >= self.mininterval:
            if (self.n - self.last_print_n) >= self.miniters:
                self.display()  # 显示更新
                self.last_print_t = time.time()
                self.last_print_n = self.n
```

### 参数说明

| 参数 | 默认值 | 修改后 | 作用 |
|------|--------|--------|------|
| `mininterval` | 0.1 | 0 | 时间间隔限制：0.1秒 → 无限制 |
| `miniters` | 自动 | 1 | 迭代次数限制：自动 → 每次都更新 |

### 为什么不是默认行为？

tqdm的默认策略是**平衡性能和信息**：
- 更新太频繁 → 性能开销（I/O、重绘）
- 更新太少 → 用户不知道进度

对于快速迭代（< 0.1秒/次），默认策略会跳过一些更新。

**我们的场景**：
- 文件数量少（7个）
- 用户明确期望看到每个文件
- I/O开销可以接受（只有7次额外输出）

→ 强制显示每个文件是合理的

## 性能影响

### 额外开销

```
修复前: 3次输出 (0/7, 1/7, 7/7)
修复后: 8次输出 (0/7, 1/7, 2/7, ..., 7/7)

额外输出: 8 - 3 = 5次
每次输出: ~100字节
总开销: ~500字节 + 5次系统调用
```

**结论**: 性能影响可忽略不计

### 用户体验提升

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| 进度可见性 | 低（只看到开始、中间、结束） | 高（每个文件都可见） |
| 用户信心 | 低（"是不是卡住了？"） | 高（清楚看到每步进展） |
| 调试能力 | 低（不知道哪个文件慢） | 高（可以看出哪个文件耗时） |

## 其他模型的对比

### 为什么有些模型显示更多行？

可能原因：

1. **文件更多**
   ```
   Llama-70B: 15个文件 → 即使跳过一些，也能看到更多
   MiDashengLM: 7个文件 → 容易被跳过
   ```

2. **文件更大/加载更慢**
   ```
   大文件: 每个1秒+ → 肯定超过0.1s间隔 → 都会显示
   小文件: 每个0.07s → 低于0.1s间隔 → 可能跳过
   ```

3. **不同的tqdm配置**
   ```
   某些实现: mininterval=0 → 全部显示
   SGLang默认: mininterval=0.1 → 可能跳过
   ```

## 相关修改历史

| 提交 | 修改 | 结果 |
|------|------|------|
| 原始代码 | 直接迭代生成器 | 可能看到7行（如果加载慢） |
| 我的错误修改 | `list(weights)` | 只看到2行 |
| 第一次修复 | 恢复直接迭代 | 看到3行（进步！） |
| **本次修复** | **添加tqdm参数** | **预期看到8行** |

## 验证命令

### 检查代码是否包含修复

```bash
grep -A 2 "Loading safetensors checkpoint shards" \
  python/sglang/srt/model_loader/weight_utils.py

# 应该看到:
#   desc="Loading safetensors checkpoint shards",
#   ...
#   mininterval=0,  # Force update every iteration
#   miniters=1,     # Update for every file
```

### 测试加载

```bash
python -m sglang.launch_server \
    --model mispeech/midashenglm-7b-0804-fp32 \
    2>&1 | grep "Loading safetensors"
```

应该看到8行输出。

## 总结

| 问题 | 原因 | 修复 |
|------|------|------|
| 只看到3行进度 | tqdm默认mininterval=0.1s，文件加载太快(0.07s/个) | 设置mininterval=0, miniters=1 |
| 期望看到7行 | 每个文件都应该显示 | 强制每次迭代都更新 |

**现在应该能看到完整的7个文件加载进度了！** 🎉

---

**修复提交**: commit f84c513
**相关文件**: `python/sglang/srt/model_loader/weight_utils.py`
**最后更新**: 2025-11-10
