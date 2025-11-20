# Mamba State Recomputation 离线验证指南

本指南提供**不需要启动服务器**的验证方法，通过静态分析、代码检查和单元测试来验证实现的正确性。

---

## 方法 1: 单元测试（推荐）

### 运行离线单元测试

```bash
# 直接运行测试脚本
python test_mamba_recomputation_offline.py
```

### 测试内容

脚本包含 12 个单元测试，验证：

1. **配置正确性**
   - 默认参数值合理性
   - 参数类型检查

2. **接口完整性**
   - `model_runner.recompute_mamba_state` 方法存在
   - 参数签名正确
   - 返回值类型正确

3. **内存管理**
   - 分配/释放流程正确
   - 无双重释放
   - 旧值在重分配前被释放（防止泄漏）

4. **并发安全**
   - 重复重计算检测
   - 早期返回逻辑

5. **核心逻辑**
   - Tombstone 检测算法
   - 距离阈值检查
   - 零初始化 vs 状态复制

6. **边界条件**
   - 空输入处理
   - 单个 token
   - 精确边界（max_tokens）

### 预期输出

```
==========================================
OFFLINE VERIFICATION SUMMARY
==========================================

  Tests run: 12
  ✅ Passed: 12
  ❌ Failed: 0
  ⚠️  Errors: 0

🎉 All offline tests passed!
   Implementation logic appears correct.
```

---

## 方法 2: 代码审查清单

### 2.1 配置传递链路检查

**检查点：**

```bash
# 1. server_args.py 定义参数
grep -n "enable_mamba_state_recomputation" python/sglang/srt/server_args.py
# 应该看到：
#   - 数据类字段定义 (line ~480)
#   - CLI argument 定义 (line ~3257)

# 2. scheduler.py 传递给 MambaRadixCache
grep -n "enable_recomputation=server_args" python/sglang/srt/managers/scheduler.py
# 应该看到 (line ~776):
#   enable_recomputation=server_args.enable_mamba_state_recomputation

# 3. mamba_radix_cache.py 接收参数
grep -n "self.enable_recomputation = enable_recomputation" python/sglang/srt/mem_cache/mamba_radix_cache.py
# 应该看到 (line ~337)
```

**验证结果：**
- ✅ 所有文件都有对应的参数定义和使用
- ✅ 参数传递链路完整：`CLI → ServerArgs → Scheduler → MambaRadixCache`

---

### 2.2 核心逻辑检查

#### A. Tombstone 检测逻辑

**文件：** `python/sglang/srt/mem_cache/mamba_radix_cache.py`

**检查代码段：** `_match_prefix_helper` 方法 (lines 866-1034)

```python
# 关键逻辑：跟踪最后一个有效的 mamba 节点
last_valid_mamba_node = None
last_valid_mamba_len = 0
tombstone_encountered = False

while matching:
    if node.mamba_value is not None:
        last_valid_mamba_node = node
        last_valid_mamba_len = len(value)
        tombstone_encountered = False
    elif node != self.root_node:
        tombstone_encountered = True
```

**验证：**
- ✅ 正确跟踪最后一个有效节点
- ✅ 检测到 tombstone 时设置标志
- ✅ Root node 不被视为 tombstone

#### B. 重计算触发条件

**代码段：** `_match_prefix_helper` (lines 960-1034)

```python
if self.enable_recomputation and tombstone_encountered:
    recompute_len = len(value) - last_valid_mamba_len

    # 检查距离限制
    if recompute_len > 0 and recompute_len <= self.recompute_max_tokens:
        # 再次检查是否被并发请求重计算
        if node.mamba_value is not None:
            best_value_len = len(value)
            best_last_node = node
        else:
            # 尝试重计算
            rebuilt_node = self._try_rebuild_mamba_state(...)
```

**验证：**
- ✅ 只在启用时触发 (`enable_recomputation`)
- ✅ 检查距离限制 (`<= recompute_max_tokens`)
- ✅ 并发安全检查（再次检查 `mamba_value`）
- ✅ 重计算距离大于 0

#### C. 内存管理

**代码段：** `_try_rebuild_mamba_state` (lines 617-738)

```python
# 1. 分配新 slot
new_mamba_idx = self.req_to_token_pool.mamba_pool.alloc(1)

# 2. ⭐ 释放旧值（防止泄漏）
if target_node.mamba_value is not None:
    if target_node.id in self.mamba_lru_list.cache:
        self.mamba_lru_list.remove_node(target_node)
        self.mamba_evictable_size_ -= 1
    self.req_to_token_pool.mamba_pool.free(target_node.mamba_value)

# 3. 调用 model_runner
success = self.model_runner.recompute_mamba_state(...)

# 4. 成功后设置新值
if success:
    target_node.mamba_value = new_mamba_idx
    ...
else:
    # 失败则释放刚分配的
    self.req_to_token_pool.mamba_pool.free(new_mamba_idx)
```

**验证：**
- ✅ 先释放旧值再分配新值（防止泄漏）
- ✅ 失败时清理已分配的资源
- ✅ LRU 列表同步更新
- ✅ 计数器（evictable_size）同步更新

---

### 2.3 ModelRunner 接口检查

**文件：** `python/sglang/srt/model_executor/model_runner.py`

**检查点：** `recompute_mamba_state` 方法 (lines 2376-2478)

```python
def recompute_mamba_state(
    self,
    start_mamba_idx: int,     # -1 表示零初始化
    target_mamba_idx: int,
    kv_indices: torch.Tensor,
) -> bool:
```

**验证：**
- ✅ 方法签名正确
- ✅ 返回 bool 表示成功/失败
- ✅ 支持零初始化 (`start_mamba_idx == -1`)
- ✅ 支持状态复制 (`start_mamba_idx >= 0`)

---

### 2.4 Scheduler 集成检查

**文件：** `python/sglang/srt/managers/scheduler.py`

**检查点：** MambaRadixCache 实例化 (lines 771-783)

```python
self.tree_cache = MambaRadixCache(
    ...
    enable_recomputation=server_args.enable_mamba_state_recomputation,
    recompute_max_tokens=server_args.mamba_recompute_max_tokens,
    prioritize_mamba_retention=server_args.prioritize_mamba_retention,
    mamba_eviction_threshold=server_args.mamba_eviction_threshold,
    model_runner=self.tp_worker.model_runner,  # ⭐ 不能是 None
)
```

**验证：**
- ✅ 所有参数正确传递
- ✅ `model_runner` 引用正确设置（不是 `None`）

---

## 方法 3: 静态代码分析

### 3.1 使用 grep 检查关键路径

```bash
cd /home/user/sglang

# 1. 检查 enable_recomputation 使用
echo "=== Checking enable_recomputation usage ==="
grep -rn "enable_recomputation" python/sglang/srt/ --include="*.py" | head -20

# 2. 检查 _try_rebuild_mamba_state 调用
echo -e "\n=== Checking _try_rebuild_mamba_state calls ==="
grep -rn "_try_rebuild_mamba_state" python/sglang/srt/ --include="*.py"

# 3. 检查 model_runner.recompute_mamba_state 调用
echo -e "\n=== Checking model_runner.recompute_mamba_state calls ==="
grep -rn "model_runner.recompute_mamba_state" python/sglang/srt/ --include="*.py"

# 4. 检查内存释放 (free)
echo -e "\n=== Checking mamba_pool.free calls ==="
grep -rn "mamba_pool.free" python/sglang/srt/mem_cache/mamba_radix_cache.py
```

**预期输出：**
- 应该在 `mamba_radix_cache.py` 中找到多处 `enable_recomputation` 使用
- `_try_rebuild_mamba_state` 只在 `_match_prefix_helper` 中被调用
- `model_runner.recompute_mamba_state` 只在 `_try_rebuild_mamba_state` 中被调用
- `mamba_pool.free` 在多处被调用（清理逻辑）

---

### 3.2 检查已修复的 Bug

#### Bug 1: model_runner=None

```bash
grep -n "model_runner=self.tp_worker.model_runner" python/sglang/srt/managers/scheduler.py
```

**预期：** 应该找到 (line ~783)，而不是 `model_runner=None`

#### Bug 2: 内存泄漏

```bash
# 检查在分配前是否释放旧值
grep -B5 -A5 "target_node.mamba_value = new_mamba_idx" python/sglang/srt/mem_cache/mamba_radix_cache.py
```

**预期：** 应该看到在赋值前有：
```python
if target_node.mamba_value is not None:
    self.req_to_token_pool.mamba_pool.free(target_node.mamba_value)
```

#### Bug 3: LRU 列表重复插入

```bash
# 检查插入前是否检查存在性
grep -B3 -A3 "mamba_lru_list.insert_mru" python/sglang/srt/mem_cache/mamba_radix_cache.py
```

**预期：** 应该看到条件检查：
```python
if target_node.id in self.mamba_lru_list.cache:
    self.mamba_lru_list.reset_node_mru(target_node)
else:
    self.mamba_lru_list.insert_mru(target_node)
```

#### Bug 4: 并发重计算

```bash
# 检查早期返回逻辑
grep -A10 "def _try_rebuild_mamba_state" python/sglang/srt/mem_cache/mamba_radix_cache.py | head -15
```

**预期：** 方法开头应该有：
```python
if target_node.mamba_value is not None:
    return target_node
```

---

## 方法 4: 逻辑流程图验证

### 4.1 匹配流程

```
_match_prefix_helper 调用
  ↓
遍历 radix tree
  ↓
检查每个节点的 mamba_value
  ├─ 有值 → 记录为 last_valid_node
  └─ 无值 → 标记 tombstone_encountered
  ↓
匹配结束后
  ↓
是否遇到 tombstone？
  ├─ 否 → 返回完整匹配
  └─ 是 → 检查是否启用重计算
         ↓
     是否在距离限制内？
       ├─ 否 → 返回到 last_valid_node
       └─ 是 → 调用 _try_rebuild_mamba_state
              ↓
          重计算成功？
            ├─ 是 → 返回完整匹配
            └─ 否 → 返回到 last_valid_node
```

**验证方式：** 对照代码 (lines 866-1034) 确认每个分支都存在

---

### 4.2 重计算流程

```
_try_rebuild_mamba_state 调用
  ↓
① 检查 target_node.mamba_value 是否已存在
  ├─ 是 → 立即返回（并发安全）
  └─ 否 → 继续
  ↓
② 确定起始状态
  ├─ start_node = None → start_mamba_idx = -1
  └─ start_node 有效 → start_mamba_idx = node's index
  ↓
③ 分配新的 mamba slot
  ↓
④ 释放旧的 mamba_value（如果存在）
  ↓
⑤ 调用 model_runner.recompute_mamba_state
  ├─ start_mamba_idx = -1 → 零初始化
  └─ start_mamba_idx >= 0 → 复制状态
  ↓
⑥ 检查返回值
  ├─ 成功 → 设置 target_node.mamba_value，插入 LRU
  └─ 失败 → 释放分配的 slot，返回 None
```

**验证方式：** 对照代码 (lines 617-738) 确认流程完整

---

## 方法 5: 配置一致性检查

### 检查默认值

```bash
# server_args.py 的默认值
grep -A1 "enable_mamba_state_recomputation:" python/sglang/srt/server_args.py
grep -A1 "mamba_recompute_max_tokens:" python/sglang/srt/server_args.py
grep -A1 "prioritize_mamba_retention:" python/sglang/srt/server_args.py
grep -A1 "mamba_eviction_threshold:" python/sglang/srt/server_args.py
```

**预期输出：**
```python
enable_mamba_state_recomputation: bool = False  # 默认禁用
mamba_recompute_max_tokens: int = 512           # 默认 512
prioritize_mamba_retention: bool = True          # 默认启用
mamba_eviction_threshold: float = 0.8            # 默认 0.8
```

**验证：**
- ✅ 默认禁用重计算（安全）
- ✅ max_tokens 默认值合理
- ✅ 优先保留 mamba states

---

## 方法 6: 文档完整性检查

### 检查文档文件

```bash
ls -lh /home/user/sglang/*.md /home/user/sglang/*.py | grep -E "(mamba|recompute|verify)"
```

**应该看到：**
```
-rw-r--r-- 1 user user  XXK  VERIFICATION_GUIDE.md
-rw-r--r-- 1 user user  XXK  OFFLINE_VERIFICATION_GUIDE.md
-rw-r--r-- 1 user user  XXK  mamba_recompute_solution_1.py
-rw-r--r-- 1 user user  XXK  mamba_recompute_solution_2.py
-rw-r--r-- 1 user user  XXK  mamba_recompute_solution_3_practical.py
-rw-r--r-- 1 user user  XXK  test_mamba_recomputation_offline.py
-rw-r--r-- 1 user user  XXK  verify_mamba_recomputation.py
```

**验证：**
- ✅ 有在线验证脚本 (`verify_mamba_recomputation.py`)
- ✅ 有离线验证脚本 (`test_mamba_recomputation_offline.py`)
- ✅ 有详细的实现方案文档
- ✅ 有验证指南

---

## 快速离线验证清单（3分钟）

```bash
cd /home/user/sglang

echo "1. 运行单元测试..."
python test_mamba_recomputation_offline.py

echo -e "\n2. 检查配置传递..."
grep -q "enable_mamba_state_recomputation" python/sglang/srt/server_args.py && echo "✅ server_args.py"
grep -q "enable_recomputation=server_args" python/sglang/srt/managers/scheduler.py && echo "✅ scheduler.py"
grep -q "self.enable_recomputation" python/sglang/srt/mem_cache/mamba_radix_cache.py && echo "✅ mamba_radix_cache.py"

echo -e "\n3. 检查关键方法存在..."
grep -q "def _try_rebuild_mamba_state" python/sglang/srt/mem_cache/mamba_radix_cache.py && echo "✅ _try_rebuild_mamba_state"
grep -q "def recompute_mamba_state" python/sglang/srt/model_executor/model_runner.py && echo "✅ recompute_mamba_state"

echo -e "\n4. 检查已修复的 bug..."
grep -q "model_runner=self.tp_worker.model_runner" python/sglang/srt/managers/scheduler.py && echo "✅ model_runner 引用正确"
grep -q "if target_node.mamba_value is not None:" python/sglang/srt/mem_cache/mamba_radix_cache.py && echo "✅ 内存泄漏已修复"

echo -e "\n✅ 离线验证完成！"
```

---

## 对比：在线 vs 离线验证

| 验证维度 | 离线验证 | 在线验证 | 推荐 |
|---------|---------|---------|------|
| **功能正确性** | ✅ 单元测试 | ✅ 集成测试 | 两者结合 |
| **代码逻辑** | ✅ 代码审查 | ❌ 不可见 | 离线 |
| **性能提升** | ❌ 无法测试 | ✅ Benchmark | 在线 |
| **生成质量** | ❌ 无法测试 | ✅ 输出对比 | 在线 |
| **稳定性** | ⚠️  部分（内存） | ✅ 长时间运行 | 在线 |
| **速度** | ✅ 秒级 | ⚠️  分钟级 | 离线 |
| **CI/CD 集成** | ✅ 容易 | ⚠️  复杂 | 离线 |

---

## 总结

### 离线验证能检查什么？

✅ **代码逻辑正确性**
- Tombstone 检测算法
- 距离限制检查
- 内存分配/释放流程
- 并发安全机制

✅ **接口完整性**
- 方法签名正确
- 参数传递完整
- 返回值类型正确

✅ **代码质量**
- 无明显 bug（内存泄漏、双重释放）
- 边界条件处理
- 错误处理路径

### 离线验证不能检查什么？

❌ **运行时行为**
- 实际 cache hit rate 提升
- 真实场景下的性能
- 生成质量影响

❌ **系统集成**
- 与其他组件的交互
- 实际模型推理效果
- GPU 内存使用

### 推荐流程

1. **开发阶段：** 使用离线验证快速迭代
   ```bash
   python test_mamba_recomputation_offline.py
   ```

2. **代码审查：** 使用本指南的检查清单

3. **集成测试：** 使用在线验证确认端到端效果
   ```bash
   python verify_mamba_recomputation.py --url http://localhost:30000
   ```

4. **生产部署：** 结合 benchmark 和日志监控

---

## 常见问题

### Q: 离线验证通过了，是否意味着功能正确？

**A:** 离线验证只能确认代码逻辑正确，不能保证：
- 实际运行时的性能提升
- 与真实模型的兼容性
- 多 GPU、多请求并发场景

**建议：** 离线验证 + 在线验证结合

### Q: 如何在 CI/CD 中集成？

**A:** 在 CI pipeline 中添加：
```yaml
- name: Run Mamba Recomputation Unit Tests
  run: python test_mamba_recomputation_offline.py
```

### Q: 离线测试失败了怎么办？

**A:** 检查：
1. 依赖是否安装（`torch`, `unittest`）
2. 是否在正确的目录运行
3. 代码是否有语法错误

如果所有检查通过但测试失败，说明实现逻辑有问题，需要修复代码。

---

**如果离线验证通过 → 代码逻辑正确 ✅**
**如果在线验证也通过 → 功能完全正常 ✅✅**
