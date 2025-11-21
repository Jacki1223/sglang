# 代码修改详解：如何让 cached token 大幅增加

本文档通过代码对比、示意图和实际执行流程，详细解释修改的原理。

---

## 📋 目录

1. [核心代码对比](#1-核心代码对比)
2. [执行流程图](#2-执行流程图)
3. [数据结构变化图](#3-数据结构变化图)
4. [完整示例演示](#4-完整示例演示)
5. [关键代码行详解](#5-关键代码行详解)

---

## 1. 核心代码对比

### 1.1 原始代码 (`_match_prefix_helper`)

**文件**: `python/sglang/srt/mem_cache/mamba_radix_cache.py`

```python
# ========== 原始代码（简化版）==========
def _match_prefix_helper(self, key: RadixKey) -> Tuple[List[torch.Tensor], TreeNode]:
    """原始的前缀匹配逻辑"""

    node = self.root_node
    child_key = self.get_child_key_fn(key)

    value = []                    # 累积的 KV cache indices
    best_value_len = 0           # ⭐ 关键变量：能返回多少 KV
    best_last_node = node        # ⭐ 关键变量：最后可用的节点

    # ========== 遍历树，匹配 key ==========
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]

        # ⭐⭐⭐ 关键检查点 ⭐⭐⭐
        if node.mamba_value is not None:
            # ✅ 有 mamba_value，可以用
            best_value_len = len(value)
            best_last_node = node
        # ❌ 如果 mamba_value 是 None，什么都不做
        # ❌ best_value_len 不更新，保持之前的值

        # 匹配 key，累积 value
        prefix_len = self.key_match_fn(child.key, key)
        if prefix_len < len(child.key):
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)  # ⭐ 无条件累积 KV
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

    # 检查最后一个节点
    if node.mamba_value is not None:
        best_value_len = len(value)
        best_last_node = node

    # ========== 返回结果 ==========
    # ❌ 问题：只返回到 best_value_len，后面的 KV 被丢弃
    return value[:best_value_len], best_last_node
    #            ^^^^^^^^^^^^^^^^
    #            截断位置！
```

**问题分析**：

```python
# 执行示例：
# Tree 结构:
#   A: [tok1,2,3], mamba_value=[42] ✅
#   └─ B: [tok4,5], mamba_value=None ❌ (Tombstone)
#       └─ C: [tok6,7], mamba_value=[99] ✅

# 匹配过程:
匹配 A:
  node = A
  if A.mamba_value is not None:  # [42] ✅
      best_value_len = 1          # ← 更新
      best_last_node = A
  value = [kv_A]

匹配 B:
  node = B
  if B.mamba_value is not None:  # None ❌
      # 这个 if 不执行！
      # best_value_len 仍然是 1 ❌
  value = [kv_A, kv_B]            # ← 继续累积

匹配 C:
  node = C
  if C.mamba_value is not None:  # [99] ✅
      # 但已经太晚了！
      # 因为 B 已经是 None，这里不会执行
  value = [kv_A, kv_B, kv_C]

返回:
  return value[:best_value_len]  # value[:1]
  # 只返回 [kv_A]
  # [kv_B, kv_C] 被丢弃！❌
```

---

### 1.2 修改后的代码 (`_match_prefix_helper`)

**文件**: `python/sglang/srt/mem_cache/mamba_radix_cache.py` (lines 920-1050)

```python
# ========== 修改后的代码 ==========
def _match_prefix_helper(self, key: RadixKey) -> Tuple[List[torch.Tensor], TreeNode]:
    """增强的前缀匹配，支持 tombstone 重计算"""

    node = self.root_node
    child_key = self.get_child_key_fn(key)

    value = []
    best_value_len = 0
    best_last_node = node

    # ========== 新增：状态跟踪变量 ==========
    last_valid_mamba_node = None   # 最后一个有效的 mamba 节点
    last_valid_mamba_len = 0       # 对应的 value 长度
    tombstone_encountered = False  # 是否遇到过 tombstone

    # ========== 遍历树 ==========
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]

        # ⭐⭐⭐ 改进的检查点 ⭐⭐⭐
        if node.mamba_value is not None:
            # ✅ 有 mamba_value
            best_value_len = len(value)
            best_last_node = node
            # 新增：记录回退点
            last_valid_mamba_node = node        # Line 949
            last_valid_mamba_len = len(value)   # Line 950
            tombstone_encountered = False       # Line 951 (重置)
        elif node != self.root_node and not tombstone_encountered:
            # ⭐ 遇到 tombstone (第一次)
            tombstone_encountered = True        # Line 954 (标记)
            # ⭐ 关键：不 break，继续遍历！

        # 继续匹配和累积 (同原始代码)
        prefix_len = self.key_match_fn(child.key, key)
        if prefix_len < len(child.key):
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)  # ⭐ 继续累积所有 KV
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

    # 检查最后一个节点
    if node.mamba_value is not None:
        best_value_len = len(value)
        best_last_node = node

    # ========== 新增：重计算逻辑 ==========
    if self.enable_recomputation and tombstone_encountered:  # Line 976
        recompute_len = len(value) - last_valid_mamba_len   # Line 977

        # 打印调试信息
        logger.debug(
            f"Tombstone detected: recompute_len={recompute_len}, "
            f"max_tokens={self.recompute_max_tokens}"
        )  # Lines 979-985

        # 检查是否已经被重计算（并发安全）
        if node.mamba_value is not None:                     # Line 988
            # 已有值，直接使用
            best_value_len = len(value)                      # Line 995
            best_last_node = node                            # Line 996
        elif recompute_len > 0 and recompute_len <= self.recompute_max_tokens:  # Line 997
            # 需要重计算

            # 确定起始节点
            if last_valid_mamba_node is None:                # Line 999
                start_node = None                            # Line 1005 (从零开始)
                kv_to_recompute = value                      # Line 1006
            else:
                start_node = last_valid_mamba_node           # Line 1012
                kv_to_recompute = value[last_valid_mamba_len:]  # Line 1013

            # ⭐⭐⭐ 调用重计算 ⭐⭐⭐
            rebuilt_node = self._try_rebuild_mamba_state(    # Line 1016
                start_node,
                kv_to_recompute,
                node,
            )

            if rebuilt_node is not None:
                # ✅ 重计算成功
                best_value_len = len(value)     # ⭐ 使用完整长度！
                best_last_node = rebuilt_node   # ⭐ 使用重建的节点！
                self.recompute_hit_count += 1

                logger.info(
                    f"✓ Mamba state recomputed successfully: "
                    f"{len(kv_to_recompute)} tokens, total hits: {self.recompute_hit_count}"
                )
            else:
                # ❌ 重计算失败
                self.recompute_miss_count += 1

    # ========== 返回结果 ==========
    return value[:best_value_len], best_last_node
    #            ^^^^^^^^^^^^^^^^
    #            现在可能是完整长度！
```

**执行示例**：

```python
# 同样的 Tree 结构:
#   A: [tok1,2,3], mamba_value=[42] ✅
#   └─ B: [tok4,5], mamba_value=None ❌ (Tombstone)
#       └─ C: [tok6,7], mamba_value=[99] ✅

# 修改后的匹配过程:
匹配 A:
  if A.mamba_value is not None:  # [42] ✅
      best_value_len = 1
      last_valid_mamba_node = A          # ← 记录回退点
      last_valid_mamba_len = 1
  value = [kv_A]

匹配 B:
  if B.mamba_value is not None:  # None ❌
      # False
  elif node != self.root_node and not tombstone_encountered:
      tombstone_encountered = True       # ← 标记
      # ⭐ 不 break，继续！
  value = [kv_A, kv_B]                  # ← 继续累积

匹配 C:
  if C.mamba_value is not None:  # [99] ✅
      best_value_len = 3                 # ← 能执行了！
      last_valid_mamba_node = C
  value = [kv_A, kv_B, kv_C]

重计算检查:
  if tombstone_encountered:  # True ✅
      recompute_len = 3 - 1 = 2

      # 调用 _try_rebuild_mamba_state
      rebuilt_node = _try_rebuild_mamba_state(
          start_node=A,          # 从 A 开始
          kv_to_recompute=[kv_B],  # 重计算 B
          node=B,                # 目标节点是 B
      )

      # 内部执行 (关键！):
      new_mamba_idx = alloc(1)        # [50]
      B.mamba_value = [50]            # ⭐ Line 718

      # 返回成功
      best_value_len = 3              # ← 使用完整长度！
      best_last_node = B

返回:
  return value[:3]  # [kv_A, kv_B, kv_C]
  # 返回所有 KV！✅
```

---

### 1.3 关键的 `_try_rebuild_mamba_state` 函数

**文件**: `python/sglang/srt/mem_cache/mamba_radix_cache.py` (lines 632-738)

```python
def _try_rebuild_mamba_state(
    self,
    start_node: Optional[TreeNode],
    kv_indices_list: List[torch.Tensor],
    target_node: TreeNode,
) -> Optional[TreeNode]:
    """
    尝试重建 mamba state

    Args:
        start_node: 起始节点（有有效的 mamba_value）或 None
        kv_indices_list: 需要重计算的 KV indices
        target_node: 目标节点（需要分配 mamba_value）

    Returns:
        成功返回 target_node，失败返回 None
    """

    if self.model_runner is None:
        return None

    # ========== 并发安全检查 ==========
    if target_node.mamba_value is not None:     # Line 655
        # 已经有值了（可能被其他线程重计算了）
        logger.debug(f"Target node {target_node.id} already has mamba_value")
        return target_node                       # Line 660

    try:
        # ========== 准备 KV indices ==========
        if not kv_indices_list:
            return None
        kv_indices = torch.cat(kv_indices_list)  # Line 667
        num_tokens = len(kv_indices)             # Line 668

        if num_tokens == 0:
            return None

        # ========== 确定起始 mamba state ==========
        if start_node is not None and start_node.mamba_value is not None:
            start_mamba_idx = start_node.mamba_value[0].item()  # Line 675
        else:
            start_mamba_idx = -1  # 特殊值：从零初始化    # Line 677

        # ========== 分配新的 mamba state slot ==========
        new_mamba_idx = self.req_to_token_pool.mamba_pool.alloc(1)  # Line 681
        if new_mamba_idx is None:
            # 分配失败，尝试驱逐
            self.evict_mamba(1)                                      # Line 684
            new_mamba_idx = self.req_to_token_pool.mamba_pool.alloc(1)
            if new_mamba_idx is None:
                return None

        # ========== 调用模型的重计算方法 ==========
        success = self.model_runner.recompute_mamba_state(          # Line 691
            start_mamba_idx=start_mamba_idx,
            target_mamba_idx=new_mamba_idx[0].item(),
            kv_indices=kv_indices,
        )

        if not success:
            # 重计算失败，释放分配的 slot
            self.req_to_token_pool.mamba_pool.free(new_mamba_idx)   # Line 699
            return None

        # ========== 内存泄漏防护 ==========
        if target_node.mamba_value is not None:                     # Line 704
            # 不应该发生，但为了安全
            logger.warning(f"Target node {target_node.id} already has mamba_value")
            # 先释放旧值
            if target_node.id in self.mamba_lru_list.cache:
                self.mamba_lru_list.remove_node(target_node)        # Line 712
            self.req_to_token_pool.mamba_pool.free(target_node.mamba_value)  # Line 715

        # ========== ⭐⭐⭐ 关键操作！⭐⭐⭐ ==========
        target_node.mamba_value = new_mamba_idx                     # Line 718
        # 这一行让 tombstone 从 None 变成非 None！

        # ========== LRU 管理 ==========
        if target_node.id in self.mamba_lru_list.cache:
            # 不应该在列表中，但检查一下
            logger.error(f"BUG: Node {target_node.id} already in mamba_lru_list")
            self.mamba_lru_list.reset_node_mru(target_node)         # Line 726
        else:
            self.mamba_lru_list.insert_mru(target_node)             # Line 728
            self.mamba_evictable_size_ += 1                         # Line 729

        return target_node                                          # Line 731

    except Exception as e:
        logger.warning(f"Mamba state recomputation failed: {e}")
        return None
```

---

## 2. 执行流程图

### 2.1 原始代码的执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                     原始代码执行流程                          │
└─────────────────────────────────────────────────────────────┘

Tree 结构:
    Root
     └─ A: [tok1,2,3]  mamba_value=[42] ✅
         └─ B: [tok4,5]  mamba_value=None ❌ (Tombstone)
             └─ C: [tok6,7]  mamba_value=[99] ✅

用户查询: [tok1,2,3,4,5,6,7] (7个token)

┌──────────────────────────────────────────────────────────────┐
│ Step 1: 初始化                                                │
├──────────────────────────────────────────────────────────────┤
│ node = Root                                                  │
│ value = []                                                   │
│ best_value_len = 0                                          │
│ best_last_node = Root                                       │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: 匹配节点 A                                            │
├──────────────────────────────────────────────────────────────┤
│ node = A                                                     │
│ value.append(A.value)  →  value = [kv_A]                   │
│                                                              │
│ if A.mamba_value is not None:  # [42] ≠ None ✅            │
│     best_value_len = 1         # ← 更新                     │
│     best_last_node = A         # ← 更新                     │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3: 匹配节点 B (Tombstone)                               │
├──────────────────────────────────────────────────────────────┤
│ node = B                                                     │
│ value.append(B.value)  →  value = [kv_A, kv_B]             │
│                                                              │
│ if B.mamba_value is not None:  # None ≠ None ❌            │
│     # ❌ False，不执行                                       │
│     # best_value_len 仍然是 1                               │
│     # best_last_node 仍然是 A                               │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 4: 匹配节点 C                                            │
├──────────────────────────────────────────────────────────────┤
│ node = C                                                     │
│ value.append(C.value)  →  value = [kv_A, kv_B, kv_C]       │
│                                                              │
│ if C.mamba_value is not None:  # [99] ≠ None ✅            │
│     # ⚠️ 但这行不会执行！                                    │
│     # 因为循环已经结束                                       │
│     # best_value_len 仍然是 1                               │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 5: 返回结果                                              │
├──────────────────────────────────────────────────────────────┤
│ return value[:best_value_len], best_last_node               │
│        ^^^^^^^^^^^^^^^^^^^^^^                               │
│        value[:1] = [kv_A]                                   │
│                                                              │
│ ❌ 只返回 1 个 KV！                                          │
│ ❌ [kv_B, kv_C] 被截断丢弃！                                 │
│                                                              │
│ cached_tokens = 3 (只有 A 的 3 个 token)                    │
└──────────────────────────────────────────────────────────────┘
```

---

### 2.2 修改后代码的执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                    修改后代码执行流程                         │
└─────────────────────────────────────────────────────────────┘

Tree 结构:
    Root
     └─ A: [tok1,2,3]  mamba_value=[42] ✅
         └─ B: [tok4,5]  mamba_value=None ❌ (Tombstone)
             └─ C: [tok6,7]  mamba_value=[99] ✅

用户查询: [tok1,2,3,4,5,6,7] (7个token)

┌──────────────────────────────────────────────────────────────┐
│ Step 1: 初始化                                                │
├──────────────────────────────────────────────────────────────┤
│ node = Root                                                  │
│ value = []                                                   │
│ best_value_len = 0                                          │
│ best_last_node = Root                                       │
│                                                              │
│ ⭐ 新增状态变量:                                             │
│ last_valid_mamba_node = None                                │
│ last_valid_mamba_len = 0                                    │
│ tombstone_encountered = False                               │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: 匹配节点 A                                            │
├──────────────────────────────────────────────────────────────┤
│ node = A                                                     │
│ value.append(A.value)  →  value = [kv_A]                   │
│                                                              │
│ if A.mamba_value is not None:  # [42] ≠ None ✅            │
│     best_value_len = 1                                      │
│     best_last_node = A                                      │
│     ⭐ last_valid_mamba_node = A      # 记录回退点          │
│     ⭐ last_valid_mamba_len = 1       # 记录长度            │
│     ⭐ tombstone_encountered = False  # 重置标记            │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3: 匹配节点 B (Tombstone)                               │
├──────────────────────────────────────────────────────────────┤
│ node = B                                                     │
│ value.append(B.value)  →  value = [kv_A, kv_B]             │
│                                                              │
│ if B.mamba_value is not None:  # None ≠ None ❌            │
│     # False                                                 │
│ elif node != self.root_node and not tombstone_encountered: │
│     ⭐ tombstone_encountered = True  # 标记遇到 Tombstone   │
│     ⭐ 继续执行，不 break！                                  │
│                                                              │
│ # best_value_len 仍然是 1                                   │
│ # 但我们记住了 last_valid_mamba_node = A                    │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 4: 匹配节点 C                                            │
├──────────────────────────────────────────────────────────────┤
│ node = C                                                     │
│ value.append(C.value)  →  value = [kv_A, kv_B, kv_C]       │
│                                                              │
│ if C.mamba_value is not None:  # [99] ≠ None ✅            │
│     # ✅ 这次能执行了！                                      │
│     # 但暂时不更新 best_value_len（等重计算后）             │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 5: 重计算检查                                            │
├──────────────────────────────────────────────────────────────┤
│ if self.enable_recomputation and tombstone_encountered:    │
│     # True ✅                                               │
│                                                              │
│     recompute_len = len(value) - last_valid_mamba_len      │
│                   = 3 - 1 = 2                               │
│                                                              │
│     logger.debug("Tombstone detected: recompute_len=2")    │
│                                                              │
│     if recompute_len <= self.recompute_max_tokens:         │
│         # True (假设 max_tokens=512)                        │
│                                                              │
│         start_node = last_valid_mamba_node  # A            │
│         kv_to_recompute = value[1:]  # [kv_B, kv_C]        │
│         target_node = B                                     │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 6: 调用 _try_rebuild_mamba_state                        │
├──────────────────────────────────────────────────────────────┤
│ rebuilt_node = self._try_rebuild_mamba_state(               │
│     start_node=A,                                           │
│     kv_to_recompute=[kv_B, kv_C],                          │
│     target_node=B,                                          │
│ )                                                            │
│                                                              │
│ 内部执行:                                                    │
│   1. 并发安全检查: B.mamba_value == None ✅                 │
│   2. 准备 KV: kv_indices = cat([kv_B, kv_C])               │
│   3. 起始 state: start_mamba_idx = A.mamba_value[0] = 42   │
│   4. ⭐ 分配新 slot: new_mamba_idx = [50]                   │
│   5. 调用重计算:                                             │
│      model_runner.recompute_mamba_state(                    │
│          start_mamba_idx=42,                                │
│          target_mamba_idx=50,                               │
│          kv_indices=[kv_B, kv_C]                            │
│      )                                                       │
│      # 可能只是 copy 或 zero-init                            │
│      # 返回 True                                            │
│   6. ⭐⭐⭐ 关键操作 ⭐⭐⭐                                     │
│      B.mamba_value = [50]  # Line 718                       │
│      # B 从 Tombstone (None) 变成有效节点！                 │
│   7. 插入 LRU: mamba_lru_list.insert_mru(B)                │
│   8. 返回: return B                                         │
│                                                              │
│ rebuilt_node = B ✅                                         │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 7: 更新返回长度                                          │
├──────────────────────────────────────────────────────────────┤
│ if rebuilt_node is not None:                                │
│     ⭐ best_value_len = len(value)  # 3 (完整长度！)        │
│     ⭐ best_last_node = B            # 使用重建的节点        │
│     self.recompute_hit_count += 1                           │
│                                                              │
│     logger.info("✓ Mamba state recomputed successfully")   │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 8: 返回结果                                              │
├──────────────────────────────────────────────────────────────┤
│ return value[:best_value_len], best_last_node               │
│        ^^^^^^^^^^^^^^^^^^^^^^                               │
│        value[:3] = [kv_A, kv_B, kv_C]                       │
│                                                              │
│ ✅ 返回所有 3 个 KV！                                        │
│ ✅ 没有截断！                                                │
│                                                              │
│ cached_tokens = 7 (所有 7 个 token)                         │
│ 提升 = (7-3)/3 = +133%                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 数据结构变化图

### 3.1 执行前的树结构

```
┌────────────────────────────────────────────────────────────────┐
│                        执行前的 Radix Tree                      │
└────────────────────────────────────────────────────────────────┘

                        Root
                         │
                         │
                    ┌────▼────┐
                    │    A    │
                    │ [1,2,3] │
                    │ mamba:  │
                    │   [42]  │ ✅ 有效
                    └────┬────┘
                         │
                         │
                    ┌────▼────┐
                    │    B    │
                    │ [4,5]   │
                    │ mamba:  │
                    │   None  │ ❌ Tombstone
                    └────┬────┘
                         │
                         │
                    ┌────▼────┐
                    │    C    │
                    │ [6,7]   │
                    │ mamba:  │
                    │   [99]  │ ✅ 有效
                    └─────────┘

Legend:
  ✅ = 有 mamba_value (可用)
  ❌ = mamba_value=None (Tombstone，不可用)
```

---

### 3.2 原始代码的匹配过程

```
┌────────────────────────────────────────────────────────────────┐
│                    原始代码：匹配和返回过程                      │
└────────────────────────────────────────────────────────────────┘

Step 1: 匹配 A
┌─────────────────┐
│   A [1,2,3]     │
│   mamba: [42]✅ │ ← 检查：有值
│                 │    best_value_len = 1 ✅
└─────────────────┘    value = [kv_A]

         ↓

Step 2: 匹配 B
┌─────────────────┐
│   B [4,5]       │
│   mamba: None❌ │ ← 检查：无值
│                 │    best_value_len 仍是 1 ❌
└─────────────────┘    value = [kv_A, kv_B]
         │
         │ ⚠️ 问题：best_value_len 不更新
         ↓

Step 3: 匹配 C
┌─────────────────┐
│   C [6,7]       │
│   mamba: [99]✅ │ ← 检查：有值
│                 │    但循环已结束
└─────────────────┘    value = [kv_A, kv_B, kv_C]

         ↓

返回: value[:1] = [kv_A]
      ^^^^^^^^^^^^^^^^^^^
      只返回到上一个有效节点 (A)

┌─────────────────────────────────────────────────────────────┐
│  返回的数据：                                                │
│  ┌─────────┐                                                │
│  │  kv_A   │ ✅ 返回                                        │
│  └─────────┘                                                │
│  ┌─────────┐                                                │
│  │  kv_B   │ ❌ 被截断                                      │
│  └─────────┘                                                │
│  ┌─────────┐                                                │
│  │  kv_C   │ ❌ 被截断                                      │
│  └─────────┘                                                │
│                                                              │
│  cached_tokens = 3                                          │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.3 修改后代码的匹配和重计算过程

```
┌────────────────────────────────────────────────────────────────┐
│                  修改后代码：匹配和重计算过程                    │
└────────────────────────────────────────────────────────────────┘

Step 1: 匹配 A
┌─────────────────┐
│   A [1,2,3]     │
│   mamba: [42]✅ │ ← 检查：有值
│                 │    best_value_len = 1 ✅
└─────────────────┘    last_valid_mamba_node = A ⭐
                       last_valid_mamba_len = 1
                       value = [kv_A]
         ↓

Step 2: 匹配 B
┌─────────────────┐
│   B [4,5]       │
│   mamba: None❌ │ ← 检查：无值
│                 │    tombstone_encountered = True ⭐
└─────────────────┘    best_value_len 仍是 1
         │             value = [kv_A, kv_B]
         │ ⭐ 关键：标记但继续！
         ↓

Step 3: 匹配 C
┌─────────────────┐
│   C [6,7]       │
│   mamba: [99]✅ │ ← 检查：有值
│                 │    继续累积
└─────────────────┘    value = [kv_A, kv_B, kv_C]

         ↓

Step 4: 重计算检查
┌────────────────────────────────────────────────────────────┐
│ if tombstone_encountered:  # True ✅                       │
│     recompute_len = len(value) - last_valid_mamba_len     │
│                   = 3 - 1 = 2                              │
│                                                             │
│     调用 _try_rebuild_mamba_state(                         │
│         start_node = A,        ← 从这里开始                │
│         kv = [kv_B, kv_C],     ← 重计算这些                │
│         target_node = B        ← 目标是 B                  │
│     )                                                       │
└────────────────────────────────────────────────────────────┘

         ↓

Step 5: 重计算执行
┌────────────────────────────────────────────────────────────┐
│ _try_rebuild_mamba_state 内部:                             │
│                                                             │
│  1. start_mamba_idx = A.mamba_value[0] = 42               │
│                                                             │
│  2. new_mamba_idx = alloc(1)  →  [50]  ⭐                  │
│                                                             │
│  3. model_runner.recompute_mamba_state(                    │
│         start=42,                                           │
│         target=50,                                          │
│         kv=[kv_B, kv_C]                                    │
│     )                                                       │
│     # 可能只是 copy(42 → 50) 或 zero-init                  │
│     # 返回 True                                            │
│                                                             │
│  4. ⭐⭐⭐ B.mamba_value = [50]  ⭐⭐⭐                       │
│                                                             │
│  5. mamba_lru_list.insert_mru(B)                           │
│                                                             │
│  6. return B                                               │
└────────────────────────────────────────────────────────────┘

         ↓

Step 6: B 的状态变化
┌──────────────────────────────────────────────────────────┐
│  Before:                  After:                          │
│  ┌─────────────────┐      ┌─────────────────┐           │
│  │   B [4,5]       │      │   B [4,5]       │           │
│  │   mamba: None❌ │  →   │   mamba: [50]✅ │           │
│  └─────────────────┘      └─────────────────┘           │
│                                                           │
│  Tombstone              有效节点                         │
└──────────────────────────────────────────────────────────┘

         ↓

Step 7: 更新返回长度
┌────────────────────────────────────────────────────────────┐
│ if rebuilt_node is not None:  # B ≠ None ✅               │
│     best_value_len = len(value)  # 3 ⭐                   │
│     best_last_node = B           # ⭐                      │
└────────────────────────────────────────────────────────────┘

         ↓

返回: value[:3] = [kv_A, kv_B, kv_C]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      返回所有 KV！

┌─────────────────────────────────────────────────────────────┐
│  返回的数据：                                                │
│  ┌─────────┐                                                │
│  │  kv_A   │ ✅ 返回                                        │
│  └─────────┘                                                │
│  ┌─────────┐                                                │
│  │  kv_B   │ ✅ 返回（因为 B 现在有 mamba_value 了）        │
│  └─────────┘                                                │
│  ┌─────────┐                                                │
│  │  kv_C   │ ✅ 返回                                        │
│  └─────────┘                                                │
│                                                              │
│  cached_tokens = 7                                          │
│  提升 = (7-3)/3 = +133%                                     │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.4 执行后的树结构对比

```
┌────────────────────────────────────────────────────────────────┐
│                执行后的树结构对比                               │
└────────────────────────────────────────────────────────────────┘

原始代码（执行后）:                修改后（执行后）:

        Root                              Root
         │                                 │
    ┌────▼────┐                       ┌────▼────┐
    │    A    │                       │    A    │
    │ [1,2,3] │                       │ [1,2,3] │
    │ mamba:  │                       │ mamba:  │
    │   [42]  │ ✅                    │   [42]  │ ✅
    └────┬────┘                       └────┬────┘
         │                                 │
    ┌────▼────┐                       ┌────▼────┐
    │    B    │                       │    B    │
    │ [4,5]   │                       │ [4,5]   │
    │ mamba:  │                       │ mamba:  │
    │   None  │ ❌ Tombstone          │   [50]  │ ✅ 重计算后
    └────┬────┘                       └────┬────┘
         │                                 │
    ┌────▼────┐                       ┌────▼────┐
    │    C    │                       │    C    │
    │ [6,7]   │                       │ [6,7]   │
    │ mamba:  │                       │ mamba:  │
    │   [99]  │ ✅                    │   [99]  │ ✅
    └─────────┘                       └─────────┘

返回的 KV:                           返回的 KV:
  只到 A (1 个)                         到 C (3 个)
  cached_tokens = 3                    cached_tokens = 7

  提升 = 0%                            提升 = +133%
```

---

## 4. 完整示例演示

### 4.1 代码级别的对比

**原始代码关键逻辑**：

```python
# mamba_radix_cache.py (原始版本)

def _match_prefix_helper(self, key):
    best_value_len = 0

    while matching:
        # 检查点
        if node.mamba_value is not None:
            best_value_len = len(value)  # ⭐ 只有这里更新
        # ❌ 如果是 None，什么都不做

        value.append(child.value)  # 继续累积

    # ❌ 返回截断的结果
    return value[:best_value_len]
```

**修改后代码关键逻辑**：

```python
# mamba_radix_cache.py (修改后)

def _match_prefix_helper(self, key):
    best_value_len = 0
    last_valid_mamba_node = None  # ⭐ 新增
    tombstone_encountered = False # ⭐ 新增

    while matching:
        if node.mamba_value is not None:
            best_value_len = len(value)
            last_valid_mamba_node = node  # ⭐ 记录
        elif node != self.root_node:
            tombstone_encountered = True  # ⭐ 标记

        value.append(child.value)

    # ⭐ 新增：重计算逻辑
    if tombstone_encountered:
        rebuilt_node = _try_rebuild_mamba_state(...)
        if rebuilt_node:
            best_value_len = len(value)  # ⭐ 使用完整长度

    return value[:best_value_len]
```

---

### 4.2 实际运行示例

**场景设置**：

```python
# 树中已有这些缓存
tree:
  A: tokens=[1,2,3], mamba_value=[42]
  └─ B: tokens=[4,5], mamba_value=None (Tombstone，由节点分裂产生)
      └─ C: tokens=[6,7], mamba_value=[99]

# 用户请求
request: tokens=[1,2,3,4,5,6,7]
```

**原始代码执行**：

```python
# 调用 _match_prefix_helper([1,2,3,4,5,6,7])

# 匹配过程
node=A: value=[kv_A]
        A.mamba_value=[42] ✅
        best_value_len = 1
        best_last_node = A

node=B: value=[kv_A, kv_B]
        B.mamba_value=None ❌
        # best_value_len 仍是 1

node=C: value=[kv_A, kv_B, kv_C]
        # 循环结束

# 返回
return value[:1], A
# [kv_A], A

# 结果
cached_tokens = 3
remaining_tokens = 4 (需要重新计算)
```

**修改后代码执行**：

```python
# 调用 _match_prefix_helper([1,2,3,4,5,6,7])

# 匹配过程
node=A: value=[kv_A]
        A.mamba_value=[42] ✅
        best_value_len = 1
        last_valid_mamba_node = A
        last_valid_mamba_len = 1

node=B: value=[kv_A, kv_B]
        B.mamba_value=None ❌
        tombstone_encountered = True ⭐

node=C: value=[kv_A, kv_B, kv_C]
        # 继续累积

# 重计算检查
if tombstone_encountered:  # True
    recompute_len = 3 - 1 = 2

    rebuilt_node = _try_rebuild_mamba_state(
        start_node=A,
        kv=[kv_B, kv_C],
        target_node=B
    )

    # _try_rebuild_mamba_state 内部:
    new_idx = alloc(1)  # [50]
    model_runner.recompute_mamba_state(42, 50, [kv_B, kv_C])
    B.mamba_value = [50]  # ⭐⭐⭐ 关键！
    return B

    # 继续
    best_value_len = 3  # ⭐ 使用完整长度
    best_last_node = B

# 返回
return value[:3], B
# [kv_A, kv_B, kv_C], B

# 结果
cached_tokens = 7
remaining_tokens = 0
提升 = (7-3)/3 = +133%
```

---

## 5. 关键代码行详解

### Line 718: 最关键的一行

```python
# python/sglang/srt/mem_cache/mamba_radix_cache.py:718

target_node.mamba_value = new_mamba_idx
```

**为什么这一行如此关键**？

```python
# Before (Tombstone):
B.mamba_value = None
→ 系统判断：if B.mamba_value is not None → False
→ 结论：B 不可用
→ 返回时截断到 A

# After (执行 Line 718):
B.mamba_value = [50]
→ 系统判断：if B.mamba_value is not None → True
→ 结论：B 可用
→ 返回时可以包含 B

# 效果：
cached_tokens: 3 → 7 (+133%)
```

---

### Lines 946-954: Tombstone 检测逻辑

```python
# python/sglang/srt/mem_cache/mamba_radix_cache.py:946-954

if node.mamba_value is not None:
    best_value_len = len(value)
    best_last_node = node
    last_valid_mamba_node = node        # Line 949 ⭐
    last_valid_mamba_len = len(value)   # Line 950 ⭐
    tombstone_encountered = False       # Line 951
elif node != self.root_node and not tombstone_encountered:
    tombstone_encountered = True        # Line 954 ⭐
    # ⭐ 不 break，继续遍历
```

**作用**：

1. **Line 949-950**: 记录"回退点"，知道从哪里开始重计算
2. **Line 954**: 标记遇到了 Tombstone，但不终止遍历
3. **关键**: 让系统能完整遍历树，收集所有 KV

---

### Lines 976-1030: 重计算决策逻辑

```python
# python/sglang/srt/mem_cache/mamba_radix_cache.py:976-1030

if self.enable_recomputation and tombstone_encountered:  # Line 976
    recompute_len = len(value) - last_valid_mamba_len

    if recompute_len > 0 and recompute_len <= self.recompute_max_tokens:
        # 决定起始点
        if last_valid_mamba_node is None:
            start_node = None  # 从零开始
        else:
            start_node = last_valid_mamba_node  # 从有效节点开始

        # 调用重计算
        rebuilt_node = self._try_rebuild_mamba_state(...)

        if rebuilt_node is not None:
            best_value_len = len(value)  # ⭐ 使用完整长度
            best_last_node = rebuilt_node
```

**作用**：

1. 检查是否需要重计算
2. 确定重计算的起点和范围
3. 调用重计算
4. **成功后使用完整长度**（关键！）

---

## 6. 总结

### 6.1 核心改动

**3 个关键变量**：
```python
last_valid_mamba_node = None   # 记录回退点
last_valid_mamba_len = 0       # 记录回退位置
tombstone_encountered = False  # 标记是否遇到 tombstone
```

**1 个关键函数**：
```python
_try_rebuild_mamba_state(start_node, kv_indices, target_node)
# 给 tombstone 分配 mamba_value
```

**1 行关键代码**：
```python
target_node.mamba_value = new_mamba_idx  # Line 718
# 让 tombstone 变成有效节点
```

---

### 6.2 提升原理

```
原始代码:
  遇到 Tombstone → 停止计数 → 返回部分 KV
  ↓
  cached_tokens = 3

修改后:
  遇到 Tombstone → 标记 → 继续遍历 → 重计算 → 分配 mamba_value → 返回完整 KV
  ↓
  cached_tokens = 7

提升 = (7-3)/3 = +133%
```

---

### 6.3 关键点

1. **不是重计算本身提升了性能**（重计算可能只是 copy 或 zero-init）
2. **是给 Tombstone 分配了 mamba_value**，让系统认为它"可用"
3. **系统只检查 `is not None`**，不检查内容
4. **Line 718 是整个优化的核心**

---

这就是完整的代码修改详解！通过对比、流程图和示例，应该能清楚地看到修改是如何让 cached token 大幅增加的。
