# ARC (Adaptive Replacement Cache) Implementation in SGLang

## 概述

本文档描述了在SGLang中实现的自适应替换缓存（ARC）算法。ARC是一种自适应缓存替换策略，它能够根据工作负载自动平衡最近访问（Recency）和频繁访问（Frequency）两种模式。

## ARC算法原理

### 核心思想

ARC维护四个列表来管理缓存：

1. **T1 (Recent Cache)**: 存储最近访问一次的页面
2. **T2 (Frequent Cache)**: 存储最近访问多次的页面
3. **B1 (Ghost T1)**: 存储从T1中驱逐的页面的元数据（不占用实际缓存空间）
4. **B2 (Ghost T2)**: 存储从T2中驱逐的页面的元数据（不占用实际缓存空间）

### 自适应参数 p

ARC使用自适应参数 `p` 来动态调整T1的目标大小：
- `|T1| + |T2| ≤ c` （c是缓存总容量）
- `p` 是T1的目标大小
- `c - p` 是T2的目标大小

参数 `p` 根据工作负载动态调整：
- 如果在B1中发生缓存未命中（说明需要更多recency保护），增加 `p`
- 如果在B2中发生缓存未命中（说明需要更多frequency保护），减少 `p`

### 优势

相比传统的LRU和LFU算法：
- **自适应性**: 自动适应不同的工作负载模式
- **无需调参**: 不需要手动设置参数来平衡recency和frequency
- **扫描抗性**: 对顺序扫描等访问模式具有更好的抗性
- **性能保证**: 在最坏情况下不会比LRU差

## 实现细节

### 文件修改

#### 1. `evict_policy.py`

新增了两个类：

**ARCManager**: ARC缓存管理器
```python
class ARCManager:
    def __init__(self, cache_size: int)
    def register_node(self, node: TreeNode)
    def unregister_node(self, node: TreeNode)
    def on_cache_hit(self, node: TreeNode)
    def on_cache_miss(self, node: TreeNode)
    def should_evict_from_t1(self) -> bool
    def on_eviction(self, node: TreeNode, keep_ghost: bool = True)
    def get_stats(self) -> Dict[str, int]
```

**ARCStrategy**: ARC驱逐策略
```python
class ARCStrategy(EvictionStrategy):
    def __init__(self, arc_manager: ARCManager)
    def get_priority(self, node: TreeNode) -> Tuple[int, float, float]
```

#### 2. `radix_cache.py`

**TreeNode扩展**: 添加了ARC相关属性
```python
class TreeNode:
    # ARC specific attributes
    self.arc_list_type: Optional[str] = None  # 'T1', 'T2', 'B1', 'B2', or None
    self.in_ghost: bool = False  # Whether this is a ghost entry
```

**RadixCache修改**: 集成ARC策略
- 在`__init__`中支持`eviction_policy="arc"`
- 在`_match_prefix_helper`中调用`arc_manager.on_cache_hit()`
- 在`evict`方法中调用`arc_manager.on_eviction()`

### 数据结构

```
┌─────────────────────────────────────────────────────────┐
│                    ARC Cache Structure                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  T1 (Recent)          T2 (Frequent)                    │
│  ┌─────────┐          ┌─────────┐                      │
│  │ Node A  │          │ Node X  │  ← Actual cache data │
│  │ Node B  │          │ Node Y  │                      │
│  │ Node C  │          │ Node Z  │                      │
│  └─────────┘          └─────────┘                      │
│       ↓                    ↓                            │
│  B1 (Ghost T1)       B2 (Ghost T2)                     │
│  ┌─────────┐          ┌─────────┐                      │
│  │ Node D  │          │ Node W  │  ← Metadata only     │
│  │ Node E  │          │ Node V  │                      │
│  └─────────┘          └─────────┘                      │
│                                                         │
│  |T1| + |T2| ≤ cache_size                              │
│  |B1| ≤ cache_size                                     │
│  |B2| ≤ cache_size                                     │
│                                                         │
│  Adaptive parameter p:                                 │
│  Target |T1| = p                                       │
│  Target |T2| = cache_size - p                          │
└─────────────────────────────────────────────────────────┘
```

## 使用方法

### 基本使用

在创建RadixCache时，指定`eviction_policy="arc"`:

```python
from sglang.srt.mem_cache.radix_cache import RadixCache

cache = RadixCache(
    req_to_token_pool=req_to_token_pool,
    token_to_kv_pool_allocator=allocator,
    page_size=1,
    eviction_policy="arc",  # 使用ARC策略
)
```

### 启动服务器时使用ARC

在启动SGLang服务器时，可以通过参数指定使用ARC策略：

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --eviction-policy arc \
    --port 30000
```

### 查看ARC统计信息

```python
if cache.arc_manager:
    stats = cache.arc_manager.get_stats()
    print(f"T1 size: {stats['T1_size']}")
    print(f"T2 size: {stats['T2_size']}")
    print(f"B1 size: {stats['B1_size']}")
    print(f"B2 size: {stats['B2_size']}")
    print(f"Adaptive parameter p: {stats['p']}")
```

## 测试

运行测试脚本：

```bash
cd /home/user/sglang
python python/sglang/srt/mem_cache/test_arc_cache.py
```

测试包括：
1. 基本ARC功能测试
2. ARC驱逐行为测试
3. ARC自适应参数调整测试
4. ARC vs LRU vs LFU性能对比

## 性能分析

### 时间复杂度

- **缓存命中**: O(1) - 更新列表状态
- **缓存未命中**: O(1) - 更新ghost列表和参数p
- **驱逐**: O(log n) - 使用堆选择驱逐节点

### 空间复杂度

- **缓存数据**: O(c) - c是缓存容量
- **Ghost条目**: O(c) - 最多存储c个ghost条目的元数据
- **总空间**: O(c) - ghost条目只存储元数据，不存储实际数据

## 适用场景

ARC特别适合以下场景：

1. **混合工作负载**: 同时包含顺序访问和随机访问
2. **未知访问模式**: 访问模式在运行时变化
3. **长对话**: 某些prompt会被频繁重用
4. **多用户场景**: 不同用户有不同的访问模式

### vs LRU

- LRU适合主要是最近访问的场景
- ARC在混合场景下表现更好，能自动适应

### vs LFU

- LFU适合主要是频繁访问的场景
- ARC能同时处理recency和frequency

## 配置参数

### eviction_policy

- **类型**: str
- **可选值**: "lru", "lfu", "fifo", "mru", "filo", "arc"
- **默认值**: "lru"
- **描述**: 缓存驱逐策略

### cache_size

ARC的cache_size会自动从`token_to_kv_pool_allocator.size`获取。

## 调试和监控

### 启用调试日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 监控ARC状态

可以定期调用`arc_manager.get_stats()`来监控ARC的状态：

```python
import time

while True:
    if cache.arc_manager:
        stats = cache.arc_manager.get_stats()
        print(f"[{time.time()}] ARC Stats: {stats}")
    time.sleep(10)  # 每10秒监控一次
```

## 已知限制

1. **Ghost条目大小**: 当前实现中，B1和B2的大小限制为cache_size，在某些极端情况下可能不够
2. **分布式场景**: 当前实现是单机版本，分布式场景下需要额外的同步机制
3. **页面大小**: 当page_size > 1时，ARC的粒度是以页为单位

## 未来改进方向

1. **自适应ghost列表大小**: 根据工作负载动态调整B1和B2的大小
2. **分布式ARC**: 支持多机器间的ARC协调
3. **ARC-2**: 实现ARC的改进版本，如CAR（Clock with Adaptive Replacement）
4. **性能优化**: 使用更高效的数据结构来管理列表

## 参考文献

1. Megiddo, N., & Modha, D. S. (2003). "ARC: A Self-Tuning, Low Overhead Replacement Cache." FAST'03.
2. Bansal, S., & Modha, D. S. (2004). "CAR: Clock with Adaptive Replacement." FAST'04.

## 贡献者

本实现由Claude AI助手完成，集成到SGLang项目中。

## 许可证

本实现遵循SGLang项目的Apache License 2.0许可证。
