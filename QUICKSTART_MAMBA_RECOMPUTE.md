# Mamba State Recomputation - 快速开始

## 🚀 5分钟快速上手

### 步骤1: 检查文件 (30秒)

确认以下文件已创建：

```bash
ls -la mamba_recompute_*
```

你应该看到：
- `mamba_recompute_patch_1_server_args.py` - 配置参数
- `mamba_recompute_patch_2_radix_cache.py` - Cache增强
- `mamba_recompute_patch_3_model_runner.py` - 重计算接口
- `apply_mamba_recompute_patches.sh` - 应用脚本
- `test_mamba_recompute.py` - 测试脚本

### 步骤2: 应用Patches (2分钟)

```bash
# 运行自动应用脚本
bash apply_mamba_recompute_patches.sh
```

### 步骤3: 手动集成核心代码 (2分钟)

#### 3.1 修改 MambaRadixCache

打开 `python/sglang/srt/mem_cache/mamba_radix_cache.py`

**在`__init__`方法中添加**（约323行）：

```python
# 在参数列表中添加
enable_recomputation: bool = False,
recompute_max_tokens: int = 512,
prioritize_mamba_retention: bool = True,
mamba_eviction_threshold: float = 0.8,
model_runner=None,

# 在方法体中添加
self.enable_recomputation = enable_recomputation
self.recompute_max_tokens = recompute_max_tokens
self.prioritize_mamba_retention = prioritize_mamba_retention
self.mamba_eviction_threshold = mamba_eviction_threshold
self.model_runner = model_runner
self.recompute_hit_count = 0
self.recompute_miss_count = 0
self.recompute_skip_count = 0
```

**在文件末尾添加** `_try_rebuild_mamba_state` 方法：

从 `mamba_recompute_patch_2_radix_cache.py` 复制该方法的完整实现。

#### 3.2 修改 ModelRunner

打开 `python/sglang/srt/model_executor/model_runner.py`

**在ModelRunner类中添加** `recompute_mamba_state` 方法：

从 `mamba_recompute_patch_3_model_runner.py` 复制该方法的完整实现。

**在`_init_cache_engine`方法中添加**：

```python
if isinstance(self.tree_cache, MambaRadixCache):
    self.tree_cache.model_runner = self
    self.tree_cache.enable_recomputation = self.server_args.enable_mamba_state_recomputation
    # ... 其他配置
```

### 步骤4: 启动测试 (30秒)

```bash
# 启动server
python -m sglang.launch_server \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --enable-mamba-state-recomputation \
    --port 30000

# 运行测试（另一个终端）
python test_mamba_recompute.py --url http://localhost:30000
```

---

## ⚡ 最小化示例

如果你只想快速验证概念，可以只做核心修改：

### 1. 添加配置参数到ServerArgs

```python
# 在 python/sglang/srt/server_args.py 的 ServerArgs 类中添加：
enable_mamba_state_recomputation: bool = False
mamba_recompute_max_tokens: int = 512
```

### 2. 在MambaRadixCache中启用基础重计算

```python
# 修改 _match_prefix_helper，在返回前添加：
if self.enable_recomputation and len(tombstone_path) > 0:
    logger.info(f"Detected {len(tombstone_path)} tombstone nodes")
    # 基础重计算逻辑
```

### 3. 启动并观察日志

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --enable-mamba-state-recomputation

# 查看日志中是否有 tombstone 检测信息
```

---

## 📊 验证结果

### 成功的标志：

1. **启动时**：日志显示
   ```
   MambaRadixCache recomputation enabled: True
   ```

2. **运行时**：日志显示
   ```
   Mamba state recomputed successfully: 128 tokens, total hits: 5
   ```

3. **统计信息**：
   ```python
   {
     "recompute_hit_count": 10,
     "recompute_miss_count": 2,
     "hit_rate": 0.833
   }
   ```

### 失败的标志：

1. ❌ Cache hit 仍然为 0
2. ❌ 没有 "recomputed successfully" 日志
3. ❌ 频繁出现 "recomputation failed" 警告

**解决方法**：查看详细的 [故障排查指南](MAMBA_RECOMPUTE_README.md#-故障排查)

---

## 🎯 下一步

1. **调优参数**: 根据你的workload调整 `mamba_recompute_max_tokens`
2. **监控性能**: 使用profiling工具观察实际性能影响
3. **生产部署**: 在测试环境充分验证后再部署到生产

详细信息请查看 [完整文档](MAMBA_RECOMPUTE_README.md)

---

## 💡 提示

- 首次集成建议先在小模型上测试
- 使用 `--enable-profile` 查看详细性能数据
- 定期检查 recomputation stats 调整参数
- 遇到问题先查看 server 日志

---

**快速链接**:
- [完整文档](MAMBA_RECOMPUTE_README.md)
- [Patch文件说明](mamba_recompute_patch_1_server_args.py)
- [测试脚本](test_mamba_recompute.py)
