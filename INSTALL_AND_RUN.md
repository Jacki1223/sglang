# 安装和运行带ARC缓存的SGLang

## 问题诊断

如果你看到这样的错误：
```
TypeError: RadixCache.__init__() got an unexpected keyword argument 'enable_deterministic_inference'
```

这说明你运行的是系统安装的旧版SGLang，而不是我们修改后的本地版本。

## 解决方案：从本地源码安装

### 步骤1: 卸载旧版本（如果有）

```bash
pip uninstall sglang -y
```

### 步骤2: 从本地源码安装

```bash
cd /home/user/sglang

# 确保在正确的分支
git checkout claude/arc-adaptive-caching-01Hg8bDBw626MiuEFygs6cxd
git pull origin claude/arc-adaptive-caching-01Hg8bDBw626MiuEFygs6cxd

# 安装开发模式（推荐，修改代码后无需重新安装）
pip install -e "python[all]"

# 或者正常安装
# pip install "python[all]"
```

### 步骤3: 验证安装

```bash
# 验证sglang使用的是本地版本
python -c "import sglang; print('SGLang location:', sglang.__file__)"

# 应该显示类似：
# SGLang location: /home/user/sglang/python/sglang/__init__.py

# 验证ARC支持
python -c "from sglang.srt.mem_cache.evict_policy import ARCManager, ARCStrategy; print('ARC classes imported successfully!')"
```

### 步骤4: 运行测试

```bash
cd /home/user/sglang

# 运行ARC单元测试
python python/sglang/srt/mem_cache/test_arc_cache.py

# 应该看到所有测试通过✓
```

### 步骤5: 运行benchmark

现在你可以运行benchmark了：

```bash
# 简单测试
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-prompts 50 \
    --radix-eviction-policy arc

# 完整对比测试
chmod +x benchmark_arc_example.sh
./benchmark_arc_example.sh meta-llama/Meta-Llama-3.1-8B-Instruct 100
```

## 替代方案：使用开发环境

如果pip install有问题，你也可以直接设置PYTHONPATH：

```bash
cd /home/user/sglang

# 设置PYTHONPATH
export PYTHONPATH=/home/user/sglang/python:$PYTHONPATH

# 验证
python -c "import sglang; print(sglang.__file__)"

# 运行benchmark
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-prompts 50 \
    --radix-eviction-policy arc
```

## 常见问题

### Q1: pip install -e 失败怎么办？

如果看到权限错误，可以加 `--user` 标志：
```bash
pip install -e "python[all]" --user
```

或者使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate
pip install -e "python[all]"
```

### Q2: 如何确认使用的是本地版本？

运行：
```bash
python -c "import sglang; print(sglang.__file__)"
```

应该显示 `/home/user/sglang/python/sglang/__init__.py`，而不是 `/usr/local/python3/lib/...`

### Q3: 修改代码后需要重新安装吗？

如果使用 `pip install -e`（开发模式），不需要重新安装。
如果使用 `pip install`（正常模式），需要重新安装。

## 快速检查清单

- [ ] 卸载旧版SGLang: `pip uninstall sglang -y`
- [ ] 切换到正确的分支: `git checkout claude/arc-adaptive-caching-01Hg8bDBw626MiuEFygs6cxd`
- [ ] 安装本地版本: `pip install -e "python[all]"`
- [ ] 验证安装位置: `python -c "import sglang; print(sglang.__file__)"`
- [ ] 运行测试: `python python/sglang/srt/mem_cache/test_arc_cache.py`
- [ ] 运行benchmark: `python -m sglang.bench_offline_throughput --radix-eviction-policy arc ...`

## 成功标志

当你看到benchmark运行并显示类似输出时，说明成功了：

```
============================================================
 Offline Throughput Benchmark Result
============================================================
Backend:                                 engine
Successful requests:                     100
Benchmark duration (s):                  45.23
Total input tokens:                      51200
Total generated tokens:                  25600
...
============================================================
```

现在你已经准备好测试ARC缓存了！🎉
