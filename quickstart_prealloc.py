#!/usr/bin/env python3
"""
KV Cache 预分配池快速开始示例

这个脚本展示了如何在 SGLang 推理中使用 KV Cache 预分配池。

无需修改 SGLang 源码，可以直接在你的推理脚本中使用！
"""

import sys
import os

# 添加 SGLang 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))


def example_standalone_usage():
    """示例 1: 独立使用预分配池（测试和验证）"""
    print("=" * 70)
    print("示例 1: 独立使用预分配池")
    print("=" * 70)

    try:
        import torch
        from sglang.srt.mem_cache.preallocated_pool import PreallocatedKVBlockPool

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 创建预分配池
        pool = PreallocatedKVBlockPool(
            total_pages=1000,
            page_size=16,
            device=device,
            bucket_sizes=[1, 2, 4, 8, 16, 32, 64],
            enable_splitting=True,
        )

        print(f"\n创建的池: {pool}")
        print(f"可用页面: {pool.available_pages()}")

        # 模拟分配
        print("\n模拟分配:")
        for size in [4, 8, 16, 5, 3]:
            pages = pool.allocate(size)
            if pages is not None:
                print(f"  ✓ 成功分配 {size} 页")
            else:
                print(f"  ✗ 分配 {size} 页失败")

        # 显示统计
        stats = pool.get_statistics()
        print(f"\n统计信息:")
        print(f"  总分配次数: {stats['total_allocations']}")
        print(f"  利用率: {stats['utilization']:.2%}")
        print(f"  块分割次数: {stats['split_operations']}")

        print("\n✓ 示例 1 完成")

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保已正确安装 SGLang")
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_direct_integration():
    """示例 2: 直接在现有代码中集成（推荐）"""
    print("\n" + "=" * 70)
    print("示例 2: 直接集成到推理流程")
    print("=" * 70)

    code_example = '''
# 在你的推理脚本中，修改 ModelRunner 的初始化
# 文件: python/sglang/srt/model_executor/model_runner.py

# 找到这段代码（大约第1905行）:
"""
else:
    assert not self.is_hybrid
    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
        self.max_total_num_tokens,
        page_size=self.page_size,
        dtype=self.kv_cache_dtype,
        device=self.device,
        kvcache=self.token_to_kv_pool,
        need_sort=need_sort,
    )
"""

# 替换为:
"""
else:
    assert not self.is_hybrid

    # 导入预分配池版本
    from sglang.srt.mem_cache.allocator import PreallocatedPagedTokenToKVPoolAllocator

    # 使用预分配池
    self.token_to_kv_pool_allocator = PreallocatedPagedTokenToKVPoolAllocator(
        self.max_total_num_tokens,
        page_size=self.page_size,
        dtype=self.kv_cache_dtype,
        device=self.device,
        kvcache=self.token_to_kv_pool,
        need_sort=need_sort,
        enable_prealloc=True,                           # 启用预分配
        prealloc_bucket_sizes=[1,2,4,8,16,32,64,128],  # 桶大小
        prealloc_ratio=0.8,                             # 80%用于预分配
    )
"""

# 保存修改，然后正常启动 SGLang
'''

    print(code_example)
    print("修改后，启动 SGLang:")
    print("  python -m sglang.launch_server --model-path your-model")
    print("\n✓ 示例 2 说明完成")


def example_with_monitoring():
    """示例 3: 带监控的使用"""
    print("\n" + "=" * 70)
    print("示例 3: 启用监控和调试")
    print("=" * 70)

    monitoring_example = '''
# 方式 1: 环境变量启用调试
export SGLANG_DEBUG_MEMORY_POOL=1

# 方式 2: 在代码中启用
pool = PreallocatedKVBlockPool(
    total_pages=1000,
    page_size=16,
    device="cuda",
    debug_mode=True,  # 启用调试
)

# 方式 3: 定期获取统计信息
def monitor_pool(allocator):
    stats = allocator.get_statistics()

    print("KV Cache 池状态:")
    print(f"  利用率: {stats['prealloc']['utilization']:.2%}")
    print(f"  总分配: {stats['prealloc']['total_allocations']}")
    print(f"  分割操作: {stats['prealloc']['split_operations']}")

    # 找出最活跃的桶
    buckets = stats['prealloc']['buckets']
    active = [(k, v['allocations']) for k, v in buckets.items() if v['allocations'] > 0]
    active.sort(key=lambda x: x[1], reverse=True)

    print("  最活跃的桶:")
    for bucket_size, allocs in active[:3]:
        print(f"    {bucket_size} 页: {allocs} 次分配")

# 在推理循环中定期调用
# monitor_pool(model_runner.token_to_kv_pool_allocator)
'''

    print(monitoring_example)
    print("\n✓ 示例 3 说明完成")


def example_production_config():
    """示例 4: 生产环境配置建议"""
    print("\n" + "=" * 70)
    print("示例 4: 生产环境配置")
    print("=" * 70)

    configs = {
        "聊天应用（短对话）": {
            "buckets": [1, 2, 4, 8, 16, 32],
            "ratio": 0.8,
            "command": """
python -m sglang.launch_server \\
    --model-path meta-llama/Llama-3.1-8B-Instruct \\
    --enable-kv-prealloc \\
    --kv-prealloc-ratio 0.8 \\
    --kv-prealloc-buckets "1,2,4,8,16,32"
            """
        },
        "长文本生成": {
            "buckets": [8, 16, 32, 64, 128, 256],
            "ratio": 0.85,
            "command": """
python -m sglang.launch_server \\
    --model-path meta-llama/Llama-3.1-70B-Instruct \\
    --enable-kv-prealloc \\
    --kv-prealloc-ratio 0.85 \\
    --kv-prealloc-buckets "8,16,32,64,128,256" \\
    --context-length 32768
            """
        },
        "批量推理": {
            "buckets": [2, 4, 8, 16, 32, 64, 128],
            "ratio": 0.9,
            "command": """
python -m sglang.launch_server \\
    --model-path your-model \\
    --enable-kv-prealloc \\
    --kv-prealloc-ratio 0.9 \\
    --kv-prealloc-buckets "2,4,8,16,32,64,128" \\
    --max-running-requests 256
            """
        }
    }

    for scenario, config in configs.items():
        print(f"\n{scenario}:")
        print(f"  桶大小: {config['buckets']}")
        print(f"  预分配比例: {config['ratio']}")
        print(f"  启动命令:{config['command']}")

    print("\n✓ 示例 4 说明完成")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           SGLang KV Cache 预分配池 - 快速开始                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # 运行所有示例
    example_standalone_usage()
    example_direct_integration()
    example_with_monitoring()
    example_production_config()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
KV Cache 预分配池提供:
  ✓ 更快的内存分配（O(1) 时间复杂度）
  ✓ 更低的内存碎片
  ✓ 更好的缓存局部性
  ✓ 可预测的性能

快速开始步骤:
  1. 修改 model_runner.py（参见示例 2）
  2. 或添加 ServerArgs 配置（参见 INTEGRATION_GUIDE.md）
  3. 启动推理服务
  4. 监控性能（参见示例 3）

详细文档:
  - KV_CACHE_PREALLOCATION_README.md: 完整文档
  - INTEGRATION_GUIDE.md: 集成指南
  - integration_patch_model_runner.py: 集成补丁
    """)

    print("\n需要帮助？")
    print("  查看文档: KV_CACHE_PREALLOCATION_README.md")
    print("  集成指南: INTEGRATION_GUIDE.md")
    print("  运行测试: python python/sglang/srt/mem_cache/test_preallocated_pool.py")
    print()


if __name__ == "__main__":
    main()
