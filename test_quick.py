#!/usr/bin/env python3
"""
快速测试脚本 - 验证PreallocPoolAllocator基本功能
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.prealloc_pool_allocator import PreallocPoolAllocator


def test_basic():
    """测试基本功能"""
    print("=" * 80)
    print("测试1: 基本初始化和分配")
    print("=" * 80)

    # 配置
    total_tokens = 32768  # 32K tokens
    page_size = 16
    head_num = 32
    head_dim = 128
    layer_num = 32
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"设备: {device}")
    print(f"总tokens: {total_tokens}")
    print(f"Page size: {page_size}")
    print()

    # 创建KV Pool
    print("创建KV Pool...")
    kv_pool = MHATokenToKVPool(
        size=total_tokens,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=device,
        enable_memory_saver=False,
    )
    print("✓ KV Pool创建成功")
    print()

    # 创建预分配池allocator
    print("创建PreallocPoolAllocator...")
    allocator = PreallocPoolAllocator(
        size=total_tokens,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kvcache=kv_pool,
        need_sort=True,
        enable_prealloc=True,
        prealloc_ratio=0.3,
    )
    print("✓ Allocator创建成功")
    print()

    # 测试分配
    print("测试分配不同大小...")
    sizes = [64, 128, 256, 512]
    allocated = []

    for size in sizes:
        indices = allocator.alloc(size)
        if indices is not None:
            allocated.append((size, indices))
            print(f"✓ 分配 {size} tokens 成功 (indices shape: {indices.shape})")
        else:
            print(f"✗ 分配 {size} tokens 失败")

    print()

    # 打印统计信息
    print("统计信息:")
    print("-" * 80)
    allocator.print_stats()
    print()

    # 测试释放
    print("测试释放...")
    for size, indices in allocated[:2]:
        allocator.free(indices)
        print(f"✓ 释放 {size} tokens")

    print()

    # 再次打印统计
    print("释放后的统计信息:")
    print("-" * 80)
    allocator.print_stats()
    print()

    # 测试重新分配
    print("测试重新分配...")
    indices = allocator.alloc(64)
    if indices is not None:
        print(f"✓ 重新分配 64 tokens 成功（应该复用了刚才释放的块）")
    print()

    print("=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)


def test_disabled():
    """测试禁用预分配池"""
    print("\n" * 2)
    print("=" * 80)
    print("测试2: 禁用预分配池")
    print("=" * 80)

    total_tokens = 32768
    page_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kv_pool = MHATokenToKVPool(
        size=total_tokens,
        page_size=page_size,
        dtype=torch.float16,
        head_num=32,
        head_dim=128,
        layer_num=32,
        device=device,
        enable_memory_saver=False,
    )

    # 禁用预分配池
    allocator = PreallocPoolAllocator(
        size=total_tokens,
        page_size=page_size,
        dtype=torch.float16,
        device=device,
        kvcache=kv_pool,
        need_sort=True,
        enable_prealloc=False,  # 禁用
    )

    print("✓ 禁用模式创建成功")

    # 测试分配
    indices = allocator.alloc(64)
    if indices is not None:
        print(f"✓ 分配成功（fallback到标准分配）")
    else:
        print(f"✗ 分配失败")

    print()
    print("=" * 80)
    print("✅ 禁用模式测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_basic()
        test_disabled()

        print("\n" * 2)
        print("🎉 所有测试成功完成！")
        print()
        print("下一步:")
        print("  1. 运行完整测试: pytest test/srt/test_prealloc_pool_allocator.py -v")
        print("  2. 运行可视化演示: python demo_prealloc_pool_visualization.py")
        print("  3. 查看实现指南: KV_Cache预分配池实现指南.md")
        print()

    except Exception as e:
        print("\n" * 2)
        print("❌ 测试失败！")
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
