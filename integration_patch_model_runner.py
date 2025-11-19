"""
补丁文件: 将 KV Cache 预分配池集成到 model_runner.py

使用方法:
1. 查看这个文件了解需要修改的位置
2. 手动应用这些修改到 python/sglang/srt/model_executor/model_runner.py
3. 或者运行此脚本自动应用补丁（需谨慎，建议先备份）

补丁内容:
"""

# ============================================================================
# 修改 1: 在文件开头添加导入
# ============================================================================
IMPORT_PATCH = """
# 在 python/sglang/srt/model_executor/model_runner.py 顶部的导入区域添加:

from sglang.srt.mem_cache.allocator import (
    # ... 现有导入 ...
    PagedTokenToKVPoolAllocator,
    PreallocatedPagedTokenToKVPoolAllocator,  # <-- 新增这一行
    TokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
"""

# ============================================================================
# 修改 2: 在 init_token_to_kv_pool_allocator 方法中集成预分配池
# ============================================================================
ALLOCATOR_INIT_PATCH = """
# 在 init_token_to_kv_pool_allocator 方法中（大约第1905行）
# 找到 PagedTokenToKVPoolAllocator 的初始化

# 原代码:
# --------
# else:
#     assert not self.is_hybrid
#     self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
#         self.max_total_num_tokens,
#         page_size=self.page_size,
#         dtype=self.kv_cache_dtype,
#         device=self.device,
#         kvcache=self.token_to_kv_pool,
#         need_sort=need_sort,
#     )

# 替换为:
# --------
else:
    assert not self.is_hybrid

    # 检查是否启用KV预分配池
    enable_prealloc = getattr(self.server_args, 'enable_kv_prealloc', False)

    if enable_prealloc:
        # 获取配置参数
        prealloc_ratio = getattr(self.server_args, 'kv_prealloc_ratio', 0.8)
        prealloc_buckets_str = getattr(self.server_args, 'kv_prealloc_buckets', None)

        # 解析桶大小配置
        if prealloc_buckets_str:
            try:
                prealloc_buckets = [int(x.strip()) for x in prealloc_buckets_str.split(',')]
            except ValueError:
                logger.warning(f"Invalid kv_prealloc_buckets format: {prealloc_buckets_str}, using defaults")
                prealloc_buckets = None
        else:
            prealloc_buckets = None  # 使用默认值 [1,2,4,8,16,32,64,128]

        # 使用预分配池版本
        self.token_to_kv_pool_allocator = PreallocatedPagedTokenToKVPoolAllocator(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=self.token_to_kv_pool,
            need_sort=need_sort,
            enable_prealloc=True,
            prealloc_bucket_sizes=prealloc_buckets,
            prealloc_ratio=prealloc_ratio,
        )

        logger.info(
            f"KV Cache preallocation enabled: ratio={prealloc_ratio}, "
            f"buckets={prealloc_buckets or 'default'}"
        )
    else:
        # 使用标准版本
        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=self.token_to_kv_pool,
            need_sort=need_sort,
        )
"""

# ============================================================================
# 修改 3: 添加统计信息打印（可选）
# ============================================================================
STATS_PATCH = """
# 在模型初始化完成后添加统计信息打印（可选）
# 例如在 __init__ 方法的末尾或合适的位置

def print_kv_cache_stats(self):
    '''打印KV Cache统计信息'''
    if hasattr(self.token_to_kv_pool_allocator, 'get_statistics'):
        stats = self.token_to_kv_pool_allocator.get_statistics()
        logger.info("KV Cache Allocator Statistics:")
        logger.info(f"  Total pages: {stats['total_pages']}")
        logger.info(f"  Page size: {stats['page_size']}")
        logger.info(f"  Total available: {stats['total_available_size']}")

        if 'prealloc' in stats and stats['enable_prealloc']:
            prealloc = stats['prealloc']
            logger.info(f"  Prealloc enabled: Yes")
            logger.info(f"  Prealloc utilization: {prealloc['utilization']:.2%}")
            logger.info(f"  Total allocations: {prealloc['total_allocations']}")
            logger.info(f"  Split operations: {prealloc['split_operations']}")

            # 打印最活跃的桶
            if prealloc['buckets']:
                sorted_buckets = sorted(
                    prealloc['buckets'].items(),
                    key=lambda x: x[1]['allocations'],
                    reverse=True
                )
                logger.info("  Top 3 active buckets:")
                for bucket_size, bucket_stats in sorted_buckets[:3]:
                    logger.info(
                        f"    Bucket {bucket_size}: "
                        f"{bucket_stats['allocations']} allocs, "
                        f"{bucket_stats['free_pages']} free pages"
                    )
"""

# ============================================================================
# 自动应用脚本
# ============================================================================
def apply_patch_automatically():
    """自动应用补丁（谨慎使用）"""
    import os
    import shutil

    model_runner_path = "python/sglang/srt/model_executor/model_runner.py"

    if not os.path.exists(model_runner_path):
        print(f"错误: 找不到 {model_runner_path}")
        return False

    # 备份原文件
    backup_path = model_runner_path + ".backup"
    shutil.copy2(model_runner_path, backup_path)
    print(f"已备份原文件到: {backup_path}")

    # 读取文件
    with open(model_runner_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 应用补丁 1: 添加导入
    if "PreallocatedPagedTokenToKVPoolAllocator" not in content:
        # 找到 PagedTokenToKVPoolAllocator 的导入位置
        import_marker = "from sglang.srt.mem_cache.allocator import"
        if import_marker in content:
            # 简单的导入添加（实际应用中需要更精确的处理）
            content = content.replace(
                "from sglang.srt.mem_cache.allocator import (",
                "from sglang.srt.mem_cache.allocator import (\n    PreallocatedPagedTokenToKVPoolAllocator,  # KV Cache预分配池"
            )
            print("✓ 已添加 PreallocatedPagedTokenToKVPoolAllocator 导入")

    # 应用补丁 2: 修改 allocator 初始化
    # 这部分较复杂，建议手动应用

    print("\n警告: 自动补丁只能部分应用。")
    print("请手动完成以下步骤:")
    print("1. 检查导入是否正确添加")
    print("2. 在 init_token_to_kv_pool_allocator 方法中添加预分配池逻辑")
    print("3. 参考 ALLOCATOR_INIT_PATCH 中的代码")

    # 写回文件
    with open(model_runner_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n已修改: {model_runner_path}")
    print(f"备份文件: {backup_path}")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache 预分配池集成补丁")
    print("=" * 70)
    print()
    print("此补丁文件包含将 KV Cache 预分配池集成到 model_runner.py 的说明")
    print()

    print("修改 1: 添加导入")
    print("-" * 70)
    print(IMPORT_PATCH)
    print()

    print("修改 2: 集成预分配池到 allocator 初始化")
    print("-" * 70)
    print(ALLOCATOR_INIT_PATCH)
    print()

    print("修改 3: 添加统计信息（可选）")
    print("-" * 70)
    print(STATS_PATCH)
    print()

    print("=" * 70)
    print("建议:")
    print("1. 手动应用这些修改以确保准确性")
    print("2. 或运行 apply_patch_automatically() 进行自动应用（需要手动完成部分步骤）")
    print("3. 应用后记得重启服务器测试")
    print("=" * 70)
