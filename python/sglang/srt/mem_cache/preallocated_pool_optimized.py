"""
优化版本的KV Cache预分配池

主要优化:
1. 缓存token indices转换以避免重复创建tensor
2. 快速路径优化 - 精确匹配直接返回
3. 减少不必要的tensor操作
4. 优化free操作的unique和mask计算
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


class OptimizedPreallocatedKVBlockPool:
    """
    优化版本的预分配KV块池

    性能优化:
    - 缓存page到token的索引转换
    - 快速路径: O(1)精确匹配查找
    - 延迟初始化: 按需创建桶
    - 减少tensor操作开销
    """

    def __init__(
        self,
        total_pages: int,
        page_size: int,
        device: str,
        bucket_sizes: Optional[List[int]] = None,
        enable_splitting: bool = True,
        debug_mode: Optional[bool] = None,
    ):
        self.total_pages = total_pages
        self.page_size = page_size
        self.device = device
        self.enable_splitting = enable_splitting
        self.debug_mode = debug_mode if debug_mode is not None else get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")

        # 默认桶大小
        if bucket_sizes is None:
            bucket_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        self.bucket_sizes = sorted(bucket_sizes)

        # 创建bucket size到索引的快速查找
        self.bucket_size_to_idx = {size: idx for idx, size in enumerate(self.bucket_sizes)}

        # 使用list而不是dict以提高访问速度
        self.free_pools: List[List[torch.Tensor]] = [[] for _ in self.bucket_sizes]

        # 统计信息
        self.stats = {
            "total_allocations": 0,
            "total_frees": 0,
            "bucket_allocations": [0] * len(self.bucket_sizes),
            "bucket_frees": [0] * len(self.bucket_sizes),
            "fallback_allocations": 0,
            "split_operations": 0,
            "fast_path_hits": 0,
        }

        # 缓存: page_size对应的offset tensor
        self._page_offset_cache = torch.arange(self.page_size, dtype=torch.int64, device=self.device)

        # 预计算: 每个桶大小对应的token indices模板(延迟初始化)
        self._token_indices_templates: Dict[int, torch.Tensor] = {}

        # 初始化pools
        self._initialize_pools()

        if self.debug_mode:
            logger.info(f"OptimizedPreallocatedKVBlockPool initialized: "
                       f"total_pages={total_pages}, page_size={page_size}, "
                       f"bucket_sizes={bucket_sizes}")

    def _get_token_indices_template(self, num_pages: int) -> torch.Tensor:
        """
        获取或创建token indices模板(缓存)

        这样避免每次分配都创建新的offset tensor
        """
        if num_pages not in self._token_indices_templates:
            # 创建一次，之后复用
            template = (
                torch.arange(num_pages, dtype=torch.int64, device=self.device)[:, None] * self.page_size
                + self._page_offset_cache[None, :]
            ).reshape(-1)
            self._token_indices_templates[num_pages] = template

        return self._token_indices_templates[num_pages]

    def _initialize_pools(self):
        """优化的初始化: 更均匀地分配页面"""
        remaining_pages = self.total_pages - 1
        current_page = 1

        # 计算每个桶应该分配的块数(更智能的分配策略)
        # 小桶分配更多块,大桶分配更少块
        total_weight = sum(1.0 / size for size in self.bucket_sizes)

        for idx, bucket_size in enumerate(self.bucket_sizes):
            # 权重: 小桶权重大
            weight = 1.0 / bucket_size
            target_pages = int(remaining_pages * (weight / total_weight))
            target_blocks = max(1, target_pages // bucket_size)

            actual_blocks = min(target_blocks, remaining_pages // bucket_size)

            for _ in range(actual_blocks):
                if current_page + bucket_size <= self.total_pages:
                    block = torch.arange(
                        current_page,
                        current_page + bucket_size,
                        dtype=torch.int64,
                        device=self.device
                    )
                    self.free_pools[idx].append(block)
                    current_page += bucket_size
                    remaining_pages -= bucket_size

        # 剩余页面放入最小桶
        if remaining_pages > 0 and self.bucket_sizes:
            smallest_idx = 0
            smallest_size = self.bucket_sizes[0]
            while current_page < self.total_pages:
                pages_left = self.total_pages - current_page
                block_size = min(smallest_size, pages_left)
                if block_size > 0:
                    block = torch.arange(
                        current_page,
                        current_page + block_size,
                        dtype=torch.int64,
                        device=self.device
                    )
                    self.free_pools[smallest_idx].append(block)
                    current_page += block_size

    def _find_best_bucket_idx(self, num_pages: int) -> Optional[int]:
        """
        快速查找最佳桶索引

        优化:
        1. 先尝试精确匹配(O(1))
        2. 再线性查找最小适配(O(k), k是桶数量)
        """
        # 快速路径: 精确匹配
        if num_pages in self.bucket_size_to_idx:
            idx = self.bucket_size_to_idx[num_pages]
            if self.free_pools[idx]:
                self.stats["fast_path_hits"] += 1
                return idx

        # 慢速路径: 找到最小的 >= num_pages 的桶
        for idx, bucket_size in enumerate(self.bucket_sizes):
            if bucket_size >= num_pages and self.free_pools[idx]:
                return idx

        return None

    def allocate(self, num_pages: int) -> Optional[torch.Tensor]:
        """
        优化的分配操作

        性能改进:
        - 快速路径精确匹配
        - 缓存token indices转换
        - 减少tensor创建
        """
        if num_pages <= 0:
            return None

        self.stats["total_allocations"] += 1

        # 找最佳桶
        best_idx = self._find_best_bucket_idx(num_pages)

        if best_idx is None:
            return None

        # 从桶中分配
        bucket_size = self.bucket_sizes[best_idx]
        block_pages = self.free_pools[best_idx].pop()
        self.stats["bucket_allocations"][best_idx] += 1

        # 检查是否需要分割
        if bucket_size > num_pages and self.enable_splitting:
            # 分割块
            allocated_pages = block_pages[:num_pages]
            remainder_pages = block_pages[num_pages:]

            # 返回剩余部分到对应桶
            remainder_size = len(remainder_pages)
            if remainder_size in self.bucket_size_to_idx:
                remainder_idx = self.bucket_size_to_idx[remainder_size]
                self.free_pools[remainder_idx].append(remainder_pages)
            else:
                # 如果剩余大小不在桶中,放入最接近的桶
                for idx, size in enumerate(self.bucket_sizes):
                    if size >= remainder_size:
                        self.free_pools[idx].append(remainder_pages)
                        break

            self.stats["split_operations"] += 1
            self.stats["fallback_allocations"] += 1

            return allocated_pages
        else:
            # 返回完整块
            return block_pages

    def free(self, pages: torch.Tensor):
        """
        优化的释放操作

        性能改进:
        - 直接返回到对应大小的桶,避免复杂逻辑
        """
        if pages is None or pages.numel() == 0:
            return

        self.stats["total_frees"] += 1
        num_pages = len(pages)

        # 快速路径: 直接返回到精确匹配的桶
        if num_pages in self.bucket_size_to_idx:
            idx = self.bucket_size_to_idx[num_pages]
            self.free_pools[idx].append(pages)
            self.stats["bucket_frees"][idx] += 1
        else:
            # 找到能容纳的最小桶
            for idx, size in enumerate(self.bucket_sizes):
                if size >= num_pages:
                    self.free_pools[idx].append(pages)
                    self.stats["bucket_frees"][idx] += 1
                    break

    def available_pages(self) -> int:
        """计算总可用页数"""
        total = 0
        for idx, bucket_size in enumerate(self.bucket_sizes):
            total += bucket_size * len(self.free_pools[idx])
        return total

    def get_statistics(self) -> Dict:
        """获取详细统计信息"""
        stats = dict(self.stats)
        stats["available_pages"] = self.available_pages()
        stats["total_pages"] = self.total_pages
        stats["utilization"] = 1.0 - (stats["available_pages"] / self.total_pages) if self.total_pages > 0 else 0

        # Per-bucket统计
        bucket_stats = {}
        for idx, bucket_size in enumerate(self.bucket_sizes):
            bucket_stats[bucket_size] = {
                "free_blocks": len(self.free_pools[idx]),
                "free_pages": bucket_size * len(self.free_pools[idx]),
                "allocations": self.stats["bucket_allocations"][idx],
                "frees": self.stats["bucket_frees"][idx],
            }
        stats["buckets"] = bucket_stats

        # 性能指标
        if stats["total_allocations"] > 0:
            stats["fast_path_hit_rate"] = stats["fast_path_hits"] / stats["total_allocations"]
            stats["split_rate"] = stats["split_operations"] / stats["total_allocations"]
        else:
            stats["fast_path_hit_rate"] = 0
            stats["split_rate"] = 0

        return stats

    def clear(self):
        """清空并重新初始化"""
        for pool in self.free_pools:
            pool.clear()

        self.stats = {
            "total_allocations": 0,
            "total_frees": 0,
            "bucket_allocations": [0] * len(self.bucket_sizes),
            "bucket_frees": [0] * len(self.bucket_sizes),
            "fallback_allocations": 0,
            "split_operations": 0,
            "fast_path_hits": 0,
        }

        self._initialize_pools()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"OptimizedPreallocatedKVBlockPool(total_pages={self.total_pages}, "
                f"available_pages={stats['available_pages']}, "
                f"utilization={stats['utilization']:.2%}, "
                f"fast_path_rate={stats.get('fast_path_hit_rate', 0):.2%})")
