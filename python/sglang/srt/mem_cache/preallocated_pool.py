"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
KV Cache Preallocation Pool - Inspired by vLLM's BlockManager.

This module implements a size-based preallocation strategy for KV cache blocks
to reduce allocation overhead and improve memory efficiency.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


class PreallocatedKVBlockPool:
    """
    Preallocation pool for KV cache blocks with size-based buckets.

    This class manages multiple pools of preallocated KV blocks of common sizes,
    inspired by vLLM's BlockManager design. Each pool contains blocks of a specific
    size to enable O(1) allocation and deallocation.

    Key features:
    - Multiple size buckets for common block sizes (e.g., 16, 32, 64, 128, 256 pages)
    - Fast allocation from appropriate size bucket
    - Fallback to larger buckets when exact size unavailable
    - Memory statistics tracking per bucket
    - Support for block splitting and merging

    Args:
        total_pages: Total number of pages available
        page_size: Size of each page in tokens
        device: Device to allocate tensors on
        bucket_sizes: List of bucket sizes in pages (default: [1, 2, 4, 8, 16, 32, 64, 128])
        enable_splitting: Whether to split larger blocks when exact size unavailable
        debug_mode: Enable debug logging and assertions
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

        # Default bucket sizes (in pages): 1, 2, 4, 8, 16, 32, 64, 128
        # These cover common allocation patterns from small decode (1 page) to large prefill (128 pages)
        if bucket_sizes is None:
            bucket_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        self.bucket_sizes = sorted(bucket_sizes)

        # Initialize pools for each bucket size
        # Each pool is a list of free page ranges (start_page, num_pages)
        self.free_pools: Dict[int, List[torch.Tensor]] = defaultdict(list)

        # Statistics tracking
        self.stats = {
            "total_allocations": 0,
            "total_frees": 0,
            "bucket_allocations": defaultdict(int),
            "bucket_frees": defaultdict(int),
            "fallback_allocations": 0,
            "split_operations": 0,
        }

        # Initialize the pools with available pages
        self._initialize_pools()

        if self.debug_mode:
            logger.info(f"PreallocatedKVBlockPool initialized: "
                       f"total_pages={total_pages}, page_size={page_size}, "
                       f"bucket_sizes={bucket_sizes}, enable_splitting={enable_splitting}")

    def _initialize_pools(self):
        """
        Initialize the free pools by distributing available pages across buckets.

        Strategy:
        1. Allocate larger buckets first to maximize large contiguous allocations
        2. Use remaining pages for smaller buckets
        3. Reserve slot 0 for dummy outputs (consistent with existing allocator)
        """
        # Start from page 1 (skip page 0 which is reserved for dummy outputs)
        remaining_pages = self.total_pages - 1
        current_page = 1

        # Distribute pages across buckets (largest first)
        for bucket_size in reversed(self.bucket_sizes):
            # Calculate how many blocks of this size we can create
            # Reserve some pages for smaller buckets
            target_blocks = max(1, remaining_pages // (bucket_size * len(self.bucket_sizes)))
            actual_blocks = min(target_blocks, remaining_pages // bucket_size)

            if actual_blocks > 0:
                # Create free blocks for this bucket
                for _ in range(actual_blocks):
                    if current_page + bucket_size <= self.total_pages:
                        block = torch.arange(
                            current_page,
                            current_page + bucket_size,
                            dtype=torch.int64,
                            device=self.device
                        )
                        self.free_pools[bucket_size].append(block)
                        current_page += bucket_size
                        remaining_pages -= bucket_size

                if self.debug_mode:
                    logger.debug(f"Initialized bucket {bucket_size}: {actual_blocks} blocks")

        # Put any remaining pages into the smallest bucket
        if remaining_pages > 0 and self.bucket_sizes:
            smallest_bucket = self.bucket_sizes[0]
            while current_page < self.total_pages:
                pages_left = self.total_pages - current_page
                block_size = min(smallest_bucket, pages_left)
                if block_size > 0:
                    block = torch.arange(
                        current_page,
                        current_page + block_size,
                        dtype=torch.int64,
                        device=self.device
                    )
                    self.free_pools[block_size].append(block)
                    current_page += block_size

    def _find_best_bucket(self, num_pages: int) -> Optional[int]:
        """
        Find the best bucket size for the requested number of pages.

        Strategy:
        1. Try to find exact match
        2. If no exact match, find the smallest bucket >= num_pages

        Args:
            num_pages: Number of pages needed

        Returns:
            Best bucket size, or None if no suitable bucket found
        """
        # Try exact match first
        if num_pages in self.bucket_sizes and self.free_pools[num_pages]:
            return num_pages

        # Find smallest bucket >= num_pages with available blocks
        for bucket_size in self.bucket_sizes:
            if bucket_size >= num_pages and self.free_pools[bucket_size]:
                return bucket_size

        return None

    def _split_block(self, block: torch.Tensor, bucket_size: int, needed_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split a block into needed portion and remainder.

        Args:
            block: The block to split
            bucket_size: Original bucket size
            needed_size: Needed size in pages

        Returns:
            Tuple of (allocated_block, remainder_block)
        """
        if self.debug_mode:
            assert len(block) == bucket_size, f"Block size mismatch: {len(block)} != {bucket_size}"
            assert needed_size < bucket_size, f"Cannot split: needed_size {needed_size} >= bucket_size {bucket_size}"

        allocated = block[:needed_size]
        remainder = block[needed_size:]

        self.stats["split_operations"] += 1

        return allocated, remainder

    def allocate(self, num_pages: int) -> Optional[torch.Tensor]:
        """
        Allocate pages from the preallocation pool.

        Args:
            num_pages: Number of pages to allocate

        Returns:
            Tensor of allocated page indices, or None if allocation failed
        """
        if num_pages <= 0:
            return None

        self.stats["total_allocations"] += 1

        # Find best bucket
        best_bucket = self._find_best_bucket(num_pages)

        if best_bucket is None:
            # No suitable bucket found
            return None

        # Allocate from the bucket
        block = self.free_pools[best_bucket].pop()
        self.stats["bucket_allocations"][best_bucket] += 1

        # Check if we need to split the block
        if best_bucket > num_pages and self.enable_splitting:
            allocated, remainder = self._split_block(block, best_bucket, num_pages)

            # Return remainder to appropriate bucket
            remainder_size = len(remainder)
            self.free_pools[remainder_size].append(remainder)

            self.stats["fallback_allocations"] += 1

            if self.debug_mode:
                logger.debug(f"Split block: allocated {num_pages} pages from bucket {best_bucket}, "
                           f"returned {remainder_size} pages")

            return allocated
        else:
            # Return full block
            return block

    def free(self, pages: torch.Tensor):
        """
        Free pages back to the preallocation pool.

        Args:
            pages: Tensor of page indices to free
        """
        if pages is None or pages.numel() == 0:
            return

        self.stats["total_frees"] += 1

        num_pages = len(pages)

        # Return to appropriate bucket
        # For simplicity, we return to the bucket matching the block size
        # In a more sophisticated implementation, we could try to merge with adjacent blocks
        self.free_pools[num_pages].append(pages)
        self.stats["bucket_frees"][num_pages] += 1

        if self.debug_mode:
            logger.debug(f"Freed {num_pages} pages to bucket {num_pages}")

    def available_pages(self) -> int:
        """
        Calculate total available pages across all buckets.

        Returns:
            Total number of available pages
        """
        total = 0
        for bucket_size, blocks in self.free_pools.items():
            total += bucket_size * len(blocks)
        return total

    def get_statistics(self) -> Dict:
        """
        Get detailed statistics about pool usage.

        Returns:
            Dictionary containing pool statistics
        """
        stats = dict(self.stats)
        stats["available_pages"] = self.available_pages()
        stats["total_pages"] = self.total_pages
        stats["utilization"] = 1.0 - (stats["available_pages"] / self.total_pages)

        # Per-bucket statistics
        bucket_stats = {}
        for bucket_size in self.bucket_sizes:
            bucket_stats[bucket_size] = {
                "free_blocks": len(self.free_pools[bucket_size]),
                "free_pages": bucket_size * len(self.free_pools[bucket_size]),
                "allocations": self.stats["bucket_allocations"][bucket_size],
                "frees": self.stats["bucket_frees"][bucket_size],
            }
        stats["buckets"] = bucket_stats

        return stats

    def clear(self):
        """Clear all pools and reinitialize."""
        self.free_pools.clear()
        self.stats = {
            "total_allocations": 0,
            "total_frees": 0,
            "bucket_allocations": defaultdict(int),
            "bucket_frees": defaultdict(int),
            "fallback_allocations": 0,
            "split_operations": 0,
        }
        self._initialize_pools()

        if self.debug_mode:
            logger.debug("PreallocatedKVBlockPool cleared and reinitialized")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"PreallocatedKVBlockPool(total_pages={self.total_pages}, "
                f"available_pages={stats['available_pages']}, "
                f"utilization={stats['utilization']:.2%}, "
                f"buckets={len(self.bucket_sizes)})")
