from __future__ import annotations

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
Page-aligned memory pool.
"""

import abc
import logging
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        device: str,
        kvcache: SWAKVPool,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        assert isinstance(kvcache, SWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.full_attn_allocator = TokenToKVPoolAllocator(
            size,
            dtype,
            device,
            kvcache.full_kv_pool,
            need_sort,
        )
        self.swa_attn_allocator = TokenToKVPoolAllocator(
            size_swa,
            dtype,
            device,
            kvcache.swa_kv_pool,
            need_sort,
        )
        self.full_to_swa_index_mapping = torch.empty(
            size + size_swa + 1,
            dtype=torch.int64,
            device=device,
        )
        self.clear()

        self._kvcache.full_to_swa_index_mapping = self.full_to_swa_index_mapping

    def available_size(self):
        raise NotImplementedError()

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size_full(self):
        return self._size_full

    @property
    def size_swa(self):
        return self._size_swa

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def alloc(self, need_size: int):
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = 0

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping.fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []


@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    offset_many_page = tl.arange(0, max_num_extend_tokens)
    page_start = tl.load(
        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
        mask=offset_many_page < num_part2,
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + offset_many_page,
        page_start * page_size + offset_many_page % page_size,
        mask=offset_many_page < num_part2,
    )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.seen_max_num_extend_tokens_next_power_of_2 = 1
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        self.seen_max_num_extend_tokens_next_power_of_2 = max(
            self.seen_max_num_extend_tokens_next_power_of_2,
            next_power_of_2(extend_num_tokens),
        )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
            self.seen_max_num_extend_tokens_next_power_of_2,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


class PreallocatedPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    """
    Enhanced paged allocator with preallocation pool support.

    This allocator extends PagedTokenToKVPoolAllocator by adding a preallocation
    pool for common block sizes, inspired by vLLM's BlockManager. This reduces
    allocation overhead and improves memory efficiency for common allocation patterns.

    The allocator maintains both:
    1. Preallocation pool: For fast allocation of common sizes
    2. Fallback to parent allocator: For sizes not in preallocation pool

    Args:
        size: Total size of the pool
        page_size: Size of each page
        dtype: Data type for tensors
        device: Device to allocate on
        kvcache: KV cache instance
        need_sort: Whether to sort free pages
        enable_prealloc: Whether to enable preallocation pool
        prealloc_bucket_sizes: List of bucket sizes for preallocation pool
        prealloc_ratio: Ratio of pages to use for preallocation (0.0-1.0)
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
        enable_prealloc: bool = True,
        prealloc_bucket_sizes: Optional[list] = None,
        prealloc_ratio: float = 0.8,
    ):
        # Initialize parent allocator
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

        self.enable_prealloc = enable_prealloc
        self.prealloc_ratio = prealloc_ratio

        if self.enable_prealloc:
            from sglang.srt.mem_cache.preallocated_pool import PreallocatedKVBlockPool

            # Calculate pages to use for preallocation pool
            prealloc_pages = int(self.num_pages * prealloc_ratio)

            # Initialize preallocation pool
            self.prealloc_pool = PreallocatedKVBlockPool(
                total_pages=prealloc_pages,
                page_size=page_size,
                device=device,
                bucket_sizes=prealloc_bucket_sizes,
                enable_splitting=True,
                debug_mode=self.debug_mode,
            )

            # Reduce parent allocator's free pages by the amount used for preallocation
            # The preallocation pool manages pages [1, prealloc_pages]
            # Parent allocator manages pages [prealloc_pages+1, num_pages]
            self.free_pages = torch.arange(
                prealloc_pages + 1, self.num_pages + 1, dtype=torch.int64, device=self.device
            )

            logger.info(
                f"PreallocatedPagedTokenToKVPoolAllocator initialized: "
                f"total_pages={self.num_pages}, prealloc_pages={prealloc_pages} ({prealloc_ratio:.1%}), "
                f"fallback_pages={len(self.free_pages)}"
            )
        else:
            self.prealloc_pool = None
            logger.info("PreallocatedPagedTokenToKVPoolAllocator initialized with preallocation disabled")

    def alloc(self, need_size: int):
        """
        Allocate pages with preallocation pool support.

        Strategy:
        1. Try to allocate from preallocation pool if enabled
        2. Fallback to parent allocator if preallocation fails or disabled

        Args:
            need_size: Number of tokens to allocate

        Returns:
            Allocated page indices, or None if allocation failed
        """
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), f"The allocation size should be page-aligned: {need_size} % {self.page_size} != 0"

        num_pages = need_size // self.page_size

        # Try preallocation pool first
        if self.enable_prealloc and self.prealloc_pool is not None:
            prealloc_pages = self.prealloc_pool.allocate(num_pages)
            if prealloc_pages is not None:
                # Convert page indices to token indices
                out_indices = (
                    prealloc_pages[:, None] * self.page_size
                    + torch.arange(self.page_size, device=self.device)
                ).reshape(-1)

                if self.debug_mode:
                    logger.debug(f"Allocated {num_pages} pages from preallocation pool")

                return out_indices

        # Fallback to parent allocator
        return super().alloc(need_size)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """
        Allocate for extend operation.

        For extend operations, we use the parent allocator's kernel-based approach
        as it's optimized for this use case and handles complex page alignment.

        Args:
            prefix_lens: Prefix lengths tensor
            prefix_lens_cpu: Prefix lengths on CPU
            seq_lens: Sequence lengths tensor
            seq_lens_cpu: Sequence lengths on CPU
            last_loc: Last location tensor
            extend_num_tokens: Number of tokens to extend

        Returns:
            Allocated indices
        """
        # For extend, use parent allocator's optimized kernel
        return super().alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        """
        Allocate for decode operation.

        For decode operations, we use the parent allocator's kernel-based approach
        as it's optimized for this use case.

        Args:
            seq_lens: Sequence lengths tensor
            seq_lens_cpu: Sequence lengths on CPU
            last_loc: Last location tensor

        Returns:
            Allocated indices
        """
        # For decode, use parent allocator's optimized kernel
        return super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    def free(self, free_index: torch.Tensor):
        """
        Free pages back to appropriate pool.

        Strategy:
        1. Determine if pages belong to preallocation pool or parent allocator
        2. Return pages to the appropriate pool

        Args:
            free_index: Indices to free
        """
        if free_index.numel() == 0:
            return

        if not self.enable_prealloc or self.prealloc_pool is None:
            # No preallocation pool, use parent allocator
            return super().free(free_index)

        # Convert token indices to page indices
        free_page_indices = torch.unique(free_index // self.page_size)

        # Calculate preallocation pool boundary
        prealloc_pages = int(self.num_pages * self.prealloc_ratio)

        # Split pages into prealloc and fallback
        prealloc_mask = free_page_indices <= prealloc_pages
        prealloc_pages_to_free = free_page_indices[prealloc_mask]
        fallback_pages_to_free = free_page_indices[~prealloc_mask]

        # Free to preallocation pool
        if prealloc_pages_to_free.numel() > 0:
            self.prealloc_pool.free(prealloc_pages_to_free)

            if self.debug_mode:
                logger.debug(f"Freed {len(prealloc_pages_to_free)} pages to preallocation pool")

        # Free to parent allocator
        if fallback_pages_to_free.numel() > 0:
            if self.is_not_in_free_group:
                if self.need_sort:
                    self.release_pages = torch.cat((fallback_pages_to_free, self.release_pages))
                else:
                    self.free_pages = torch.cat((fallback_pages_to_free, self.free_pages))
            else:
                self.free_group.append(fallback_pages_to_free)

            if self.debug_mode:
                logger.debug(f"Freed {len(fallback_pages_to_free)} pages to fallback pool")

    def available_size(self):
        """
        Calculate total available size.

        Returns:
            Total available size in tokens
        """
        if not self.enable_prealloc or self.prealloc_pool is None:
            return super().available_size()

        # Sum of preallocation pool and parent allocator
        prealloc_available = self.prealloc_pool.available_pages() * self.page_size
        fallback_available = (len(self.free_pages) + len(self.release_pages)) * self.page_size

        return prealloc_available + fallback_available

    def clear(self):
        """Clear all pools and reinitialize."""
        super().clear()

        if self.enable_prealloc and self.prealloc_pool is not None:
            self.prealloc_pool.clear()

            # Reset free pages for fallback allocator
            prealloc_pages = int(self.num_pages * self.prealloc_ratio)
            self.free_pages = torch.arange(
                prealloc_pages + 1, self.num_pages + 1, dtype=torch.int64, device=self.device
            )

    def get_statistics(self) -> dict:
        """
        Get detailed statistics about allocator usage.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_pages": self.num_pages,
            "page_size": self.page_size,
            "enable_prealloc": self.enable_prealloc,
        }

        if self.enable_prealloc and self.prealloc_pool is not None:
            stats["prealloc"] = self.prealloc_pool.get_statistics()
            stats["fallback_available_pages"] = len(self.free_pages) + len(self.release_pages)
        else:
            stats["fallback_available_pages"] = len(self.free_pages) + len(self.release_pages)

        stats["total_available_size"] = self.available_size()

        return stats

    def __repr__(self) -> str:
        if self.enable_prealloc and self.prealloc_pool is not None:
            return (f"PreallocatedPagedTokenToKVPoolAllocator("
                   f"total_pages={self.num_pages}, "
                   f"prealloc_pool={self.prealloc_pool}, "
                   f"fallback_pages={len(self.free_pages)})")
        else:
            return f"PreallocatedPagedTokenToKVPoolAllocator(prealloc_disabled, total_pages={self.num_pages})"
