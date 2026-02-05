"""
Copyright 2023-2024 SGLang Team
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
Optimized Triton kernels for RadixCache operations.

This module provides high-performance GPU kernels to replace Python loops in the
RadixCache implementation, significantly reducing overhead for cache operations.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _token_match_kernel(
    key0_ptr,  # pointer to first token sequence
    key1_ptr,  # pointer to second token sequence
    output_ptr,  # pointer to output (single int64 value)
    len0,  # length of first sequence
    len1,  # length of second sequence
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for finding the longest matching prefix between two token sequences.

    This replaces the Python loop in _key_match_page_size1 with a vectorized GPU implementation.

    Args:
        key0_ptr: Pointer to first token sequence (int32/int64)
        key1_ptr: Pointer to second token sequence (int32/int64)
        output_ptr: Pointer to output buffer for match length
        len0: Length of first sequence
        len1: Length of second sequence
        BLOCK_SIZE: Block size for processing (must be power of 2)
    """
    pid = tl.program_id(0)

    # Calculate minimum length
    min_len = tl.minimum(len0, len1)

    # Process in blocks
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid positions
    mask = offsets < min_len

    # Load tokens from both sequences
    tokens0 = tl.load(key0_ptr + offsets, mask=mask, other=-1)
    tokens1 = tl.load(key1_ptr + offsets, mask=mask, other=-2)

    # Compare tokens
    matches = tokens0 == tokens1

    # Find first mismatch using bit manipulation
    # Convert bool to int: True -> 1, False -> 0
    match_ints = matches.to(tl.int32)

    # Store intermediate results for reduction
    # We need to find the first 0 in the match_ints array
    # Store the match result atomically
    for i in range(BLOCK_SIZE):
        offset = block_start + i
        if offset < min_len:
            if not tl.load(key0_ptr + offset) == tl.load(key1_ptr + offset):
                # First mismatch found - atomically update if this is smaller
                tl.atomic_min(output_ptr, offset)
                return

    # All tokens in this block matched, update progress
    tl.atomic_max(output_ptr, block_start + BLOCK_SIZE)


@triton.jit
def _token_match_vectorized_kernel(
    key0_ptr,
    key1_ptr,
    output_ptr,
    len0,
    len1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorized token matching kernel with better performance.

    This version processes tokens in parallel and uses reduction to find
    the first mismatch efficiently.
    """
    # Use single thread for small sequences
    pid = tl.program_id(0)

    if pid > 0:
        return

    min_len = tl.minimum(len0, len1)

    # Process in chunks
    match_len = 0
    for block_start in range(0, min_len, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < min_len

        # Load and compare
        tokens0 = tl.load(key0_ptr + offsets, mask=mask, other=-1)
        tokens1 = tl.load(key1_ptr + offsets, mask=mask, other=-2)
        matches = tokens0 == tokens1

        # Find first mismatch in this block
        for i in range(BLOCK_SIZE):
            offset = block_start + i
            if offset < min_len:
                if tokens0[i] != tokens1[i]:
                    # Found mismatch
                    tl.store(output_ptr, offset)
                    return

        # All matched in this block
        match_len = block_start + BLOCK_SIZE

    # All tokens matched
    tl.store(output_ptr, min_len)


@triton.jit
def _batch_lock_update_kernel(
    node_indices_ptr,  # indices of nodes in the path
    lock_refs_ptr,  # current lock_ref values
    key_lens_ptr,  # length of key for each node
    output_lock_refs_ptr,  # output updated lock refs
    output_evictable_delta_ptr,  # output change in evictable size
    output_protected_delta_ptr,  # output change in protected size
    num_nodes,  # number of nodes in path
    increment: tl.constexpr,  # 1 for inc, -1 for dec
    BLOCK_SIZE: tl.constexpr,
):
    """
    Batch update lock references for all nodes in a path.

    This replaces the Python while loop in inc_lock_ref/dec_lock_ref with
    a single GPU kernel call that processes all nodes in parallel.

    Args:
        node_indices_ptr: Indices of nodes in the path (from leaf to root)
        lock_refs_ptr: Current lock_ref values for each node
        key_lens_ptr: Length of key for each node
        output_lock_refs_ptr: Updated lock_ref values
        output_evictable_delta_ptr: Change in evictable size
        output_protected_delta_ptr: Change in protected size
        num_nodes: Number of nodes to process
        increment: 1 for increment, -1 for decrement
        BLOCK_SIZE: Block size for parallel processing
    """
    pid = tl.program_id(0)

    # Calculate which nodes this thread processes
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_nodes

    # Load node data
    lock_refs = tl.load(lock_refs_ptr + offsets, mask=mask, other=0)
    key_lens = tl.load(key_lens_ptr + offsets, mask=mask, other=0)

    # Calculate new lock refs
    new_lock_refs = lock_refs + increment

    # Calculate size deltas
    evictable_delta = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    protected_delta = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    if increment > 0:
        # Incrementing: nodes going from 0 to 1 move from evictable to protected
        was_zero = lock_refs == 0
        evictable_delta = tl.where(was_zero & mask, -key_lens, evictable_delta)
        protected_delta = tl.where(was_zero & mask, key_lens, protected_delta)
    else:
        # Decrementing: nodes going from 1 to 0 move from protected to evictable
        was_one = lock_refs == 1
        evictable_delta = tl.where(was_one & mask, key_lens, evictable_delta)
        protected_delta = tl.where(was_one & mask, -key_lens, protected_delta)

    # Store results
    tl.store(output_lock_refs_ptr + offsets, new_lock_refs, mask=mask)

    # Atomic add for size deltas
    for i in range(BLOCK_SIZE):
        if offsets[i] < num_nodes:
            tl.atomic_add(output_evictable_delta_ptr, evictable_delta[i])
            tl.atomic_add(output_protected_delta_ptr, protected_delta[i])


def token_match_fast(
    key0: torch.Tensor,
    key1: torch.Tensor,
) -> int:
    """
    Fast GPU-accelerated token sequence matching.

    Finds the longest common prefix between two token sequences using Triton.
    This is a drop-in replacement for _key_match_page_size1.

    Args:
        key0: First token sequence (torch.Tensor of int32/int64)
        key1: Second token sequence (torch.Tensor of int32/int64)

    Returns:
        Length of matching prefix (int)
    """
    # Handle edge cases
    if len(key0) == 0 or len(key1) == 0:
        return 0

    # Ensure tensors are on GPU
    if not key0.is_cuda:
        key0 = key0.cuda()
    if not key1.is_cuda:
        key1 = key1.cuda()

    # Ensure contiguous
    key0 = key0.contiguous()
    key1 = key1.contiguous()

    min_len = min(len(key0), len(key1))

    # For very short sequences, use CPU
    if min_len < 32:
        key0_cpu = key0.cpu().tolist() if key0.is_cuda else key0.tolist()
        key1_cpu = key1.cpu().tolist() if key1.is_cuda else key1.tolist()
        for i, (t0, t1) in enumerate(zip(key0_cpu, key1_cpu)):
            if t0 != t1:
                return i
        return min_len

    # Allocate output
    output = torch.tensor([min_len], dtype=torch.int64, device=key0.device)

    # Launch kernel
    BLOCK_SIZE = 128
    grid = (1,)  # Single block for simplicity

    _token_match_vectorized_kernel[grid](
        key0,
        key1,
        output,
        len(key0),
        len(key1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.item()


def batch_lock_update(
    node_indices: torch.Tensor,
    lock_refs: torch.Tensor,
    key_lens: torch.Tensor,
    increment: bool = True,
) -> Tuple[torch.Tensor, int, int]:
    """
    Batch update lock references for multiple nodes in a path.

    This optimizes inc_lock_ref/dec_lock_ref by processing all nodes in parallel.

    Args:
        node_indices: Indices of nodes in the path (not used directly, for interface)
        lock_refs: Current lock_ref values for each node
        key_lens: Length of key for each node
        increment: True to increment, False to decrement

    Returns:
        Tuple of (updated_lock_refs, evictable_delta, protected_delta)
    """
    if len(lock_refs) == 0:
        return lock_refs, 0, 0

    # Ensure on GPU
    if not lock_refs.is_cuda:
        lock_refs = lock_refs.cuda()
    if not key_lens.is_cuda:
        key_lens = key_lens.cuda()

    num_nodes = len(lock_refs)

    # Allocate outputs
    output_lock_refs = torch.empty_like(lock_refs)
    output_evictable_delta = torch.zeros(1, dtype=torch.int64, device=lock_refs.device)
    output_protected_delta = torch.zeros(1, dtype=torch.int64, device=lock_refs.device)

    # Launch kernel
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_nodes, BLOCK_SIZE),)

    _batch_lock_update_kernel[grid](
        node_indices,
        lock_refs,
        key_lens,
        output_lock_refs,
        output_evictable_delta,
        output_protected_delta,
        num_nodes,
        increment=1 if increment else -1,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (
        output_lock_refs,
        output_evictable_delta.item(),
        output_protected_delta.item(),
    )


class OptimizedRadixCacheOps:
    """
    Collection of optimized operations for RadixCache.

    This class provides drop-in replacements for hot-path Python functions
    in RadixCache with GPU-accelerated Triton kernels.
    """

    @staticmethod
    def key_match(key0_tokens, key1_tokens):
        """
        Optimized token sequence matching.

        Drop-in replacement for _key_match_page_size1.
        """
        # Convert lists to tensors if needed
        if isinstance(key0_tokens, list):
            key0_tokens = torch.tensor(key0_tokens, dtype=torch.int64)
        if isinstance(key1_tokens, list):
            key1_tokens = torch.tensor(key1_tokens, dtype=torch.int64)

        return token_match_fast(key0_tokens, key1_tokens)

    @staticmethod
    def should_use_gpu_ops(sequence_length: int, use_gpu: bool = True) -> bool:
        """
        Determine whether to use GPU ops based on sequence length and availability.

        Args:
            sequence_length: Length of sequences being processed
            use_gpu: Whether GPU ops are enabled

        Returns:
            True if GPU ops should be used
        """
        # Use GPU for sequences longer than 32 tokens
        # For shorter sequences, Python overhead is minimal
        return use_gpu and torch.cuda.is_available() and sequence_length >= 32


# Fallback implementations for when Triton is not available
def _fallback_token_match(key0_tokens, key1_tokens):
    """CPU fallback for token matching."""
    i = 0
    for t0, t1 in zip(key0_tokens, key1_tokens):
        if t0 != t1:
            break
        i += 1
    return i
