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
    key0_ptr,
    key1_ptr,
    output_ptr,
    len0,
    len1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Find the longest matching prefix between two token sequences.

    This kernel uses a simple parallel approach where each program processes
    one block and finds the first mismatch within its range.
    """
    pid = tl.program_id(0)

    # Calculate minimum length
    min_len = tl.minimum(len0, len1)

    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid positions
    mask = offsets < min_len

    # Load tokens from both sequences
    tokens0 = tl.load(key0_ptr + offsets, mask=mask, other=0)
    tokens1 = tl.load(key1_ptr + offsets, mask=mask, other=0)

    # Compare tokens
    matches = tokens0 == tokens1

    # Find first mismatch position in this block
    # Use cumsum to find the first False
    match_mask = matches & mask

    # Count consecutive matches from start
    # If all tokens match in this block, store block_end
    # Otherwise, find the first mismatch
    if tl.sum(match_mask.to(tl.int32)) == tl.sum(mask.to(tl.int32)):
        # All tokens in this block matched
        if block_start + BLOCK_SIZE >= min_len:
            # This is the last block
            tl.atomic_min(output_ptr, min_len)
        else:
            # Signal that this block fully matched
            tl.atomic_min(output_ptr, block_start + BLOCK_SIZE)
    else:
        # Find first mismatch
        # Use a simple linear scan within the block
        mismatch_found = False
        UNROLL_SIZE: tl.constexpr = 128
        for i in tl.static_range(UNROLL_SIZE):
            if i < BLOCK_SIZE:
                offset = block_start + i
                if offset < min_len and not mismatch_found:
                    t0 = tl.load(key0_ptr + offset)
                    t1 = tl.load(key1_ptr + offset)
                    if t0 != t1:
                        tl.atomic_min(output_ptr, offset)
                        mismatch_found = True


def token_match_fast(
    key0: torch.Tensor,
    key1: torch.Tensor,
) -> int:
    """
    Fast GPU-accelerated token sequence matching.

    Finds the longest common prefix between two token sequences using Triton.
    For short sequences, uses CPU fallback for better performance.

    Args:
        key0: First token sequence (torch.Tensor of int32/int64)
        key1: Second token sequence (torch.Tensor of int32/int64)

    Returns:
        Length of matching prefix (int)
    """
    # Handle edge cases
    if len(key0) == 0 or len(key1) == 0:
        return 0

    min_len = min(len(key0), len(key1))

    # For very short sequences, use CPU (faster than GPU kernel launch overhead)
    if min_len < 64:
        # CPU fallback
        if key0.is_cuda:
            key0 = key0.cpu()
        if key1.is_cuda:
            key1 = key1.cpu()

        key0_list = key0.tolist()
        key1_list = key1.tolist()

        for i in range(min_len):
            if key0_list[i] != key1_list[i]:
                return i
        return min_len

    # Ensure tensors are on GPU
    if not key0.is_cuda:
        key0 = key0.cuda()
    if not key1.is_cuda:
        key1 = key1.cuda()

    # Ensure contiguous
    key0 = key0.contiguous()
    key1 = key1.contiguous()

    # Initialize output with max possible value
    output = torch.tensor([min_len], dtype=torch.int64, device=key0.device)

    # Launch kernel
    BLOCK_SIZE = 128
    num_blocks = triton.cdiv(min_len, BLOCK_SIZE)
    grid = (num_blocks,)

    _token_match_kernel[grid](
        key0,
        key1,
        output,
        len(key0),
        len(key1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.item()


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
        # Use GPU for sequences longer than 64 tokens
        # For shorter sequences, CPU is actually faster due to kernel launch overhead
        return use_gpu and torch.cuda.is_available() and sequence_length >= 64


# Fallback implementations for when Triton is not available
def _fallback_token_match(key0_tokens, key1_tokens):
    """CPU fallback for token matching."""
    i = 0
    for t0, t1 in zip(key0_tokens, key1_tokens):
        if t0 != t1:
            break
        i += 1
    return i
