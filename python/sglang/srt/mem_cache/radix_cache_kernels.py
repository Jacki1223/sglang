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

This module provides high-performance GPU kernels to replace Python loops
in the RadixCache implementation.
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
    min_len,
    num_chunks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for token matching using compile-time loop unrolling.

    Uses a single program to process all tokens sequentially in chunks.
    The number of chunks is a compile-time constant for proper unrolling.
    """
    # Only program 0 does the work
    pid = tl.program_id(0)
    if pid > 0:
        return

    # Process chunks using static_range for compile-time unrolling
    # We process up to num_chunks, but stop early if we find a mismatch
    for chunk_idx in tl.static_range(num_chunks):
        chunk_start = chunk_idx * BLOCK_SIZE

        # Early exit if beyond min_len
        if chunk_start >= min_len:
            tl.store(output_ptr, min_len)
            return

        # Calculate offsets for this chunk
        offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < min_len

        # Load tokens
        tokens0 = tl.load(key0_ptr + offsets, mask=mask, other=0)
        tokens1 = tl.load(key1_ptr + offsets, mask=mask, other=0)

        # Compare
        matches = tokens0 == tokens1

        # Check if all tokens in this chunk match
        match_mask = matches & mask
        num_matched = tl.sum(match_mask.to(tl.int32))
        num_valid = tl.sum(mask.to(tl.int32))

        if num_matched < num_valid:
            # Found a mismatch in this chunk
            # Do a sequential scan to find exactly where
            for i in tl.static_range(BLOCK_SIZE):
                offset = chunk_start + i
                if offset < min_len:
                    t0 = tl.load(key0_ptr + offset)
                    t1 = tl.load(key1_ptr + offset)
                    if t0 != t1:
                        # Found the mismatch
                        tl.store(output_ptr, offset)
                        return

    # All tokens matched
    tl.store(output_ptr, min_len)


def token_match_fast(
    key0: torch.Tensor,
    key1: torch.Tensor,
) -> int:
    """
    Fast GPU-accelerated token sequence matching using Triton.

    Finds the longest common prefix between two token sequences.
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

    # For very short sequences, use CPU (faster than GPU transfer + kernel launch)
    if min_len < 512:
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

    # Ensure tensors are on GPU and contiguous
    if not key0.is_cuda:
        key0 = key0.cuda()
    if not key1.is_cuda:
        key1 = key1.cuda()

    key0 = key0.contiguous()
    key1 = key1.contiguous()

    # Calculate number of chunks needed
    BLOCK_SIZE = 256
    num_chunks = triton.cdiv(min_len, BLOCK_SIZE)

    # Limit max chunks to avoid excessive kernel size
    # For very long sequences, we'll need multiple kernel launches
    MAX_CHUNKS_PER_KERNEL = 64  # Up to 16K tokens per kernel

    offset = 0
    while offset < min_len:
        # Process up to MAX_CHUNKS_PER_KERNEL in this launch
        remaining = min_len - offset
        chunks_this_launch = min(num_chunks - (offset // BLOCK_SIZE), MAX_CHUNKS_PER_KERNEL)
        len_this_launch = min(remaining, chunks_this_launch * BLOCK_SIZE)

        # Allocate output
        output = torch.tensor([len_this_launch], dtype=torch.int64, device=key0.device)

        # Launch kernel with single program
        grid = (1,)

        # Use JIT compilation with the specific num_chunks value
        _token_match_kernel[grid](
            key0[offset:],
            key1[offset:],
            output,
            len_this_launch,
            num_chunks=chunks_this_launch,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        result = output.item()

        # If we found a mismatch, return the absolute position
        if result < len_this_launch:
            return offset + result

        # Move to next chunk
        offset += len_this_launch

    # All tokens matched
    return min_len


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
        # Use GPU for sequences longer than 512 tokens
        # For shorter sequences, CPU is faster due to overhead
        return use_gpu and torch.cuda.is_available() and sequence_length >= 512


# Fallback implementations for when Triton is not available
def _fallback_token_match(key0_tokens, key1_tokens):
    """CPU fallback for token matching."""
    i = 0
    for t0, t1 in zip(key0_tokens, key1_tokens):
        if t0 != t1:
            break
        i += 1
    return i
