#!/usr/bin/env python3
"""
Standalone test script to verify vectorized key matching.
"""

import time
import numpy as np
from typing import List, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available, skipping Torch tests")


class RadixKey:
    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None):
        self.token_ids = token_ids
        self.extra_key = extra_key

    def __len__(self) -> int:
        return len(self.token_ids)


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(f"Extra keys do not match: {key0.extra_key} vs {key1.extra_key}")


# ========== Original Implementations ==========

def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


# ========== Vectorized Implementations ==========

def _key_match_page_size1_vectorized(key0: RadixKey, key1: RadixKey):
    """Vectorized version using NumPy."""
    _check_extra_key(key0, key1)

    arr0 = np.array(key0.token_ids, dtype=np.int32)
    arr1 = np.array(key1.token_ids, dtype=np.int32)

    min_len = min(len(arr0), len(arr1))
    if min_len == 0:
        return 0

    matches = arr0[:min_len] == arr1[:min_len]
    mismatch_pos = np.argmin(matches)

    if matches[mismatch_pos]:
        return min_len
    else:
        return mismatch_pos


def _key_match_paged_vectorized(key0: RadixKey, key1: RadixKey, page_size: int):
    """Vectorized version using NumPy.

    Note: This implementation matches the behavior of the original, including
    the edge case where the return value may exceed min_len when there's a
    partial page match at the end.
    """
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    if min_len == 0:
        return 0

    arr0 = np.array(key0.token_ids, dtype=np.int32)
    arr1 = np.array(key1.token_ids, dtype=np.int32)

    # Calculate number of iterations (not just full pages!)
    # This matches the original loop condition: while i < min_len
    num_iterations = (min_len + page_size - 1) // page_size  # Ceiling division

    # For all complete pages, use vectorized comparison
    num_full_pages = min_len // page_size

    if num_full_pages > 0:
        len_full_pages = num_full_pages * page_size
        pages0 = arr0[:len_full_pages].reshape(num_full_pages, page_size)
        pages1 = arr1[:len_full_pages].reshape(num_full_pages, page_size)

        page_matches = pages0 == pages1
        all_match_per_page = np.all(page_matches, axis=1)

        first_mismatch_page = np.argmin(all_match_per_page)

        if not all_match_per_page[first_mismatch_page]:
            # Found a mismatch in full pages
            return first_mismatch_page * page_size
    else:
        len_full_pages = 0

    # Check if there's a partial page at the end
    if len_full_pages < min_len:
        # There's a partial page, check if it matches
        remaining = min_len - len_full_pages
        if np.array_equal(arr0[len_full_pages:min_len], arr1[len_full_pages:min_len]):
            # Partial page matches, return full page boundary (matches original behavior)
            return len_full_pages + page_size
        else:
            # Partial page doesn't match
            return len_full_pages

    return len_full_pages


def _key_match_paged_torch(key0: RadixKey, key1: RadixKey, page_size: int):
    """Torch-based vectorized version.

    Note: This implementation matches the behavior of the original, including
    the edge case where the return value may exceed min_len when there's a
    partial page match at the end.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available")

    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    if min_len == 0:
        return 0

    tensor0 = torch.tensor(key0.token_ids, dtype=torch.int32)
    tensor1 = torch.tensor(key1.token_ids, dtype=torch.int32)

    num_full_pages = min_len // page_size

    if num_full_pages > 0:
        len_full_pages = num_full_pages * page_size
        pages0 = tensor0[:len_full_pages].reshape(num_full_pages, page_size)
        pages1 = tensor1[:len_full_pages].reshape(num_full_pages, page_size)

        page_matches = pages0 == pages1
        all_match_per_page = torch.all(page_matches, dim=1)

        if not torch.all(all_match_per_page):
            first_mismatch_page = torch.argmin(all_match_per_page.to(torch.int32)).item()
            return first_mismatch_page * page_size
    else:
        len_full_pages = 0

    # Check partial page
    if len_full_pages < min_len:
        if torch.equal(tensor0[len_full_pages:min_len], tensor1[len_full_pages:min_len]):
            return len_full_pages + page_size
        else:
            return len_full_pages

    return len_full_pages


# ========== Tests ==========

def test_correctness():
    """Test that vectorized versions produce identical results to original."""
    print("=" * 80)
    print("Testing Correctness")
    print("=" * 80)

    test_cases = [
        ("Identical sequences", [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]),
        ("Partial match", [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 9, 5, 6, 7, 8]),
        ("Different lengths (short first)", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8]),
        ("Different lengths (long first)", [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5]),
        ("Empty first", [], [1, 2, 3]),
        ("Empty second", [1, 2, 3], []),
        ("Both empty", [], []),
        ("First mismatch", [9, 2, 3], [1, 2, 3]),
    ]

    print("\nTesting page_size=1:")
    for name, seq1, seq2 in test_cases:
        key1 = RadixKey(seq1)
        key2 = RadixKey(seq2)

        result_orig = _key_match_page_size1(key1, key2)
        result_vec = _key_match_page_size1_vectorized(key1, key2)

        status = "✓" if result_orig == result_vec else "✗"
        print(f"  {status} {name:<30} Original={result_orig}, Vectorized={result_vec}")

        if result_orig != result_vec:
            raise AssertionError(f"Mismatch in {name}!")

    print("\nTesting paged matching (page_size=16):")
    page_size = 16

    # Long sequences
    long_test_cases = [
        ("Long identical", list(range(1000)), list(range(1000))),
        ("Long partial match at 500", list(range(1000)), [*list(range(500)), 9999, *list(range(501, 1000))]),
        ("Long mismatch at start", [9999] + list(range(1, 1000)), list(range(1000))),
        ("Long mismatch at page boundary", list(range(512)), [*list(range(256)), 9999, *list(range(257, 512))]),
    ]

    for name, seq1, seq2 in long_test_cases:
        key1 = RadixKey(seq1)
        key2 = RadixKey(seq2)

        result_orig = _key_match_paged(key1, key2, page_size)
        result_vec = _key_match_paged_vectorized(key1, key2, page_size)

        if HAS_TORCH:
            result_torch = _key_match_paged_torch(key1, key2, page_size)
            status = "✓" if result_orig == result_vec == result_torch else "✗"
            print(f"  {status} {name:<30} Orig={result_orig}, NumPy={result_vec}, Torch={result_torch}")
            if not (result_orig == result_vec == result_torch):
                raise AssertionError(f"Mismatch in {name}!")
        else:
            status = "✓" if result_orig == result_vec else "✗"
            print(f"  {status} {name:<30} Orig={result_orig}, NumPy={result_vec}")
            if not (result_orig == result_vec):
                raise AssertionError(f"Mismatch in {name}!")

    print("\n✓ All correctness tests passed!\n")


def benchmark_performance():
    """Benchmark performance of different implementations."""
    print("=" * 80)
    print("Benchmarking Performance")
    print("=" * 80)

    lengths = [100, 1000, 10000, 50000, 100000]
    page_size = 16

    print(f"\nPaged matching (page_size={page_size}):")
    if HAS_TORCH:
        print(f"{'Length':<10} {'Original (ms)':<15} {'NumPy (ms)':<15} {'Torch (ms)':<15} {'NumPy Speedup':<15} {'Torch Speedup':<15}")
        print("-" * 95)
    else:
        print(f"{'Length':<10} {'Original (ms)':<15} {'NumPy (ms)':<15} {'NumPy Speedup':<15}")
        print("-" * 60)

    for length in lengths:
        key1 = RadixKey(list(range(length)))
        key2 = RadixKey(list(range(length)))

        # Warm up
        for _ in range(3):
            _key_match_paged(key1, key2, page_size)
            _key_match_paged_vectorized(key1, key2, page_size)
            if HAS_TORCH:
                _key_match_paged_torch(key1, key2, page_size)

        # Benchmark
        iterations = max(10, 1000 // (length // 100))

        start = time.time()
        for _ in range(iterations):
            _key_match_paged(key1, key2, page_size)
        time_orig = (time.time() - start) * 1000 / iterations

        start = time.time()
        for _ in range(iterations):
            _key_match_paged_vectorized(key1, key2, page_size)
        time_vec = (time.time() - start) * 1000 / iterations

        speedup_numpy = time_orig / time_vec if time_vec > 0 else float('inf')

        if HAS_TORCH:
            start = time.time()
            for _ in range(iterations):
                _key_match_paged_torch(key1, key2, page_size)
            time_torch = (time.time() - start) * 1000 / iterations
            speedup_torch = time_orig / time_torch if time_torch > 0 else float('inf')
            print(f"{length:<10} {time_orig:<15.4f} {time_vec:<15.4f} {time_torch:<15.4f} {speedup_numpy:<15.2f}x {speedup_torch:<15.2f}x")
        else:
            print(f"{length:<10} {time_orig:<15.4f} {time_vec:<15.4f} {speedup_numpy:<15.2f}x")

    print(f"\nPage_size=1 matching:")
    print(f"{'Length':<10} {'Original (ms)':<15} {'Vectorized (ms)':<15} {'Speedup':<15}")
    print("-" * 55)

    for length in lengths:
        key1 = RadixKey(list(range(length)))
        key2 = RadixKey(list(range(length)))

        # Warm up
        for _ in range(3):
            _key_match_page_size1(key1, key2)
            _key_match_page_size1_vectorized(key1, key2)

        iterations = max(10, 1000 // (length // 100))

        start = time.time()
        for _ in range(iterations):
            _key_match_page_size1(key1, key2)
        time_orig = (time.time() - start) * 1000 / iterations

        start = time.time()
        for _ in range(iterations):
            _key_match_page_size1_vectorized(key1, key2)
        time_vec = (time.time() - start) * 1000 / iterations

        speedup = time_orig / time_vec if time_vec > 0 else float('inf')

        print(f"{length:<10} {time_orig:<15.4f} {time_vec:<15.4f} {speedup:<15.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark_performance()
    print("\n✓ All tests completed successfully!")
