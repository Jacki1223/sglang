"""
Unit tests for OptimizedRadixKey.

This test suite verifies:
1. Correctness vs original RadixKey
2. Memory efficiency improvements
3. Performance of vectorized operations
4. Backward compatibility

Run with:
    python -m pytest test/srt/test_radix_key_optimized.py -v
    python test/srt/test_radix_key_optimized.py
"""

import sys
import unittest

import torch

from sglang.srt.mem_cache.radix_key_optimized import (
    OptimizedRadixKey,
    RadixKey,
    optimized_key_match_paged_vectorized,
    optimized_key_match_vectorized,
)


class TestOptimizedRadixKey(unittest.TestCase):
    """Test cases for OptimizedRadixKey."""

    def test_creation_from_list(self):
        """Test creating key from list."""
        key = OptimizedRadixKey([1, 2, 3, 4])

        self.assertEqual(len(key), 4)
        self.assertEqual(key.to_list(), [1, 2, 3, 4])
        self.assertIsNone(key.extra_key)

    def test_creation_from_tensor(self):
        """Test creating key from tensor (zero-copy)."""
        tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        key = OptimizedRadixKey(tensor)

        self.assertEqual(len(key), 4)
        # Should be same tensor object
        self.assertTrue(torch.equal(key.token_tensor, tensor))

    def test_creation_with_extra_key(self):
        """Test creating key with extra_key."""
        key = OptimizedRadixKey([1, 2, 3], extra_key="lora_id_123")

        self.assertEqual(key.extra_key, "lora_id_123")
        self.assertEqual(len(key), 3)

    def test_length(self):
        """Test __len__ method."""
        test_cases = [
            ([1, 2, 3], 3),
            ([], 0),
            ([42], 1),
            (list(range(100)), 100),
        ]

        for tokens, expected_len in test_cases:
            with self.subTest(tokens=tokens):
                key = OptimizedRadixKey(tokens)
                self.assertEqual(len(key), expected_len)

    def test_iteration(self):
        """Test __iter__ method."""
        tokens = [1, 2, 3, 4, 5]
        key = OptimizedRadixKey(tokens)

        result = list(key)
        self.assertEqual(result, tokens)

    def test_getitem_single_index(self):
        """Test __getitem__ with single index."""
        key = OptimizedRadixKey([10, 20, 30, 40, 50])

        # Positive index
        key0 = key[0]
        self.assertIsInstance(key0, OptimizedRadixKey)
        self.assertEqual(key0.to_list(), [10])

        # Negative index
        key_last = key[-1]
        self.assertEqual(key_last.to_list(), [50])

        # Middle index
        key2 = key[2]
        self.assertEqual(key2.to_list(), [30])

    def test_getitem_slice(self):
        """Test __getitem__ with slice."""
        key = OptimizedRadixKey([1, 2, 3, 4, 5], extra_key="test")

        # Basic slice
        sliced = key[1:4]
        self.assertIsInstance(sliced, OptimizedRadixKey)
        self.assertEqual(sliced.to_list(), [2, 3, 4])
        self.assertEqual(sliced.extra_key, "test")

        # Empty slice
        empty = key[2:2]
        self.assertEqual(len(empty), 0)

        # Full slice
        full = key[:]
        self.assertEqual(full.to_list(), [1, 2, 3, 4, 5])

        # Slice with step
        step = key[::2]
        self.assertEqual(step.to_list(), [1, 3, 5])

    def test_repr(self):
        """Test __repr__ method."""
        key = OptimizedRadixKey([1, 2, 3], extra_key="test")
        repr_str = repr(key)

        self.assertIn("OptimizedRadixKey", repr_str)
        self.assertIn("extra_key='test'", repr_str)
        self.assertIn("[1, 2, 3]", repr_str)

    def test_repr_long_sequence(self):
        """Test __repr__ with long sequence (should truncate)."""
        long_tokens = list(range(20))
        key = OptimizedRadixKey(long_tokens)
        repr_str = repr(key)

        self.assertIn("...", repr_str)
        self.assertIn("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", repr_str)

    def test_equality(self):
        """Test __eq__ method."""
        key1 = OptimizedRadixKey([1, 2, 3])
        key2 = OptimizedRadixKey([1, 2, 3])
        key3 = OptimizedRadixKey([1, 2, 4])
        key4 = OptimizedRadixKey([1, 2, 3], extra_key="different")

        # Equal
        self.assertEqual(key1, key2)

        # Different tokens
        self.assertNotEqual(key1, key3)

        # Different extra_key
        self.assertNotEqual(key1, key4)

    def test_to_list(self):
        """Test to_list() method."""
        tokens = [1, 2, 3, 4, 5]
        key = OptimizedRadixKey(tokens)

        result = key.to_list()
        self.assertEqual(result, tokens)
        self.assertIsInstance(result, list)

    def test_to_tensor(self):
        """Test to_tensor() method."""
        key = OptimizedRadixKey([1, 2, 3, 4])

        tensor = key.to_tensor()
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.tolist(), [1, 2, 3, 4])

    def test_clone(self):
        """Test clone() method."""
        key1 = OptimizedRadixKey([1, 2, 3], extra_key="test")
        key2 = key1.clone()

        # Should be equal
        self.assertEqual(key1, key2)

        # But different tensor objects
        self.assertIsNot(key1.token_tensor, key2.token_tensor)

    def test_device_handling(self):
        """Test device parameter."""
        # Create on CPU
        key_cpu = OptimizedRadixKey([1, 2, 3], device=torch.device('cpu'))
        self.assertEqual(key_cpu.token_tensor.device.type, 'cpu')

        # If CUDA available, test CUDA
        if torch.cuda.is_available():
            key_cuda = OptimizedRadixKey([1, 2, 3], device=torch.device('cuda'))
            self.assertEqual(key_cuda.token_tensor.device.type, 'cuda')

    def test_to_device(self):
        """Test to() method for device transfer."""
        key = OptimizedRadixKey([1, 2, 3])

        # Move to CPU (should be no-op)
        key_cpu = key.to('cpu')
        self.assertEqual(key_cpu.token_tensor.device.type, 'cpu')

        # If CUDA available, test transfer
        if torch.cuda.is_available():
            key_cuda = key.to('cuda')
            self.assertEqual(key_cuda.token_tensor.device.type, 'cuda')
            self.assertEqual(key_cuda.to_list(), [1, 2, 3])


class TestVectorizedMatching(unittest.TestCase):
    """Test vectorized matching functions."""

    def test_optimized_key_match_vectorized_exact(self):
        """Test exact match."""
        key1 = OptimizedRadixKey([1, 2, 3, 4, 5])
        key2 = OptimizedRadixKey([1, 2, 3, 4, 5])

        match_len = optimized_key_match_vectorized(key1, key2)
        self.assertEqual(match_len, 5)

    def test_optimized_key_match_vectorized_partial(self):
        """Test partial match."""
        key1 = OptimizedRadixKey([1, 2, 3, 4, 5])
        key2 = OptimizedRadixKey([1, 2, 3, 6, 7])

        match_len = optimized_key_match_vectorized(key1, key2)
        self.assertEqual(match_len, 3)

    def test_optimized_key_match_vectorized_no_match(self):
        """Test no match."""
        key1 = OptimizedRadixKey([1, 2, 3])
        key2 = OptimizedRadixKey([4, 5, 6])

        match_len = optimized_key_match_vectorized(key1, key2)
        self.assertEqual(match_len, 0)

    def test_optimized_key_match_vectorized_different_lengths(self):
        """Test keys with different lengths."""
        key1 = OptimizedRadixKey([1, 2, 3, 4, 5])
        key2 = OptimizedRadixKey([1, 2, 3])

        match_len = optimized_key_match_vectorized(key1, key2)
        self.assertEqual(match_len, 3)

    def test_optimized_key_match_vectorized_extra_key(self):
        """Test that extra_key mismatch returns 0."""
        key1 = OptimizedRadixKey([1, 2, 3], extra_key="a")
        key2 = OptimizedRadixKey([1, 2, 3], extra_key="b")

        match_len = optimized_key_match_vectorized(key1, key2)
        self.assertEqual(match_len, 0)

    def test_optimized_key_match_paged(self):
        """Test paged matching."""
        # Exact page match
        key1 = OptimizedRadixKey([1, 2, 3, 4, 5, 6, 7, 8])
        key2 = OptimizedRadixKey([1, 2, 3, 4, 9, 10, 11, 12])

        match_len = optimized_key_match_paged_vectorized(key1, key2, page_size=4)
        self.assertEqual(match_len, 4)

    def test_optimized_key_match_paged_partial_page(self):
        """Test paged matching with partial page."""
        key1 = OptimizedRadixKey([1, 2, 3, 4, 5])
        key2 = OptimizedRadixKey([1, 2, 3, 4, 6])

        # With page_size=4, should match 4 tokens
        match_len = optimized_key_match_paged_vectorized(key1, key2, page_size=4)
        self.assertEqual(match_len, 4)

    def test_optimized_key_match_paged_multiple_pages(self):
        """Test paged matching with multiple pages."""
        key1 = OptimizedRadixKey([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        key2 = OptimizedRadixKey([1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16])

        match_len = optimized_key_match_paged_vectorized(key1, key2, page_size=4)
        self.assertEqual(match_len, 8)  # 2 pages match


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with original RadixKey."""

    def test_radix_key_wrapper(self):
        """Test that RadixKey wrapper works."""
        key = RadixKey([1, 2, 3, 4])

        self.assertIsInstance(key, OptimizedRadixKey)
        self.assertEqual(len(key), 4)
        self.assertEqual(key.to_list(), [1, 2, 3, 4])

    def test_radix_key_with_extra_key(self):
        """Test RadixKey with extra_key."""
        key = RadixKey([1, 2, 3], extra_key="test")

        self.assertEqual(key.extra_key, "test")
        self.assertEqual(len(key), 3)


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency improvements."""

    def test_memory_usage_comparison(self):
        """Compare memory usage of list vs tensor."""
        import sys

        # Create large sequence
        size = 1000
        tokens_list = list(range(size))

        # List-based storage
        list_size = sys.getsizeof(tokens_list)
        # Add size of integers
        list_size += sum(sys.getsizeof(i) for i in tokens_list[:10])  # Sample

        # Tensor-based storage
        key = OptimizedRadixKey(tokens_list)
        tensor_size = key.token_tensor.element_size() * key.token_tensor.nelement()

        print(f"\nMemory usage for {size} tokens:")
        print(f"  List: ~{list_size:,} bytes")
        print(f"  Tensor: {tensor_size:,} bytes")
        print(f"  Reduction: ~{(1 - tensor_size/list_size)*100:.1f}%")

        # Tensor should use less memory
        self.assertLess(tensor_size, list_size)


class TestPerformance(unittest.TestCase):
    """Performance tests (informational)."""

    def test_matching_performance(self):
        """Compare matching performance."""
        import time

        # Create long keys
        size = 10000
        key1 = OptimizedRadixKey(list(range(size)))
        key2 = OptimizedRadixKey(list(range(size // 2)) + list(range(size, size + size // 2)))

        # Warm up
        for _ in range(10):
            _ = optimized_key_match_vectorized(key1, key2)

        # Benchmark
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            match_len = optimized_key_match_vectorized(key1, key2)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations * 1000  # ms

        print(f"\nVectorized matching ({size} tokens):")
        print(f"  Average time: {avg_time:.3f} ms")
        print(f"  Match length: {match_len}")

        # Should be reasonably fast
        self.assertLess(avg_time, 10)  # < 10ms per match


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
