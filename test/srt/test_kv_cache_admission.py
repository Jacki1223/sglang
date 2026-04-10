"""
Unit tests for KV cache admission policies.

Tests for sglang.srt.mem_cache.admission_policy — these are pure-Python
unit tests that do NOT require a GPU or a running server.
"""

import unittest
from typing import List
from unittest.mock import MagicMock

from sglang.srt.mem_cache.admission_policy import (
    ADMISSION_POLICY_CHOICES,
    AlwaysAdmitPolicy,
    ContentHashAdmissionPolicy,
    KVAdmissionPolicy,
    NeverAdmitPolicy,
    NgramFrequencyAdmissionPolicy,
    PrefixPopularityAdmissionPolicy,
    create_admission_policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(hit_count: int = 0):
    """Return a minimal mock TreeNode with a configurable hit_count."""
    node = MagicMock()
    node.hit_count = hit_count
    return node


def tokens(n: int, start: int = 1) -> List[int]:
    """Return a simple integer token sequence of length n."""
    return list(range(start, start + n))


# ---------------------------------------------------------------------------
# AlwaysAdmitPolicy
# ---------------------------------------------------------------------------


class TestAlwaysAdmitPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = AlwaysAdmitPolicy()

    def test_admits_all_tokens(self):
        ids = tokens(10)
        self.assertEqual(self.policy.compute_admit_length(ids, 0, 1), 10)

    def test_admits_all_with_page_size(self):
        # 12 tokens, page_size 4 → floor_page(12, 4) = 12
        ids = tokens(12)
        self.assertEqual(self.policy.compute_admit_length(ids, 0, 4), 12)

    def test_paged_truncation(self):
        # 11 tokens, page_size 4 → floor_page(11, 4) = 8
        ids = tokens(11)
        self.assertEqual(self.policy.compute_admit_length(ids, 0, 4), 8)

    def test_cached_prefix_included(self):
        ids = tokens(10)
        result = self.policy.compute_admit_length(ids, 5, 1)
        self.assertEqual(result, 10)

    def test_empty_tokens(self):
        self.assertEqual(self.policy.compute_admit_length([], 0, 1), 0)

    def test_update_noop(self):
        # update() must not raise
        self.policy.update(tokens(5), 5, 0)


# ---------------------------------------------------------------------------
# NeverAdmitPolicy
# ---------------------------------------------------------------------------


class TestNeverAdmitPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = NeverAdmitPolicy()

    def test_admits_nothing_beyond_cached(self):
        ids = tokens(10)
        self.assertEqual(self.policy.compute_admit_length(ids, 0, 1), 0)

    def test_preserves_cached_len(self):
        ids = tokens(10)
        self.assertEqual(self.policy.compute_admit_length(ids, 6, 1), 6)

    def test_paged_cached_len_respected(self):
        # cached_len 5, page_size 4 → floor_page(5, 4) = 4
        ids = tokens(12)
        result = self.policy.compute_admit_length(ids, 5, 4)
        # NeverAdmit: floor_page(cached_len=5, 4) = 4
        self.assertEqual(result, 4)

    def test_empty_tokens(self):
        self.assertEqual(self.policy.compute_admit_length([], 0, 1), 0)


# ---------------------------------------------------------------------------
# PrefixPopularityAdmissionPolicy
# ---------------------------------------------------------------------------


class TestPrefixPopularityAdmissionPolicy(unittest.TestCase):
    def test_admits_all_when_node_is_hot(self):
        policy = PrefixPopularityAdmissionPolicy(min_hit_count=1)
        ids = tokens(10)
        node = make_node(hit_count=2)
        self.assertEqual(policy.compute_admit_length(ids, 0, 1, node), 10)

    def test_rejects_new_pages_when_node_is_cold(self):
        policy = PrefixPopularityAdmissionPolicy(min_hit_count=1)
        ids = tokens(10)
        node = make_node(hit_count=0)
        # Cold node → only admit cached_len (0 here)
        self.assertEqual(policy.compute_admit_length(ids, 0, 1, node), 0)

    def test_preserves_cached_len_even_when_cold(self):
        policy = PrefixPopularityAdmissionPolicy(min_hit_count=2)
        ids = tokens(10)
        node = make_node(hit_count=1)
        # Cold: only cached part (4) is kept
        result = policy.compute_admit_length(ids, 4, 1, node)
        self.assertEqual(result, 4)

    def test_always_admit_pages_covers_unconditional(self):
        policy = PrefixPopularityAdmissionPolicy(
            min_hit_count=5, always_admit_pages=3
        )
        ids = tokens(10)
        cold_node = make_node(hit_count=0)
        # always_admit_pages=3, page_size=1 → first 3 tokens unconditional
        result = policy.compute_admit_length(ids, 0, 1, cold_node)
        self.assertEqual(result, 3)

    def test_always_admit_pages_with_page_size(self):
        policy = PrefixPopularityAdmissionPolicy(
            min_hit_count=5, always_admit_pages=2
        )
        # always_admit_pages=2 × page_size=4 = 8 tokens unconditional
        ids = tokens(20)
        cold_node = make_node(hit_count=0)
        result = policy.compute_admit_length(ids, 0, 4, cold_node)
        self.assertEqual(result, 8)

    def test_fallback_to_full_admit_when_no_node(self):
        policy = PrefixPopularityAdmissionPolicy(min_hit_count=999)
        ids = tokens(10)
        result = policy.compute_admit_length(ids, 0, 1, node=None)
        self.assertEqual(result, 10)

    def test_min_hit_count_zero_always_admits(self):
        # min_hit_count=0 means even cold nodes pass
        policy = PrefixPopularityAdmissionPolicy(min_hit_count=0)
        ids = tokens(8)
        node = make_node(hit_count=0)
        self.assertEqual(policy.compute_admit_length(ids, 0, 1, node), 8)

    def test_page_aligned_output(self):
        policy = PrefixPopularityAdmissionPolicy(min_hit_count=1)
        ids = tokens(11)
        node = make_node(hit_count=3)
        # floor_page(11, 4) = 8
        self.assertEqual(policy.compute_admit_length(ids, 0, 4, node), 8)

    def test_invalid_min_hit_count(self):
        with self.assertRaises(ValueError):
            PrefixPopularityAdmissionPolicy(min_hit_count=-1)


# ---------------------------------------------------------------------------
# NgramFrequencyAdmissionPolicy
# ---------------------------------------------------------------------------


class TestNgramFrequencyAdmissionPolicy(unittest.TestCase):
    def test_cold_cache_admits_nothing_beyond_always_prefix(self):
        policy = NgramFrequencyAdmissionPolicy(min_frequency=2.0)
        ids = tokens(10)
        # No updates → all counts are 0 → score < 2 for every page
        result = policy.compute_admit_length(ids, 0, 1)
        self.assertEqual(result, 0)

    def test_always_admit_prefix_len_respected(self):
        policy = NgramFrequencyAdmissionPolicy(
            min_frequency=5.0, always_admit_prefix_len=6
        )
        ids = tokens(10)
        # First 6 tokens always admitted (page_size=1, so aligned=6)
        result = policy.compute_admit_length(ids, 0, 1)
        self.assertEqual(result, 6)

    def test_frequent_content_is_admitted_page_size_4(self):
        """With page_size=4 each page has internal bigrams that are trackable."""
        policy = NgramFrequencyAdmissionPolicy(
            min_frequency=2.0, window_size=10000
        )
        ids = tokens(4)
        # Simulate seeing these bigrams 3 times
        for _ in range(3):
            policy.update(ids, len(ids), 0, page_size=4)

        result = policy.compute_admit_length(ids, 0, 4)
        self.assertEqual(result, 4)

    def test_frequent_content_is_admitted_page_size_1_with_context(self):
        """With page_size=1, entry bigrams are used; first cached token provides context."""
        policy = NgramFrequencyAdmissionPolicy(
            min_frequency=2.0, window_size=10000
        )
        ids = tokens(5)  # [1, 2, 3, 4, 5]
        # Simulate seeing these bigrams 3 times
        for _ in range(3):
            policy.update(ids, len(ids), 0, page_size=1)

        # cached_len=1: first token already cached; remaining tokens scored via entry bigrams
        result = policy.compute_admit_length(ids, 1, 1)
        self.assertEqual(result, 5)

    def test_infrequent_content_stopped_at_first_miss(self):
        policy = NgramFrequencyAdmissionPolicy(
            min_frequency=3.0, window_size=10000
        )
        # tokens 1-4 seen twice, tokens 5-8 seen only once
        common_ids = tokens(4, start=1)
        rare_ids = tokens(4, start=5)
        for _ in range(2):
            policy.update(common_ids, 4, 0, page_size=4)
        policy.update(rare_ids, 4, 0, page_size=4)

        # page_size=4: first page=common_ids (freq >= 2), second page=rare_ids (freq=1 < 3)
        all_ids = common_ids + rare_ids
        result = policy.compute_admit_length(all_ids, 0, 4)
        # Only the first page is admitted because second page's score is too low
        self.assertEqual(result, 4)

    def test_window_eviction(self):
        # Tiny window — old entries should decay.
        policy = NgramFrequencyAdmissionPolicy(
            min_frequency=2.0, window_size=3
        )
        ids = [1, 2, 3, 4]
        # Three updates, then window = 3; the oldest bigrams are evicted.
        for _ in range(3):
            policy.update(ids, 4, 0, page_size=1)
        # New data floods out old data; policy should still not crash.
        policy.update(tokens(10, start=100), 10, 0, page_size=1)

    def test_update_noop_for_short_sequence(self):
        policy = NgramFrequencyAdmissionPolicy()
        # Single token → no bigrams; must not crash
        policy.update([42], 1, 0)

    def test_invalid_min_frequency(self):
        with self.assertRaises(ValueError):
            NgramFrequencyAdmissionPolicy(min_frequency=-0.1)

    def test_page_aligned_output(self):
        policy = NgramFrequencyAdmissionPolicy(min_frequency=0.0)
        # freq 0 → every page admitted (frequency >= 0)
        ids = tokens(11)
        result = policy.compute_admit_length(ids, 0, 4)
        # floor_page(11, 4) = 8
        self.assertEqual(result, 8)


# ---------------------------------------------------------------------------
# ContentHashAdmissionPolicy
# ---------------------------------------------------------------------------


class TestContentHashAdmissionPolicy(unittest.TestCase):
    def test_cold_cache_admits_nothing(self):
        policy = ContentHashAdmissionPolicy(min_count=1)
        ids = tokens(8)
        result = policy.compute_admit_length(ids, 0, 4)
        self.assertEqual(result, 0)

    def test_admitted_after_second_occurrence(self):
        policy = ContentHashAdmissionPolicy(min_count=1)
        ids = tokens(8)
        # First occurrence — update hashes
        policy.update(ids, 8, 0, page_size=4)
        # Now counts for each 4-token page are 1 >= min_count=1 → admit all
        result = policy.compute_admit_length(ids, 0, 4)
        self.assertEqual(result, 8)

    def test_partial_admission(self):
        policy = ContentHashAdmissionPolicy(min_count=2)
        page_a = tokens(4, start=1)
        page_b = tokens(4, start=5)
        all_ids = page_a + page_b

        # page_a seen twice, page_b seen once
        policy.update(page_a, 4, 0, page_size=4)
        policy.update(page_a, 4, 0, page_size=4)
        policy.update(page_b, 4, 0, page_size=4)

        result = policy.compute_admit_length(all_ids, 0, 4)
        # page_a has count=2 >= 2, page_b has count=1 < 2 → stop after page_a
        self.assertEqual(result, 4)

    def test_always_admit_prefix_len(self):
        policy = ContentHashAdmissionPolicy(min_count=5, always_admit_prefix_len=8)
        ids = tokens(16)
        # No history → but first 8 tokens unconditional
        result = policy.compute_admit_length(ids, 0, 4)
        self.assertEqual(result, 8)

    def test_window_eviction(self):
        policy = ContentHashAdmissionPolicy(min_count=1, window_size=2)
        ids = tokens(8)
        # 2 pages × 2 occurrences → fills window; old entries evicted
        policy.update(ids, 8, 0, page_size=4)
        policy.update(ids, 8, 0, page_size=4)
        # More data floods out old pages
        policy.update(tokens(8, start=100), 8, 0, page_size=4)
        # Must not crash; result can be anything valid
        result = policy.compute_admit_length(ids, 0, 4)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 8)

    def test_cached_prefix_always_preserved(self):
        policy = ContentHashAdmissionPolicy(min_count=999)
        ids = tokens(8)
        # cached_len=4 → result must be >= 4
        result = policy.compute_admit_length(ids, 4, 4)
        self.assertGreaterEqual(result, 4)

    def test_invalid_min_count(self):
        with self.assertRaises(ValueError):
            ContentHashAdmissionPolicy(min_count=-1)

    def test_empty_token_ids(self):
        policy = ContentHashAdmissionPolicy(min_count=1)
        result = policy.compute_admit_length([], 0, 4)
        self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# create_admission_policy factory
# ---------------------------------------------------------------------------


class TestCreateAdmissionPolicyFactory(unittest.TestCase):
    def test_always(self):
        p = create_admission_policy("always")
        self.assertIsInstance(p, AlwaysAdmitPolicy)

    def test_never(self):
        p = create_admission_policy("never")
        self.assertIsInstance(p, NeverAdmitPolicy)

    def test_prefix_popularity(self):
        p = create_admission_policy("prefix_popularity", min_hit_count=3)
        self.assertIsInstance(p, PrefixPopularityAdmissionPolicy)
        self.assertEqual(p.min_hit_count, 3)

    def test_ngram_frequency(self):
        p = create_admission_policy("ngram_frequency", min_frequency=4.5)
        self.assertIsInstance(p, NgramFrequencyAdmissionPolicy)
        self.assertEqual(p.min_frequency, 4.5)

    def test_content_hash(self):
        p = create_admission_policy("content_hash", min_count=2)
        self.assertIsInstance(p, ContentHashAdmissionPolicy)
        self.assertEqual(p.min_count, 2)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            create_admission_policy("banana_policy")

    def test_case_insensitive(self):
        p = create_admission_policy("ALWAYS")
        self.assertIsInstance(p, AlwaysAdmitPolicy)

    def test_all_registered_choices_creatable(self):
        for name in ADMISSION_POLICY_CHOICES:
            p = create_admission_policy(name)
            self.assertIsInstance(p, KVAdmissionPolicy)


# ---------------------------------------------------------------------------
# floor_page utility
# ---------------------------------------------------------------------------


class TestFloorPage(unittest.TestCase):
    def test_page_size_1(self):
        self.assertEqual(KVAdmissionPolicy.floor_page(7, 1), 7)

    def test_exact_multiple(self):
        self.assertEqual(KVAdmissionPolicy.floor_page(8, 4), 8)

    def test_rounds_down(self):
        self.assertEqual(KVAdmissionPolicy.floor_page(9, 4), 8)
        self.assertEqual(KVAdmissionPolicy.floor_page(7, 4), 4)
        self.assertEqual(KVAdmissionPolicy.floor_page(3, 4), 0)

    def test_zero_length(self):
        self.assertEqual(KVAdmissionPolicy.floor_page(0, 4), 0)


# ---------------------------------------------------------------------------
# Invariant: result is always page-aligned and in [cached_len, floor_page(N)]
# ---------------------------------------------------------------------------


class TestAdmitLengthInvariant(unittest.TestCase):
    """
    All concrete policies must satisfy:
      floor_page(cached_len, ps) <= result <= floor_page(len(ids), ps)
    and result must itself be page-aligned.
    """

    POLICIES = [
        AlwaysAdmitPolicy(),
        NeverAdmitPolicy(),
        PrefixPopularityAdmissionPolicy(min_hit_count=1),
        PrefixPopularityAdmissionPolicy(min_hit_count=0),
        NgramFrequencyAdmissionPolicy(min_frequency=2.0),
        NgramFrequencyAdmissionPolicy(min_frequency=0.0),
        ContentHashAdmissionPolicy(min_count=1),
        ContentHashAdmissionPolicy(min_count=0),
    ]

    def _check_invariant(self, policy, ids, cached_len, page_size, node=None):
        result = policy.compute_admit_length(ids, cached_len, page_size, node)
        max_admit = KVAdmissionPolicy.floor_page(len(ids), page_size)
        min_admit = KVAdmissionPolicy.floor_page(cached_len, page_size)

        self.assertGreaterEqual(
            result,
            min_admit,
            f"{policy.__class__.__name__}: result {result} < cached_len "
            f"(floor) {min_admit}",
        )
        self.assertLessEqual(
            result,
            max_admit,
            f"{policy.__class__.__name__}: result {result} > max {max_admit}",
        )
        if page_size > 1:
            self.assertEqual(
                result % page_size,
                0,
                f"{policy.__class__.__name__}: result {result} not page-aligned "
                f"(page_size={page_size})",
            )

    def test_invariant_page_size_1(self):
        for policy in self.POLICIES:
            for n in (0, 1, 5, 10, 100):
                for cached in (0, 2, min(4, n)):
                    if cached > n:
                        continue  # invalid state: cached_len > len(token_ids)
                    self._check_invariant(policy, tokens(n), cached, 1)

    def test_invariant_page_size_4(self):
        for policy in self.POLICIES:
            for n in (0, 4, 8, 12, 13, 20):
                for cached in (0, 4, min(8, n)):
                    if cached > n:
                        continue
                    self._check_invariant(
                        policy, tokens(n), cached, 4, node=make_node(0)
                    )

    def test_invariant_with_hot_node(self):
        for policy in self.POLICIES:
            node = make_node(hit_count=10)
            self._check_invariant(policy, tokens(16), 4, 4, node)


if __name__ == "__main__":
    unittest.main()
