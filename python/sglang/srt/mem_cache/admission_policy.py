from __future__ import annotations

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KV Cache Admission Policies.

Implements selective KV cache admission to replace blind write-in, inspired by:
  "KV Admission: Learning What to Write for Efficient Long-Context Inference"

Key idea: not every computed KV page is worth writing into the prefix cache.
An *admission policy* decides, for each page, whether storing it is likely
to pay off through a future cache hit.  By being selective we can:
  - Reduce cache pollution from one-off requests.
  - Preserve cache space for genuinely reusable prefixes.
  - Improve the effective hit-rate for popular content.

Hierarchy:
    KVAdmissionPolicy (abstract)
    ├── AlwaysAdmitPolicy          — current default: admit everything
    ├── NeverAdmitPolicy           — admit nothing (baseline / testing)
    ├── PrefixPopularityAdmissionPolicy — use parent-node hit count
    ├── NgramFrequencyAdmissionPolicy  — bigram frequency over a sliding window
    └── ContentHashAdmissionPolicy     — exact-match page deduplication

All policies expose two methods:
  compute_admit_length(token_ids, cached_len, page_size, node) -> int
  update(token_ids, admitted_len, cached_len, page_size)

``compute_admit_length`` is called synchronously before writing to the radix
tree.  ``update`` is called afterwards so the policy can adjust its internal
statistics for the next decision.
"""

import abc
import collections
import hashlib
import struct
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class KVAdmissionPolicy(abc.ABC):
    """
    Abstract base for all KV cache admission policies.

    Each policy answers the question: given a completed request whose full
    token sequence is ``token_ids``, how many of those tokens' KV pairs
    should we commit to the radix-tree prefix cache?

    The returned count is *page-aligned* and lies in the closed interval
    ``[cached_len, floor_page(len(token_ids), page_size)]``.

    * ``cached_len`` tokens are already protected inside the tree (they were
      matched by a previous ``match_prefix`` call); these are never dropped.
    * Tokens beyond the returned length are freed immediately (their KV was
      computed for the current request but will not be reused).
    """

    @abc.abstractmethod
    def compute_admit_length(
        self,
        token_ids: List[int],
        cached_len: int,
        page_size: int,
        node: Optional["TreeNode"] = None,
    ) -> int:
        """
        Compute how many tokens to admit into the prefix cache.

        Args:
            token_ids:  Full, page-aligned token sequence for this request.
            cached_len: Number of tokens already locked in the tree.
            page_size:  KV cache page size (tokens per page).
            node:       Last matched ``TreeNode`` from ``match_prefix``;
                        carries ``hit_count`` and ancestry information.

        Returns:
            Page-aligned token count to cache.  Must satisfy
            ``cached_len <= result <= floor_page(len(token_ids), page_size)``.
        """
        ...

    def update(
        self,
        token_ids: List[int],
        admitted_len: int,
        cached_len: int,
        page_size: int = 1,
    ) -> None:
        """
        Optional callback: update internal statistics after a request.

        Default implementation is a no-op; override to implement online
        learning / frequency tracking.

        Args:
            token_ids:    Full token sequence of the completed request.
            admitted_len: Number of tokens that were actually admitted.
            cached_len:   Tokens that were already in the cache.
            page_size:    KV cache page size.
        """

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def floor_page(length: int, page_size: int) -> int:
        """Round *length* down to the nearest page boundary."""
        if page_size <= 1:
            return length
        return (length // page_size) * page_size

    def _clamp_admit_len(
        self, admit_len: int, cached_len: int, max_len: int, page_size: int
    ) -> int:
        """Clamp and align an admission length to a valid, page-aligned value."""
        admit_len = self.floor_page(admit_len, page_size)
        return max(cached_len, min(admit_len, self.floor_page(max_len, page_size)))


# ---------------------------------------------------------------------------
# Trivial policies
# ---------------------------------------------------------------------------


class AlwaysAdmitPolicy(KVAdmissionPolicy):
    """
    Unconditionally admits all computed KV pages.

    This reproduces SGLang's historical behaviour and serves as the default
    when no admission policy is explicitly configured.
    """

    def compute_admit_length(
        self, token_ids, cached_len, page_size, node=None
    ) -> int:
        return self.floor_page(len(token_ids), page_size)


class NeverAdmitPolicy(KVAdmissionPolicy):
    """
    Admits nothing beyond what is already cached.

    Useful as a lower-bound baseline to measure how much the prefix cache
    contributes to a workload.  All freshly computed KV is freed immediately.
    """

    def compute_admit_length(
        self, token_ids, cached_len, page_size, node=None
    ) -> int:
        return self.floor_page(cached_len, page_size)


# ---------------------------------------------------------------------------
# Prefix-popularity policy
# ---------------------------------------------------------------------------


class PrefixPopularityAdmissionPolicy(KVAdmissionPolicy):
    """
    Admit new pages only when the node at the insertion point is popular.

    Rationale: if a radix-tree node has been accessed frequently
    (``node.hit_count >= min_hit_count``), it is a *hot* prefix that many
    requests share.  Extending such a node with new content is valuable
    because future requests that match the same popular prefix are likely
    to keep reading beyond it.

    Conversely, if the insertion node has never been a cache-hit, the
    sequence is probably unique to this request and not worth caching.

    An ``always_admit_pages`` guard ensures that the first N pages of every
    request are unconditionally cached.  This covers system prompts and
    other *shared preambles* that are inserted only once (hit_count == 0 at
    that moment) but are extremely valuable to cache for subsequent reuse.

    Args:
        min_hit_count:      Minimum ``hit_count`` of the last-matched node
                            for new pages to be admitted.  Default 1 means
                            "admit only if this prefix was a cache-hit before."
        always_admit_pages: Unconditionally admit the first this-many pages
                            (measured from the beginning of the sequence).
                            Default 0 disables the guard.
    """

    def __init__(
        self,
        min_hit_count: int = 1,
        always_admit_pages: int = 0,
    ) -> None:
        if min_hit_count < 0:
            raise ValueError("min_hit_count must be >= 0")
        self.min_hit_count = min_hit_count
        self.always_admit_pages = always_admit_pages

    def compute_admit_length(
        self, token_ids, cached_len, page_size, node=None
    ) -> int:
        max_admit = self.floor_page(len(token_ids), page_size)
        always_end = self.floor_page(self.always_admit_pages * page_size, page_size)

        # Unconditionally admit up to always_end.
        unconditional = min(always_end, max_admit)

        # If no node information is available, fall back to full admission.
        if node is None:
            return max_admit

        # Admit beyond unconditional only if the insertion-point node is hot.
        if node.hit_count >= self.min_hit_count:
            return max_admit

        # Node is cold: cap at the unconditional prefix.
        return max(self.floor_page(cached_len, page_size), unconditional)


# ---------------------------------------------------------------------------
# N-gram frequency policy
# ---------------------------------------------------------------------------


class NgramFrequencyAdmissionPolicy(KVAdmissionPolicy):
    """
    Admit pages whose token bigram content appears frequently across requests.

    For each candidate page we compute an *average bigram frequency* score.
    Pages whose score meets or exceeds ``min_frequency`` are admitted;
    scanning stops at the first page that falls below the threshold, which
    preserves the *contiguous-prefix* invariant required by the radix tree.

    Bigram frequencies are maintained in a bounded sliding-window counter
    so that old, stale observations decay away automatically.

    This policy approximates the offline-trained "learned admission" from the
    KV-Admission paper using purely online, zero-shot statistics.

    Args:
        min_frequency:          Minimum average bigram count for a page to be
                                admitted.  A value of 2 means "this pair of
                                tokens must have appeared together at least
                                twice in recent requests."
        window_size:            Maximum number of bigrams retained in memory.
                                Older bigrams are evicted FIFO once the window
                                is full.
        always_admit_prefix_len: Unconditionally admit the first N tokens
                                (page-aligned).  Defaults to 0.
    """

    def __init__(
        self,
        min_frequency: float = 2.0,
        window_size: int = 200_000,
        always_admit_prefix_len: int = 0,
    ) -> None:
        if min_frequency < 0:
            raise ValueError("min_frequency must be >= 0")
        self.min_frequency = min_frequency
        self.window_size = window_size
        self.always_admit_prefix_len = always_admit_prefix_len

        # token-bigram → occurrence count (within sliding window)
        self._bigram_counts: collections.Counter = collections.Counter()
        # FIFO queue of recently seen bigram keys for sliding-window eviction
        self._bigram_window: collections.deque = collections.deque()

    # ------------------------------------------------------------------

    @staticmethod
    def _bigram_key(t0: int, t1: int) -> int:
        """Pack two 32-bit token IDs into a single 64-bit integer key."""
        return (int(t0) & 0xFFFF_FFFF) | ((int(t1) & 0xFFFF_FFFF) << 32)

    def _page_score(self, token_ids: List[int], start: int, end: int) -> float:
        """Compute the average bigram frequency for tokens[start:end].

        For pages with >= 2 tokens: average of internal bigrams.
        For single-token pages: use the *entry bigram*
          (token_ids[start-1], token_ids[start]) if a predecessor exists,
          because that cross-boundary bigram is what gets recorded by
          ``update()`` and serves as a proxy for "how predictable is this
          token given its predecessor."  If no predecessor is available
          (start == 0) the score is 0 — the first token of a sequence has
          no context to be scored against.
        """
        page = token_ids[start:end]
        n = len(page)
        if n == 0:
            return 0.0
        if n == 1:
            if start > 0:
                key = self._bigram_key(token_ids[start - 1], page[0])
                return float(self._bigram_counts.get(key, 0))
            return 0.0
        total = sum(
            self._bigram_counts.get(self._bigram_key(page[i], page[i + 1]), 0)
            for i in range(n - 1)
        )
        return total / (n - 1)

    # ------------------------------------------------------------------

    def compute_admit_length(
        self, token_ids, cached_len, page_size, node=None
    ) -> int:
        page_size = max(1, page_size)
        always_end = self.floor_page(self.always_admit_prefix_len, page_size)
        max_admit = self.floor_page(len(token_ids), page_size)

        # Pages up to always_end are unconditional.
        admit_len = min(max(self.floor_page(cached_len, page_size), always_end), max_admit)

        # Scan remaining pages from max(cached_start, always_end).
        scan_start = max(self.floor_page(cached_len, page_size), always_end)
        for page_start in range(scan_start, max_admit, page_size):
            page_end = page_start + page_size
            score = self._page_score(token_ids, page_start, page_end)
            if score >= self.min_frequency:
                admit_len = page_end
            else:
                break  # radix tree requires a contiguous prefix

        return admit_len

    def update(
        self,
        token_ids: List[int],
        admitted_len: int,
        cached_len: int,
        page_size: int = 1,
    ) -> None:
        """Record all bigrams from this request into the sliding window."""
        n = len(token_ids)
        if n < 2:
            return
        for i in range(n - 1):
            key = self._bigram_key(token_ids[i], token_ids[i + 1])
            self._bigram_counts[key] += 1
            self._bigram_window.append(key)

        # Evict oldest bigrams that overflow the window.
        while len(self._bigram_window) > self.window_size:
            old_key = self._bigram_window.popleft()
            self._bigram_counts[old_key] -= 1
            if self._bigram_counts[old_key] <= 0:
                del self._bigram_counts[old_key]


# ---------------------------------------------------------------------------
# Content-hash (exact-repeat) policy
# ---------------------------------------------------------------------------


class ContentHashAdmissionPolicy(KVAdmissionPolicy):
    """
    Admit a page only once its exact token content has been seen before.

    We hash each page-sized slice of the token sequence (using a fast
    non-cryptographic hash) and track per-hash occurrence counts in a
    bounded sliding-window counter.  A page is admitted when its hash
    count reaches ``min_count``.

    This is the most *precise* heuristic: it requires no approximation,
    no hyper-parameter tuning beyond the threshold, and it naturally
    handles:
      * system prompts (same tokens every request → quickly admitted)
      * RAG document chunks (same chunk re-embedded across queries)
      * one-off queries (never reach the threshold → never cached)

    It is the SGLang analogue of the offline-trained learned predictor in
    the KV-Admission paper, implemented with purely online exact statistics.

    Args:
        min_count:              A page is admitted once its content-hash has
                                been observed this many times.
                                ``min_count=1`` → admit on *second* occurrence.
                                ``min_count=0`` → equivalent to AlwaysAdmit.
        window_size:            Maximum number of page-hash entries to remember.
        always_admit_prefix_len: Unconditionally admit the first N tokens.
    """

    def __init__(
        self,
        min_count: int = 1,
        window_size: int = 50_000,
        always_admit_prefix_len: int = 0,
    ) -> None:
        if min_count < 0:
            raise ValueError("min_count must be >= 0")
        self.min_count = min_count
        self.window_size = window_size
        self.always_admit_prefix_len = always_admit_prefix_len

        self._hash_counts: collections.Counter = collections.Counter()
        self._hash_window: collections.deque = collections.deque()

    # ------------------------------------------------------------------

    @staticmethod
    def _page_hash(token_ids: List[int], start: int, end: int) -> int:
        """Compute a 64-bit integer hash of the token slice [start, end)."""
        page = token_ids[start:end]
        # Pack tokens as big-endian unsigned 32-bit integers.
        data = struct.pack(f">{len(page)}I", *page)
        digest = hashlib.md5(data, usedforsecurity=False).digest()  # type: ignore[call-arg]
        return struct.unpack(">q", digest[:8])[0]

    # ------------------------------------------------------------------

    def compute_admit_length(
        self, token_ids, cached_len, page_size, node=None
    ) -> int:
        page_size = max(1, page_size)
        always_end = self.floor_page(self.always_admit_prefix_len, page_size)
        max_admit = self.floor_page(len(token_ids), page_size)

        # Unconditional prefix.
        admit_len = min(max(self.floor_page(cached_len, page_size), always_end), max_admit)

        # Scan from the first page not already covered.
        scan_start = max(self.floor_page(cached_len, page_size), always_end)
        for page_start in range(scan_start, max_admit, page_size):
            page_end = page_start + page_size
            ph = self._page_hash(token_ids, page_start, page_end)
            if self._hash_counts.get(ph, 0) >= self.min_count:
                admit_len = page_end
            else:
                break  # stop at first uncached page

        return admit_len

    def update(
        self,
        token_ids: List[int],
        admitted_len: int,
        cached_len: int,
        page_size: int = 1,
    ) -> None:
        """Record all page-hashes from this request into the sliding window."""
        page_size = max(1, page_size)
        max_len = self.floor_page(len(token_ids), page_size)

        for page_start in range(0, max_len, page_size):
            page_end = page_start + page_size
            ph = self._page_hash(token_ids, page_start, page_end)
            self._hash_counts[ph] += 1
            self._hash_window.append(ph)

        # Evict oldest entries that overflow the window.
        while len(self._hash_window) > self.window_size:
            old = self._hash_window.popleft()
            self._hash_counts[old] -= 1
            if self._hash_counts[old] <= 0:
                del self._hash_counts[old]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: All registered policy names (used for argument validation).
ADMISSION_POLICY_CHOICES: List[str] = [
    "always",
    "never",
    "prefix_popularity",
    "ngram_frequency",
    "content_hash",
]


def create_admission_policy(
    policy: str,
    # PrefixPopularityAdmissionPolicy
    min_hit_count: int = 1,
    always_admit_pages: int = 0,
    # NgramFrequencyAdmissionPolicy
    min_frequency: float = 2.0,
    ngram_window_size: int = 200_000,
    # ContentHashAdmissionPolicy
    min_count: int = 1,
    hash_window_size: int = 50_000,
    # Shared
    always_admit_prefix_len: int = 0,
) -> KVAdmissionPolicy:
    """
    Factory that creates a ``KVAdmissionPolicy`` by name.

    Args:
        policy:                 One of ``ADMISSION_POLICY_CHOICES``.
        min_hit_count:          For ``prefix_popularity``.
        always_admit_pages:     For ``prefix_popularity``.
        min_frequency:          For ``ngram_frequency``.
        ngram_window_size:      For ``ngram_frequency``.
        min_count:              For ``content_hash``.
        hash_window_size:       For ``content_hash``.
        always_admit_prefix_len: Shared unconditional-prefix guard (tokens).

    Returns:
        Configured ``KVAdmissionPolicy`` instance.
    """
    policy = policy.lower().strip()

    if policy == "always":
        return AlwaysAdmitPolicy()

    if policy == "never":
        return NeverAdmitPolicy()

    if policy == "prefix_popularity":
        return PrefixPopularityAdmissionPolicy(
            min_hit_count=min_hit_count,
            always_admit_pages=always_admit_pages,
        )

    if policy == "ngram_frequency":
        return NgramFrequencyAdmissionPolicy(
            min_frequency=min_frequency,
            window_size=ngram_window_size,
            always_admit_prefix_len=always_admit_prefix_len,
        )

    if policy == "content_hash":
        return ContentHashAdmissionPolicy(
            min_count=min_count,
            window_size=hash_window_size,
            always_admit_prefix_len=always_admit_prefix_len,
        )

    raise ValueError(
        f"Unknown KV cache admission policy: {policy!r}. "
        f"Supported policies: {ADMISSION_POLICY_CHOICES}"
    )
