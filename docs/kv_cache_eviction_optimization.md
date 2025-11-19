# KV Cache Eviction Optimization

This document describes the value-aware eviction strategies implemented to improve KV cache efficiency in SGLang.

## Overview

The KV cache eviction mechanism has been enhanced with three new strategies that consider the structural value of cached nodes, not just their access patterns. These strategies help protect important cache entries (such as common prefixes) from premature eviction.

## New Eviction Strategies

### 1. Value-Aware LRU (`value_aware_lru`)

An enhanced LRU strategy that considers node value in addition to recency.

**Key Features:**
- Protects nodes with longer prefixes (likely to be shared by multiple requests)
- Protects nodes with children (common prefix nodes in the radix tree)
- Balances recency with structural importance

**When to Use:**
- Workloads with many shared prefixes (e.g., batch processing with similar prompts)
- Applications where cache hit rate is critical
- General-purpose workloads as a drop-in replacement for standard LRU

**Configuration:**
```python
cache = RadixCache(
    req_to_token_pool=req_pool,
    token_to_kv_pool_allocator=allocator,
    page_size=16,
    eviction_policy="value_aware_lru"
)
```

**Default Parameters:**
- `prefix_weight`: 0.001 - Weight for prefix length contribution
- `subtree_weight`: 0.01 - Weight for subtree size contribution
- `value_weight`: 1000.0 - Overall value score multiplier

### 2. Adaptive LFU (`adaptive_lfu`)

An LFU strategy that addresses the cold-start problem by protecting new nodes.

**Key Features:**
- Two-phase approach: protection phase → mature phase
- New nodes use LRU during protection period
- Mature nodes use frequency-based eviction
- Prevents new requests from being immediately evicted

**When to Use:**
- Workloads with varying access patterns
- Applications where new content should get fair treatment
- Scenarios where traditional LFU evicts new entries too aggressively

**Configuration:**
```python
cache = RadixCache(
    req_to_token_pool=req_pool,
    token_to_kv_pool_allocator=allocator,
    page_size=16,
    eviction_policy="adaptive_lfu"
)
```

**Default Parameters:**
- `protection_period`: 60.0 seconds - Time to protect new nodes
- `min_hit_threshold`: 3 hits - Minimum hits before graduating to mature phase

### 3. Value-Aware Adaptive LFU (`value_aware_adaptive_lfu`)

A combined strategy incorporating both value-awareness and adaptive behavior.

**Key Features:**
- Protects new nodes during protection period (adaptive)
- Considers structural value (prefix length, subtree size)
- Uses frequency-based eviction for mature nodes
- Provides comprehensive protection for both new and valuable nodes

**When to Use:**
- Production workloads with diverse access patterns
- Applications requiring maximum cache efficiency
- Scenarios with both shared prefixes and varying request patterns

**Configuration:**
```python
cache = RadixCache(
    req_to_token_pool=req_pool,
    token_to_kv_pool_allocator=allocator,
    page_size=16,
    eviction_policy="value_aware_adaptive_lfu"
)
```

**Default Parameters:**
- `protection_period`: 60.0 seconds
- `min_hit_threshold`: 3 hits
- `prefix_weight`: 0.001
- `subtree_weight`: 0.01
- `value_weight`: 1000.0

## Using with SGLang Server

You can specify the eviction policy when starting the SGLang server:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --radix-eviction-policy value_aware_lru \
    --port 30000
```

Available policies:
- `lru` (default) - Standard Least Recently Used
- `lfu` - Least Frequently Used
- `fifo` - First In First Out
- `mru` - Most Recently Used
- `filo` - First In Last Out
- `value_aware_lru` - Value-aware LRU (new)
- `adaptive_lfu` - Adaptive LFU (new)
- `value_aware_adaptive_lfu` - Combined strategy (new)

## Performance Expectations

Based on the optimization design, the new strategies should provide:

### Cache Hit Rate Improvements
- **Value-Aware LRU**: 10-15% improvement in workloads with shared prefixes
- **Adaptive LFU**: 5-10% improvement in workloads with varying patterns
- **Value-Aware Adaptive LFU**: 10-20% improvement in mixed workloads

### Throughput Improvements
- Overall: 15-25% throughput increase due to higher cache hit rates
- Most significant gains in batch processing scenarios
- Reduced latency variance due to better cache retention

### When to Expect Maximum Benefit

**High Benefit Scenarios:**
- Batch processing with similar prompts (high prefix sharing)
- Chat applications with conversation history
- Code generation with common prefixes
- RAG applications with repeated context

**Moderate Benefit Scenarios:**
- Mixed workloads with some pattern repetition
- General-purpose LLM serving

**Low Benefit Scenarios:**
- Completely random, non-repetitive queries
- Very small cache sizes where eviction is rare

## Implementation Details

### Value Score Calculation

For `ValueAwareLRUStrategy` and `ValueAwareAdaptiveLFUStrategy`:

```python
value_score = prefix_weight * prefix_length + subtree_weight * num_children

priority = recency - (value_score * value_weight)
```

Lower priority = evicted first, so high-value nodes get lower priority (protected).

### Phase Transition in Adaptive Strategies

```python
if age < protection_period OR hit_count < min_hit_threshold:
    phase = 0  # Protection phase (use LRU)
else:
    phase = 1  # Mature phase (use LFU)
```

Priority tuple: `(phase, hit_count, recency)` sorted in ascending order.

## Monitoring and Tuning

### Metrics to Monitor

The RadixCache exposes metrics to help monitor eviction behavior:

- `evictable_size()` - Tokens available for eviction
- `protected_size()` - Tokens protected by lock references
- `total_size()` - Total cached tokens

### Tuning Parameters

If you need to customize the behavior, you can modify the strategy parameters in `evict_policy.py`:

```python
# Example: Increase protection for very long prefixes
strategy = ValueAwareLRUStrategy(
    prefix_weight=0.002,  # Double the default
    subtree_weight=0.01,
    value_weight=1000.0
)
```

### Debug Mode

To understand eviction behavior, you can enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# RadixCache will log eviction events
cache.evict(num_tokens=100)
```

## Migration Guide

### From Standard LRU

Simply change the eviction policy:

```python
# Before
cache = RadixCache(..., eviction_policy="lru")

# After
cache = RadixCache(..., eviction_policy="value_aware_lru")
```

No other code changes required. The new strategy is a drop-in replacement.

### From Standard LFU

If you're currently using LFU and experiencing cold-start issues:

```python
# Before
cache = RadixCache(..., eviction_policy="lfu")

# After
cache = RadixCache(..., eviction_policy="adaptive_lfu")
```

### Choosing the Right Strategy

**Decision Tree:**

1. Do you have significant prefix sharing?
   - Yes → Try `value_aware_lru` or `value_aware_adaptive_lfu`
   - No → Continue to step 2

2. Do you need frequency-based eviction?
   - Yes → Try `adaptive_lfu` or `value_aware_adaptive_lfu`
   - No → Use `value_aware_lru`

3. Is cold-start a problem?
   - Yes → Use `adaptive_lfu` or `value_aware_adaptive_lfu`
   - No → Use `lfu` or `value_aware_lru`

**Recommended Starting Point:**
For most workloads, we recommend starting with `value_aware_lru` as it provides good benefits with minimal complexity.

## Testing

Comprehensive unit tests are provided in `test/srt/test_value_aware_eviction.py`:

```bash
# Run tests
python -m pytest test/srt/test_value_aware_eviction.py -v

# Run specific test
python -m pytest test/srt/test_value_aware_eviction.py::TestValueAwareLRUStrategy -v
```

## Future Enhancements

Planned improvements for future releases:

1. **Incremental Eviction Queue** - Maintain persistent priority queue to avoid O(N) scans
2. **Async Pre-eviction** - Background thread for proactive eviction
3. **Partitioned Cache** - Different regions with different strategies
4. **ML-based Prediction** - Learn access patterns for predictive eviction
5. **Configurable Parameters** - Runtime tuning via server arguments

## References

- Main implementation: `python/sglang/srt/mem_cache/evict_policy.py`
- Integration point: `python/sglang/srt/mem_cache/radix_cache.py`
- Tests: `test/srt/test_value_aware_eviction.py`

## Questions and Support

For questions or issues with the new eviction strategies:
1. Check existing GitHub issues: https://github.com/sgl-project/sglang/issues
2. Open a new issue with the `kv-cache` label
3. Include your workload characteristics and observed behavior
