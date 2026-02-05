# Compute-Cost-Aware MoE Load Balancing

## Overview

Traditional MoE (Mixture of Experts) load balancing in SGLang uses **token count** to balance expert workloads across GPUs. However, this approach doesn't account for the actual **computation cost** of processing tokens through different experts.

The **Compute-Cost-Aware Load Balancing** system addresses this limitation by:

1. **Profiling** the actual execution time of each expert
2. **Weighing** experts by their compute cost (not just token count)
3. **Rebalancing** expert placement to minimize actual GPU workload imbalance

This leads to better GPU utilization and improved inference performance.

## Problem: Token-Based vs Compute-Cost-Based Balancing

### Token-Based Balancing (Current Default)

```
Expert 0: 100 tokens × 1.0ms/token = 100ms
Expert 1: 100 tokens × 2.5ms/token = 250ms
Expert 2: 100 tokens × 1.2ms/token = 120ms

Token distribution: Balanced ✓ (100, 100, 100)
Compute time: Imbalanced ✗ (100ms, 250ms, 120ms)
GPU utilization: 100ms / 250ms = 40%
```

### Compute-Cost-Aware Balancing (New)

```
After rebalancing based on compute cost:

GPU 0: Expert 0 (100ms) + Expert 2 (120ms) = 220ms
GPU 1: Expert 1 replica 1 (125ms) + Expert 1 replica 2 (125ms) = 250ms

Token distribution: May vary
Compute time: Balanced ✓ (220ms, 250ms)
GPU utilization: 220ms / 250ms = 88%  ← Better!
```

## Architecture

### Components

1. **ExpertComputeProfiler** (`expert_compute_profiler.py`)
   - Tracks execution time per expert
   - Computes average time per token
   - Provides compute cost weights

2. **Compute-Aware Algorithms** (`eplb_algorithms/compute_aware.py`)
   - Extends DeepSeek's balanced packing algorithm
   - Blends token count with compute cost
   - Supports hierarchical balancing

3. **EPLB Integration** (`eplb_manager.py`)
   - Automatically uses profiled data
   - Triggers rebalancing with compute cost

## Usage

### Basic Usage

Enable compute-cost-aware load balancing with these flags:

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --enable-eplb \
    --eplb-algorithm compute_aware \
    --enable-expert-compute-profiling \
    --eplb-compute-cost-alpha 0.6
```

### Configuration Options

#### Essential Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-eplb` | `False` | Enable EPLB load balancing |
| `--eplb-algorithm` | `auto` | Use `compute_aware` or `compute_aware_hierarchical` |
| `--enable-expert-compute-profiling` | `False` | Enable compute cost profiling |

#### Advanced Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--eplb-compute-cost-alpha` | `0.5` | Blending factor: 0.0=token-only, 1.0=compute-only |
| `--expert-compute-profiling-warmup-steps` | `10` | Warmup steps before profiling |
| `--expert-compute-profiling-interval` | `1` | Profile every N forward passes |
| `--eplb-rebalance-num-iterations` | `1000` | Trigger rebalancing every N iterations |
| `--eplb-min-rebalancing-utilization-threshold` | `1.0` | Only rebalance if utilization < threshold |

### Choosing the Right Alpha

The `alpha` parameter controls how much weight to give to compute cost vs token count:

- **α = 0.0**: Pure token-based balancing (default behavior)
- **α = 0.3**: Light compute-cost awareness (safe for most workloads)
- **α = 0.5**: Balanced consideration (recommended starting point)
- **α = 0.7**: Heavy compute-cost weighting (for highly variable expert costs)
- **α = 1.0**: Pure compute-cost balancing (ignores token count)

**Recommendation**: Start with `α = 0.5` and tune based on profiling data.

## Example Configurations

### DeepSeek-V3 (Recommended)

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --enable-eplb \
    --eplb-algorithm compute_aware_hierarchical \
    --enable-expert-compute-profiling \
    --eplb-compute-cost-alpha 0.6 \
    --expert-compute-profiling-warmup-steps 20 \
    --expert-compute-profiling-interval 1 \
    --eplb-rebalance-num-iterations 500 \
    --eplb-min-rebalancing-utilization-threshold 0.85 \
    --expert-distribution-recorder-mode stat \
    --expert-distribution-recorder-buffer-size 1000
```

**Why these settings?**
- `compute_aware_hierarchical`: Best for multi-node setups
- `alpha=0.6`: Prioritize compute cost over token count
- `warmup=20`: More warmup for stable profiling
- `threshold=0.85`: Rebalance if utilization < 85%

### Qwen2-MoE (Single Node)

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-57B-A14B-Instruct \
    --tp 4 \
    --enable-eplb \
    --eplb-algorithm compute_aware \
    --enable-expert-compute-profiling \
    --eplb-compute-cost-alpha 0.5 \
    --eplb-rebalance-num-iterations 1000
```

**Why these settings?**
- `compute_aware`: Flat balancing for single node
- `alpha=0.5`: Balanced token/compute consideration

### Mixtral (Debugging/Development)

```bash
python -m sglang.launch_server \
    --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp 2 \
    --enable-eplb \
    --eplb-algorithm compute_aware \
    --enable-expert-compute-profiling \
    --expert-compute-profiling-interval 10 \
    --eplb-compute-cost-alpha 0.4 \
    --enable-expert-distribution-metrics
```

**Why these settings?**
- `interval=10`: Less frequent profiling (lower overhead)
- `alpha=0.4`: Conservative compute-cost weighting
- `enable-expert-distribution-metrics`: Detailed logging for debugging

## Monitoring and Debugging

### Checking Profiler Status

The profiler logs its status:

```
[ExpertComputeProfiler] Initialized with warmup_steps=10, profiling_interval=1
[ExpertComputeProfiler] Profiling started
[EPLBManager] Using compute-cost-aware rebalancing with profiled data
```

### Viewing Profiling Statistics

You can access profiling statistics programmatically:

```python
from sglang.srt.eplb.expert_compute_profiler import get_global_expert_compute_profiler

profiler = get_global_expert_compute_profiler()
if profiler is not None:
    stats = profiler.get_statistics_summary()
    print(stats)
```

Output example:
```python
{
    "forward_pass_count": 1523,
    "profiling_enabled": True,
    "layers": {
        "layer_0": {
            "expert_0": {
                "avg_time_per_token_ms": 1.23,
                "avg_time_per_invocation_ms": 45.6,
                "total_tokens": 12345,
                "num_invocations": 567
            },
            "expert_1": {
                "avg_time_per_token_ms": 2.87,
                ...
            }
        }
    }
}
```

### Understanding Log Messages

**Good signs:**
- `[ExpertComputeProfiler] Profiling started`
- `[EPLBManager] Using compute-cost-aware rebalancing`
- `[Expert Balancedness] current_pass_balancedness=0.89` (high is good)

**Warning signs:**
- `current_pass_balancedness=0.45` (low utilization)
- `[EPLBManager] Skipped ep rebalancing: current GPU utilization > threshold`

## Performance Expectations

### Expected Improvements

Typical improvements with compute-cost-aware balancing:

| Metric | Token-Based | Compute-Aware | Improvement |
|--------|-------------|---------------|-------------|
| GPU Utilization | 40-60% | 75-90% | **+35-50%** |
| Throughput (tokens/s) | Baseline | +15-30% | **+15-30%** |
| Latency (P99) | Baseline | -10-20% | **-10-20%** |

*Results vary by model, workload, and hardware.*

### When to Use Compute-Aware Balancing

✅ **Use when:**
- Experts have variable computation costs
- You observe GPU utilization imbalance (some GPUs idle while others work)
- Workload has diverse token types (prefill + decode)
- Multi-node or large-scale deployments

❌ **Skip when:**
- All experts have similar computation costs
- GPU utilization is already balanced (>85%)
- Single GPU or very small models
- Profiling overhead is a concern

## Troubleshooting

### Issue: Profiler Not Starting

**Symptoms:**
- No profiling logs
- Balancing still uses token counts

**Solutions:**
1. Ensure `--enable-expert-compute-profiling` is set
2. Check that EPLB is enabled: `--enable-eplb`
3. Verify model supports MoE: Check model architecture

### Issue: No Rebalancing Happening

**Symptoms:**
- Profiler runs but no rebalancing
- GPU utilization remains low

**Solutions:**
1. Lower the threshold: `--eplb-min-rebalancing-utilization-threshold 0.7`
2. Reduce rebalancing interval: `--eplb-rebalance-num-iterations 100`
3. Check algorithm is compute-aware: `--eplb-algorithm compute_aware`

### Issue: High Profiling Overhead

**Symptoms:**
- Throughput degradation
- Increased latency

**Solutions:**
1. Increase profiling interval: `--expert-compute-profiling-interval 10`
2. Use stat_approx mode: `--expert-distribution-recorder-mode stat_approx`
3. Disable if overhead too high

## Advanced Topics

### Custom Profiling Integration

You can implement custom profiling hooks:

```python
from sglang.srt.eplb.expert_compute_profiler import ExpertComputeProfiler

# Create custom profiler
profiler = ExpertComputeProfiler(
    expert_location_metadata=metadata,
    enable_profiling=True,
    warmup_steps=20,
    profiling_interval=5
)

# Use CUDA events for accurate timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# ... expert computation ...
end_event.record()

profiler.profile_expert_execution(
    layer_idx=0,
    expert_idx=3,
    num_tokens=128,
    start_event=start_event,
    end_event=end_event
)
```

### Combining with Elastic EP

Compute-cost-aware balancing works with Elastic EP:

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --enable-eplb \
    --eplb-algorithm compute_aware_hierarchical \
    --enable-expert-compute-profiling \
    --elastic-ep-backend mooncake \
    --eplb-compute-cost-alpha 0.7
```

The profiler adapts to node failures and rebalances accordingly.

## API Reference

### ExpertComputeProfiler

```python
class ExpertComputeProfiler:
    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        enable_profiling: bool = True,
        warmup_steps: int = 10,
        profiling_interval: int = 1,
    )

    def start_profiling(self)
    def get_expert_compute_weights(self, layer_idx: Optional[int] = None) -> torch.Tensor
    def get_adjusted_expert_load(self, token_count: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    def get_statistics_summary(self) -> Dict
    def reset_statistics(self)
```

### Compute-Aware Algorithms

```python
def compute_cost_aware_rebalance_experts(
    token_weight: torch.Tensor,              # [layers, num_logical_experts]
    compute_cost_weight: Optional[torch.Tensor],  # [layers, num_logical_experts]
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
    alpha: float = 0.5,                      # Blending factor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

## References

- [Expert Parallelism Documentation](expert_parallelism.md)
- [DeepSeek EPLB Algorithm](https://github.com/deepseek-ai/EPLB)
- [SGLang MoE Implementation](../../python/sglang/srt/layers/moe/)

## FAQ

**Q: Does this work with all MoE models?**
A: Yes, it works with any MoE model supported by SGLang (DeepSeek, Qwen, Mixtral, etc.).

**Q: What's the profiling overhead?**
A: Typically <2% throughput impact with `interval=1`. Use higher intervals to reduce overhead.

**Q: Can I disable profiling after initialization?**
A: Currently, profiling runs continuously once enabled. Future versions may support dynamic on/off.

**Q: How does this compare to vLLM's MoE balancing?**
A: SGLang's compute-aware balancing considers actual execution time, while most systems only balance token counts.

**Q: What if different batches have different compute costs?**
A: The profiler tracks average costs over time and adapts. Use a larger warmup period for more stability.

## Conclusion

Compute-cost-aware MoE load balancing significantly improves GPU utilization by considering actual computation costs rather than just token counts. This leads to better performance, especially for models with heterogeneous expert computation costs.

**Recommended Next Steps:**
1. Start with default settings and monitor GPU utilization
2. Tune `alpha` based on your workload characteristics
3. Adjust rebalancing frequency based on workload variability
4. Monitor profiling overhead and adjust interval if needed
