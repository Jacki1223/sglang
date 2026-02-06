"""
Tuning script for the fused Activation+GEMM2 MoE kernel.

This script generates optimal Triton kernel configurations for the fused
Activation+GEMM2 kernel, which fuses activation(gate)*up computation into
GEMM2's A-tile loading prologue.

The script sweeps over BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
num_warps, and num_stages for each batch size, and saves the best configuration
to a JSON file that can be used at runtime.

Usage:
    # Tune for a specific model with tensor parallelism
    python benchmark/kernels/fused_moe_triton/tuning_fused_act_gemm2.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2

    # Tune for specific dimensions directly
    python benchmark/kernels/fused_moe_triton/tuning_fused_act_gemm2.py \\
        --num-experts 8 --hidden-size 4096 --intermediate-size 14336 --topk 2

    # Tune for a single batch size
    python benchmark/kernels/fused_moe_triton/tuning_fused_act_gemm2.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2 --batch-size 32

Notes:
    - Requires Ray for distributed benchmarking across multiple GPUs
    - The generated configs apply to the fused_moe_act_gemm2_kernel
    - The kernel's K dimension is intermediate_size (w2.shape[2]),
      same as the original GEMM2, but the A input is full-width (2*K)
      since it reads both gate and up halves
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Tuple

import ray
import torch
import triton
from common_utils import (
    BenchmarkConfig,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    save_configs,
    sort_config,
)
from ray.experimental.tqdm_ray import tqdm

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_file_name,
    get_default_config,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
    invoke_fused_moe_act_gemm2_kernel,
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_hip

_is_hip = is_hip()


def benchmark_fused_act_gemm2_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    num_iters: int = 100,
) -> float:
    """
    Benchmark the fused Act+GEMM2 kernel with a given config.

    Simulates the full pipeline: GEMM1 output -> fused Act+GEMM2 -> output.
    Only times the fused Act+GEMM2 kernel.
    """
    import triton.language as tl

    device = "cuda"
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

    # Simulate intermediate_cache1 output from GEMM1: [total_tokens, 2*intermediate_size]
    total_tokens_approx = num_tokens * topk
    intermediate_cache1 = torch.randn(
        total_tokens_approx, 2 * intermediate_size, dtype=dtype, device=device
    )

    # w2 shape: [E, hidden_size, intermediate_size]
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )

    # Output buffer
    output = torch.empty(
        total_tokens_approx, hidden_size, dtype=dtype, device=device
    )

    # Generate routing info
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_weights = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk)
    topk_weights = topk_weights.to(dtype=torch.float32)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )

    # Warmup
    for _ in range(3):
        invoke_fused_moe_act_gemm2_kernel(
            intermediate_cache1,
            w2,
            None,  # bias
            output,
            None,  # A_scale
            None,  # B_scale
            None,  # B_zp
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,  # mul_routed_weight
            1,  # top_k (for GEMM2, each sorted token maps to one expert)
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            filter_expert=True,
            activation="silu",
        )
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        invoke_fused_moe_act_gemm2_kernel(
            intermediate_cache1,
            w2,
            None,
            output,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            filter_expert=True,
            activation="silu",
        )
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg = sum(times) / len(times) * 1000  # convert ms to us
    return avg


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed
        self.device_id = int(ray.get_gpu_ids()[0])

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        best_config = None
        best_time = float("inf")
        with torch.cuda.device(self.device_id) if _is_hip else nullcontext():
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_fused_act_gemm2_config(
                        config,
                        num_tokens,
                        num_experts,
                        intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        num_iters=10,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError):
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config

        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def main(args: argparse.Namespace):
    print(args)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    if args.model:
        model_config = get_model_config(args.model, args.tp_size, args.ep_size)
        E = model_config["num_experts"]
        topk = model_config["topk"]
        hidden_size = model_config["hidden_size"]
        shard_intermediate_size = model_config["shard_intermediate_size"]
        dtype = model_config["dtype"]
    else:
        E = args.num_experts
        topk = args.topk
        hidden_size = args.hidden_size
        shard_intermediate_size = 2 * args.intermediate_size
        dtype = torch.bfloat16

    intermediate_size = shard_intermediate_size // 2

    if args.batch_size is None:
        batch_sizes = get_default_batch_sizes()
    else:
        batch_sizes = [args.batch_size]

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    search_space = get_configs_compute_bound()

    # Filter configs where BLOCK_SIZE_K > intermediate_size
    search_space = [c for c in search_space if c["BLOCK_SIZE_K"] <= intermediate_size]

    print(f"\nTuning fused Act+GEMM2 kernel")
    print(f"  E={E}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, topk={topk}")
    print(f"  Search space: {len(search_space)} configs")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  GPUs: {num_gpus}")
    print()

    best_configs = {}
    for i, batch_size in enumerate(batch_sizes):
        worker = workers[i % num_gpus]
        result = worker.tune.remote(
            batch_size,
            E,
            intermediate_size,
            hidden_size,
            topk,
            dtype,
            search_space,
        )
        best_configs[batch_size] = result

    # Collect results
    final_configs = {}
    for batch_size, result_ref in best_configs.items():
        config = ray.get(result_ref)
        final_configs[batch_size] = sort_config(config)
        print(f"  batch_size={batch_size}: {sort_config(config)}")

    # Save configs
    from sglang.srt.utils import get_device_name

    device_name = get_device_name().replace(" ", "_")
    output_file = (
        f"fused_act_gemm2_E={E},K={intermediate_size},N={hidden_size},"
        f"device_name={device_name}.json"
    )
    if args.output:
        output_file = args.output

    save_configs(final_configs, output_file)
    print(f"\nSaved configs to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune the fused Activation+GEMM2 MoE kernel"
    )
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model name")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallelism size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--intermediate-size", type=int, default=14336, help="Intermediate size")
    parser.add_argument("--topk", type=int, default=2, help="Top-k experts")
    parser.add_argument("--batch-size", type=int, default=None, help="Single batch size to tune")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")

    args = parser.parse_args()
    main(args)
