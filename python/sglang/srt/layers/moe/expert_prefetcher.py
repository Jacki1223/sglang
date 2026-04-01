"""
Async Expert Prefetcher for MoE Models.

Overlaps expert weight loading (CPU → GPU) with attention computation by
leveraging cross-layer routing correlation for prediction.

Architecture Overview:
    Layer N:   attn_N → gate_N → topk_ids_N → experts_N → [record topk_ids_N]
    Layer N+1: [predict from topk_ids_N → start async prefetch]
               → attn_N+1 (overlapped with prefetch on alt_stream)
               → gate_N+1 → topk_ids_N+1
               → [wait prefetch, sync-load mispredictions]
               → experts_N+1 (weights already on GPU)

Key Design Decisions:
    - GPU buffers are full-sized [num_experts, ...] tensors so expert_id indexing
      works unchanged in fused MoE kernels (uninitialized slots are never accessed).
    - Prediction uses previous layer's routing (cross-layer correlation ~40-70%).
    - Mispredictions are handled by synchronous fallback loading — correctness
      is never sacrificed.
    - Uses a dedicated CUDA stream (alt_stream) for async CPU→GPU transfers.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class _LayerExpertState:
    """Tracks per-layer expert weight state for prefetching."""

    layer_id: int
    num_experts: int
    # CPU pinned weight copies: param_name → [num_experts, ...]
    cpu_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    # GPU buffers: param_name → [num_experts, ...] (full-sized for direct indexing)
    gpu_buffers: Dict[str, torch.Tensor] = field(default_factory=dict)
    # Which expert IDs are currently valid on GPU
    loaded_experts: Set[int] = field(default_factory=set)
    # Last forward's routing decision (for cross-layer prediction)
    last_topk_ids: Optional[torch.Tensor] = None
    # Async prefetch synchronization
    prefetch_event: Optional[torch.cuda.Event] = None
    # Stats
    total_predictions: int = 0
    correct_predictions: int = 0


class ExpertPrefetcher:
    """
    Manages async expert weight prefetching overlapped with attention.

    Usage in decoder layer forward:
        1. prefetcher.start_prefetch(layer_id)    # before attention
        2. ... attention computation ...           # overlapped with prefetch
        3. ... gate + topk → topk_ids ...          # after attention
        4. prefetcher.wait_and_ensure(layer_id, topk_ids)  # ensure experts ready
        5. ... expert computation ...              # weights on GPU
        6. prefetcher.record_routing(layer_id, topk_ids)   # for next layer
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_prefetch_experts: int = 64,
    ):
        """
        Args:
            device: CUDA device for GPU buffers.
            max_prefetch_experts: Max experts to prefetch per layer (caps prediction set).
        """
        self.device = device or torch.device("cuda")
        self.alt_stream = torch.cuda.Stream(device=self.device)
        self.max_prefetch_experts = max_prefetch_experts
        self.layers: Dict[int, _LayerExpertState] = {}
        # Running frequency for cold-start / fallback prediction
        self.expert_frequency: Dict[int, torch.Tensor] = {}

    def register_experts(
        self,
        layer_id: int,
        expert_module: nn.Module,
        weight_names: List[str],
        num_experts: int,
    ):
        """
        Register a layer's expert weights. Moves them to CPU pinned memory
        and allocates GPU buffers for async loading.

        Args:
            layer_id: The transformer layer index.
            expert_module: The FusedMoE module containing expert weights.
            weight_names: Parameter names to offload (e.g. ["w13_weight", "w2_weight"]).
            num_experts: Number of local experts.
        """
        state = _LayerExpertState(layer_id=layer_id, num_experts=num_experts)

        for name in weight_names:
            param = getattr(expert_module, name, None)
            if param is None:
                logger.warning(
                    f"Layer {layer_id}: param '{name}' not found, skipping"
                )
                continue
            if param.data.device.type == "meta":
                logger.warning(
                    f"Layer {layer_id}: param '{name}' already on meta, skipping"
                )
                continue

            # Create CPU pinned copy for fast async transfer
            cpu_data = torch.empty(
                param.data.shape,
                dtype=param.data.dtype,
                device="cpu",
                pin_memory=True,
            )
            cpu_data.copy_(param.data)
            state.cpu_weights[name] = cpu_data

            # GPU buffer: full-sized so expert_id indexing works unchanged
            gpu_buf = torch.empty(
                param.data.shape,
                dtype=param.data.dtype,
                device=self.device,
            )
            state.gpu_buffers[name] = gpu_buf

            # Free GPU memory — param becomes a meta tensor
            param.data = torch.empty(
                param.data.shape, dtype=param.data.dtype, device="meta"
            )

        self.layers[layer_id] = state
        self.expert_frequency[layer_id] = torch.zeros(num_experts)

        total_bytes = sum(
            t.numel() * t.element_size() for t in state.cpu_weights.values()
        )
        logger.info(
            f"ExpertPrefetcher: layer {layer_id}, {num_experts} experts, "
            f"params={weight_names}, cpu_pinned={total_bytes / 1e9:.2f}GB"
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_experts(self, layer_id: int) -> Set[int]:
        """
        Predict which experts this layer will activate.

        Strategy priority:
        1. Cross-layer: use previous layer's actual topk_ids (strongest signal).
        2. Historical frequency: accumulated from previous forwards.
        3. Cold start: first N experts.
        """
        state = self.layers[layer_id]

        # Strategy 1: Cross-layer correlation
        prev_layer = layer_id - 1
        if prev_layer in self.layers:
            prev_state = self.layers[prev_layer]
            if prev_state.last_topk_ids is not None:
                predicted = set(prev_state.last_topk_ids.reshape(-1).unique().cpu().tolist())
                # Cap to budget
                if len(predicted) > self.max_prefetch_experts:
                    predicted = set(list(predicted)[: self.max_prefetch_experts])
                return predicted

        # Strategy 2: Historical frequency
        freq = self.expert_frequency.get(layer_id)
        if freq is not None and freq.sum() > 0:
            n = min(self.max_prefetch_experts, state.num_experts)
            _, top_ids = freq.topk(n)
            return set(top_ids.tolist())

        # Strategy 3: Cold start — load a reasonable initial set
        n = min(self.max_prefetch_experts, state.num_experts)
        return set(range(n))

    # ------------------------------------------------------------------
    # Async Prefetch
    # ------------------------------------------------------------------

    def start_prefetch(self, layer_id: int):
        """
        Start async prefetch of predicted experts on alt_stream.
        Call BEFORE attention computation begins.
        """
        if layer_id not in self.layers:
            return

        state = self.layers[layer_id]
        predicted = self._predict_experts(layer_id)

        # Only transfer experts not already loaded
        to_load = predicted - state.loaded_experts
        if not to_load:
            state.prefetch_event = None
            return

        current_stream = torch.cuda.current_stream(self.device)
        self.alt_stream.wait_stream(current_stream)

        with torch.cuda.stream(self.alt_stream):
            for name, cpu_w in state.cpu_weights.items():
                gpu_buf = state.gpu_buffers[name]
                for eid in to_load:
                    if eid < state.num_experts:
                        gpu_buf[eid].copy_(cpu_w[eid], non_blocking=True)

            event = torch.cuda.Event()
            event.record()
            state.prefetch_event = event

        state.loaded_experts |= to_load

    def wait_and_ensure(
        self, layer_id: int, topk_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Wait for async prefetch to complete, then synchronously load any
        mispredicted experts. Returns GPU weight buffers.

        Call AFTER gate/topk produces actual expert IDs, BEFORE expert computation.

        Args:
            layer_id: Layer index.
            topk_ids: Actual expert IDs from the gate [batch, top_k].

        Returns:
            Dict of param_name → GPU tensor with all needed experts loaded.
        """
        if layer_id not in self.layers:
            return {}

        state = self.layers[layer_id]

        # Wait for async prefetch
        if state.prefetch_event is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_event(state.prefetch_event)
            state.prefetch_event = None

        # Determine actually needed experts
        needed = set(topk_ids.reshape(-1).unique().cpu().tolist())
        missing = needed - state.loaded_experts

        # Update prediction stats
        state.total_predictions += len(needed)
        state.correct_predictions += len(needed) - len(missing)

        if missing:
            # Synchronous fallback for mispredicted experts
            logger.debug(
                f"Layer {layer_id}: {len(missing)}/{len(needed)} experts "
                f"missed prediction, sync loading: {sorted(missing)}"
            )
            for name, cpu_w in state.cpu_weights.items():
                gpu_buf = state.gpu_buffers[name]
                for eid in missing:
                    if eid < state.num_experts:
                        gpu_buf[eid].copy_(cpu_w[eid])
            state.loaded_experts |= missing

        return state.gpu_buffers

    # ------------------------------------------------------------------
    # Routing Recording
    # ------------------------------------------------------------------

    def record_routing(self, layer_id: int, topk_ids: torch.Tensor):
        """
        Record this layer's routing decision for cross-layer prediction.
        Call AFTER expert computation.
        """
        if layer_id not in self.layers:
            return
        state = self.layers[layer_id]
        state.last_topk_ids = topk_ids.detach()

        # Update frequency statistics
        freq = self.expert_frequency.get(layer_id)
        if freq is not None:
            for eid in topk_ids.reshape(-1).unique().cpu().tolist():
                if eid < len(freq):
                    freq[eid] += 1

    # ------------------------------------------------------------------
    # Weight Swap for Expert Computation
    # ------------------------------------------------------------------

    def swap_weights_in(self, layer_id: int, expert_module: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Swap expert module's parameter .data to point to GPU buffers.
        Returns the original (meta) data for later restoration.

        After this call, expert_module.w13_weight.data is the GPU buffer,
        so the fused MoE kernel can access it normally.
        """
        if layer_id not in self.layers:
            return {}

        state = self.layers[layer_id]
        saved = {}
        for name, gpu_buf in state.gpu_buffers.items():
            param = getattr(expert_module, name, None)
            if param is not None:
                saved[name] = param.data
                param.data = gpu_buf
        return saved

    def swap_weights_out(
        self, layer_id: int, expert_module: nn.Module, saved: Dict[str, torch.Tensor]
    ):
        """
        Restore expert module's parameter .data to the original (meta) tensors
        and clear the loaded expert set for this layer.
        """
        for name, original_data in saved.items():
            param = getattr(expert_module, name, None)
            if param is not None:
                param.data = original_data

        if layer_id in self.layers:
            self.layers[layer_id].loaded_experts.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_prediction_stats(self) -> Dict[int, float]:
        """Return per-layer prediction hit rate."""
        stats = {}
        for layer_id, state in self.layers.items():
            if state.total_predictions > 0:
                stats[layer_id] = (
                    state.correct_predictions / state.total_predictions
                )
        return stats

    def log_prediction_stats(self):
        """Log prediction accuracy for all layers."""
        stats = self.get_prediction_stats()
        if stats:
            avg = sum(stats.values()) / len(stats)
            logger.info(
                f"ExpertPrefetcher prediction accuracy: "
                f"avg={avg:.1%}, per_layer={{{', '.join(f'{k}:{v:.1%}' for k, v in sorted(stats.items()))}}}"
            )
