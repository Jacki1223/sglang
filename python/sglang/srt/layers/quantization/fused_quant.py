"""Fused quantization methods that combine linear projection with activation.

This module extends SGLang's quantization framework to support kernel fusion,
combining Linear+Activation operations while maintaining compatibility with
all existing quantization methods (FP8, INT8, AWQ, GPTQ, etc.).

Architecture:
    FusedLinearMethod wraps existing quant methods and adds activation fusion.
    This allows gradual adoption without breaking existing code.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.base_config import LinearMethodBase


class FusedActivationLinearMethod(LinearMethodBase):
    """Wrapper that adds activation fusion to any LinearMethod.

    This wraps an existing quantization method (UnquantizedLinearMethod,
    Fp8LinearMethod, etc.) and fuses the activation function.

    Architecture:
        Original: quant_method.apply(layer, x) → activation(output)
        Fused: FusedActivationLinearMethod.apply(layer, x) → fused output

    Args:
        base_method: The underlying quantization method to wrap
        activation: Activation function name ("silu", "gelu", etc.)

    Example:
        >>> # Wrap FP8 quantization with SiLU fusion
        >>> from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
        >>> base_method = Fp8LinearMethod(quant_config)
        >>> fused_method = FusedActivationLinearMethod(base_method, "silu")
        >>>
        >>> # Create layer
        >>> layer.quant_method = fused_method
        >>> fused_method.create_weights(layer, ...)
        >>>
        >>> # Forward pass with fusion
        >>> output = fused_method.apply(layer, x)  # Linear + SiLU fused
    """

    def __init__(
        self,
        base_method: LinearMethodBase,
        activation: str = "silu",
    ):
        super().__init__()
        self.base_method = base_method
        self.activation = activation.lower()

        # Validate activation
        valid_activations = ["silu", "gelu", "gelu_tanh"]
        if self.activation not in valid_activations:
            raise ValueError(
                f"Unsupported activation: {self.activation}. "
                f"Supported: {valid_activations}"
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Delegate weight creation to base method."""
        self.base_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    @torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply linear projection + activation with fusion.

        torch.compile will automatically fuse the linear operation
        and activation into optimized kernels.

        Args:
            layer: The linear layer
            x: Input tensor
            bias: Optional bias

        Returns:
            Output with activation applied
        """
        # Linear projection using base quantization method
        output = self.base_method.apply(layer, x, bias)

        # Apply activation (will be fused by torch.compile)
        if self.activation == "silu":
            output = F.silu(output)
        elif self.activation == "gelu":
            output = F.gelu(output, approximate="none")
        elif self.activation == "gelu_tanh":
            output = F.gelu(output, approximate="tanh")

        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Delegate to base method."""
        self.base_method.process_weights_after_loading(layer)


class FusedSiluMulLinearMethod(LinearMethodBase):
    """Fused Linear → SiluAndMul for MLP layers.

    This is specifically designed for the gate_up_proj pattern in SGLang:
        gate_up_proj: Linear that outputs [gate, up] concatenated
        SiluAndMul: silu(gate) * up

    Instead of:
        gate_up = linear(x)       # Write to memory
        output = silu_and_mul(gate_up)  # Read from memory

    We fuse to:
        output = fused_linear_silu_mul(x)  # Direct computation

    Args:
        base_method: Underlying quantization method

    Example:
        >>> # In LlamaMLP
        >>> from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        >>> base_method = UnquantizedLinearMethod()
        >>> fused_method = FusedSiluMulLinearMethod(base_method)
        >>> layer.quant_method = fused_method
    """

    def __init__(self, base_method: LinearMethodBase):
        super().__init__()
        self.base_method = base_method

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Delegate weight creation to base method."""
        self.base_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    @torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply fused Linear + SiluAndMul.

        torch.compile will fuse:
        1. Linear projection → [gate, up]
        2. Split into gate and up
        3. SiLU(gate) * up

        Into optimized kernels.

        Args:
            layer: The MergedColumnParallelLinear layer
            x: Input tensor
            bias: Optional bias

        Returns:
            SiLU(gate) * up
        """
        # Linear projection: outputs [gate, up] concatenated
        gate_up = self.base_method.apply(layer, x, bias)

        # Split into gate and up
        # Assumes output is [batch, intermediate_size * 2]
        d = gate_up.shape[-1] // 2
        gate = gate_up[..., :d]
        up = gate_up[..., d:]

        # SiLU(gate) * up (will be fused by torch.compile)
        output = F.silu(gate) * up

        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Delegate to base method."""
        self.base_method.process_weights_after_loading(layer)


def wrap_with_fused_activation(
    quant_method: Optional[LinearMethodBase],
    activation: str = "silu",
) -> Optional[LinearMethodBase]:
    """Helper to wrap a quantization method with activation fusion.

    Args:
        quant_method: Original quantization method (or None)
        activation: Activation function to fuse

    Returns:
        Wrapped method with fusion, or None if quant_method is None

    Example:
        >>> from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
        >>> quant_method = Fp8LinearMethod(quant_config)
        >>> fused_method = wrap_with_fused_activation(quant_method, "silu")
    """
    if quant_method is None:
        return None

    return FusedActivationLinearMethod(quant_method, activation)


def wrap_with_silu_mul_fusion(
    quant_method: Optional[LinearMethodBase],
) -> Optional[LinearMethodBase]:
    """Helper to wrap a quantization method with SiluAndMul fusion.

    Specifically for gate_up_proj layers.

    Args:
        quant_method: Original quantization method (or None)

    Returns:
        Wrapped method with SiluAndMul fusion

    Example:
        >>> # In LlamaMLP __init__
        >>> gate_up_quant_method = wrap_with_silu_mul_fusion(quant_config.get_linear_method())
        >>> self.gate_up_proj = MergedColumnParallelLinear(
        ...     ...,
        ...     quant_config=QuickQuantConfig(gate_up_quant_method)
        ... )
    """
    if quant_method is None:
        return None

    return FusedSiluMulLinearMethod(quant_method)


# ====================================================================================
# Configuration helper for enabling fusion globally
# ====================================================================================

class FusionConfig:
    """Global configuration for kernel fusion.

    This controls whether to enable fusion across the model.

    Usage:
        >>> # Enable fusion globally
        >>> FusionConfig.enable_linear_activation_fusion = True
        >>>
        >>> # Disable for debugging
        >>> FusionConfig.enable_linear_activation_fusion = False
    """

    enable_linear_activation_fusion: bool = False
    enable_silu_mul_fusion: bool = False
    fusion_mode: str = "max-autotune"  # torch.compile mode

    @classmethod
    def configure_torch_compile(cls):
        """Configure torch.compile for optimal fusion."""
        if not (cls.enable_linear_activation_fusion or cls.enable_silu_mul_fusion):
            return

        import logging

        logger = logging.getLogger(__name__)

        try:
            # Enable inductor optimizations
            torch._inductor.config.max_autotune = True
            torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True

            # Mixed precision optimizations
            torch._inductor.config.force_fuse_int_mm_with_mul = True
            torch._inductor.config.use_mixed_mm = True

            logger.info(
                f"Kernel fusion configured: "
                f"linear_activation={cls.enable_linear_activation_fusion}, "
                f"silu_mul={cls.enable_silu_mul_fusion}"
            )
        except Exception as e:
            logger.warning(f"Failed to configure kernel fusion: {e}")
