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
# ==============================================================================

"""Configuration for MiDashengLM model."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DashengAudioConfig:
    """Configuration for Dasheng audio encoder."""
    
    # Audio processing parameters
    win_length: int = 512
    n_fft: int = 512
    hop_length: int = 160
    f_min: float = 0.0
    f_max: float = 8000.0
    n_mels: int = 64
    sample_rate: int = 16000
    center: bool = True
    
    # Model parameters
    target_length: int = 1024
    embed_dim: int = 768
    input_channels: int = 1
    patch_size: tuple = (16, 16)
    patch_stride: tuple = (16, 16)
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    init_values: Optional[float] = None
    depth: int = 12


@dataclass
class MiDashengLMConfig:
    """Configuration for MiDashengLM model."""
    
    # Text model configuration
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_dropout: float = 0.0
    
    # Audio-specific configuration
    audio_encoder_config: DashengAudioConfig = None
    subsample_factor: int = 5
    
    # Model architecture
    architectures: list = None
    model_type: str = "midashenglm"
    
    def __post_init__(self):
        if self.audio_encoder_config is None:
            self.audio_encoder_config = DashengAudioConfig()
        if self.architectures is None:
            self.architectures = ["MiDashengLMModel"]
