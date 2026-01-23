# coding=utf-8
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
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
"""NemotronH model configuration for MLX."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NemotronHConfig:
    """Configuration class for NemotronH model.

    This configuration mirrors the PyTorch NemotronHConfig for compatibility
    with weight loading and model behavior.
    """

    # Model architecture
    vocab_size: int = 256000
    hidden_size: int = 4096
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None
    intermediate_size: int = 16384
    max_position_embeddings: int = 8192

    # Normalization
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True

    # Activation functions
    mlp_hidden_act: str = "silu"
    mamba_hidden_act: str = "silu"

    # Bias configuration
    attention_bias: bool = False
    mlp_bias: bool = False
    use_bias: bool = False
    use_conv_bias: bool = True

    # Mamba2 (SSM) configuration
    mamba_num_heads: int = 128
    mamba_head_dim: int = 64
    ssm_state_size: int = 128
    conv_kernel: int = 4
    n_groups: int = 8
    chunk_size: int = 256
    time_step_limit: Tuple[float, float] = (0.0, float("inf"))
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4

    # MoE (Mixture of Experts) configuration
    n_routed_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    moe_shared_expert_intermediate_size: int = 8192
    routed_scaling_factor: float = 1.0
    n_group: int = 8
    topk_group: int = 4
    norm_topk_prob: bool = True

    # Attention configuration
    attention_dropout: float = 0.0

    # Layer pattern configuration
    # M: Mamba, *: Attention, -: MLP, E: MoE
    # Example: "M*-M-E" means Mamba, Attention, MLP, Mamba, MLP, MoE
    hybrid_override_pattern: str = "M" * 52  # Default to all Mamba layers

    # Training configuration
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = False

    # Generation configuration
    use_cache: bool = True
    num_logits_to_keep: int = 1

    # Output configuration
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True

    # Internal configuration
    _attn_implementation: str = "eager"

    def __post_init__(self):
        """Derive additional configuration values."""
        # Compute head_dim if not provided
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Parse layer types from hybrid_override_pattern
        self._parse_layer_types()

    def _parse_layer_types(self):
        """Parse the hybrid_override_pattern to get layer types."""
        pattern_map = {
            "M": "mamba",
            "*": "attention",
            "-": "mlp",
            "E": "moe",
        }

        self.layers_block_type = []
        for char in self.hybrid_override_pattern:
            if char in pattern_map:
                self.layers_block_type.append(pattern_map[char])
            else:
                raise ValueError(f"Unknown layer pattern character: {char}")

        # Validate layer count matches
        if len(self.layers_block_type) != self.num_hidden_layers:
            # Extend or truncate to match num_hidden_layers
            if len(self.layers_block_type) < self.num_hidden_layers:
                # Repeat the pattern
                while len(self.layers_block_type) < self.num_hidden_layers:
                    self.layers_block_type.append(self.layers_block_type[-1])
            else:
                self.layers_block_type = self.layers_block_type[:self.num_hidden_layers]

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NemotronHConfig":
        """Create a config from a dictionary."""
        # Handle time_step_limit as a tuple
        if "time_step_limit" in config_dict and isinstance(config_dict["time_step_limit"], list):
            config_dict["time_step_limit"] = tuple(config_dict["time_step_limit"])

        # Filter out unknown keys
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """Convert config to a dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "residual_in_fp32": self.residual_in_fp32,
            "mlp_hidden_act": self.mlp_hidden_act,
            "mamba_hidden_act": self.mamba_hidden_act,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
            "use_bias": self.use_bias,
            "use_conv_bias": self.use_conv_bias,
            "mamba_num_heads": self.mamba_num_heads,
            "mamba_head_dim": self.mamba_head_dim,
            "ssm_state_size": self.ssm_state_size,
            "conv_kernel": self.conv_kernel,
            "n_groups": self.n_groups,
            "chunk_size": self.chunk_size,
            "time_step_limit": list(self.time_step_limit),
            "time_step_min": self.time_step_min,
            "time_step_max": self.time_step_max,
            "time_step_floor": self.time_step_floor,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "moe_shared_expert_intermediate_size": self.moe_shared_expert_intermediate_size,
            "routed_scaling_factor": self.routed_scaling_factor,
            "n_group": self.n_group,
            "topk_group": self.topk_group,
            "norm_topk_prob": self.norm_topk_prob,
            "attention_dropout": self.attention_dropout,
            "hybrid_override_pattern": self.hybrid_override_pattern,
            "initializer_range": self.initializer_range,
            "rescale_prenorm_residual": self.rescale_prenorm_residual,
            "use_cache": self.use_cache,
            "num_logits_to_keep": self.num_logits_to_keep,
            "output_attentions": self.output_attentions,
            "output_hidden_states": self.output_hidden_states,
            "use_return_dict": self.use_return_dict,
        }


# Predefined configurations for common model sizes
def nemotron_h_8b_config() -> NemotronHConfig:
    """Configuration for NemotronH 8B model."""
    return NemotronHConfig(
        vocab_size=256000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        mamba_num_heads=64,
        mamba_head_dim=64,
        ssm_state_size=128,
        hybrid_override_pattern="M*-M-E" * 6 + "M*",  # Repeating pattern
    )


def nemotron_h_56b_config() -> NemotronHConfig:
    """Configuration for NemotronH 56B model."""
    return NemotronHConfig(
        vocab_size=256000,
        hidden_size=8192,
        num_hidden_layers=52,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        mamba_num_heads=128,
        mamba_head_dim=64,
        ssm_state_size=128,
        hybrid_override_pattern="M" * 52,  # All Mamba layers for base
    )
