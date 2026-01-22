# coding=utf-8
# Copyright 2024 HuggingFace Inc. team.
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
"""MLX NemotronH model implementation.

This is a port of the PyTorch NemotronH model to MLX, optimized for Apple Silicon.
The model is a hybrid architecture combining Mamba2 (SSM), Attention, MLP, and MoE layers.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from nemo_mlx.models.configuration_nemotron_h import NemotronHConfig
from nemo_mlx.utils.ssm_utils import (
    apply_mask_to_padding_states,
    pad_tensor_by_size,
    repeat_kv,
    reshape_into_chunks,
    segment_sum,
    softplus,
)

# Activation functions mapping
ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,  # swish is an alias for silu
    "tanh": mx.tanh,
    "sigmoid": mx.sigmoid,
}


@dataclass
class NemotronHOutput:
    """Output class for NemotronH model."""

    last_hidden_state: Optional[mx.array] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class NemotronHCausalLMOutput:
    """Output class for NemotronH causal language model."""

    loss: Optional[mx.array] = None
    logits: Optional[mx.array] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


class NemotronHRMSNorm(nn.Module):
    """RMSNorm implementation for NemotronH.

    This is equivalent to T5LayerNorm and LlamaRMSNorm.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype

        # Convert to float32 for numerical stability
        hidden_states = hidden_states.astype(mx.float32)

        # Compute RMS
        variance = mx.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)

        # Apply weight and convert back to original dtype
        return (self.weight.astype(mx.float32) * hidden_states).astype(input_dtype)


class MambaRMSNormGated(nn.Module):
    """Gated RMSNorm for Mamba2 layers.

    This applies RMSNorm with an optional gating mechanism.
    """

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps
        self.group_size = group_size

    def __call__(self, hidden_states: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        """Forward pass with optional gating.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            gate: Optional gate tensor [batch, seq_len, hidden_size]

        Returns:
            Normalized (and optionally gated) tensor
        """
        input_dtype = hidden_states.dtype

        # Convert to float32 for numerical stability
        hidden_states = hidden_states.astype(mx.float32)

        # Compute RMS normalization
        # Group normalization: compute RMS over groups
        if self.group_size > 0:
            # Reshape for group normalization
            shape = hidden_states.shape
            hidden_size = shape[-1]
            num_groups = hidden_size // self.group_size

            # Reshape to [batch, seq_len, num_groups, group_size]
            hidden_states_grouped = hidden_states.reshape(*shape[:-1], num_groups, self.group_size)
            variance = mx.mean(hidden_states_grouped ** 2, axis=-1, keepdims=True)
            hidden_states_grouped = hidden_states_grouped * mx.rsqrt(variance + self.variance_epsilon)
            hidden_states = hidden_states_grouped.reshape(*shape)
        else:
            variance = mx.mean(hidden_states ** 2, axis=-1, keepdims=True)
            hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)

        # Apply gate if provided (element-wise multiplication before weight)
        if gate is not None:
            gate = gate.astype(mx.float32)
            # Apply SiLU activation to gate
            gate = nn.silu(gate)
            hidden_states = hidden_states * gate

        # Apply weight and convert back to original dtype
        return (self.weight.astype(mx.float32) * hidden_states).astype(input_dtype)


class NemotronHMamba2Mixer(nn.Module):
    """Mamba2 SSM mixer for NemotronH.

    Computes Δ, A, B, C, and D state space parameters and the contextualized states.
    A, D are input independent, while Δ, B, C are input-dependent (selective).
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()

        self.num_heads = config.mamba_num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN.get(config.mamba_hidden_act, nn.silu)

        self.layer_norm_epsilon = config.layer_norm_epsilon

        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # Conv1d for Mamba - implemented as grouped convolution
        # In MLX, we'll implement this manually using the weight matrix
        self.conv1d_weight = mx.zeros((self.conv_dim, 1, config.conv_kernel))
        if config.use_conv_bias:
            self.conv1d_bias = mx.zeros((self.conv_dim,))
        else:
            self.conv1d_bias = None

        # Projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)

        # Time step projection (discretization)
        self.dt_bias = mx.ones((self.num_heads,))

        # S4D real initialization - A values
        A = mx.arange(1, self.num_heads + 1, dtype=mx.float32)
        self.A_log = mx.log(A)

        # Gated RMSNorm
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=self.layer_norm_epsilon,
            group_size=self.intermediate_size // self.n_groups
        )

        # D skip connection
        self.D = mx.ones((self.num_heads,))

        # Output projection
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

    def _apply_conv1d(self, x: mx.array) -> mx.array:
        """Apply 1D convolution manually for grouped convolution.

        Args:
            x: Input tensor [batch, seq_len, conv_dim]

        Returns:
            Convolved tensor [batch, seq_len, conv_dim]
        """
        batch_size, seq_len, conv_dim = x.shape

        # Transpose to [batch, conv_dim, seq_len] for conv
        x = x.transpose(0, 2, 1)

        # Pad for causal convolution
        padding = self.conv_kernel_size - 1
        x = mx.pad(x, [(0, 0), (0, 0), (padding, 0)])

        # Apply depthwise conv (grouped with groups=conv_dim)
        # Weight shape: [conv_dim, 1, kernel_size]
        # For each channel, convolve with its own kernel
        outputs = []
        for i in range(conv_dim):
            # Extract single channel [batch, 1, seq_len + padding]
            x_channel = x[:, i:i+1, :]
            # Kernel for this channel [1, 1, kernel_size]
            kernel = self.conv1d_weight[i:i+1, :, :]

            # Manual convolution using sliding window
            channel_output = mx.zeros((batch_size, 1, seq_len))
            for k in range(self.conv_kernel_size):
                channel_output = channel_output + x_channel[:, :, k:k+seq_len] * kernel[0, 0, k]

            outputs.append(channel_output)

        # Concatenate all channels [batch, conv_dim, seq_len]
        output = mx.concatenate(outputs, axis=1)

        # Add bias if present
        if self.conv1d_bias is not None:
            output = output + self.conv1d_bias[:, None]

        # Transpose back to [batch, seq_len, conv_dim]
        return output.transpose(0, 2, 1)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass using the naive torch implementation (no CUDA kernels).

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)

        # Calculate d_mlp size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        # Split projection
        splits = [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads]
        split_indices = []
        idx = 0
        for s in splits[:-1]:
            idx += s
            split_indices.append(idx)

        parts = mx.split(projected_states, split_indices, axis=-1)
        _, _, gate, hidden_states_B_C, dt = parts

        # 2. Convolution sequence transformation
        hidden_states_B_C = self._apply_conv1d(hidden_states_B_C)
        hidden_states_B_C = self.act(hidden_states_B_C)

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)

        # Split into hidden_states, B, C
        groups_time_state_size = self.n_groups * self.ssm_state_size
        split_indices = [self.intermediate_size, self.intermediate_size + groups_time_state_size]
        parts = mx.split(hidden_states_B_C, split_indices, axis=-1)
        hidden_states_ssm, B, C = parts

        # 3. SSM transformation (naive SSD implementation)
        A = -mx.exp(self.A_log.astype(mx.float32))  # [num_heads]

        # Apply softplus to dt and clamp
        dt = softplus(dt + self.dt_bias)
        dt = mx.clip(dt, self.time_step_limit[0], self.time_step_limit[1])

        # Reshape tensors
        hidden_states_ssm = hidden_states_ssm.reshape(batch_size, seq_len, -1, self.head_dim).astype(mx.float32)
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).astype(mx.float32)
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).astype(mx.float32)

        # Repeat B and C for each head group
        num_head_groups = self.num_heads // self.n_groups
        B = mx.repeat(B, num_head_groups, axis=2)
        C = mx.repeat(C, num_head_groups, axis=2)

        # Compute padding size for chunking
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        # D residual connection
        # D shape: [num_heads] -> [1, 1, num_heads, 1] for broadcasting with [batch, seq, num_heads, head_dim]
        D_expanded = self.D.reshape(1, 1, self.num_heads, 1)
        D_residual = D_expanded * pad_tensor_by_size(hidden_states_ssm, pad_size)

        # Discretize x and A
        hidden_states_ssm = hidden_states_ssm * mx.expand_dims(dt, axis=-1)
        A_dt = A.astype(hidden_states_ssm.dtype) * dt

        # Rearrange into blocks/chunks
        hidden_states_ssm = reshape_into_chunks(hidden_states_ssm, pad_size, self.chunk_size)
        A_dt = reshape_into_chunks(A_dt, pad_size, self.chunk_size)
        B = reshape_into_chunks(B, pad_size, self.chunk_size)
        C = reshape_into_chunks(C, pad_size, self.chunk_size)

        # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
        A_dt = A_dt.transpose(0, 3, 1, 2)
        A_cumsum = mx.cumsum(A_dt, axis=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = mx.exp(segment_sum(A_dt))

        # Contraction of C and B to get G (attention-weights like)
        # G_intermediate: (b, c, l, s, h, n)
        G_intermediate = mx.expand_dims(C, axis=3) * mx.expand_dims(B, axis=2)
        G = G_intermediate.sum(axis=-1)  # (b, c, l, s, h)

        # Compute M, equivalent to applying attention mask to weights
        L_permuted = L.transpose(0, 2, 3, 4, 1)  # [bsz, chunks, l, s, num_heads]
        M_intermediate = mx.expand_dims(G, axis=-1) * mx.expand_dims(L_permuted, axis=-1)
        M = M_intermediate.sum(axis=-1)

        # Compute Y_diag (apply to values)
        Y_diag = (mx.expand_dims(M, axis=-1) * mx.expand_dims(hidden_states_ssm, axis=2)).sum(axis=3)

        # 2. Compute the state for each intra-chunk
        decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        decay_states_permuted = decay_states.transpose(0, 2, 3, 1)  # [bsz, chunks, chunk_size, num_heads]
        B_decay = B * mx.expand_dims(decay_states_permuted, axis=-1)
        states = (mx.expand_dims(B_decay, axis=-2) * mx.expand_dims(hidden_states_ssm, axis=-1)).sum(axis=2)

        # 3. Compute the inter-chunk SSM recurrence
        previous_states = mx.zeros_like(states[:, :1])
        states = mx.concatenate([previous_states, states], axis=1)
        A_cumsum_padded = mx.pad(A_cumsum[:, :, :, -1], [(0, 0), (0, 0), (1, 0)])
        decay_chunk = mx.exp(segment_sum(A_cumsum_padded))
        # segment_sum output: [bsz, num_heads, chunks+1, chunks+1]
        # transpose to: [bsz, chunks+1, chunks+1, num_heads]
        decay_chunk = decay_chunk.transpose(0, 2, 3, 1)
        new_states = (mx.expand_dims(mx.expand_dims(decay_chunk, axis=-1), axis=-1) * mx.expand_dims(states, axis=2)).sum(axis=1)
        states = new_states[:, :-1]
        # ssm_state = new_states[:, -1]  # Final state for caching (unused currently)

        # 4. Compute state -> output conversion per chunk
        state_decay_out = mx.exp(A_cumsum)
        state_decay_out_permuted = state_decay_out.transpose(0, 2, 3, 1)  # [bsz, chunks, chunk_size, num_heads]
        C_times_states = mx.expand_dims(C, axis=-2) * mx.expand_dims(states, axis=2)
        Y_off = C_times_states.sum(axis=-1) * mx.expand_dims(state_decay_out_permuted, axis=-1)

        # Add output of intra-chunk and inter-chunk terms
        y = Y_diag + Y_off

        # Reshape output
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Add D residual
        y = y + D_residual

        # Cut off padding
        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        y = y.reshape(batch_size, seq_len, -1)

        # Apply gated normalization
        scan_output = self.norm(y, gate)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.astype(dtype))

        return contextualized_states


class NemotronHAttention(nn.Module):
    """Multi-headed attention for NemotronH."""

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        if config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.num_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias)

        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional causal mask

        Returns:
            Output tensor and optional attention weights
        """
        bsz, q_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Repeat K, V for grouped query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = (query_states @ key_states.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask
        if self.is_causal and q_len > 1:
            causal_mask = mx.triu(mx.full((q_len, q_len), float("-inf")), k=1)
            attn_weights = attn_weights + causal_mask

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and apply to values
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ value_states

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class NemotronHMLP(nn.Module):
    """MLP layer for NemotronH."""

    def __init__(
        self,
        config: NemotronHConfig,
        intermediate_size: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN.get(config.mlp_hidden_act, nn.silu)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHTopkRouter(nn.Module):
    """Top-k router for MoE layer."""

    def __init__(self, config: NemotronHConfig):
        super().__init__()

        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def _get_topk_indices(self, scores: mx.array) -> mx.array:
        """Get top-k expert indices with group-based selection."""
        # Add bias for score correction
        scores_for_choice = scores.reshape(-1, self.n_routed_experts) + self.e_score_correction_bias

        # Group-based selection
        num_experts_per_group = self.n_routed_experts // self.n_group
        scores_grouped = scores_for_choice.reshape(-1, self.n_group, num_experts_per_group)

        # Get top 2 scores per group and sum them
        top2_scores = mx.sort(scores_grouped, axis=-1)[:, :, -2:]
        group_scores = top2_scores.sum(axis=-1)

        # Select top-k groups
        topk_group_indices = mx.argsort(group_scores, axis=-1)[:, -self.topk_group:]

        # Create group mask
        batch_size = scores_for_choice.shape[0]
        group_mask = mx.zeros((batch_size, self.n_group))

        # This is a simplified version - in practice you'd use scatter
        for i in range(self.topk_group):
            indices = topk_group_indices[:, i]
            # One-hot encoding for each index
            one_hot = mx.zeros((batch_size, self.n_group))
            for b in range(batch_size):
                one_hot = one_hot.at[b, int(indices[b])].add(1.0)
            group_mask = group_mask + one_hot

        # Expand mask to expert dimension
        score_mask = mx.repeat(mx.expand_dims(group_mask, axis=-1), num_experts_per_group, axis=-1)
        score_mask = score_mask.reshape(-1, self.n_routed_experts)

        # Mask scores and select top-k
        scores_masked = mx.where(score_mask > 0, scores_for_choice, mx.zeros_like(scores_for_choice))
        topk_indices = mx.argsort(scores_masked, axis=-1)[:, -self.top_k:]

        return topk_indices

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
        """Route tokens to experts.

        Args:
            hidden_states: Input tensor [batch * seq_len, hidden_size]

        Returns:
            topk_indices: Expert indices [batch * seq_len, top_k]
            topk_weights: Expert weights [batch * seq_len, top_k]
        """
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)

        # Compute router logits
        router_logits = hidden_states.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(router_logits)

        # Get top-k indices
        topk_indices = self._get_topk_indices(scores)

        # Gather top-k weights
        topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)

        # Normalize if configured
        if self.norm_topk_prob:
            denominator = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights


class NemotronHMOE(nn.Module):
    """Mixture of Experts layer for NemotronH."""

    def __init__(self, config: NemotronHConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.config = config
        self.experts = [
            NemotronHMLP(config, intermediate_size=config.moe_intermediate_size, layer_idx=layer_idx)
            for _ in range(config.n_routed_experts)
        ]
        self.gate = NemotronHTopkRouter(config)
        self.shared_experts = NemotronHMLP(
            config,
            intermediate_size=config.moe_shared_expert_intermediate_size,
            layer_idx=layer_idx,
        )

    def _moe(
        self,
        hidden_states: mx.array,
        topk_indices: mx.array,
        topk_weights: mx.array,
    ) -> mx.array:
        """Apply MoE routing.

        Args:
            hidden_states: Flattened input [batch * seq_len, hidden_size]
            topk_indices: Expert indices [batch * seq_len, top_k]
            topk_weights: Expert weights [batch * seq_len, top_k]

        Returns:
            Output tensor [batch * seq_len, hidden_size]
        """
        num_tokens, hidden_size = hidden_states.shape
        num_experts = len(self.experts)
        top_k = topk_indices.shape[1]

        # Initialize output
        final_hidden_states = mx.zeros_like(hidden_states)

        # Process each expert
        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]

            # Create mask for tokens routed to this expert (across all top-k positions)
            # Shape: [num_tokens, top_k]
            expert_mask = topk_indices == expert_idx

            # Check if any tokens are routed to this expert
            if not mx.any(expert_mask):
                continue

            # For each top-k position, accumulate weighted expert outputs
            for k in range(top_k):
                # Mask for this expert at position k: [num_tokens]
                mask_k = expert_mask[:, k]

                # Get weights for this position: [num_tokens]
                weights_k = topk_weights[:, k]

                # Apply expert to all tokens (we'll mask the output)
                # This is less efficient but avoids index gathering issues
                expert_output = expert(hidden_states)  # [num_tokens, hidden_size]

                # Weight by routing weights and mask
                # Expand mask and weights for broadcasting
                mask_expanded = mx.expand_dims(mask_k.astype(hidden_states.dtype), axis=-1)
                weights_expanded = mx.expand_dims(weights_k, axis=-1)

                # Add weighted output only for tokens routed to this expert
                final_hidden_states = final_hidden_states + expert_output * mask_expanded * weights_expanded

        return final_hidden_states

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # Route tokens
        topk_indices, topk_weights = self.gate(hidden_states)

        # Flatten for MoE processing
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        # Apply MoE
        hidden_states = self._moe(hidden_states, topk_indices, topk_weights)

        # Reshape back
        hidden_states = hidden_states.reshape(*orig_shape)

        # Add shared experts
        hidden_states = hidden_states + self.shared_experts(residuals)

        return hidden_states


class NemotronHBlock(nn.Module):
    """Single block of NemotronH model.

    Can be Mamba, Attention, MLP, or MoE depending on configuration.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Determine block type
        self.block_type = config.layers_block_type[layer_idx]

        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronHAttention(config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid layer pattern {config.hybrid_override_pattern[layer_idx]}")

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        residual = hidden_states

        # Normalize
        hidden_states = self.norm(hidden_states)

        # Convert residual to fp32 if configured
        if self.residual_in_fp32:
            residual = residual.astype(mx.float32)

        # Apply mixer based on block type
        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
        elif self.block_type == "attention":
            hidden_states, _ = self.mixer(hidden_states, attention_mask=attention_mask)
        elif self.block_type in ["mlp", "moe"]:
            hidden_states = self.mixer(hidden_states)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states


class NemotronHModel(nn.Module):
    """NemotronH base model without LM head."""

    def __init__(self, config: NemotronHConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            NemotronHBlock(config, layer_idx=idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ) -> NemotronHOutput:
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            inputs_embeds: Optional pre-computed embeddings
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states

        Returns:
            NemotronHOutput with last_hidden_state and optional hidden_states
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        # Process through all layers
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Determine mask type based on layer type
            layer_mask = attention_mask if layer.block_type in ["mamba", "attention"] else None

            hidden_states = layer(hidden_states, attention_mask=layer_mask)

        # Final normalization
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return NemotronHOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class NemotronHForCausalLM(nn.Module):
    """NemotronH model with language modeling head."""

    def __init__(self, config: NemotronHConfig):
        super().__init__()

        self.config = config
        self.backbone = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ) -> NemotronHCausalLMOutput:
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            inputs_embeds: Optional pre-computed embeddings
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            output_hidden_states: Whether to return all hidden states

        Returns:
            NemotronHCausalLMOutput with logits and optional loss
        """
        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).astype(mx.float32)

        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # Compute cross entropy loss
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="mean",
            )

        return NemotronHCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> mx.array:
        """Generate text autoregressively.

        Args:
            input_ids: Initial input token IDs [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k sampling
            top_p: Optional nucleus sampling

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Get logits for last position
            outputs = self(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_indices = mx.argsort(next_token_logits, axis=-1)[:, :-top_k]
                next_token_logits = mx.where(
                    mx.arange(next_token_logits.shape[-1]) < top_k_indices.min(axis=-1, keepdims=True),
                    mx.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Sample next token
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.expand_dims(next_token, axis=-1)

            # Append to sequence
            input_ids = mx.concatenate([input_ids, next_token], axis=-1)

        return input_ids
