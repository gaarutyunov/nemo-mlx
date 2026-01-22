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
"""Utility functions for State Space Models (SSM) in MLX."""

import mlx.core as mx


def pad_tensor_by_size(input_tensor: mx.array, pad_size: int) -> mx.array:
    """
    Pad tensor with `pad_size` on the seq_len dim (dim=1).

    Assumes that we only have tensors of either size 4 or 3.

    Args:
        input_tensor: Input tensor of shape [bsz, seq_len, ...] (3D or 4D)
        pad_size: Number of zeros to pad along sequence dimension

    Returns:
        Padded tensor
    """
    if pad_size == 0:
        return input_tensor

    ndim = input_tensor.ndim

    if ndim == 3:
        # [bsz, seq_len, dim] -> pad seq_len
        bsz, seq_len, dim = input_tensor.shape
        padding = mx.zeros((bsz, pad_size, dim), dtype=input_tensor.dtype)
        return mx.concatenate([input_tensor, padding], axis=1)
    elif ndim == 4:
        # [bsz, seq_len, dim1, dim2] -> pad seq_len
        bsz, seq_len, dim1, dim2 = input_tensor.shape
        padding = mx.zeros((bsz, pad_size, dim1, dim2), dtype=input_tensor.dtype)
        return mx.concatenate([input_tensor, padding], axis=1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {ndim}D")


def reshape_into_chunks(input_tensor: mx.array, pad_size: int, chunk_size: int) -> mx.array:
    """
    Pad input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously split it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3.

    Args:
        input_tensor: Input tensor of shape [bsz, seq_len, ...] (3D or 4D)
        pad_size: Number of zeros to pad along sequence dimension
        chunk_size: Size of each chunk

    Returns:
        Reshaped tensor with chunks
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if input_tensor.ndim == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        bsz, padded_seq_len, num_heads = input_tensor.shape
        num_chunks = padded_seq_len // chunk_size
        return input_tensor.reshape(bsz, num_chunks, chunk_size, num_heads)
    elif input_tensor.ndim == 4:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size]
        # -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        bsz, padded_seq_len, num_heads, dim = input_tensor.shape
        num_chunks = padded_seq_len // chunk_size
        return input_tensor.reshape(bsz, num_chunks, chunk_size, num_heads, dim)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {input_tensor.ndim}D")


def segment_sum(input_tensor: mx.array) -> mx.array:
    """
    More stable segment sum calculation. Uses cumulative sums and masking
    instead of direct subtractions.

    Args:
        input_tensor: Input tensor of shape [..., chunk_size]

    Returns:
        Segment sum tensor of shape [..., chunk_size, chunk_size]
    """
    chunk_size = input_tensor.shape[-1]

    # 1. Expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = mx.expand_dims(input_tensor, axis=-1)
    input_tensor = mx.tile(input_tensor, (1,) * (input_tensor.ndim - 1) + (chunk_size,))

    # 2. Create a lower triangular mask with the diagonal set to 0 to zero out elements above diag
    mask = mx.tril(mx.ones((chunk_size, chunk_size)), k=-1)
    input_tensor = mx.where(mask.astype(mx.bool_), input_tensor, mx.zeros_like(input_tensor))

    # 3. Compute actual cumsum
    tensor_segsum = mx.cumsum(input_tensor, axis=-2)

    # 4. Apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = mx.tril(mx.ones((chunk_size, chunk_size)), k=0)
    neg_inf = mx.full(tensor_segsum.shape, float("-inf"), dtype=tensor_segsum.dtype)
    tensor_segsum = mx.where(mask.astype(mx.bool_), tensor_segsum, neg_inf)

    return tensor_segsum


def apply_mask_to_padding_states(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    """
    Tunes out the hidden states for padding tokens.

    See https://github.com/state-spaces/mamba/issues/66

    Args:
        hidden_states: Hidden states tensor [bsz, seq_len, hidden_size]
        attention_mask: Attention mask tensor [bsz, seq_len]

    Returns:
        Masked hidden states
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        # Expand mask to match hidden states shape
        mask = mx.expand_dims(attention_mask, axis=-1)  # [bsz, seq_len, 1]
        hidden_states = (hidden_states * mask).astype(dtype)

    return hidden_states


def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    """
    Repeat key/value heads for grouped query attention.

    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim).

    Args:
        hidden_states: KV tensor [batch, num_key_value_heads, slen, head_dim]
        n_rep: Number of repetitions

    Returns:
        Expanded tensor [batch, num_key_value_heads * n_rep, slen, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    # [batch, num_key_value_heads, slen, head_dim]
    # -> [batch, num_key_value_heads, 1, slen, head_dim]
    # -> [batch, num_key_value_heads, n_rep, slen, head_dim]
    # -> [batch, num_key_value_heads * n_rep, slen, head_dim]
    hidden_states = mx.expand_dims(hidden_states, axis=2)
    hidden_states = mx.tile(hidden_states, (1, 1, n_rep, 1, 1))
    hidden_states = hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    return hidden_states


def softplus(x: mx.array) -> mx.array:
    """Softplus activation: log(1 + exp(x))."""
    return mx.log(1.0 + mx.exp(x))
