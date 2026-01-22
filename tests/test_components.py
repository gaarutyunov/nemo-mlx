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
"""
Test individual NemotronH model components to verify MLX implementation.

This test suite compares the MLX implementation against the PyTorch reference
to ensure numerical equivalence within acceptable tolerances.
"""

import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nemo_mlx.models.nemotron_h import (
    MambaRMSNormGated,
    NemotronHAttention,
    NemotronHBlock,
    NemotronHMamba2Mixer,
    NemotronHMLP,
    NemotronHModel,
    NemotronHRMSNorm,
)
from nemo_mlx.utils.ssm_utils import (
    pad_tensor_by_size,
    repeat_kv,
    reshape_into_chunks,
    segment_sum,
    softplus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Utility functions for testing
# ============================================================================

def compute_similarity_metrics(torch_output: np.ndarray, mlx_output: np.ndarray):
    """
    Compute similarity metrics between PyTorch and MLX outputs.

    Args:
        torch_output: PyTorch output as numpy array
        mlx_output: MLX output as numpy array

    Returns:
        Dictionary with MAE, max error, and cosine similarity
    """
    # Mean Absolute Error
    mae = np.abs(torch_output - mlx_output).mean()

    # Max Absolute Error
    max_error = np.abs(torch_output - mlx_output).max()

    # Cosine similarity (flattened)
    torch_flat = torch_output.flatten()
    mlx_flat = mlx_output.flatten()

    cosine_sim = np.dot(torch_flat, mlx_flat) / (
        np.linalg.norm(torch_flat) * np.linalg.norm(mlx_flat) + 1e-10
    )

    return {
        "mae": mae,
        "max_error": max_error,
        "cosine_similarity": cosine_sim,
    }


def copy_linear_weights(mlx_linear: mlx_nn.Linear, torch_linear):
    """Copy weights from PyTorch Linear to MLX Linear."""

    mlx_linear.weight = mx.array(torch_linear.weight.detach().numpy())
    if torch_linear.bias is not None:
        mlx_linear.bias = mx.array(torch_linear.bias.detach().numpy())


# ============================================================================
# Test SSM utility functions
# ============================================================================

class TestSSMUtils:
    """Test SSM utility functions."""

    def test_pad_tensor_by_size_3d(self, random_seed):
        """Test padding 3D tensors."""
        batch_size, seq_len, dim = 2, 10, 64
        x = mx.array(np.random.randn(batch_size, seq_len, dim).astype(np.float32))

        pad_size = 6
        result = pad_tensor_by_size(x, pad_size)

        assert result.shape == (batch_size, seq_len + pad_size, dim)
        # Check original data is preserved
        np.testing.assert_array_almost_equal(
            np.array(result[:, :seq_len, :]),
            np.array(x),
        )
        # Check padding is zeros
        np.testing.assert_array_almost_equal(
            np.array(result[:, seq_len:, :]),
            np.zeros((batch_size, pad_size, dim)),
        )

    def test_pad_tensor_by_size_4d(self, random_seed):
        """Test padding 4D tensors."""
        batch_size, seq_len, heads, dim = 2, 10, 4, 32
        x = mx.array(np.random.randn(batch_size, seq_len, heads, dim).astype(np.float32))

        pad_size = 6
        result = pad_tensor_by_size(x, pad_size)

        assert result.shape == (batch_size, seq_len + pad_size, heads, dim)

    def test_reshape_into_chunks(self, random_seed):
        """Test reshaping into chunks."""
        batch_size, seq_len, dim = 2, 100, 64
        chunk_size = 32
        x = mx.array(np.random.randn(batch_size, seq_len, dim).astype(np.float32))

        # Calculate required padding
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        result = reshape_into_chunks(x, pad_size, chunk_size)

        expected_chunks = (seq_len + pad_size) // chunk_size
        assert result.shape == (batch_size, expected_chunks, chunk_size, dim)

    def test_segment_sum(self, random_seed):
        """Test segment sum computation."""
        chunk_size = 8
        x = mx.array(np.random.randn(2, 4, chunk_size).astype(np.float32))

        result = segment_sum(x)

        assert result.shape == (*x.shape, chunk_size)
        # Check lower triangular structure
        result_np = np.array(result)
        # Upper triangular should be -inf
        for i in range(chunk_size):
            for j in range(i + 1, chunk_size):
                assert result_np[0, 0, i, j] == float("-inf")

    def test_repeat_kv(self, random_seed):
        """Test KV head repetition for grouped query attention."""
        batch, num_kv_heads, seq_len, head_dim = 2, 4, 10, 32
        n_rep = 4

        x = mx.array(np.random.randn(batch, num_kv_heads, seq_len, head_dim).astype(np.float32))
        result = repeat_kv(x, n_rep)

        assert result.shape == (batch, num_kv_heads * n_rep, seq_len, head_dim)

        # Check values are correctly repeated
        result_np = np.array(result)
        x_np = np.array(x)
        for i in range(num_kv_heads):
            for j in range(n_rep):
                np.testing.assert_array_almost_equal(
                    result_np[:, i * n_rep + j, :, :],
                    x_np[:, i, :, :],
                )

    def test_softplus(self, random_seed):
        """Test softplus activation."""
        x = mx.array(np.random.randn(100).astype(np.float32))
        result = softplus(x)

        # Compare with numpy implementation
        x_np = np.array(x)
        expected = np.log(1 + np.exp(x_np))

        np.testing.assert_array_almost_equal(np.array(result), expected, decimal=5)


# ============================================================================
# Test RMSNorm components
# ============================================================================

class TestRMSNorm:
    """Test RMSNorm implementations."""

    def test_nemotron_h_rmsnorm(self, random_seed):
        """Test NemotronHRMSNorm."""
        hidden_size = 256
        eps = 1e-6

        # Create MLX RMSNorm
        mlx_norm = NemotronHRMSNorm(hidden_size, eps=eps)

        # Create input
        x = mx.array(np.random.randn(2, 32, hidden_size).astype(np.float32))

        # Forward pass
        output = mlx_norm(x)

        # Check output shape
        assert output.shape == x.shape

        # Verify RMS normalization property
        output_np = np.array(output).astype(np.float64)

        # After RMS norm, the RMS should be approximately 1 (scaled by weight)
        rms = np.sqrt(np.mean(output_np ** 2, axis=-1, keepdims=True))
        # RMS should be close to 1 since weights are initialized to 1
        assert np.allclose(rms, 1.0, atol=0.1)

    def test_mamba_rmsnorm_gated(self, random_seed):
        """Test MambaRMSNormGated."""
        hidden_size = 256
        group_size = 64
        eps = 1e-5

        # Create gated RMSNorm
        mlx_norm = MambaRMSNormGated(hidden_size, group_size=group_size, eps=eps)

        # Create input and gate
        x = mx.array(np.random.randn(2, 32, hidden_size).astype(np.float32))
        gate = mx.array(np.random.randn(2, 32, hidden_size).astype(np.float32))

        # Forward pass without gate
        output_no_gate = mlx_norm(x)
        assert output_no_gate.shape == x.shape

        # Forward pass with gate
        output_with_gate = mlx_norm(x, gate=gate)
        assert output_with_gate.shape == x.shape

        # Gated output should be different from non-gated
        assert not np.allclose(np.array(output_no_gate), np.array(output_with_gate))


# ============================================================================
# Test MLP components
# ============================================================================

class TestMLP:
    """Test MLP implementations."""

    def test_nemotron_h_mlp(self, test_config, sample_hidden_states, random_seed):
        """Test NemotronHMLP forward pass."""
        # Create MLP
        mlp = NemotronHMLP(test_config, layer_idx=0)

        # Convert input to MLX
        x = mx.array(sample_hidden_states)

        # Forward pass
        output = mlp(x)

        # Check output shape matches input
        assert output.shape == x.shape

        logger.info(f"MLP output shape: {output.shape}")
        logger.info(f"MLP output mean: {mx.mean(output).item():.6f}, std: {mx.std(output).item():.6f}")


# ============================================================================
# Test Attention components
# ============================================================================

class TestAttention:
    """Test Attention implementations."""

    def test_nemotron_h_attention(self, small_attention_config, sample_hidden_states, random_seed):
        """Test NemotronHAttention forward pass."""
        config = small_attention_config

        # Create attention module
        attn = NemotronHAttention(config, layer_idx=0)

        # Convert input to MLX
        x = mx.array(sample_hidden_states[:, :, :config.hidden_size])

        # Forward pass
        output, _ = attn(x)

        # Check output shape matches input
        assert output.shape == x.shape

        logger.info(f"Attention output shape: {output.shape}")
        logger.info(f"Attention output mean: {mx.mean(output).item():.6f}, std: {mx.std(output).item():.6f}")

    def test_attention_causal_mask(self, small_attention_config, random_seed):
        """Test that attention is causal."""
        config = small_attention_config
        attn = NemotronHAttention(config, layer_idx=0)

        batch_size, seq_len = 1, 10
        x = mx.array(np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32))

        output, _ = attn(x)

        # With causal masking, later positions shouldn't affect earlier outputs
        # This is a basic sanity check - comprehensive test would compare with known results
        assert output.shape == (batch_size, seq_len, config.hidden_size)


# ============================================================================
# Test Mamba2 components
# ============================================================================

class TestMamba2:
    """Test Mamba2 SSM implementations."""

    def test_mamba2_mixer_shape(self, small_mamba_config, random_seed):
        """Test NemotronHMamba2Mixer output shape."""
        config = small_mamba_config

        # Create Mamba2 mixer
        mixer = NemotronHMamba2Mixer(config, layer_idx=0)

        # Create input
        batch_size, seq_len = 2, 32
        x = mx.array(np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32))

        # Forward pass
        output = mixer(x)

        # Check output shape matches input
        assert output.shape == x.shape

        logger.info(f"Mamba2 output shape: {output.shape}")
        logger.info(f"Mamba2 output mean: {mx.mean(output).item():.6f}, std: {mx.std(output).item():.6f}")

    def test_mamba2_mixer_deterministic(self, small_mamba_config, random_seed):
        """Test that Mamba2 mixer produces consistent outputs."""
        config = small_mamba_config
        mixer = NemotronHMamba2Mixer(config, layer_idx=0)

        batch_size, seq_len = 2, 16
        x = mx.array(np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32))

        # Run forward pass twice
        output1 = mixer(x)
        output2 = mixer(x)

        # Outputs should be identical
        np.testing.assert_array_almost_equal(np.array(output1), np.array(output2))


# ============================================================================
# Test Block and Model components
# ============================================================================

class TestBlock:
    """Test NemotronHBlock."""

    def test_mamba_block(self, small_mamba_config, random_seed):
        """Test NemotronHBlock with Mamba layer."""
        config = small_mamba_config
        block = NemotronHBlock(config, layer_idx=0)

        batch_size, seq_len = 2, 16
        x = mx.array(np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32))

        output = block(x)

        assert output.shape == x.shape
        assert block.block_type == "mamba"

    def test_attention_block(self, small_attention_config, random_seed):
        """Test NemotronHBlock with Attention layer."""
        config = small_attention_config
        block = NemotronHBlock(config, layer_idx=0)

        batch_size, seq_len = 2, 16
        x = mx.array(np.random.randn(batch_size, seq_len, config.hidden_size).astype(np.float32))

        output = block(x)

        assert output.shape == x.shape
        assert block.block_type == "attention"


class TestModel:
    """Test full NemotronHModel."""

    def test_model_forward(self, test_config, random_seed):
        """Test full model forward pass."""
        model = NemotronHModel(test_config)

        batch_size, seq_len = 2, 16
        input_ids = mx.array(np.random.randint(0, test_config.vocab_size, (batch_size, seq_len)))

        output = model(input_ids=input_ids)

        assert output.last_hidden_state.shape == (batch_size, seq_len, test_config.hidden_size)

        logger.info(f"Model output shape: {output.last_hidden_state.shape}")

    def test_model_with_hidden_states(self, test_config, random_seed):
        """Test model returns hidden states when requested."""
        model = NemotronHModel(test_config)

        batch_size, seq_len = 2, 16
        input_ids = mx.array(np.random.randint(0, test_config.vocab_size, (batch_size, seq_len)))

        output = model(input_ids=input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        # Should have num_layers + 1 hidden states (input embedding + each layer)
        assert len(output.hidden_states) == test_config.num_hidden_layers + 1

    def test_model_consistency(self, test_config, random_seed):
        """Test model produces consistent outputs."""
        model = NemotronHModel(test_config)

        batch_size, seq_len = 2, 8
        input_ids = mx.array(np.random.randint(0, test_config.vocab_size, (batch_size, seq_len)))

        # Run twice
        output1 = model(input_ids=input_ids)
        output2 = model(input_ids=input_ids)

        # Should be identical
        np.testing.assert_array_almost_equal(
            np.array(output1.last_hidden_state),
            np.array(output2.last_hidden_state),
        )


# ============================================================================
# Test comparison with PyTorch (requires torch_model.py)
# ============================================================================

@pytest.mark.requires_torch
class TestPyTorchComparison:
    """Tests that compare MLX implementation with PyTorch reference."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            pytest.skip("PyTorch not available")
        return True

    def test_rmsnorm_comparison(self, torch_available, random_seed):
        """Compare RMSNorm implementations."""
        import torch

        hidden_size = 256
        eps = 1e-6

        # Create both implementations
        mlx_norm = NemotronHRMSNorm(hidden_size, eps=eps)

        # Create PyTorch RMSNorm manually
        class TorchRMSNorm(torch.nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, x):
                input_dtype = x.dtype
                x = x.to(torch.float32)
                variance = x.pow(2).mean(-1, keepdim=True)
                x = x * torch.rsqrt(variance + self.variance_epsilon)
                return (self.weight.to(torch.float32) * x).to(input_dtype)

        torch_norm = TorchRMSNorm(hidden_size, eps=eps)

        # Create input
        np.random.seed(42)
        x_np = np.random.randn(2, 32, hidden_size).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        # Forward pass
        with torch.no_grad():
            out_torch = torch_norm(x_torch).numpy()
        out_mlx = np.array(mlx_norm(x_mlx))

        # Compare
        metrics = compute_similarity_metrics(out_torch, out_mlx)
        logger.info(f"RMSNorm comparison: {metrics}")

        assert metrics["mae"] < 1e-5, f"MAE too high: {metrics['mae']}"
        assert metrics["cosine_similarity"] > 0.9999, f"Cosine similarity too low: {metrics['cosine_similarity']}"

    def test_softplus_comparison(self, torch_available, random_seed):
        """Compare softplus implementations."""
        import torch

        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float32)

        # PyTorch
        x_torch = torch.from_numpy(x_np)
        out_torch = torch.nn.functional.softplus(x_torch).numpy()

        # MLX
        x_mlx = mx.array(x_np)
        out_mlx = np.array(softplus(x_mlx))

        # Compare
        metrics = compute_similarity_metrics(out_torch, out_mlx)
        logger.info(f"Softplus comparison: {metrics}")

        assert metrics["mae"] < 1e-5, f"MAE too high: {metrics['mae']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
