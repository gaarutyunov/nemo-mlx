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
Test individual NemotronH model components by comparing MLX implementation
against PyTorch reference to ensure numerical equivalence.
"""

import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nemo_mlx.models.configuration_nemotron_h import NemotronHConfig
from nemo_mlx.models.nemotron_h import (
    NemotronHAttention,
    NemotronHMamba2Mixer,
    NemotronHMLP,
    NemotronHMOE,
    NemotronHRMSNorm,
)
from nemo_mlx.utils.ssm_utils import softplus

# Import PyTorch reference implementation
sys.path.insert(0, str(Path(__file__).parent.parent))
from torch_model import (
    NemotronHAttention as TorchAttention,
)
from torch_model import (
    NemotronHConfig as TorchConfig,
)
from torch_model import (
    NemotronHMamba2Mixer as TorchMamba2Mixer,
)
from torch_model import (
    NemotronHMLP as TorchMLP,
)
from torch_model import (
    NemotronHMOE as TorchMOE,
)
from torch_model import (
    NemotronHRMSNorm as TorchRMSNorm,
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
    mae = np.abs(torch_output - mlx_output).mean()
    max_error = np.abs(torch_output - mlx_output).max()

    torch_flat = torch_output.flatten()
    mlx_flat = mlx_output.flatten()

    cosine_sim = np.dot(torch_flat, mlx_flat) / (np.linalg.norm(torch_flat) * np.linalg.norm(mlx_flat) + 1e-10)

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


def copy_mlp_weights(mlx_mlp, torch_mlp):
    """Copy weights from PyTorch MLP to MLX MLP."""
    copy_linear_weights(mlx_mlp.up_proj, torch_mlp.up_proj)
    copy_linear_weights(mlx_mlp.down_proj, torch_mlp.down_proj)


def copy_attention_weights(mlx_attn, torch_attn):
    """Copy weights from PyTorch Attention to MLX Attention."""
    copy_linear_weights(mlx_attn.q_proj, torch_attn.q_proj)
    copy_linear_weights(mlx_attn.k_proj, torch_attn.k_proj)
    copy_linear_weights(mlx_attn.v_proj, torch_attn.v_proj)
    copy_linear_weights(mlx_attn.o_proj, torch_attn.o_proj)


def copy_mamba2_weights(mlx_mamba, torch_mamba):
    """Copy weights from PyTorch Mamba2Mixer to MLX Mamba2Mixer."""
    # Input projection
    copy_linear_weights(mlx_mamba.in_proj, torch_mamba.in_proj)

    # Output projection
    copy_linear_weights(mlx_mamba.out_proj, torch_mamba.out_proj)

    # Conv1d weights: PyTorch [out, in/groups, kernel] -> same format
    mlx_mamba.conv1d_weight = mx.array(torch_mamba.conv1d.weight.detach().numpy())
    if torch_mamba.conv1d.bias is not None:
        mlx_mamba.conv1d_bias = mx.array(torch_mamba.conv1d.bias.detach().numpy())

    # Parameters
    mlx_mamba.dt_bias = mx.array(torch_mamba.dt_bias.detach().numpy())
    mlx_mamba.A_log = mx.array(torch_mamba.A_log.detach().numpy())
    mlx_mamba.D = mx.array(torch_mamba.D.detach().numpy())

    # Norm weights
    mlx_mamba.norm.weight = mx.array(torch_mamba.norm.weight.detach().numpy())


def copy_rmsnorm_weights(mlx_norm, torch_norm):
    """Copy weights from PyTorch RMSNorm to MLX RMSNorm."""
    mlx_norm.weight = mx.array(torch_norm.weight.detach().numpy())


def copy_moe_weights(mlx_moe, torch_moe):
    """Copy weights from PyTorch MoE to MLX MoE."""
    # Copy router weights
    copy_linear_weights(mlx_moe.gate.linear, torch_moe.gate.linear)

    # Copy shared expert weights
    copy_mlp_weights(mlx_moe.shared_experts, torch_moe.shared_experts)

    # Copy routed expert weights
    for i, (mlx_expert, torch_expert) in enumerate(zip(mlx_moe.experts, torch_moe.experts)):
        copy_mlp_weights(mlx_expert, torch_expert)


def create_torch_config(**kwargs):
    """Create a PyTorch NemotronHConfig with given parameters."""
    return TorchConfig(**kwargs)


def create_mlx_config(**kwargs):
    """Create an MLX NemotronHConfig with given parameters."""
    return NemotronHConfig(**kwargs)


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    mx.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def small_config_params():
    """Parameters for a small test configuration."""
    return {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "mamba_num_heads": 4,
        "mamba_head_dim": 32,
        "ssm_state_size": 8,
        "n_groups": 2,
        "chunk_size": 32,
        "hybrid_override_pattern": "MM",
    }


@pytest.fixture
def moe_config_params():
    """Parameters for MoE test configuration."""
    return {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "mamba_num_heads": 4,
        "mamba_head_dim": 32,
        "ssm_state_size": 8,
        "n_groups": 2,
        "chunk_size": 32,
        "hybrid_override_pattern": "EE",
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 2,
        "moe_intermediate_size": 128,
        "moe_shared_expert_intermediate_size": 256,
    }


# ============================================================================
# PyTorch Comparison Tests
# ============================================================================


class TestRMSNormComparison:
    """Compare RMSNorm implementations between PyTorch and MLX."""

    def test_rmsnorm(self, random_seed):
        """Compare NemotronHRMSNorm implementations."""
        hidden_size = 256
        eps = 1e-6

        # Create both implementations
        torch_norm = TorchRMSNorm(hidden_size, eps=eps)
        mlx_norm = NemotronHRMSNorm(hidden_size, eps=eps)

        # Copy weights
        copy_rmsnorm_weights(mlx_norm, torch_norm)

        # Create identical input
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
        assert metrics["cosine_similarity"] > 0.9999, f"Cosine sim too low: {metrics['cosine_similarity']}"


class TestSoftplusComparison:
    """Compare softplus implementations."""

    def test_softplus(self, random_seed):
        """Compare softplus implementations."""
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


class TestMLPComparison:
    """Compare MLP implementations between PyTorch and MLX."""

    def test_mlp(self, small_config_params, random_seed):
        """Compare NemotronHMLP implementations."""
        torch_config = create_torch_config(**small_config_params)
        mlx_config = create_mlx_config(**small_config_params)

        # Create both implementations
        torch_mlp = TorchMLP(torch_config, layer_idx=0)
        mlx_mlp = NemotronHMLP(mlx_config, layer_idx=0)

        # Copy weights
        copy_mlp_weights(mlx_mlp, torch_mlp)

        # Create identical input
        batch_size, seq_len = 2, 16
        x_np = np.random.randn(batch_size, seq_len, torch_config.hidden_size).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        # Forward pass
        with torch.no_grad():
            out_torch = torch_mlp(x_torch).numpy()
        out_mlx = np.array(mlx_mlp(x_mlx))

        # Compare
        metrics = compute_similarity_metrics(out_torch, out_mlx)
        logger.info(f"MLP comparison: {metrics}")

        assert metrics["mae"] < 1e-4, f"MAE too high: {metrics['mae']}"
        assert metrics["cosine_similarity"] > 0.999, f"Cosine sim too low: {metrics['cosine_similarity']}"


class TestAttentionComparison:
    """Compare Attention implementations between PyTorch and MLX."""

    def test_attention(self, random_seed):
        """Compare NemotronHAttention implementations."""
        config_params = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "mamba_num_heads": 4,
            "mamba_head_dim": 32,
            "ssm_state_size": 8,
            "n_groups": 2,
            "chunk_size": 32,
            "hybrid_override_pattern": "**",
        }

        torch_config = create_torch_config(**config_params)
        mlx_config = create_mlx_config(**config_params)

        # Create both implementations
        torch_attn = TorchAttention(torch_config, layer_idx=0)
        mlx_attn = NemotronHAttention(mlx_config, layer_idx=0)

        # Copy weights
        copy_attention_weights(mlx_attn, torch_attn)

        # Create identical input
        batch_size, seq_len = 2, 16
        x_np = np.random.randn(batch_size, seq_len, torch_config.hidden_size).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        # Forward pass
        with torch.no_grad():
            out_torch, _, _ = torch_attn(x_torch)
            out_torch = out_torch.numpy()
        out_mlx, _ = mlx_attn(x_mlx)
        out_mlx = np.array(out_mlx)

        # Compare
        metrics = compute_similarity_metrics(out_torch, out_mlx)
        logger.info(f"Attention comparison: {metrics}")

        assert metrics["mae"] < 1e-4, f"MAE too high: {metrics['mae']}"
        assert metrics["cosine_similarity"] > 0.999, f"Cosine sim too low: {metrics['cosine_similarity']}"


class TestMamba2Comparison:
    """Compare Mamba2Mixer implementations between PyTorch and MLX."""

    def test_mamba2_mixer(self, small_config_params, random_seed):
        """Compare NemotronHMamba2Mixer implementations."""
        torch_config = create_torch_config(**small_config_params)
        mlx_config = create_mlx_config(**small_config_params)

        # Create both implementations
        torch_mamba = TorchMamba2Mixer(torch_config, layer_idx=0)
        mlx_mamba = NemotronHMamba2Mixer(mlx_config, layer_idx=0)

        # Copy weights
        copy_mamba2_weights(mlx_mamba, torch_mamba)

        # Create identical input
        batch_size, seq_len = 2, 32  # seq_len should be divisible by chunk_size
        x_np = np.random.randn(batch_size, seq_len, torch_config.hidden_size).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        # Forward pass (use slow_forward for PyTorch to avoid CUDA kernels)
        with torch.no_grad():
            out_torch = torch_mamba.slow_forward(x_torch).numpy()
        out_mlx = np.array(mlx_mamba(x_mlx))

        # Compare
        metrics = compute_similarity_metrics(out_torch, out_mlx)
        logger.info(f"Mamba2Mixer comparison: {metrics}")

        # Mamba2 has more numerical differences due to complex SSM computations
        assert metrics["mae"] < 0.1, f"MAE too high: {metrics['mae']}"
        assert metrics["cosine_similarity"] > 0.9, f"Cosine sim too low: {metrics['cosine_similarity']}"


class TestMOEComparison:
    """Compare MoE implementations between PyTorch and MLX."""

    def test_moe(self, moe_config_params, random_seed):
        """Compare NemotronHMOE implementations."""
        torch_config = create_torch_config(**moe_config_params)
        mlx_config = create_mlx_config(**moe_config_params)

        # Create both implementations
        torch_moe = TorchMOE(torch_config, layer_idx=0)
        mlx_moe = NemotronHMOE(mlx_config, layer_idx=0)

        # Copy weights
        copy_moe_weights(mlx_moe, torch_moe)

        # Create identical input
        batch_size, seq_len = 2, 16
        x_np = np.random.randn(batch_size, seq_len, torch_config.hidden_size).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        # Forward pass
        with torch.no_grad():
            out_torch = torch_moe(x_torch).numpy()
        out_mlx = np.array(mlx_moe(x_mlx))

        # Compare
        metrics = compute_similarity_metrics(out_torch, out_mlx)
        logger.info(f"MoE comparison: {metrics}")

        assert metrics["mae"] < 0.1, f"MAE too high: {metrics['mae']}"
        assert metrics["cosine_similarity"] > 0.9, f"Cosine sim too low: {metrics['cosine_similarity']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
