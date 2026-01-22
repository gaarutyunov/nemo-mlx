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
Pytest configuration and shared fixtures for NemotronH MLX tests.
"""

import logging
import sys
from pathlib import Path

import pytest

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))


def pytest_collection_modifyitems(config, items):
    """Mark tests that require model weights or specific backends."""
    for item in items:
        # Mark tests that require model weights
        if "model_weights" in item.fixturenames or "torch_model" in item.fixturenames:
            item.add_marker(pytest.mark.requires_weights)

        # Mark tests that require PyTorch
        if "torch_model" in item.fixturenames or "torch" in str(item.fspath):
            item.add_marker(pytest.mark.requires_torch)


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_config():
    """Create a minimal test configuration."""
    from nemo_mlx.models.configuration_nemotron_h import NemotronHConfig

    return NemotronHConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        mamba_num_heads=8,
        mamba_head_dim=32,
        ssm_state_size=16,
        n_groups=2,
        chunk_size=64,
        # Simple pattern for testing: Mamba, Attention, MLP, MoE
        hybrid_override_pattern="M*-E",
        n_routed_experts=8,  # Must be >= n_group * topk_group
        num_experts_per_tok=2,
        n_group=2,  # n_routed_experts // n_group must be > 0
        topk_group=2,
        moe_intermediate_size=256,
        moe_shared_expert_intermediate_size=512,
    )


@pytest.fixture(scope="session")
def small_mamba_config():
    """Create a small Mamba-only configuration for faster testing."""
    from nemo_mlx.models.configuration_nemotron_h import NemotronHConfig

    return NemotronHConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        mamba_num_heads=4,
        mamba_head_dim=32,
        ssm_state_size=8,
        n_groups=2,
        chunk_size=32,
        hybrid_override_pattern="MM",  # All Mamba layers
    )


@pytest.fixture(scope="session")
def small_attention_config():
    """Create a small attention-only configuration for faster testing."""
    from nemo_mlx.models.configuration_nemotron_h import NemotronHConfig

    return NemotronHConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        mamba_num_heads=4,
        mamba_head_dim=32,
        ssm_state_size=8,
        n_groups=2,
        chunk_size=32,
        hybrid_override_pattern="**",  # All attention layers
    )


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    import mlx.core as mx
    import numpy as np

    seed = 42
    np.random.seed(seed)
    mx.random.seed(seed)
    return seed


@pytest.fixture
def sample_input(random_seed):
    """Create sample input for testing."""
    import numpy as np

    batch_size = 2
    seq_len = 32
    vocab_size = 1000

    # Random token IDs
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids


@pytest.fixture
def sample_hidden_states(random_seed, test_config):
    """Create sample hidden states for component testing."""
    import numpy as np

    batch_size = 2
    seq_len = 32
    hidden_size = test_config.hidden_size

    hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    return hidden_states
