# NemotronH MLX Port - Development Guide

This document describes the architecture, porting decisions, and testing approach for the MLX port of the NemotronH model.

## Overview

NemotronH is a hybrid language model architecture that combines:
- **Mamba2 (SSM)**: State Space Model layers for efficient sequence modeling
- **Attention**: Standard multi-head attention with grouped query attention (GQA)
- **MLP**: Feed-forward layers with SiLU activation
- **MoE**: Mixture of Experts with top-k routing

The model uses a configurable layer pattern (e.g., `M*-E` = Mamba, Attention, MLP, MoE) allowing flexible hybrid architectures.

## Project Structure

```
src/nemo_mlx/
├── __init__.py                 # Package exports
├── convert_weights.py          # PyTorch to MLX weight conversion
├── models/
│   ├── __init__.py
│   ├── configuration_nemotron_h.py  # Model configuration dataclass
│   └── nemotron_h.py           # All model components
└── utils/
    ├── __init__.py
    └── ssm_utils.py            # SSM utility functions

tests/
├── conftest.py                 # Pytest fixtures
└── test_components.py          # Component and comparison tests
```

## Key Porting Differences: PyTorch vs MLX

### 1. Tensor Layout Conventions

MLX uses **channels-last** format by default, while PyTorch typically uses **channels-first**:

```python
# PyTorch Conv1d weight: (out_channels, in_channels/groups, kernel_size)
# MLX: Same format for our manual implementation

# PyTorch attention: (batch, num_heads, seq_len, head_dim)
# MLX: Same format
```

### 2. Missing MLX Functions

Several PyTorch functions don't have direct MLX equivalents:

| PyTorch | MLX Equivalent |
|---------|----------------|
| `torch.full_like(x, val)` | `mx.full(x.shape, val, dtype=x.dtype)` |
| `F.silu(x)` | `mx.sigmoid(x) * x` or `nn.silu(x)` |
| `F.softplus(x)` | `mx.log(1.0 + mx.exp(x))` |
| `torch.repeat_interleave` | Manual reshape + tile + reshape |
| `F.conv1d` (grouped) | Manual implementation (see below) |

### 3. Grouped Convolution (Conv1d)

MLX doesn't have built-in grouped convolution. For depthwise conv1d (groups=channels), we implement manually:

```python
def _apply_conv1d(self, x: mx.array) -> mx.array:
    """Manual depthwise conv1d implementation."""
    # x: [batch, seq_len, conv_dim]
    # weight: [conv_dim, 1, kernel_size]

    # Transpose to [batch, conv_dim, seq_len]
    x = x.transpose(0, 2, 1)

    # Pad for causal convolution
    x = mx.pad(x, [(0, 0), (0, 0), (kernel_size - 1, 0)])

    # Manual sliding window convolution per channel
    # ... (see nemotron_h.py for full implementation)
```

### 4. Broadcasting Differences

MLX broadcasting follows NumPy rules but requires explicit shape alignment:

```python
# PyTorch: D shape [num_heads] broadcasts with [batch, seq, num_heads, dim]
# MLX: Must reshape D to [1, 1, num_heads, 1] explicitly
D_expanded = self.D.reshape(1, 1, self.num_heads, 1)
```

### 5. In-place Operations

MLX doesn't support in-place operations. All operations return new arrays:

```python
# PyTorch
x.add_(1)  # In-place

# MLX
x = x + 1  # Must reassign
```

### 6. No CUDA Kernels

The PyTorch NemotronH uses optimized CUDA kernels for Mamba2 (via `mamba_ssm` package). Our MLX port uses the **naive Python implementation** which is slower but works on all platforms.

## Component Details

### NemotronHRMSNorm

RMS normalization without mean centering:

```python
def __call__(self, x):
    variance = mx.mean(x ** 2, axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + self.eps)
    return self.weight * x
```

### MambaRMSNormGated

Gated RMS norm with optional SiLU gating for Mamba2 layers:

```python
def __call__(self, x, gate=None):
    # RMS normalize
    x = rms_norm(x)
    # Apply gate with SiLU activation
    if gate is not None:
        x = x * nn.silu(gate)
    return self.weight * x
```

### NemotronHMamba2Mixer

The Mamba2 SSM layer. Key components:

1. **Input projection**: Projects hidden_size → intermediate_size + conv_dim + num_heads
2. **Conv1d**: Causal depthwise convolution
3. **SSM computation**: Discretized state space model with chunked processing
4. **Gated output**: RMSNorm with gating + output projection

The SSM uses chunked processing for efficiency:
- Splits sequence into chunks of size `chunk_size`
- Computes intra-chunk (diagonal) and inter-chunk interactions
- Uses segment_sum for stable cumulative computations

### NemotronHAttention

Standard multi-head attention with:
- Grouped Query Attention (GQA) support via `num_key_value_heads`
- Causal masking for autoregressive generation
- No rotary position embeddings (RoPE) in base implementation

### NemotronHMOE

Mixture of Experts with:
- Top-k expert selection per token
- Group-based routing for load balancing
- Shared expert that processes all tokens
- Routed scaling factor for expert outputs

## Testing Approach

### Component Comparison Tests

Following the pattern from [vjepa2-mlx](https://github.com/gaarutyunov/vjepa2-mlx):

1. **Create identical inputs** for PyTorch and MLX
2. **Copy weights** from PyTorch to MLX
3. **Run forward pass** on both
4. **Compare outputs** using:
   - Mean Absolute Error (MAE)
   - Max Absolute Error
   - Cosine Similarity

```python
def compute_similarity_metrics(torch_output, mlx_output):
    mae = np.abs(torch_output - mlx_output).mean()
    cosine_sim = np.dot(torch_flat, mlx_flat) / (norm(torch_flat) * norm(mlx_flat))
    return {"mae": mae, "cosine_similarity": cosine_sim}
```

### Test Fixtures

```python
@pytest.fixture
def small_mamba_config():
    """Small config for fast Mamba testing."""
    return NemotronHConfig(
        hidden_size=128,
        num_hidden_layers=2,
        mamba_num_heads=4,
        hybrid_override_pattern="MM",
    )

@pytest.fixture
def small_attention_config():
    """Small config for fast attention testing."""
    return NemotronHConfig(
        hidden_size=128,
        num_hidden_layers=2,
        hybrid_override_pattern="**",
    )
```

### Running Tests

```bash
# All tests (requires MLX)
pytest tests/test_components.py -v

# Without PyTorch comparison
pytest tests/test_components.py -v -m "not requires_torch"

# With PyTorch comparison
pip install torch
pytest tests/test_components.py -v -m "requires_torch"
```

## Weight Conversion

Convert PyTorch checkpoints to MLX format:

```bash
python -m nemo_mlx.convert_weights --input model.pt --output model.safetensors
```

Key conversions:
- Linear weights: No change needed (both use [out, in] format)
- Conv1d weights: [out, in/groups, kernel] → same format
- Embeddings: No change needed

## CI/CD

GitHub Actions workflow builds MLX from source on Linux:

```yaml
- name: Build MLX from source
  run: |
    git clone --depth 1 https://github.com/ml-explore/mlx.git
    cd mlx && pip wheel . -w ~/.mlx-wheel

- name: Install MLX
  run: pip install ~/.mlx-wheel/mlx-*.whl
```

The wheel is cached between runs for faster CI.

## Known Limitations

1. **No KV caching**: Generation requires full sequence recomputation
2. **Naive SSM**: Uses Python loops instead of optimized kernels
3. **No flash attention**: Standard attention implementation
4. **MoE routing**: Simplified routing without auxiliary losses

## Future Improvements

- [ ] Add KV caching for efficient generation
- [ ] Optimize Conv1d with MLX primitives
- [ ] Add RoPE support for attention
- [ ] Implement flash attention variant
- [ ] Add model parallelism support

## References

- [Original NemotronH (PyTorch)](https://github.com/huggingface/transformers)
- [Mamba2 Paper](https://arxiv.org/abs/2405.21060)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [vjepa2-mlx](https://github.com/gaarutyunov/vjepa2-mlx) - Reference for testing patterns
