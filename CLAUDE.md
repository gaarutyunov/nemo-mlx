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

## Common Porting Pitfalls

### 1. Transpose Dimension Order

PyTorch's `transpose(dim0, dim1)` swaps two dimensions. MLX's `transpose` takes a full permutation tuple:

```python
# PyTorch: swap dims 1 and 3
# From [bsz, num_heads, chunks+1, chunks+1] to [bsz, chunks+1, chunks+1, num_heads]
x = x.transpose(1, 3)

# MLX: specify full permutation
# Wrong: transpose(0, 3, 1, 2) gives [bsz, chunks+1, num_heads, chunks+1]
# Correct: transpose(0, 2, 3, 1) gives [bsz, chunks+1, chunks+1, num_heads]
x = x.transpose(0, 2, 3, 1)
```

### 2. Missing `full_like` Function

MLX doesn't have `mx.full_like()`. Use `mx.full()` with explicit shape:

```python
# PyTorch
mask = torch.full_like(tensor, float("-inf"))

# MLX
mask = mx.full(tensor.shape, float("-inf"), dtype=tensor.dtype)
```

### 3. Broadcasting Shape Alignment

MLX requires explicit reshaping for broadcasting:

```python
# PyTorch: [num_heads] automatically broadcasts with [batch, seq, num_heads, dim]
D_residual = self.D * hidden_states

# MLX: Must reshape explicitly
D_expanded = self.D.reshape(1, 1, self.num_heads, 1)
D_residual = D_expanded * hidden_states
```

### 4. Einsum Limitations

MLX's einsum may have different performance characteristics. Consider manual implementations for complex operations:

```python
# Instead of einsum for batched matmul
# result = mx.einsum('bchd,bchs->bcds', A, B)

# Use explicit transpose and matmul
result = A.transpose(...) @ B
```

### 5. Segment Sum for SSM

The segment sum operation requires careful masking to avoid numerical instability:

```python
def segment_sum(input_tensor):
    chunk_size = input_tensor.shape[-1]

    # Expand and tile
    input_tensor = mx.expand_dims(input_tensor, axis=-1)
    input_tensor = mx.tile(input_tensor, (1,) * (input_tensor.ndim - 1) + (chunk_size,))

    # Lower triangular mask (excluding diagonal)
    mask = mx.tril(mx.ones((chunk_size, chunk_size)), k=-1)
    input_tensor = mx.where(mask.astype(mx.bool_), input_tensor, mx.zeros_like(input_tensor))

    # Cumsum
    tensor_segsum = mx.cumsum(input_tensor, axis=-2)

    # Apply final mask with -inf for upper triangle
    mask = mx.tril(mx.ones((chunk_size, chunk_size)), k=0)
    neg_inf = mx.full(tensor_segsum.shape, float("-inf"), dtype=tensor_segsum.dtype)
    tensor_segsum = mx.where(mask.astype(mx.bool_), tensor_segsum, neg_inf)

    return tensor_segsum
```

### 6. `mx.where` Single-Argument Form Not Supported

PyTorch's `torch.where(condition)` (single-argument form) returns indices where condition is true. MLX only supports the three-argument form `mx.where(condition, x, y)`. For MoE routing, use vectorized masking instead:

```python
# PyTorch: Get indices where expert is selected
# token_indices = torch.where(topk_indices == expert_idx)[0]

# MLX: Use vectorized masking instead
expert_mask = topk_indices == expert_idx  # [num_tokens, top_k]
for k in range(top_k):
    mask_k = expert_mask[:, k]  # Boolean mask for tokens selecting this expert at position k
    weights_k = topk_weights[:, k]
    expert_output = expert(hidden_states)
    # Apply mask via broadcasting
    mask_expanded = mx.expand_dims(mask_k.astype(hidden_states.dtype), axis=-1)
    weights_expanded = mx.expand_dims(weights_k, axis=-1)
    final_hidden_states = final_hidden_states + expert_output * mask_expanded * weights_expanded
```

### 7. MoE Configuration Constraints

When configuring Mixture of Experts, ensure `n_routed_experts >= n_group`. The routing logic computes `num_experts_per_group = n_routed_experts // n_group`, which must be > 0:

```python
# Wrong: Will fail with "Cannot infer the shape of an empty array"
config = NemotronHConfig(
    n_routed_experts=4,
    n_group=8,  # 4 // 8 = 0, invalid!
)

# Correct: Ensure n_routed_experts >= n_group
config = NemotronHConfig(
    n_routed_experts=8,  # Must be >= n_group * topk_group
    n_group=2,           # 8 // 2 = 4, valid
    topk_group=2,
)
```

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
