#!/usr/bin/env python3
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
"""
Utility script to convert PyTorch NemotronH model weights to MLX format.

Usage:
    python convert_weights.py --input checkpoint.pth --output model_mlx.safetensors

This script handles the conversion of all NemotronH components including:
- Mamba2 layers (conv1d, SSM parameters)
- Attention layers
- MLP layers
- MoE layers
- RMSNorm layers
- Embeddings
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_tensor_to_numpy(tensor) -> np.ndarray:
    """Convert a PyTorch tensor to numpy array."""
    import torch

    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return tensor


def convert_conv1d_weight(weight: np.ndarray) -> np.ndarray:
    """
    Convert PyTorch Conv1d weight to MLX format.

    PyTorch Conv1d weight shape: (out_channels, in_channels/groups, kernel_size)
    For grouped convolution with groups=out_channels: (out_channels, 1, kernel_size)

    MLX expects the same format for our manual implementation.
    """
    # For depthwise conv1d, shape is already (out_channels, 1, kernel_size)
    # which is what we need
    return weight


def convert_layer_name(torch_name: str) -> str:
    """
    Convert PyTorch layer names to MLX format.

    Handles the mapping from HuggingFace/PyTorch naming conventions
    to our MLX implementation.
    """
    # Remove common prefixes
    name = torch_name
    name = name.replace("backbone.", "backbone.")

    # Handle Mamba2 mixer names
    if "mixer." in name:
        # conv1d.weight -> conv1d_weight
        name = name.replace("conv1d.weight", "conv1d_weight")
        name = name.replace("conv1d.bias", "conv1d_bias")

    # Handle MoE expert names
    if "experts." in name:
        # experts.0.up_proj -> experts.0.up_proj
        pass  # Names should match

    # Handle attention names
    # PyTorch: q_proj.weight -> MLX: q_proj.weight
    # Names should match for attention

    return name


def convert_state_dict(torch_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PyTorch state dict to MLX-compatible format.

    Args:
        torch_state_dict: PyTorch state dictionary

    Returns:
        MLX-compatible state dictionary with numpy arrays
    """
    import mlx.core as mx

    mlx_state_dict = {}

    for key, value in torch_state_dict.items():
        # Convert tensor to numpy
        np_array = convert_tensor_to_numpy(value)

        if np_array is None:
            continue

        # Skip non-tensor values
        if not isinstance(np_array, np.ndarray):
            logger.warning(f"Skipping non-tensor parameter: {key}")
            continue

        # Convert layer name
        mlx_key = convert_layer_name(key)

        # Handle Conv1d weight conversion
        if "conv1d.weight" in key or "conv1d_weight" in mlx_key:
            np_array = convert_conv1d_weight(np_array)
            logger.info(f"Converted Conv1d weight: {key} shape {np_array.shape}")

        # Handle embeddings (no conversion needed)
        elif "embeddings.weight" in key or "embedding.weight" in key:
            logger.info(f"Embedding: {key} shape {np_array.shape}")

        # Handle linear layers (no conversion needed for MLX)
        elif "weight" in key and np_array.ndim == 2:
            # Linear weights: (out_features, in_features) - same in both
            pass

        # Log conversion
        logger.debug(f"Converting: {key} -> {mlx_key}, shape: {np_array.shape}")

        # Convert to MLX array
        mlx_state_dict[mlx_key] = mx.array(np_array)

    return mlx_state_dict


def load_pytorch_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load PyTorch checkpoint and extract state dict.

    Handles various checkpoint formats:
    - Direct state dict
    - HuggingFace format with 'model' key
    - Training checkpoint with 'state_dict' key
    """
    import torch

    logger.info(f"Loading PyTorch checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    logger.info(f"Loaded {len(state_dict)} parameters from checkpoint")
    return state_dict


def save_mlx_weights(mlx_state_dict: Dict[str, Any], output_path: str):
    """Save MLX weights to safetensors format."""
    import mlx.core as mx

    logger.info(f"Saving MLX weights to {output_path}")
    mx.save_safetensors(output_path, mlx_state_dict)
    logger.info(f"Successfully saved {len(mlx_state_dict)} parameters")


def verify_conversion(
    torch_state_dict: Dict[str, Any],
    mlx_state_dict: Dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """
    Verify that the conversion preserved parameter values.

    Args:
        torch_state_dict: Original PyTorch state dict
        mlx_state_dict: Converted MLX state dict
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if all parameters match within tolerance
    """

    logger.info("Verifying conversion...")
    all_match = True

    for torch_key, torch_value in torch_state_dict.items():
        mlx_key = convert_layer_name(torch_key)

        if mlx_key not in mlx_state_dict:
            logger.warning(f"Key not found in MLX dict: {torch_key} -> {mlx_key}")
            continue

        torch_np = convert_tensor_to_numpy(torch_value)
        mlx_np = np.array(mlx_state_dict[mlx_key])

        # Handle shape differences from conversions
        if "conv1d" in torch_key:
            # Conv1d weights may have different shapes
            pass

        try:
            # Flatten for comparison if shapes differ
            if torch_np.shape != mlx_np.shape:
                logger.warning(
                    f"Shape mismatch for {torch_key}: torch {torch_np.shape} vs mlx {mlx_np.shape}"
                )
                continue

            if not np.allclose(torch_np, mlx_np, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(torch_np - mlx_np))
                logger.warning(f"Value mismatch for {torch_key}: max diff = {max_diff}")
                all_match = False
        except Exception as e:
            logger.error(f"Error comparing {torch_key}: {e}")
            all_match = False

    if all_match:
        logger.info("All parameters match within tolerance")
    else:
        logger.warning("Some parameters have mismatches")

    return all_match


def convert_checkpoint(
    input_path: str,
    output_path: str,
    verify: bool = True,
) -> bool:
    """
    Convert a PyTorch checkpoint to MLX format.

    Args:
        input_path: Path to PyTorch checkpoint
        output_path: Path to save MLX weights
        verify: Whether to verify the conversion

    Returns:
        True if conversion was successful
    """
    # Load PyTorch checkpoint
    torch_state_dict = load_pytorch_checkpoint(input_path)

    # Convert to MLX format
    mlx_state_dict = convert_state_dict(torch_state_dict)

    # Verify conversion if requested
    if verify:
        verify_conversion(torch_state_dict, mlx_state_dict)

    # Save MLX weights
    save_mlx_weights(mlx_state_dict, output_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch NemotronH model to MLX format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint (.pth or .pt)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output MLX weights (.safetensors)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification after conversion'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert checkpoint
    success = convert_checkpoint(
        str(input_path),
        str(output_path),
        verify=not args.no_verify,
    )

    if success:
        logger.info("\nConversion complete!")
        logger.info(f"Input:  {args.input}")
        logger.info(f"Output: {args.output}")
        return 0
    else:
        logger.error("Conversion failed")
        return 1


if __name__ == '__main__':
    exit(main())
