# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Position Encoding Implementations for Vision Transformers

This module implements various position encoding strategies used in transformer-based
vision models like SAM (Segment Anything Model). Position encodings are crucial for
transformers as they provide spatial information about token positions since the
attention mechanism itself is position-agnostic.

The module includes:
1. PositionEmbeddingSine: Sinusoidal position encoding (similar to original Transformer)
2. PositionEmbeddingRandom: Random Fourier feature-based position encoding
3. Rotary Position Encoding (RoPE): Advanced encoding that directly modifies attention weights
"""

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal Position Embedding for 2D Images
    
    This implements a 2D version of the sinusoidal position encoding from 
    "Attention Is All You Need" paper. It generates position embeddings using
    sine and cosine functions with different frequencies for x and y coordinates.
    
    The encoding helps the model understand spatial relationships between pixels
    or patch positions in an image.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        """
        Initialize sinusoidal position embedding.
        
        Args:
            num_pos_feats: Number of position features (must be even for x,y split)
            temperature: Temperature parameter for frequency scaling (higher = lower freq)
            normalize: Whether to normalize position coordinates to [0,1]
            scale: Scaling factor for normalized coordinates (default: 2π)
        """
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        # Split features equally between x and y dimensions
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        # Cache for storing computed position embeddings to avoid recomputation
        self.cache = {}

    def _encode_xy(self, x, y):
        """
        Core encoding function that converts x,y coordinates to sinusoidal embeddings.
        
        Args:
            x, y: 1D tensors of normalized coordinates [0,1]
            
        Returns:
            pos_x, pos_y: Position embeddings using sin/cos with different frequencies
        """
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        
        # Scale coordinates by the scale factor (usually 2π)
        x_embed = x * self.scale
        y_embed = y * self.scale

        # Create frequency dimensions: higher indices = lower frequencies
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Apply sinusoidal encoding: pos / freq
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        # Alternate between sin and cos for different frequency dimensions
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        """
        Encode bounding box coordinates with position embeddings.
        
        Args:
            x, y: Center coordinates of boxes
            w, h: Width and height of boxes
            
        Returns:
            pos: Combined position encoding [pos_y, pos_x, height, width]
        """
        pos_x, pos_y = self._encode_xy(x, y)
        # Concatenate: y_pos + x_pos + height + width
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        """
        Encode point coordinates with their labels.

        Args:
            x, y: Point coordinates (batch_size, num_points)
            labels: Point labels (batch_size, num_points)

        Returns:
            pos: Position embeddings with labels [pos_y, pos_x, label]
        """
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl

        # Flatten coordinates for encoding, then reshape back
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)

        # Combine position embeddings with point labels
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Generate 2D position embeddings for image-like tensors.
        
        Args:
            x: Input tensor with shape (batch, channels, height, width)
            
        Returns:
            pos: Position embeddings (batch, embedding_dim, height, width)
        """
        # Use caching to avoid recomputing for same spatial dimensions
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
            
        # Create coordinate grids: y increases down, x increases right
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        # Normalize coordinates to [0, 1] range if requested
        if self.normalize:
            eps = 1e-6  # Small epsilon to avoid division by zero
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Generate frequency dimensions for sinusoidal encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Apply sinusoidal encoding to each spatial position
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Interleave sin and cos for different frequency components
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        
        # Concatenate y and x embeddings, permute to (batch, channels, height, width)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Cache the result for future use
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """
    Random Fourier Feature Position Encoding
    
    This approach uses random spatial frequencies to encode positions, which can
    provide better performance for some tasks compared to fixed sinusoidal patterns.
    Based on "Fourier Features Let Networks Learn High Frequency Functions".
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        """
        Initialize random position embedding.
        
        Args:
            num_pos_feats: Number of position features to generate
            scale: Scaling factor for random frequencies (default: 1.0)
        """
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
            
        # Generate random Gaussian matrix for Fourier features
        # Shape: (2, num_pos_feats) for x,y coordinates
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply random Fourier feature encoding to coordinates.
        
        Args:
            coords: Normalized coordinates in [0,1]^2 with shape (..., 2)
            
        Returns:
            Encoded features with shape (..., 2*num_pos_feats)
        """
        # Map from [0,1] to [-1,1] for better numerical properties
        coords = 2 * coords - 1
        
        # Apply random linear projection: coords @ random_matrix
        coords = coords @ self.positional_encoding_gaussian_matrix
        
        # Scale by 2π for full period coverage
        coords = 2 * np.pi * coords
        
        # Apply sinusoidal encoding: concatenate sin and cos
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate position encoding for a 2D grid of specified size.
        
        Args:
            size: (height, width) of the grid
            
        Returns:
            Position encoding tensor with shape (2*num_pos_feats, height, width)
        """
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        
        # Create coordinate grid with normalized positions
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        # Cumsum creates incremental coordinates, subtract 0.5 for pixel centers
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        
        # Normalize to [0,1] range
        y_embed = y_embed / h
        x_embed = x_embed / w

        # Apply random Fourier feature encoding
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Encode arbitrary 2D coordinates (not necessarily on a grid).
        
        Args:
            coords_input: Coordinates tensor (batch, num_points, 2)
            image_size: (height, width) for normalization
            
        Returns:
            Position encodings (batch, num_points, 2*num_pos_feats)
        """
        coords = coords_input.clone()
        # Normalize coordinates to [0,1] based on image dimensions
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # x coordinate
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # y coordinate
        return self._pe_encoding(coords.to(torch.float))


# Rotary Positional Encoding (RoPE) Implementation
# Adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch

def init_t_xy(end_x: int, end_y: int):
    """
    Initialize 2D coordinate tensors for rotary position encoding.
    
    Args:
        end_x, end_y: Grid dimensions
        
    Returns:
        t_x, t_y: Flattened coordinate tensors for x and y dimensions
    """
    # Create linear indices and convert to 2D coordinates
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()  # x coordinate (column)
    t_y = torch.div(t, end_x, rounding_mode="floor").float()  # y coordinate (row)
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """
    Compute complex exponentials for axial (2D) rotary position encoding.
    
    Rotary encoding rotates feature vectors based on their position, allowing
    the model to naturally understand relative positions in attention computation.
    
    Args:
        dim: Feature dimension (must be divisible by 4 for x,y encoding)
        end_x, end_y: Spatial grid dimensions
        theta: Base frequency for rotation (higher = slower rotation)
        
    Returns:
        Complex tensor containing rotation matrices for each position
    """
    # Generate frequency scales: lower indices = higher frequencies
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    # Get 2D coordinate grids
    t_x, t_y = init_t_xy(end_x, end_y)
    
    # Compute frequency * position for each coordinate
    freqs_x = torch.outer(t_x, freqs_x)  # (positions, frequencies)
    freqs_y = torch.outer(t_y, freqs_y)
    
    # Convert to complex exponentials: e^(i * freq * pos)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    
    # Concatenate x and y components
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor to match input tensor dimensions for broadcasting.
    
    Args:
        freqs_cis: Complex frequency tensor
        x: Input tensor to match
        
    Returns:
        Reshaped frequency tensor ready for element-wise multiplication
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    
    # Create shape that broadcasts correctly: keep last 2 dims, set others to 1
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """
    Apply rotary position encoding to query and key tensors.
    
    This function rotates the feature vectors based on their positions,
    enabling the attention mechanism to naturally capture relative positions.
    
    Args:
        xq: Query tensor (..., seq_len, dim)
        xk: Key tensor (..., seq_len_k, dim)
        freqs_cis: Complex rotation frequencies
        repeat_freqs_k: Whether to repeat frequencies for longer key sequences
        
    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Convert real tensors to complex by pairing adjacent dimensions
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0  # Handle empty key tensors (e.g., due to dropout)
        else None
    )
    
    # Reshape frequencies for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotation to queries: multiply by complex exponential
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    
    if xk_ is None:
        # No keys to rotate (empty due to dropout)
        return xq_out.type_as(xq).to(xq.device), xk
    
    # Handle different sequence lengths between queries and keys
    if repeat_freqs_k:
        # Repeat frequencies to match key sequence length
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            # Use repeat for CUDA tensors
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # Use expand + flatten for better compatibility on other devices
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    
    # Apply rotation to keys
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
