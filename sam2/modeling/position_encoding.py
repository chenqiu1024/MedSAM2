# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Position Encoding Implementations for Vision Transformers and SAM2

This module implements sophisticated position encoding strategies crucial for transformer-based
vision models. Since transformers are inherently permutation-invariant, position encodings
provide essential spatial awareness that enables models to understand geometric relationships
between image regions, points, and objects.

The module contains three main position encoding approaches:

1. **PositionEmbeddingSine**: Classical sinusoidal encoding adapted for 2D vision tasks
   - Based on "Attention Is All You Need" with 2D spatial extensions
   - Uses fixed mathematical functions (sin/cos) with multiple frequencies
   - Excellent for capturing periodic spatial patterns and relative positions
   - Provides strong inductive bias for spatial relationships

2. **PositionEmbeddingRandom**: Random Fourier feature-based encoding
   - Implements learnable position encoding through random frequency sampling
   - Based on "Fourier Features Let Networks Learn High Frequency Functions"
   - More flexible than fixed sinusoidal patterns
   - Can adapt to specific data characteristics during training

3. **Rotary Position Encoding (RoPE)**: Advanced relative position encoding
   - Directly modifies attention computation through feature rotation
   - Provides natural understanding of relative positions without explicit embeddings
   - More parameter-efficient and often more effective than additive approaches
   - Particularly strong for tasks requiring precise spatial reasoning

Key Design Principles:

- **Spatial Awareness**: Enable transformers to understand 2D image geometry
- **Translation Invariance**: Consistent encoding regardless of absolute position
- **Multi-Scale Support**: Handle different image resolutions and patch sizes
- **Efficiency**: Optimized implementations with caching and vectorization
- **Flexibility**: Support for points, boxes, grids, and arbitrary coordinates

Applications in SAM2:

- **Image Encoding**: Spatial position information for image patches
- **Prompt Encoding**: Position-aware encoding of user clicks and bounding boxes
- **Cross-Attention**: Spatial alignment between image features and prompts
- **Memory Attention**: Temporal-spatial encoding for video object tracking
- **Multi-Scale Features**: Consistent position encoding across feature pyramid levels

The choice of position encoding can significantly impact model performance:
- Sinusoidal: Best for tasks requiring strong geometric inductive bias
- Random Fourier: Good for adaptive learning of spatial patterns
- RoPE: Optimal for attention-heavy architectures requiring relative positioning
"""

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal Position Embedding for 2D Vision Tasks
    
    This class implements a 2D extension of the sinusoidal position encoding from the
    original Transformer paper. It generates position embeddings using sine and cosine
    functions with different frequencies for x and y spatial dimensions.
    
    Mathematical Foundation:
    - For position (x, y) and dimension d:
      PE(x,y,2i) = sin(x / 10000^(2i/d)) for x coordinates  
      PE(x,y,2i+1) = cos(x / 10000^(2i/d)) for x coordinates
      PE(x,y,2j) = sin(y / 10000^(2j/d)) for y coordinates
      PE(x,y,2j+1) = cos(y / 10000^(2j/d)) for y coordinates
    
    Key Properties:
    - **Deterministic**: Same positions always get same encodings
    - **Smooth**: Nearby positions have similar encodings
    - **Bounded**: All values are in [-1, 1] range
    - **Periodic**: Enables learning of periodic spatial patterns
    - **Relative**: Encodings naturally capture relative position information
    
    Use Cases:
    - Dense image feature maps (CNN backbone outputs)
    - Sparse point coordinates (user clicks, keypoints)
    - Bounding box coordinates (object detection)
    - Multi-scale feature pyramid position encoding
    
    Performance Characteristics:
    - Fast computation with vectorized operations
    - Memory efficient with optional caching
    - Stable gradients due to smooth functions
    - Good generalization to unseen positions
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        """
        Initialize sinusoidal position embedding with configurable parameters.
        
        Args:
            num_pos_feats (int): Total number of position features. Must be even to
                               split equally between x and y dimensions. Higher values
                               provide more precise position discrimination.
                               
            temperature (int): Temperature parameter controlling frequency scaling.
                             Higher values → lower frequencies → smoother gradients.
                             Lower values → higher frequencies → more precise localization.
                             Default 10000 matches original Transformer paper.
                             
            normalize (bool): Whether to normalize spatial coordinates to [0,1] range.
                            True: Coordinates normalized for scale invariance.
                            False: Use raw pixel coordinates (less stable).
                            
            scale (float, optional): Scaling factor applied to normalized coordinates.
                                   Default 2π provides full period coverage.
                                   Higher values → faster spatial variation.
                                   Lower values → smoother spatial transition.
        
        Raises:
            AssertionError: If num_pos_feats is odd (can't split x/y evenly).
            ValueError: If scale provided but normalize=False.
        """
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        
        # Split features equally between x and y spatial dimensions
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        
        # Validation and default scale setting
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi  # Full period coverage for sin/cos functions
        self.scale = scale

        # Cache for storing computed position embeddings to avoid recomputation
        # This optimization is crucial for video processing where multiple frames
        # often have the same spatial dimensions, and for transformer models where
        # the same position encodings are reused across attention layers
        # Key: (height, width) → Value: position embedding tensor
        self.cache = {}

    def _encode_xy(self, x, y):
        """
        Core sinusoidal encoding function for x,y coordinate pairs.
        
        This method implements the mathematical core of sinusoidal position encoding,
        converting normalized spatial coordinates into high-dimensional embeddings
        that capture both absolute and relative position information.
        
        The encoding process:
        1. Scale coordinates by the scale factor (typically 2π)
        2. Generate frequency dimensions with exponential decay
        3. Apply sinusoidal functions (sin/cos) with different frequencies
        4. Interleave sin and cos to create final embedding
        
        Args:
            x (torch.Tensor): 1D tensor of normalized x coordinates [0,1]
            y (torch.Tensor): 1D tensor of normalized y coordinates [0,1]
            
        Returns:
            tuple: (pos_x, pos_y) where each is a tensor of shape (len(coords), num_pos_feats)
                  pos_x: Position embeddings for x dimension
                  pos_y: Position embeddings for y dimension
                  
        Mathematical Details:
        - Frequency computation: freq_i = temperature^(2i / num_pos_feats)
        - Position encoding: PE_i = sin(coord * scale / freq_i) or cos(...)
        - Alternating pattern: [sin, cos, sin, cos, ...] for different frequencies
        """
        # Input validation: coordinates must be 1D and same length
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        
        # Scale coordinates by the scale factor for appropriate frequency range
        x_embed = x * self.scale  # Typically scales [0,1] to [0, 2π]
        y_embed = y * self.scale

        # Generate frequency dimensions with exponential decay pattern
        # Higher indices → lower frequencies → broader spatial patterns
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Apply position encoding: coordinate / frequency for each dimension
        pos_x = x_embed[:, None] / dim_t  # Shape: (num_positions, num_frequencies)
        pos_y = y_embed[:, None] / dim_t

        # Apply sinusoidal encoding with alternating sin/cos pattern
        # This creates a unique, smooth embedding for each position
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
        
        This method creates position-aware representations for bounding boxes,
        combining spatial position (center) with size information (width/height).
        The encoding provides rich spatial context for object detection and
        segmentation tasks.
        
        Args:
            x (torch.Tensor): Center x coordinates of bounding boxes
            y (torch.Tensor): Center y coordinates of bounding boxes  
            w (torch.Tensor): Width of bounding boxes
            h (torch.Tensor): Height of bounding boxes
            
        Returns:
            torch.Tensor: Combined position encoding with shape (num_boxes, encoding_dim)
                         Format: [pos_y_encoding, pos_x_encoding, height, width]
                         
        Note:
            - Coordinates should be normalized to [0,1] if normalize=True
            - The encoding combines sinusoidal position with explicit size
            - Output can be used directly in transformer attention layers
        """
        # Generate sinusoidal position encodings for box centers
        pos_x, pos_y = self._encode_xy(x, y)
        
        # Combine position encodings with explicit size information
        # Format: [y_position, x_position, height, width]
        # This provides both spatial location and scale information
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    # Backward compatibility alias
    encode = encode_boxes

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        """
        Encode point coordinates with their semantic labels.
        
        This method creates position-aware representations for sparse point inputs
        such as user clicks, keypoints, or landmarks. It combines spatial position
        with semantic label information for rich point representations.

        Args:
            x (torch.Tensor): Point x coordinates with shape (batch_size, num_points)
            y (torch.Tensor): Point y coordinates with shape (batch_size, num_points)
            labels (torch.Tensor): Point labels with shape (batch_size, num_points)
                                 Common labels: 1=foreground, 0=background, 2/3=box corners

        Returns:
            torch.Tensor: Position encodings with shape (batch_size, num_points, encoding_dim)
                         Format: [pos_y_encoding, pos_x_encoding, label]
                         
        Applications:
            - Interactive segmentation (user clicks)
            - Keypoint detection (anatomical landmarks)
            - Object detection (corner points)
            - Semi-supervised learning (labeled points)
        """
        # Validate input tensor shapes
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl

        # Flatten spatial coordinates for batch encoding, then reshape
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x = pos_x.reshape(bx, nx, -1)
        pos_y = pos_y.reshape(by, ny, -1)

        # Combine sinusoidal position encodings with semantic labels
        # The label provides discrete semantic information alongside continuous position
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor, debug_name: str = None):
        """
        Generate 2D position embeddings for dense image feature maps.
        
        This is the main forward method for encoding spatial positions of image
        patches or feature map locations. It creates a dense grid of position
        embeddings that can be added to or concatenated with image features.
        
        The method uses efficient caching to avoid recomputing embeddings for
        the same spatial dimensions, which is common in video processing or
        when processing multiple images of the same size.
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (batch, channels, height, width)
                             The spatial dimensions (H, W) determine the position grid size.
            debug_name (str, optional): Name for debug state capture. If provided and debug mode
                                      is enabled, captures intermediate states for visualization.
                             
        Returns:
            torch.Tensor: Position embeddings with shape (batch, embedding_dim, height, width)
                         Can be directly added to input features or used in attention.
                         
        Optimization Features:
        - **Caching**: Computed embeddings cached by spatial dimensions
        - **Vectorization**: Efficient batch processing of coordinate grids
        - **Memory Reuse**: Same embeddings repeated across batch dimension
        
        Usage Examples:
            # Add to image features
            image_features = backbone(image)  # (B, C, H, W)
            pos_embed = position_encoder(image_features)  # (B, embed_dim, H, W)
            enhanced_features = image_features + pos_embed
            
            # Use in attention
            pos_embed = position_encoder(features).flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Use spatial dimensions as cache key for efficient reuse
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            # Reuse cached computation and repeat for batch dimension
            pos = self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
            
            # Debug capture for cached result
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "position_encoding_sine",
                    state_name="position_embeddings_cached",
                    data=pos,
                    metadata={
                        'encoding_type': 'sine',
                        'cached': True,
                        'spatial_dims': cache_key,
                        'normalize': self.normalize,
                        'scale': self.scale,
                        'temperature': self.temperature
                    }
                )
            return pos
            
        # Create dense coordinate grids for the spatial dimensions
        # Y coordinates: increase downward (standard image convention)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        # X coordinates: increase rightward
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        # Debug capture for raw coordinate grids
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="coordinate_grids_x",
                data=x_embed,
                metadata={'encoding_type': 'sine', 'coordinate_type': 'x_raw'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="coordinate_grids_y",
                data=y_embed,
                metadata={'encoding_type': 'sine', 'coordinate_type': 'y_raw'}
            )

        # Normalize coordinates to [0, 1] range for scale invariance
        if self.normalize:
            eps = 1e-6  # Small epsilon to prevent division by zero
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            # Debug capture for normalized coordinates
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "position_encoding_sine",
                    state_name="coordinate_grids_x_normalized",
                    data=x_embed,
                    metadata={'encoding_type': 'sine', 'coordinate_type': 'x_normalized', 'scale': self.scale}
                )
                capture_debug_state(
                    component_name=debug_name or "position_encoding_sine",
                    state_name="coordinate_grids_y_normalized",
                    data=y_embed,
                    metadata={'encoding_type': 'sine', 'coordinate_type': 'y_normalized', 'scale': self.scale}
                )

        # Generate frequency dimensions for sinusoidal encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Debug capture for frequency dimensions
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="frequency_dimensions",
                data=dim_t,
                metadata={'encoding_type': 'sine', 'temperature': self.temperature, 'num_pos_feats': self.num_pos_feats}
            )

        # Apply sinusoidal encoding to every spatial position
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Create alternating sin/cos pattern for different frequency components
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        
        # Debug capture for intermediate sin/cos encodings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="sinusoidal_encoding_x",
                data=pos_x,
                metadata={'encoding_type': 'sine', 'coordinate_type': 'x_encoded'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="sinusoidal_encoding_y",
                data=pos_y,
                metadata={'encoding_type': 'sine', 'coordinate_type': 'y_encoded'}
            )
        
        # Concatenate y and x embeddings, then permute to standard format
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Debug capture for final position embeddings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="position_embeddings_final",
                data=pos,
                metadata={
                    'encoding_type': 'sine',
                    'cached': False,
                    'spatial_dims': cache_key,
                    'normalize': self.normalize,
                    'scale': self.scale,
                    'temperature': self.temperature,
                    'num_pos_feats': self.num_pos_feats
                }
            )
        
        # Cache result for future use with same spatial dimensions
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """
    Random Fourier Feature Position Encoding for Adaptive Spatial Representations
    
    This class implements position encoding using Random Fourier Features (RFF),
    which can learn to represent spatial relationships more flexibly than fixed
    sinusoidal patterns. The approach projects coordinates through random linear
    transformations followed by sinusoidal activation.
    
    Mathematical Foundation:
    1. Sample random matrix Ω ~ N(0, σ²I) where σ controls frequency scale
    2. For position p: φ(p) = [sin(2π·Ω·p), cos(2π·Ω·p)]
    3. This creates a universal approximator for smooth spatial functions
    
    Key Advantages:
    - **Adaptive**: Can learn task-specific spatial representations
    - **Universal**: Theoretically can approximate any smooth spatial function
    - **Compact**: Efficient representation with learnable frequencies
    - **Robust**: Less sensitive to specific frequency choices than fixed encoding
    
    Based on:
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    Tancik et al., NeurIPS 2020
    
    Applications:
    - Coordinate-based neural networks
    - Implicit neural representations
    - Adaptive position encoding for transformers
    - Multi-resolution spatial modeling
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        """
        Initialize random Fourier feature position encoding.
        
        Args:
            num_pos_feats (int): Number of position features to generate.
                               Higher values provide more representational capacity
                               but increase computational cost. Default 64.
                               
            scale (float, optional): Standard deviation for random frequency sampling.
                                   Higher values → higher frequency components → finer spatial details.
                                   Lower values → lower frequency components → smoother spatial variation.
                                   Default 1.0 provides balanced representation.
        
        Implementation Details:
        - Random matrix sampled once during initialization (frozen during training)
        - Gaussian distribution ensures good frequency coverage
        - 2D coordinates mapped to higher-dimensional feature space
        - Sinusoidal activation provides smooth, bounded outputs
        """
        super().__init__()
        
        # Set default scale if not provided
        if scale is None or scale <= 0.0:
            scale = 1.0
            
        # Generate random Gaussian matrix for Fourier feature projection
        # Shape: (2, num_pos_feats) - maps 2D coordinates to feature space
        # The random matrix is fixed after initialization (not trainable)
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor, debug_name: str = None) -> torch.Tensor:
        """
        Apply random Fourier feature encoding to normalized coordinates.
        
        This method implements the core RFF transformation: projecting coordinates
        through a random linear transformation followed by sinusoidal activation.
        The result is a high-dimensional representation that can capture complex
        spatial patterns while maintaining smoothness properties.
        
        Args:
            coords (torch.Tensor): Normalized coordinates in [0,1]² with shape (..., 2)
                                 Last dimension contains (x, y) coordinate pairs.
            debug_name (str, optional): Name for debug state capture.
                                 
        Returns:
            torch.Tensor: Encoded features with shape (..., 2*num_pos_feats)
                         Concatenation of sine and cosine projections.
                         
        Mathematical Steps:
        1. Normalize [0,1] → [-1,1] for better numerical properties
        2. Linear projection: coords @ random_matrix  
        3. Scale by 2π for full sinusoidal period coverage
        4. Apply sin and cos to create final embedding
        
        Properties:
        - Smooth: Small coordinate changes → small embedding changes
        - Bounded: All values in [-1, 1] range
        - Rich: Can represent complex spatial functions
        - Efficient: Single matrix multiplication + elementwise operations
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="input_coordinates",
                data=coords,
                metadata={'encoding_type': 'random', 'coordinate_range': '[0,1]'}
            )
        
        # Normalize coordinates from [0,1] to [-1,1] for symmetric representation
        coords = 2 * coords - 1
        
        # Debug capture for normalized coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="normalized_coordinates",
                data=coords,
                metadata={'encoding_type': 'random', 'coordinate_range': '[-1,1]'}
            )
        
        # Apply random linear projection to map 2D coords to higher-dimensional space
        coords = coords @ self.positional_encoding_gaussian_matrix
        
        # Debug capture for projected coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="projected_coordinates",
                data=coords,
                metadata={'encoding_type': 'random', 'projection_shape': str(self.positional_encoding_gaussian_matrix.shape)}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="gaussian_projection_matrix",
                data=self.positional_encoding_gaussian_matrix,
                metadata={'encoding_type': 'random', 'matrix_type': 'gaussian_random'}
            )
        
        # Scale by 2π for full period coverage of sinusoidal functions
        coords = 2 * np.pi * coords
        
        # Apply sinusoidal activation and concatenate sin/cos components
        sin_coords = torch.sin(coords)
        cos_coords = torch.cos(coords)
        
        # Debug capture for sin/cos components
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="sin_components",
                data=sin_coords,
                metadata={'encoding_type': 'random', 'activation': 'sine'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="cos_components",
                data=cos_coords,
                metadata={'encoding_type': 'random', 'activation': 'cosine'}
            )
        
        # This creates a rich, smooth representation of spatial position
        result = torch.cat([sin_coords, cos_coords], dim=-1)
        
        # Debug capture for final encoding
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="final_encoding",
                data=result,
                metadata={'encoding_type': 'random', 'final_dim': result.shape[-1]}
            )
        
        return result

    def forward(self, size: Tuple[int, int], debug_name: str = None) -> torch.Tensor:
        """
        Generate random Fourier position encoding for a 2D spatial grid.
        
        This method creates position embeddings for every location in a dense
        spatial grid, such as image feature maps or attention grids. The encoding
        provides adaptive spatial awareness that can be learned during training.
        
        Args:
            size (Tuple[int, int]): Spatial grid dimensions (height, width)
            debug_name (str, optional): Name for debug state capture.
            
        Returns:
            torch.Tensor: Position encoding with shape (2*num_pos_feats, height, width)
                         Each spatial location has a unique embedding vector.
                         
        Usage Examples:
            # For transformer attention
            pos_embed = encoder.forward((64, 64))  # 64x64 grid
            pos_embed = pos_embed.flatten(1, 2).transpose(0, 1)  # (4096, embed_dim)
            
            # For CNN feature augmentation  
            pos_embed = encoder.forward((32, 32))  # Match feature map size
            features = features + pos_embed  # Element-wise addition
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        
        # Create coordinate grid with pixel-centered positions
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        # Cumulative sum creates incremental coordinates [1,2,3,...]
        # Subtract 0.5 to center coordinates in pixel centers
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        
        # Debug capture for raw coordinate grids
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="coordinate_grid_x_raw",
                data=x_embed,
                metadata={'encoding_type': 'random', 'grid_size': size, 'coordinate_type': 'x_raw'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="coordinate_grid_y_raw",
                data=y_embed,
                metadata={'encoding_type': 'random', 'grid_size': size, 'coordinate_type': 'y_raw'}
            )
        
        # Normalize coordinates to [0,1] range for scale invariance
        y_embed = y_embed / h
        x_embed = x_embed / w

        # Debug capture for normalized coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="coordinate_grid_x_normalized",
                data=x_embed,
                metadata={'encoding_type': 'random', 'grid_size': size, 'coordinate_type': 'x_normalized'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="coordinate_grid_y_normalized",
                data=y_embed,
                metadata={'encoding_type': 'random', 'grid_size': size, 'coordinate_type': 'y_normalized'}
            )

        # Apply random Fourier feature encoding to coordinate pairs
        coords_stack = torch.stack([x_embed, y_embed], dim=-1)
        pe = self._pe_encoding(coords_stack, debug_name=debug_name)
        
        # Permute to channel-first format: (features, height, width)
        result = pe.permute(2, 0, 1)
        
        # Debug capture for final result
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="position_embeddings_grid",
                data=result,
                metadata={'encoding_type': 'random', 'grid_size': size, 'output_shape': result.shape}
            )
        
        return result

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int], debug_name: str = None
    ) -> torch.Tensor:
        """
        Encode arbitrary 2D coordinates with random Fourier features.
        
        This method handles sparse coordinate sets that don't necessarily lie on
        a regular grid. It's particularly useful for encoding user interaction
        points, object centers, or other sparse spatial annotations.
        
        Args:
            coords_input (torch.Tensor): Coordinate tensor with shape (batch, num_points, 2)
                                        Last dimension contains (x, y) coordinate pairs.
                                        Coordinates should be in pixel space.
                                        
            image_size (Tuple[int, int]): Reference image dimensions (height, width)
                                        Used for coordinate normalization.
            
            debug_name (str, optional): Name for debug state capture.
                                        
        Returns:
            torch.Tensor: Position encodings with shape (batch, num_points, 2*num_pos_feats)
                         Each coordinate pair gets a unique embedding vector.
                         
        Applications:
        - Interactive segmentation: Encode user click coordinates
        - Object detection: Encode bounding box centers  
        - Keypoint detection: Encode landmark positions
        - Sparse supervision: Encode labeled point annotations
        
        Example:
            # Encode user clicks for interactive segmentation
            clicks = torch.tensor([[[100, 150], [200, 300]]])  # (1, 2, 2)
            embeddings = encoder.forward_with_coords(clicks, (512, 512))  # (1, 2, 128)
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Clone input to avoid modifying original coordinates
        coords = coords_input.clone()
        
        # Debug capture for input coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="sparse_coordinates_input",
                data=coords,
                metadata={'encoding_type': 'random', 'coordinate_space': 'pixel', 'image_size': image_size}
            )
        
        # Normalize coordinates to [0,1] range based on image dimensions
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # x coordinate (width)
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # y coordinate (height)
        
        # Debug capture for normalized coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="sparse_coordinates_normalized",
                data=coords,
                metadata={'encoding_type': 'random', 'coordinate_space': 'normalized', 'image_size': image_size}
            )
        
        # Apply random Fourier feature encoding
        result = self._pe_encoding(coords.to(torch.float), debug_name=debug_name)
        
        # Debug capture for final result
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="sparse_coordinates_encoded",
                data=result,
                metadata={'encoding_type': 'random', 'output_shape': result.shape, 'num_points': coords.shape[1]}
            )
        
        return result


# ============================================================================
# Rotary Positional Encoding (RoPE) Implementation
# ============================================================================
#
# Rotary Position Encoding provides an elegant solution for relative position
# modeling by directly rotating feature vectors based on their positions.
# Unlike additive position encodings, RoPE modifies the attention computation
# itself to naturally incorporate relative position information.
#
# Key advantages:
# - Relative position awareness without explicit position embeddings
# - Better length extrapolation (can handle longer sequences than trained on)
# - More parameter efficient (no additional position parameters)
# - Natural integration with attention mechanism
#
# References:
# 1. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al.)
# 2. https://github.com/meta-llama/codellama/blob/main/llama/model.py  
# 3. https://github.com/naver-ai/rope-vit
# 4. https://github.com/lucidrains/rotary-embedding-torch
# ============================================================================


def init_t_xy(end_x: int, end_y: int):
    """
    Initialize 2D coordinate tensors for rotary position encoding.
    
    Creates flattened coordinate grids that map 2D spatial positions to 1D
    sequences, enabling efficient batch processing of rotary encodings for
    vision transformers and spatial attention mechanisms.
    
    Args:
        end_x (int): Width of the spatial grid (number of columns)
        end_y (int): Height of the spatial grid (number of rows)
        
    Returns:
        tuple: (t_x, t_y) where:
            t_x (torch.Tensor): Flattened x coordinates with shape (end_x * end_y,)
            t_y (torch.Tensor): Flattened y coordinates with shape (end_x * end_y,)
            
    Example:
        For a 2x3 grid:
        t_x = [0, 1, 0, 1, 0, 1]  # Column indices
        t_y = [0, 0, 1, 1, 2, 2]  # Row indices
        
        This creates a row-major flattening pattern suitable for vision transformers
        that process image patches in scanline order.
    """
    # Create linear indices for flattened 2D grid
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    
    # Convert to 2D coordinates using row-major ordering
    t_x = (t % end_x).float()  # x coordinate (column index)
    t_y = torch.div(t, end_x, rounding_mode="floor").float()  # y coordinate (row index)
    
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """
    Compute complex exponentials for 2D Rotary Position Encoding (RoPE) in vision tasks.
    
    This function is a cornerstone of SAM2's spatial understanding, implementing 2D RoPE
    for vision transformers. It extends the original RoPE concept from 1D sequences to
    2D spatial grids, enabling transformers to understand spatial relationships in images.
    
    **Theoretical Foundation from RoFormer:**
    RoPE encodes position information by rotating feature vectors in complex space:
    - For 1D: f(x,m) = x * e^(i*m*θ) where m is position, θ is frequency
    - For 2D: f(x,m,n) = x * e^(i*m*θ_x) * e^(i*n*θ_y) for position (m,n)
    
    **Key Innovation for Vision Tasks:**
    Unlike language models with 1D token sequences, vision requires 2D spatial awareness.
    This function creates separate rotation frequencies for horizontal (x) and vertical (y)
    dimensions, allowing the model to distinguish:
    - Horizontal relationships: objects side-by-side
    - Vertical relationships: objects above/below  
    - Diagonal relationships: combination of x,y rotations
    
    **Mathematical Details:**
    For each spatial position (x, y) and frequency index k:
    
    Frequency calculation (geometric decay):
    ω_k = 1 / (theta^(2k/dim)) where k ∈ [0, dim/4)
    
    Complex rotation for position (x, y):
    - X-component: e^(i * ω_k * x) = cos(ω_k*x) + i*sin(ω_k*x)  
    - Y-component: e^(i * ω_k * y) = cos(ω_k*y) + i*sin(ω_k*y)
    
    **Benefits over Traditional Position Encoding:**
    1. **Relative Awareness**: Attention naturally depends on relative positions
    2. **Extrapolation**: Works on image sizes not seen during training
    3. **Parameter Efficiency**: No learnable position parameters needed
    4. **Rotation Equivariance**: Consistent under coordinate transformations
    
    **Applications in SAM2:**
    - Image patch attention with spatial awareness
    - Cross-attention between prompts and image regions
    - Memory attention for temporal-spatial consistency
    - Multi-scale feature fusion with position preservation
    
    Args:
        dim (int): Feature dimension per attention head, must be divisible by 4.
                  Constraint needed because we encode 2D positions (x,y) using
                  complex numbers, requiring dim/2 complex values = dim/4 per dimension.
                  Typical values: 64, 128, 256 (matching transformer head dimensions).
                  
        end_x (int): Width of the spatial grid (number of columns).
                    Corresponds to width of feature maps or image patches.
                    For 512x512 image with 16x16 patches: end_x = 32.
                    
        end_y (int): Height of the spatial grid (number of rows).
                    Corresponds to height of feature maps or image patches.
                    Should match the spatial dimensions of input features.
                    
        theta (float): Base frequency parameter controlling rotation speed.
                      Higher values → slower rotation → better for longer sequences.
                      Lower values → faster rotation → finer position discrimination.
                      Default 10000 follows RoFormer paper and works well for vision.
                      
    Returns:
        torch.Tensor: Complex exponentials with shape (end_x * end_y, dim//2).
                     Each row contains rotation coefficients for one spatial position.
                     First dim//4 elements: x-dimension rotations
                     Last dim//4 elements: y-dimension rotations
                     Ready for use in apply_rotary_enc() function.
                     
    Raises:
        AssertionError: If dim is not divisible by 4. This constraint ensures
                       equal allocation of features to x and y spatial dimensions.
                       
    Example:
        # For 32x32 feature map with 128-dim features per head
        freqs_cis = compute_axial_cis(dim=128, end_x=32, end_y=32)
        # freqs_cis.shape = (1024, 64) for all spatial positions
        
        # Use in attention layer
        q_rot, k_rot = apply_rotary_enc(queries, keys, freqs_cis)
    """
    # Generate frequency scales with geometric progression
    # Lower indices → higher frequencies → finer spatial discrimination
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    # Get 2D coordinate grids for the spatial domain
    t_x, t_y = init_t_xy(end_x, end_y)
    
    # Compute frequency × position for each spatial location
    # Outer product creates (positions × frequencies) matrix
    freqs_x = torch.outer(t_x, freqs_x)  # Shape: (end_x * end_y, dim//4)
    freqs_y = torch.outer(t_y, freqs_y)  # Shape: (end_x * end_y, dim//4)
    
    # Convert to complex exponentials: e^(i * frequency * position)
    # torch.polar(magnitude, phase) creates complex numbers
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    
    # Concatenate x and y rotation components
    # Final shape: (end_x * end_y, dim//2) for both x and y rotations
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor to enable broadcasting with input tensor.
    
    This utility function adjusts the shape of rotation frequencies to match
    the dimensionality of query/key tensors, enabling efficient element-wise
    multiplication for rotary position encoding.
    
    Args:
        freqs_cis (torch.Tensor): Complex rotation frequencies 
        x (torch.Tensor): Input tensor to match dimensionally
        
    Returns:
        torch.Tensor: Reshaped frequency tensor ready for broadcasting
        
    Broadcasting Rules:
    - Keeps spatial dimensions (last 2) unchanged  
    - Sets all other dimensions to 1 for broadcasting
    - Enables efficient batched rotation computation
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim  # Validate tensor has sufficient dimensions
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])  # Match spatial dims
    
    # Create broadcast-compatible shape: [1, 1, ..., spatial_dims]
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """
    Apply rotary position encoding to query and key tensors in attention computation.
    
    This function implements the core RoPE mechanism: rotating feature vectors based
    on their positions to inject relative position information directly into the
    attention computation. This approach is more efficient and often more effective
    than additive position embeddings.
    
    Mathematical Operation:
    For complex representation q = q_real + i*q_imag:
    RoPE(q, pos) = q * e^(i*θ*pos) = q * (cos(θ*pos) + i*sin(θ*pos))
    
    This rotation preserves vector magnitude while changing direction based on position,
    creating a natural way to encode relative positions in the attention mechanism.
    
    Args:
        xq (torch.Tensor): Query tensor with shape (..., seq_len, dim)
                          Features to be rotated based on their positions.
                          
        xk (torch.Tensor): Key tensor with shape (..., seq_len_k, dim)  
                          Features to be rotated for relative position computation.
                          May have different sequence length than queries.
                          
        freqs_cis (torch.Tensor): Complex rotation frequencies from compute_axial_cis()
                                 Contains position-dependent rotation matrices.
                                 
        repeat_freqs_k (bool): Whether to repeat frequencies for longer key sequences.
                              True: Handle cases where keys are longer than queries.
                              False: Assume matching sequence lengths (default).
                              
    Returns:
        tuple: (rotated_queries, rotated_keys) where:
            rotated_queries (torch.Tensor): Position-encoded queries, same shape as xq
            rotated_keys (torch.Tensor): Position-encoded keys, same shape as xk
            
    Key Benefits:
    - **Relative Position**: Naturally encodes relative distances between positions
    - **Extrapolation**: Can handle longer sequences than seen during training
    - **Efficiency**: No additional parameters needed for position encoding
    - **Integration**: Seamlessly integrates with existing attention mechanisms
    
    Applications:
    - Vision transformers with spatial position awareness
    - Cross-attention between image regions and text tokens
    - Multi-scale feature fusion with position consistency
    - Video understanding with temporal-spatial position encoding
    """
    # Convert real tensors to complex representation by pairing adjacent dimensions
    # This enables efficient rotation computation using complex multiplication
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    
    # Handle potential empty key tensors (e.g., from attention dropout)
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0  # Check for non-empty key sequence
        else None
    )
    
    # Reshape rotation frequencies for broadcasting with query tensor
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotation to queries via complex multiplication
    # This efficiently computes the 2D rotation without explicit matrix operations
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    
    # Handle empty key case (no rotation needed)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    
    # Handle different sequence lengths between queries and keys
    if repeat_freqs_k:
        # Repeat rotation frequencies to match longer key sequence
        r = xk_.shape[-2] // xq_.shape[-2]  # Repetition factor
        
        if freqs_cis.is_cuda:
            # Use tensor.repeat() for CUDA tensors (better performance)
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # Use expand + flatten for better compatibility on CPU/other devices
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    
    # Apply rotation to keys with potentially different frequencies
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # Convert back to original tensor types and devices
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
