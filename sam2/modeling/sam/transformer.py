# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Two-Way Transformer Architecture for SAM2 (Segment Anything Model 2)

This module implements the transformer-based mask decoder component of SAM2, which
processes image embeddings and prompt embeddings to generate accurate segmentation masks.
The architecture employs a novel "two-way" attention mechanism that enables bidirectional
information flow between sparse prompt tokens (points, boxes) and dense image tokens.

Key Architectural Components:

1. TwoWayTransformer: Main decoder that orchestrates the attention process
2. TwoWayAttentionBlock: Core building block implementing bidirectional attention
3. Attention: Standard multi-head attention with optional downsampling for efficiency
4. RoPEAttention: Enhanced attention with Rotary Position Encoding for better spatial understanding

The Two-Way Attention Mechanism:
- Sparse-to-Sparse: Prompt tokens attend to each other (self-attention)
- Sparse-to-Dense: Prompt tokens attend to image tokens (cross-attention)
- Dense-to-Sparse: Image tokens attend to prompt tokens (reverse cross-attention)

This bidirectional flow allows the model to:
- Refine prompt representations based on image context
- Update image representations based on user prompts
- Achieve better alignment between user intent and visual features

Performance Optimizations:
- Flash Attention support for faster computation on modern GPUs
- Attention downsampling to reduce computational complexity
- Efficient tensor reshaping and memory management
"""

import contextlib
import math
import warnings
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import get_sdpa_settings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Attention optimization settings for different hardware configurations
# These settings are automatically detected based on GPU capabilities
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()

# Global fallback flag for attention kernel selection
# Set to True if Flash Attention fails, allowing all available kernels
ALLOW_ALL_KERNELS = False


def sdp_kernel_context(dropout_p):
    """
    Create context manager for scaled dot-product attention kernel selection.
    
    This function manages the selection of attention computation kernels based on
    hardware capabilities and performance characteristics. It prioritizes Flash
    Attention for optimal performance but provides fallbacks for compatibility.
    
    Args:
        dropout_p (float): Dropout probability for attention weights
        
    Returns:
        Context manager for attention kernel configuration
        
    Kernel Selection Logic:
    - Flash Attention: Fastest on modern GPUs with sufficient memory
    - Math Kernel: Most compatible, required for older GPUs with dropout
    - Memory Efficient: Optimized for older hardware with limited memory
    
    The context manager ensures graceful fallback if preferred kernels fail.
    """
    if ALLOW_ALL_KERNELS:
        return contextlib.nullcontext()

    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        # Enable math kernel for older GPUs or when dropout is used during training
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        # Enable memory-efficient kernel for older hardware
        enable_mem_efficient=OLD_GPU,
    )


class TwoWayTransformer(nn.Module):
    """
    Transformer decoder with bidirectional attention for mask prediction in SAM2.
    
    This is the main component of the mask decoder that processes image embeddings
    and prompt embeddings through a series of two-way attention blocks. The architecture
    enables sophisticated interaction between user prompts and image features, allowing
    for precise mask generation based on various input modalities.
    
    Key Design Features:
    - Bidirectional attention between prompts and image features
    - Stack of transformer blocks for iterative refinement
    - Final attention layer for comprehensive prompt-image fusion
    - Support for variable numbers of prompt tokens
    
    The two-way mechanism allows:
    1. Prompts to gather relevant information from the image
    2. Image features to be modulated by prompt context
    3. Iterative refinement through multiple layers
    4. Final alignment between prompts and visual features
    """
    
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        Initialize the TwoWayTransformer with specified architecture parameters.

        Args:
            depth (int): Number of transformer layers to stack. More layers allow
                for more sophisticated prompt-image interactions but increase
                computational cost.
            embedding_dim (int): Channel dimension for all embeddings. Must be
                consistent across the entire model for proper tensor operations.
            num_heads (int): Number of attention heads in multi-head attention.
                Must divide embedding_dim evenly. More heads allow the model to
                attend to different types of relationships simultaneously.
            mlp_dim (int): Hidden dimension in the MLP blocks within each transformer
                layer. Typically 2-4x the embedding_dim for optimal capacity.
            activation (nn.Module): Activation function for MLP blocks. ReLU is
                standard, but GELU can provide smoother gradients.
            attention_downsample_rate (int): Factor by which to reduce the internal
                attention dimension for computational efficiency. Higher values
                reduce memory usage but may impact model capacity.
        """
        super().__init__()
        # Store architecture parameters for reference
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        
        # Create stack of two-way attention blocks
        # Each block performs bidirectional attention between prompts and image
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    # Skip positional encoding in first layer to avoid double-adding
                    skip_first_layer_pe=(i == 0),
                )
            )

        # Final attention layer for comprehensive prompt-to-image alignment
        # This layer ensures that prompt tokens have fully integrated image context
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process image and prompt embeddings through two-way transformer blocks.
        
        This method performs the core computation of the mask decoder, enabling
        sophisticated interaction between user prompts and image features through
        bidirectional attention mechanisms.

        Args:
            image_embedding (torch.Tensor): Dense image features from the image encoder
                with shape [B, embedding_dim, H, W]. These represent visual content
                at spatial locations in the image.
            image_pe (torch.Tensor): Positional encoding for image features with the
                same shape as image_embedding. Provides spatial awareness to the model.
            point_embedding (torch.Tensor): Sparse prompt embeddings (points, boxes,
                mask tokens) with shape [B, N_prompts, embedding_dim]. These encode
                user intentions and guidance for segmentation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - processed_prompts: Refined prompt embeddings with shape 
                  [B, N_prompts, embedding_dim] containing image-aware prompt representations
                - processed_image: Updated image embeddings with shape 
                  [B, H*W, embedding_dim] containing prompt-aware visual features
                  
        Processing Flow:
        1. Reshape image embeddings from spatial to sequence format
        2. Apply stacked two-way attention blocks for iterative refinement
        3. Perform final prompt-to-image attention for comprehensive alignment
        4. Return refined embeddings for mask generation
        
        The bidirectional attention allows prompts to gather relevant visual context
        while image features are modulated by user intent, resulting in highly
        accurate and contextually appropriate segmentation masks.
        """
        # Convert image embeddings from spatial to sequence format
        # Transform from [B, C, H, W] to [B, H*W, C] for transformer processing
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Initialize queries and keys for attention operations
        # Queries: prompt embeddings that will attend to image features
        # Keys: image embeddings that will be attended to by prompts
        queries = point_embedding  # Prompt tokens seeking visual information
        keys = image_embedding     # Image tokens providing visual context

        # Apply transformer blocks with bidirectional attention
        # Each block refines both prompt and image representations
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # Original prompt positions (unchanged)
                key_pe=image_pe,          # Image spatial positions
            )

        # Final attention layer: comprehensive prompt-to-image alignment
        # This ensures prompt tokens have fully integrated visual context
        q = queries + point_embedding  # Add positional info to refined queries
        k = keys + image_pe           # Add positional info to refined keys
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        
        # Apply residual connection and normalization
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """
    Single transformer block implementing bidirectional attention between prompts and image.
    
    This is the core building block of the TwoWayTransformer, implementing a sophisticated
    attention mechanism that allows information to flow in both directions between sparse
    prompt tokens and dense image tokens. Each block consists of four main operations:
    
    1. Self-Attention on Prompts: Prompt tokens attend to each other
    2. Cross-Attention (Prompts → Image): Prompts gather visual information  
    3. MLP Processing: Non-linear transformation of prompt features
    4. Cross-Attention (Image → Prompts): Image features incorporate prompt context
    
    This bidirectional design enables:
    - Prompt refinement based on visual context
    - Image feature modulation based on user intent
    - Iterative improvement through multiple blocks
    - Rich interaction between different modalities
    
    The architecture follows the principle that both prompt and image representations
    should be mutually informative, leading to better segmentation accuracy.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Initialize a two-way attention block with specified parameters.

        Args:
            embedding_dim (int): Channel dimension of embeddings. Must be consistent
                across all components for proper tensor operations.
            num_heads (int): Number of attention heads. More heads allow the model
                to capture different types of relationships between tokens.
            mlp_dim (int): Hidden dimension in the MLP block. Typically 2-4x the
                embedding dimension for optimal expressivity.
            activation (nn.Module): Activation function in MLP. ReLU is standard
                for stability, GELU for smoother gradients.
            attention_downsample_rate (int): Factor to reduce internal attention
                dimension for computational efficiency.
            skip_first_layer_pe (bool): Whether to skip positional encoding in
                the first attention operation. Used to avoid double-adding PE.
        """
        super().__init__()
        
        # Step 1: Self-attention for prompt tokens to interact with each other
        # This allows different prompts (e.g., multiple points) to coordinate
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Step 2: Cross-attention from prompts to image features
        # Prompts gather relevant visual information from the image
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Step 3: MLP processing for prompt feature transformation
        # Non-linear transformation to increase model expressivity
        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        # Step 4: Cross-attention from image features to prompt tokens
        # Image features incorporate prompt context for better alignment
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        # Flag to control positional encoding usage in the first layer
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply four-stage bidirectional attention between prompts and image features.
        
        This method implements the core two-way attention computation that enables
        sophisticated interaction between user prompts and visual features.
        
        Args:
            queries (Tensor): Prompt token embeddings [B, N_prompts, C]
            keys (Tensor): Image token embeddings [B, N_image, C] 
            query_pe (Tensor): Positional encoding for prompts [B, N_prompts, C]
            key_pe (Tensor): Positional encoding for image [B, N_image, C]
            
        Returns:
            Tuple[Tensor, Tensor]: Updated (queries, keys) after bidirectional attention
            
        Four-Stage Process:
        1. Self-attention: Prompts attend to each other for coordination
        2. Cross-attention: Prompts gather visual context from image
        3. MLP: Non-linear transformation of prompt features
        4. Reverse cross-attention: Image features incorporate prompt context
        """
        
        # Stage 1: Self-attention among prompt tokens
        # This allows different prompts to coordinate and share information
        # Skip PE in first layer to avoid double-adding positional information
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            # Add positional encoding to queries for spatial awareness
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out  # Residual connection
        queries = self.norm1(queries)

        # Stage 2: Cross-attention from prompts to image features
        # Prompts attend to image to gather relevant visual information
        q = queries + query_pe  # Prompt positions
        k = keys + key_pe      # Image positions
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out  # Residual connection
        queries = self.norm2(queries)

        # Stage 3: MLP processing of prompt features
        # Non-linear transformation to increase representational capacity
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out  # Residual connection
        queries = self.norm3(queries)

        # Stage 4: Cross-attention from image features to prompt tokens
        # Image features attend to prompts to incorporate user intent
        # Note: q and k are swapped to reverse the attention direction
        q = queries + query_pe  # Updated prompt positions
        k = keys + key_pe      # Image positions
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out  # Update image features with prompt context
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    Multi-head attention layer with optional downsampling for computational efficiency.
    
    This is a flexible attention implementation that supports:
    - Standard multi-head attention computation
    - Optional dimension downsampling to reduce memory usage
    - Different input dimensions for keys/values vs queries
    - Dropout regularization during training
    - Hardware-optimized attention kernels (Flash Attention)
    
    The downsampling capability is particularly useful for processing high-resolution
    image features, allowing the model to maintain reasonable computational costs
    while preserving important attention patterns.
    
    Mathematical Operation:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    Where Q, K, V are projected from input embeddings and optionally downsampled.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        """
        Initialize the attention layer with specified parameters.
        
        Args:
            embedding_dim (int): Dimension of input embeddings and output
            num_heads (int): Number of attention heads. Must divide internal_dim evenly
            downsample_rate (int): Factor by which to reduce internal computation
                dimension. Higher values reduce memory but may impact capacity
            dropout (float): Dropout probability for attention weights
            kv_in_dim (int): Input dimension for keys and values if different from
                embedding_dim. Useful for cross-attention scenarios
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        # Allow different input dimensions for keys/values (useful for cross-attention)
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        # Reduce internal dimension for computational efficiency
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        
        # Ensure heads divide internal dimension evenly
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide internal_dim evenly."

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        # Output projection to restore original embedding dimension
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """
        Reshape tensor to separate attention heads for parallel computation.
        
        Args:
            x: Input tensor [B, N, C]
            num_heads: Number of attention heads
            
        Returns:
            Reshaped tensor [B, num_heads, N, C_per_head]
            
        This reshaping allows each attention head to operate on a subset of
        the feature dimensions, enabling the model to attend to different
        types of relationships simultaneously.
        """
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """
        Recombine attention heads back into single tensor.
        
        Args:
            x: Multi-head tensor [B, num_heads, N, C_per_head]
            
        Returns:
            Combined tensor [B, N, C]
            
        This operation concatenates the outputs from all attention heads,
        allowing the model to leverage information from different attention
        patterns simultaneously.
        """
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Compute multi-head attention with optional hardware optimization.
        
        Args:
            q: Query tensor [B, N_q, C]
            k: Key tensor [B, N_k, C_kv]  
            v: Value tensor [B, N_v, C_kv]
            
        Returns:
            Attention output [B, N_q, C]
            
        Implementation Details:
        - Projects inputs to internal dimension for efficiency
        - Separates into multiple attention heads
        - Uses optimized attention kernels when available
        - Provides graceful fallback for compatibility
        - Applies dropout during training only
        """
        # Project inputs to internal dimension
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into multiple attention heads for parallel processing
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply dropout only during training
        dropout_p = self.dropout_p if self.training else 0.0
        
        # Compute attention with hardware optimization
        try:
            # Use optimized attention kernels when available (Flash Attention)
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Graceful fallback if optimized kernels fail
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            # Enable all available kernels for compatibility
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Recombine attention heads and project to output dimension
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """
    Enhanced attention layer with Rotary Position Encoding (RoPE) for better spatial understanding.
    
    RoPE is an advanced positional encoding technique that directly encodes position
    information into the attention computation by rotating query and key vectors.
    This approach provides several advantages over traditional additive position encodings:
    
    Benefits of RoPE:
    - Better relative position modeling
    - Improved extrapolation to longer sequences
    - More robust spatial understanding for 2D data
    - Maintains rotation equivariance properties
    
    This is particularly beneficial for vision tasks where spatial relationships
    are crucial for accurate segmentation and object understanding.
    
    The implementation supports:
    - 2D positional encoding for image features
    - Flexible frequency computation for different resolutions
    - Optional key repetition for cross-attention scenarios
    - Partial RoPE application (excluding certain tokens)
    """

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        """
        Initialize RoPE attention with rotary position encoding.
        
        Args:
            rope_theta (float): Base frequency for rotary encoding. Lower values
                provide finer angular resolution but may reduce long-range modeling
            rope_k_repeat (bool): Whether to repeat query RoPE frequencies to match
                key length. Needed for cross-attention to memory features
            feat_sizes (tuple): Expected feature map dimensions [width, height]
                used to precompute rotary frequencies
            *args, **kwargs: Additional arguments passed to parent Attention class
        """
        super().__init__(*args, **kwargs)

        # Create function to compute rotary frequencies for 2D coordinates
        self.compute_cis = partial(
            compute_axial_cis, 
            dim=self.internal_dim // self.num_heads,  # Per-head dimension
            theta=rope_theta  # Base frequency for rotation
        )
        
        # Precompute rotary frequencies for expected feature size
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        """
        Compute attention with rotary position encoding applied to queries and keys.
        
        Args:
            q: Query tensor [B, N_q, C]
            k: Key tensor [B, N_k, C_kv]
            v: Value tensor [B, N_v, C_kv]
            num_k_exclude_rope: Number of key tokens to exclude from RoPE
                (e.g., special tokens that don't represent spatial positions)
                
        Returns:
            Attention output with enhanced spatial understanding [B, N_q, C]
            
        Implementation Flow:
        1. Project inputs and separate into heads
        2. Compute rotary frequencies for current sequence length
        3. Apply rotary encoding to queries and spatial keys
        4. Compute attention with spatially-aware representations
        5. Recombine heads and project to output
        """
        # Project inputs to internal dimension and separate heads
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Compute rotary position encoding
        # Assume square feature maps for simplicity
        w = h = math.sqrt(q.shape[-2])
        
        # Ensure frequencies are on the correct device
        self.freqs_cis = self.freqs_cis.to(q.device)
        
        # Recompute frequencies if sequence length doesn't match
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
            
        # Handle different sequence lengths between queries and keys
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat, "rope_k_repeat must be True for different q/k lengths"

        # Apply rotary encoding to spatial tokens only
        # Some tokens (like special embeddings) may not represent spatial positions
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        # Compute attention with spatially-enhanced representations
        dropout_p = self.dropout_p if self.training else 0.0
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fallback for compatibility
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Recombine heads and project to output dimension
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
