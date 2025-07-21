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
        
        # Create stack of two-way attention blocks following transformer design principles
        # Each block performs bidirectional attention between prompts and image features,
        # enabling sophisticated interaction between user intent and visual content.
        # 
        # Architecture inspired by Vision Transformer (arXiv:2010.11929) self-attention
        # but extended with cross-attention for prompt-image interaction. The depth
        # parameter controls the number of refinement iterations, with more layers
        # allowing more sophisticated prompt-image alignment at higher computational cost.
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
                    # since input embeddings may already contain positional information
                    skip_first_layer_pe=(i == 0),
                )
            )

        # Final attention layer for comprehensive prompt-to-image alignment
        # This critical layer ensures that prompt tokens have fully integrated visual context
        # before mask generation. Following the MAE asymmetric design principle (arXiv:2111.06377),
        # this final attention focuses computational resources on the most important alignment
        # between refined prompt representations and visual features.
        # 
        # The final attention serves as a "readout" mechanism where prompt tokens gather
        # their final visual context, similar to how MAE's decoder attends to encoder features.
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        debug_name: str = None,
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
            debug_name (str, optional): Name for debug state capture.

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
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input embeddings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="input_image_embedding",
                data=image_embedding,
                metadata={'component_type': 'two_way_transformer', 'stage': 'input', 'tensor_type': 'image_embedding'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="input_image_pe",
                data=image_pe,
                metadata={'component_type': 'two_way_transformer', 'stage': 'input', 'tensor_type': 'image_pe'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="input_point_embedding",
                data=point_embedding,
                metadata={'component_type': 'two_way_transformer', 'stage': 'input', 'tensor_type': 'point_embedding'}
            )
        
        # Convert image embeddings from spatial to sequence format
        # Transform from [B, C, H, W] to [B, H*W, C] for transformer processing
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Debug capture for reshaped embeddings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="reshaped_image_embedding",
                data=image_embedding,
                metadata={'component_type': 'two_way_transformer', 'stage': 'reshape', 'tensor_type': 'image_embedding'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="reshaped_image_pe",
                data=image_pe,
                metadata={'component_type': 'two_way_transformer', 'stage': 'reshape', 'tensor_type': 'image_pe'}
            )

        # Initialize queries and keys for attention operations
        # Queries: prompt embeddings that will attend to image features
        # Keys: image embeddings that will be attended to by prompts
        queries = point_embedding  # Prompt tokens seeking visual information
        keys = image_embedding     # Image tokens providing visual context

        # Apply transformer blocks with bidirectional attention
        # Each block refines both prompt and image representations
        for i, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # Original prompt positions (unchanged)
                key_pe=image_pe,          # Image spatial positions
                debug_name=f"{debug_name or 'two_way_transformer'}_layer_{i}" if debug_name else None,
            )
            
            # Debug capture for layer outputs
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "two_way_transformer",
                    state_name=f"layer_{i}_queries_output",
                    data=queries,
                    metadata={'component_type': 'two_way_transformer', 'stage': f'layer_{i}_output', 'tensor_type': 'queries'}
                )
                capture_debug_state(
                    component_name=debug_name or "two_way_transformer",
                    state_name=f"layer_{i}_keys_output",
                    data=keys,
                    metadata={'component_type': 'two_way_transformer', 'stage': f'layer_{i}_output', 'tensor_type': 'keys'}
                )

        # Final attention layer: comprehensive prompt-to-image alignment
        # This ensures prompt tokens have fully integrated visual context
        q = queries + point_embedding  # Add positional info to refined queries
        k = keys + image_pe           # Add positional info to refined keys
        
        # Debug capture for final attention inputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="final_attention_queries",
                data=q,
                metadata={'component_type': 'two_way_transformer', 'stage': 'final_attention_input', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="final_attention_keys",
                data=k,
                metadata={'component_type': 'two_way_transformer', 'stage': 'final_attention_input', 'tensor_type': 'keys'}
            )
        
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys, debug_name=f"{debug_name or 'two_way_transformer'}_final_attn" if debug_name else None)
        
        # Apply residual connection and normalization
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        # Debug capture for final outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="final_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_transformer', 'stage': 'final_output', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="final_keys_output",
                data=keys,
                metadata={'component_type': 'two_way_transformer', 'stage': 'final_output', 'tensor_type': 'keys'}
            )

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
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, debug_name: str = None
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
            debug_name (str, optional): Name for debug state capture.
            
        Returns:
            Tuple[Tensor, Tensor]: Updated (queries, keys) after bidirectional attention
            
        Four-Stage Process:
        1. Self-attention: Prompts attend to each other for coordination
        2. Cross-attention: Prompts gather visual context from image
        3. MLP: Non-linear transformation of prompt features
        4. Reverse cross-attention: Image features incorporate prompt context
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input tensors
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="input_queries",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'input', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="input_keys",
                data=keys,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'input', 'tensor_type': 'keys'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="input_query_pe",
                data=query_pe,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'input', 'tensor_type': 'query_pe'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="input_key_pe",
                data=key_pe,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'input', 'tensor_type': 'key_pe'}
            )
        
        # Stage 1: Self-attention among prompt tokens
        # This allows different prompts to coordinate and share information
        # Skip PE in first layer to avoid double-adding positional information
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries, debug_name=f"{debug_name or 'two_way_attention_block'}_self_attn" if debug_name else None)
        else:
            # Add positional encoding to queries for spatial awareness
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries, debug_name=f"{debug_name or 'two_way_attention_block'}_self_attn" if debug_name else None)
            queries = queries + attn_out  # Residual connection
        queries = self.norm1(queries)

        # Debug capture after stage 1
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="stage1_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'stage1_output', 'tensor_type': 'queries'}
            )

        # Stage 2: Cross-attention from prompts to image features
        # Prompts attend to image to gather relevant visual information
        q = queries + query_pe  # Prompt positions
        k = keys + key_pe      # Image positions
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, debug_name=f"{debug_name or 'two_way_attention_block'}_cross_attn_token_to_image" if debug_name else None)
        queries = queries + attn_out  # Residual connection
        queries = self.norm2(queries)

        # Debug capture after stage 2
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="stage2_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'stage2_output', 'tensor_type': 'queries'}
            )

        # Stage 3: MLP processing of prompt features
        # Non-linear transformation to increase representational capacity
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out  # Residual connection
        queries = self.norm3(queries)

        # Debug capture after stage 3
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="stage3_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'stage3_output', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="stage3_mlp_output",
                data=mlp_out,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'stage3_mlp', 'tensor_type': 'mlp_output'}
            )

        # Stage 4: Cross-attention from image features to prompt tokens
        # Image features attend to prompts to incorporate user intent
        # Note: q and k are swapped to reverse the attention direction
        q = queries + query_pe  # Updated prompt positions
        k = keys + key_pe      # Image positions
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries, debug_name=f"{debug_name or 'two_way_attention_block'}_cross_attn_image_to_token" if debug_name else None)
        keys = keys + attn_out  # Update image features with prompt context
        keys = self.norm4(keys)

        # Debug capture for final outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="final_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'final_output', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="final_keys_output",
                data=keys,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'final_output', 'tensor_type': 'keys'}
            )

        return queries, keys


class Attention(nn.Module):
    """
    Multi-head attention layer with computational optimizations for vision applications.
    
    This implementation extends the standard attention mechanism from Vision Transformers
    (arXiv:2010.11929) with several enhancements for efficient processing of high-resolution
    image features and cross-modal prompt-image interactions.
    
    Core Mathematical Operation (from "Attention Is All You Need"):
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    Where:
    - Q (queries): What information is being sought
    - K (keys): What information is available for matching
    - V (values): The actual information content to be retrieved
    - d_k: Key dimension for scaling (prevents saturation in softmax)
    
    Multi-Head Extension (from ViT paper):
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    This allows the model to attend to different representation subspaces
    simultaneously, capturing various types of relationships (e.g., spatial,
    semantic, geometric) in parallel.
    
    Key Features for SAM2:
    - Computational downsampling: Reduces memory from O(N²) to manageable levels
    - Cross-modal support: Different input dimensions for prompt vs image features
    - Hardware optimization: Automatic kernel selection (Flash Attention, etc.)
    - Flexible projections: Adaptable to various feature dimensions
    - Dropout regularization: Prevents overfitting during training
    
    The downsampling capability is crucial for processing high-resolution features
    (e.g., 64x64 = 4096 tokens) without computational explosion, while maintaining
    the essential attention patterns needed for accurate segmentation.
    
    Applications in SAM2:
    - Prompt-to-image cross-attention for gathering visual context
    - Image-to-prompt cross-attention for incorporating user intent
    - Self-attention within prompt tokens for coordination
    - Memory attention for temporal consistency in video processing
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

    def forward(self, q: Tensor, k: Tensor, v: Tensor, debug_name: str = None) -> Tensor:
        """
        Compute multi-head attention with optional hardware optimization.
        
        Args:
            q: Query tensor [B, N_q, C]
            k: Key tensor [B, N_k, C_kv]  
            v: Value tensor [B, N_v, C_kv]
            debug_name: Optional name for debug state capture
            
        Returns:
            Attention output [B, N_q, C]
            
        Implementation Details:
        - Projects inputs to internal dimension for efficiency
        - Separates into multiple attention heads
        - Uses optimized attention kernels when available
        - Provides graceful fallback for compatibility
        - Applies dropout during training only
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input tensors
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="input_queries",
                data=q,
                metadata={'component_type': 'attention', 'stage': 'input', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="input_keys",
                data=k,
                metadata={'component_type': 'attention', 'stage': 'input', 'tensor_type': 'keys'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="input_values",
                data=v,
                metadata={'component_type': 'attention', 'stage': 'input', 'tensor_type': 'values'}
            )
        
        # Project inputs to internal dimension
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Debug capture for projected tensors
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="projected_queries",
                data=q,
                metadata={'component_type': 'attention', 'stage': 'projection', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="projected_keys",
                data=k,
                metadata={'component_type': 'attention', 'stage': 'projection', 'tensor_type': 'keys'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="projected_values",
                data=v,
                metadata={'component_type': 'attention', 'stage': 'projection', 'tensor_type': 'values'}
            )

        # Separate into multiple attention heads for parallel processing
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Debug capture for multi-head tensors
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="multihead_queries",
                data=q,
                metadata={'component_type': 'attention', 'stage': 'multihead', 'tensor_type': 'queries', 'num_heads': self.num_heads}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="multihead_keys",
                data=k,
                metadata={'component_type': 'attention', 'stage': 'multihead', 'tensor_type': 'keys', 'num_heads': self.num_heads}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="multihead_values",
                data=v,
                metadata={'component_type': 'attention', 'stage': 'multihead', 'tensor_type': 'values', 'num_heads': self.num_heads}
            )

        # Apply dropout only during training
        dropout_p = self.dropout_p if self.training else 0.0
        
        # Compute attention weights manually for debug capture if needed
        if debug_name and is_debug_enabled():
            # Compute attention scores manually for debugging
            scale = (q.shape[-1]) ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Capture attention weights
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="attention_scores",
                data=attn_scores,
                metadata={'component_type': 'attention', 'stage': 'attention_computation', 'tensor_type': 'scores'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="attention_weights",
                data=attn_weights,
                metadata={'component_type': 'attention', 'stage': 'attention_computation', 'tensor_type': 'weights', 'num_heads': self.num_heads}
            )
        
        # Compute attention with hardware-optimized kernels for maximum efficiency
        # This uses PyTorch's scaled_dot_product_attention which automatically selects
        # the most efficient implementation based on hardware capabilities:
        # 
        # Flash Attention: Memory-efficient attention for modern GPUs
        # - Reduces memory usage from O(N²) to O(N) through recomputation
        # - Significantly faster on A100, H100, and similar architectures
        # - Essential for processing high-resolution images and long sequences
        # 
        # Math Kernel: Standard attention implementation
        # - Most compatible across different hardware
        # - Required for older GPUs or when dropout is used during training
        # - Provides numerical stability guarantees
        # 
        # Memory Efficient: Optimized for memory-constrained scenarios
        # - Useful for inference on devices with limited GPU memory
        # - Balances speed and memory usage
        try:
            # Use optimized attention kernels with automatic selection
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Graceful fallback ensures compatibility across all hardware
            # This is crucial for deployment in diverse environments
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            # Enable all available kernels for maximum compatibility
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Debug capture for attention output
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="attention_output_multihead",
                data=out,
                metadata={'component_type': 'attention', 'stage': 'attention_output', 'tensor_type': 'multihead_output'}
            )

        # Recombine attention heads and project to output dimension
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        # Debug capture for final output
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="final_output",
                data=out,
                metadata={'component_type': 'attention', 'stage': 'final_output', 'tensor_type': 'output'}
            )

        return out


class RoPEAttention(Attention):
    """
    Enhanced attention layer with Rotary Position Encoding (RoPE) for superior spatial understanding.
    
    RoPE represents a significant advancement in positional encoding, originally proposed for
    language models but particularly effective for vision tasks. Unlike traditional additive
    position encodings used in Vision Transformers (arXiv:2010.11929), RoPE directly modifies
    the attention computation by rotating query and key vectors in complex space.
    
    Theoretical Foundation:
    RoPE encodes position information by rotating feature vectors using:
    q_m = R_m * q, k_n = R_n * k, where R_θ is a rotation matrix
    This preserves relative position information: <q_m, k_n> depends only on (m-n)
    
    Benefits over traditional Position Encoding (from ViT paper):
    - Better relative position modeling: Natural encoding of spatial relationships
    - Improved extrapolation: Works with sequence lengths not seen during training
    - More robust spatial understanding: Direct integration into attention computation
    - Rotation equivariance: Maintains consistency under coordinate transformations
    - No additive interference: Doesn't compete with content representations
    
    Applications in SAM2:
    - Spatial relationship modeling between image patches (similar to ViT improvements)
    - Enhanced cross-attention for prompt-image alignment
    - Better handling of multi-scale features in Feature Pyramid Networks
    - Improved memory attention for temporal consistency (SAM2 innovation)
    
    This is crucial for video segmentation where spatial-temporal relationships
    determine object tracking accuracy and consistency across frames.
    
    Implementation Features:
    - 2D coordinate system support for image feature maps
    - Flexible frequency computation for different input resolutions
    - Optional key repetition for cross-attention to memory features
    - Partial application (excluding non-spatial tokens like object pointers)
    - Hardware-optimized computation with fallback compatibility
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
