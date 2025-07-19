# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Memory Attention Mechanism for SAM2 Video Object Tracking

This module implements the memory attention system that enables SAM2 to maintain
temporal consistency across video frames. The memory attention mechanism allows
the model to selectively attend to relevant information from past frames while
processing the current frame, creating robust object tracking capabilities.

Key Components:

1. **MemoryAttentionLayer**: Individual transformer layer with memory cross-attention
   - Self-attention on current frame features  
   - Cross-attention to memory bank from previous frames
   - Feed-forward processing with residual connections
   - Flexible position encoding integration

2. **MemoryAttention**: Stack of memory attention layers
   - Multiple layers for hierarchical temporal reasoning
   - Batch processing with optional format conversion
   - Object pointer token support for multi-object tracking
   - Configurable position encoding strategies

Memory Architecture:

The memory attention operates on two types of inputs:
- **Current Frame Features**: Features from the frame being processed
- **Memory Bank**: Compressed representations from previous frames

The attention mechanism enables the model to:
- Maintain object appearance consistency across time
- Handle temporal occlusions and re-appearances  
- Adapt to gradual appearance changes
- Distinguish between multiple objects in the same scene

Applications in SAM2:
- Video object segmentation with temporal consistency
- Multi-object tracking with separate memory banks
- Interactive video editing with memory-based propagation
- Robust handling of object appearance changes over time
"""

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):
    """
    Single layer of memory attention for temporal feature processing.
    
    This layer implements a transformer block specifically designed for video
    understanding, combining self-attention on current frame features with
    cross-attention to a memory bank of past frame information. The architecture
    enables robust temporal modeling while maintaining computational efficiency.
    
    Architecture:
    1. **Self-Attention**: Processes spatial relationships within current frame
    2. **Memory Cross-Attention**: Integrates information from previous frames  
    3. **Feed-Forward Network**: Non-linear feature transformation
    4. **Residual Connections**: Gradient flow and training stability
    5. **Layer Normalization**: Feature stabilization and convergence
    
    The layer supports flexible position encoding strategies, allowing position
    information to be injected at different stages of the attention computation
    for optimal spatial-temporal understanding.
    """

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        """
        Initialize memory attention layer with configurable components.
        
        Args:
            activation (str): Activation function name for feed-forward network.
                            Common choices: 'relu', 'gelu', 'swish'
                            
            cross_attention (nn.Module): Cross-attention module for memory integration.
                                       Typically RoPEAttention for spatial awareness.
                                       
            d_model (int): Model dimension for feature representations.
                          Must match input feature dimensions.
                          
            dim_feedforward (int): Hidden dimension in feed-forward network.
                                 Usually 2-4x larger than d_model.
                                 
            dropout (float): Dropout probability for regularization.
                           Applied to attention outputs and feed-forward.
                           
            pos_enc_at_attn (bool): Whether to add position encoding to self-attention.
                                  True: Enhanced spatial awareness in current frame.
                                  False: Pure content-based self-attention.
                                  
            pos_enc_at_cross_attn_keys (bool): Whether to add position encoding to memory keys.
                                             True: Position-aware memory retrieval.
                                             False: Content-only memory access.
                                             
            pos_enc_at_cross_attn_queries (bool): Whether to add position encoding to queries.
                                                 True: Spatial-aware memory queries.
                                                 False: Content-based memory queries.
                                                 
            self_attention (nn.Module): Self-attention module for current frame processing.
                                      Usually standard or RoPE attention.
        """
        super().__init__()
        
        # Store configuration parameters
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        
        # Attention modules
        self.self_attn = self_attention          # Current frame self-attention
        self.cross_attn_image = cross_attention  # Memory cross-attention

        # Feed-forward network for non-linear transformation
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization for stable training
        self.norm1 = nn.LayerNorm(d_model)  # Pre-self-attention norm
        self.norm2 = nn.LayerNorm(d_model)  # Pre-cross-attention norm  
        self.norm3 = nn.LayerNorm(d_model)  # Pre-feed-forward norm
        
        # Dropout for regularization at each residual connection
        self.dropout1 = nn.Dropout(dropout)  # Self-attention dropout
        self.dropout2 = nn.Dropout(dropout)  # Cross-attention dropout
        self.dropout3 = nn.Dropout(dropout)  # Feed-forward dropout

        # Activation function setup
        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Position encoding configuration flags
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        """
        Forward pass for self-attention on current frame features.
        
        This method processes spatial relationships within the current frame,
        allowing features to interact and refine their representations based
        on spatial context. Optional position encoding enhances spatial awareness.
        
        Args:
            tgt (torch.Tensor): Current frame features to process
            query_pos (torch.Tensor): Position encodings for spatial awareness
            
        Returns:
            torch.Tensor: Self-attended features with residual connection
        """
        # Pre-normalization for stable training
        tgt2 = self.norm1(tgt)
        
        # Add position encoding if configured
        # For self-attention, queries and keys come from the same source
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        
        # Apply self-attention to model spatial relationships
        tgt2 = self.self_attn(q, k, v=tgt2)
        
        # Residual connection with dropout for regularization
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        """
        Forward pass for cross-attention between current frame and memory.
        
        This method integrates information from the memory bank into current
        frame features, enabling temporal consistency and object tracking
        across video frames. The cross-attention allows selective retrieval
        of relevant past information.
        
        Args:
            tgt (torch.Tensor): Current frame features (queries)
            memory (torch.Tensor): Memory bank features (keys and values)
            query_pos (torch.Tensor): Position encoding for current frame
            pos (torch.Tensor): Position encoding for memory features
            num_k_exclude_rope (int): Number of keys to exclude from RoPE encoding
                                    Used for object pointer tokens that don't need
                                    spatial position encoding
            
        Returns:
            torch.Tensor: Memory-enhanced features with residual connection
        """
        # Prepare keyword arguments for RoPE attention if needed
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Pre-normalization for stable training
        tgt2 = self.norm2(tgt)
        
        # Cross-attention between current frame (queries) and memory (keys/values)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,  # Values always come from memory without position encoding
            **kwds,
        )
        
        # Residual connection with dropout
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """
        Complete forward pass through memory attention layer.
        
        Processes current frame features through self-attention, integrates
        memory information via cross-attention, and applies feed-forward
        transformation. The three-stage architecture enables both spatial
        and temporal reasoning.
        
        Args:
            tgt (torch.Tensor): Current frame features to process
            memory (torch.Tensor): Memory bank from previous frames
            pos (torch.Tensor, optional): Position encoding for memory features
            query_pos (torch.Tensor, optional): Position encoding for current features
            num_k_exclude_rope (int): Number of memory keys to exclude from RoPE
            
        Returns:
            torch.Tensor: Enhanced features combining current frame and memory information
        """
        # Stage 1: Self-attention on current frame for spatial reasoning
        tgt = self._forward_sa(tgt, query_pos)
        
        # Stage 2: Cross-attention to memory for temporal integration  
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        
        # Stage 3: Feed-forward network for non-linear transformation
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt


class MemoryAttention(nn.Module):
    """
    Multi-layer memory attention module for robust temporal modeling.
    
    This class stacks multiple MemoryAttentionLayer modules to create a deep
    architecture capable of complex temporal reasoning. The multi-layer design
    enables hierarchical processing where early layers capture basic temporal
    patterns and deeper layers model complex object interactions over time.
    
    Key Features:
    - **Hierarchical Processing**: Multiple layers for complex temporal patterns
    - **Batch Processing**: Efficient handling of multiple frames/objects
    - **Format Flexibility**: Automatic handling of sequence-first vs batch-first
    - **Object Pointer Support**: Special handling for multi-object tracking tokens
    - **Position Encoding Control**: Flexible position encoding strategies
    
    The module serves as the core temporal reasoning component in SAM2's video
    processing pipeline, enabling the model to maintain object consistency
    and handle complex temporal dynamics in video sequences.
    """
    
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,
    ):
        """
        Initialize multi-layer memory attention module.
        
        Args:
            d_model (int): Model dimension for all feature representations.
                          Must be consistent across all layers.
                          
            pos_enc_at_input (bool): Whether to add position encoding at input.
                                   True: Inject spatial information early.
                                   False: Rely on attention-level position encoding.
                                   
            layer (nn.Module): Template layer to replicate for the stack.
                             Usually a configured MemoryAttentionLayer.
                             
            num_layers (int): Number of attention layers in the stack.
                            More layers enable more complex temporal reasoning.
                            
            batch_first (bool): Expected input format for efficient processing.
                              True: Input shape (batch, sequence, features).
                              False: Input shape (sequence, batch, features).
        """
        super().__init__()
        self.d_model = d_model
        
        # Create stack of identical attention layers
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        
        # Final normalization for stable outputs
        self.norm = nn.LayerNorm(d_model)
        
        # Configuration flags
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # Current frame features for self-attention
        memory: torch.Tensor,  # Memory bank features for cross-attention
        curr_pos: Optional[Tensor] = None,  # Position encoding for current frame
        memory_pos: Optional[Tensor] = None,  # Position encoding for memory
        num_obj_ptr_tokens: int = 0,  # Number of object pointer tokens
    ):
        """
        Process current frame features with memory attention.
        
        This method applies the full stack of memory attention layers to integrate
        current frame features with memory bank information. The processing enables
        robust temporal consistency and object tracking across video frames.
        
        Args:
            curr (torch.Tensor): Current frame features with shape:
                               (batch, seq_len, d_model) if batch_first=True
                               (seq_len, batch, d_model) if batch_first=False
                               
            memory (torch.Tensor): Memory bank features with same format as curr.
                                 Contains compressed information from previous frames.
                                 
            curr_pos (torch.Tensor, optional): Position encodings for current frame.
                                             Should match curr's spatial dimensions.
                                             
            memory_pos (torch.Tensor, optional): Position encodings for memory features.
                                               Should match memory's spatial dimensions.
                                               
            num_obj_ptr_tokens (int): Number of object pointer tokens in memory.
                                    These tokens represent object-specific information
                                    and may need special handling in RoPE attention.
                                    
        Returns:
            torch.Tensor: Enhanced current frame features after memory integration.
                         Same shape as input curr tensor.
                         
        Processing Flow:
        1. Handle input format conversion if needed
        2. Add position encoding at input level if configured
        3. Apply all memory attention layers sequentially  
        4. Apply final normalization
        5. Convert back to original format
        """
        # Handle legacy list input format (backward compatibility)
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        # Validate batch size consistency between current and memory
        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        # Initialize output with current frame features
        output = curr
        
        # Add position encoding at input level if configured
        # Small scaling factor (0.1) prevents overwhelming the features
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        # Convert to batch-first format if needed for efficient processing
        if self.batch_first:
            output = output.transpose(0, 1)      # seq, batch -> batch, seq
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        # Apply all memory attention layers sequentially
        for layer in self.layers:
            # Prepare special arguments for RoPE attention layers
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                # Exclude object pointer tokens from RoPE position encoding
                # since they represent semantic rather than spatial information
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            # Process through current attention layer
            output = layer(
                tgt=output,            # Current frame features (queries)
                memory=memory,         # Memory bank (keys and values)
                pos=memory_pos,        # Memory position encodings
                query_pos=curr_pos,    # Current frame position encodings
                **kwds,
            )
            
        # Apply final layer normalization for stable outputs
        normed_output = self.norm(output)

        # Convert back to original format if needed
        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)  # batch, seq -> seq, batch
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
