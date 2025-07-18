# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
PromptEncoder for SAM2 (Segment Anything Model 2)

This module implements the prompt encoding component of SAM2, which is responsible for
converting various types of user inputs (prompts) into embeddings that can be processed
by the mask decoder. The encoder handles three main types of prompts:

1. Point prompts: Individual points with positive/negative labels for inclusion/exclusion
2. Box prompts: Bounding boxes that define regions of interest
3. Mask prompts: Dense pixel-wise masks that provide detailed spatial information

The encoder produces two types of outputs:
- Sparse embeddings: For point and box prompts (discrete spatial locations)
- Dense embeddings: For mask prompts (continuous spatial information)

This design allows SAM2 to handle flexible user interactions while maintaining
computational efficiency through appropriate representations.
"""

from typing import Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom

from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    """
    Encodes different types of prompts (points, boxes, masks) into embeddings for SAM2.
    
    This encoder is a crucial component that bridges user interactions with the model's
    internal representations. It transforms human-interpretable prompts into high-dimensional
    embeddings that the mask decoder can process to generate accurate segmentation masks.
    
    Key Design Principles:
    - Unified interface for heterogeneous prompt types
    - Positional encoding to preserve spatial relationships
    - Separate pathways for sparse (point/box) and dense (mask) prompts
    - Learnable embeddings for different prompt semantics
    """
    
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Initialize the PromptEncoder with the specified configuration.

        Arguments:
          embed_dim (int): The prompts' embedding dimension. This should match the
            dimension expected by the mask decoder for consistency.
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding from the image encoder, as (H, W). Used for creating
            appropriately sized positional encodings.
          input_image_size (tuple(int, int)): The padded size of the image as input
            to the image encoder, as (H, W). Used for coordinate normalization.
          mask_in_chans (int): The number of hidden channels used for encoding
            input masks. Controls the capacity of the mask encoding pathway.
          activation (nn.Module): The activation function to use when encoding
            input masks. Typically GELU for better gradient flow.
        """
        super().__init__()
        # Store core dimensions for coordinate transformations and embedding sizing
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        
        # Positional encoding layer for spatial awareness
        # Uses random Fourier features to encode 2D coordinates
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Point embedding setup
        # We support 4 types of point embeddings:
        # 0: Negative point (background/exclude)
        # 1: Positive point (foreground/include) 
        # 2: Top-left corner of bounding box
        # 3: Bottom-right corner of bounding box
        self.num_point_embeddings: int = 4
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        
        # Special embedding for padding tokens (when no point is provided)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Mask encoding pathway
        # Input masks are typically high-resolution, so we need to downsample
        # them to match the spatial resolution of image embeddings
        self.mask_input_size = (
            4 * image_embedding_size[0],  # Masks are 4x the image embedding size
            4 * image_embedding_size[1],
        )
        
        # Convolutional downsampling network for mask encoding
        # This progressively reduces spatial resolution while increasing channel depth
        # to create a dense embedding that matches the image embedding spatial size
        self.mask_downscaling = nn.Sequential(
            # First downsampling: reduce spatial size by 2x, increase channels
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),  # Normalize for stable training
            activation(),
            # Second downsampling: further reduce spatial size, increase channels
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            # Final projection: map to embedding dimension
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        
        # Default embedding when no mask is provided
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Generate dense positional encoding for the entire image embedding space.
        
        This creates a positional encoding tensor that covers every spatial location
        in the image embedding. It's used to provide spatial context for dense
        operations like mask processing.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
            
        Note: The positional encoding helps the model understand spatial relationships
        and is crucial for maintaining spatial coherence in the generated masks.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor, # Indicates the coordinates of the points
        labels: torch.Tensor, # Indicates if the point is foreground or background
        pad: bool,
    ) -> torch.Tensor:
        """
        Convert point coordinates and labels into embeddings.
        
        This method handles the encoding of discrete point prompts, which are the
        most common type of user interaction in interactive segmentation tasks.
        
        Args:
            points (torch.Tensor): Point coordinates in format [batch, num_points, 2]
                where the last dimension contains (x, y) coordinates
            labels (torch.Tensor): Point labels in format [batch, num_points]
                where 0=negative point, 1=positive point, -1=padding
            pad (bool): Whether to add padding points. Used when no box prompts
                are provided to maintain consistent tensor dimensions
                
        Returns:
            torch.Tensor: Point embeddings with shape [batch, num_points, embed_dim]
            
        Implementation Details:
        - Coordinates are shifted by 0.5 to represent pixel centers rather than corners
        - Positional encoding provides spatial context for each point
        - Different learnable embeddings are added based on point labels
        - Padding points (label=-1) receive special "not a point" embeddings
        """
        # Shift coordinates to pixel centers for more accurate spatial representation
        points = points + 0.5
        
        # Add padding points if needed(to keep the sequence length constant)
        if pad:
            # Create zero-coordinate padding points
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            # Label padding points with -1 to distinguish them from real points
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
            
        # Generate positional encoding based on spatial coordinates
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        
        # Apply label-specific embeddings to encode point semantics
        # Start by zeroing out padding points
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        
        # Add semantic embeddings based on point labels
        point_embedding[labels == 0] += self.point_embeddings[0].weight  # Negative points
        point_embedding[labels == 1] += self.point_embeddings[1].weight  # Positive points
        point_embedding[labels == 2] += self.point_embeddings[2].weight  # Box corners (top-left)
        point_embedding[labels == 3] += self.point_embeddings[3].weight  # Box corners (bottom-right)
        
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert bounding box coordinates into embeddings.
        
        Bounding boxes are represented as two corner points (top-left and bottom-right),
        which are then encoded similarly to point prompts but with specific semantic
        labels to distinguish them from regular point clicks.
        
        Args:
            boxes (torch.Tensor): Box coordinates in format [batch, num_boxes, 4]
                where the last dimension contains (x1, y1, x2, y2) coordinates
                
        Returns:
            torch.Tensor: Box embeddings with shape [batch, num_boxes*2, embed_dim]
                Each box is represented as two corner point embeddings
                
        Implementation Details:
        - Boxes are decomposed into top-left and bottom-right corner points
        - Each corner gets positional encoding based on its coordinates
        - Corner-specific learnable embeddings distinguish box corners from point clicks
        - This representation allows the model to understand both box extent and semantics
        """
        # Shift coordinates to pixel centers
        boxes = boxes + 0.5
        
        # Reshape boxes into corner point format: [batch, num_boxes, 2, 2]
        # where the last two dimensions represent [corner_idx, (x,y)]
        coords = boxes.reshape(-1, 2, 2)
        
        # Generate positional encoding for corner coordinates
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        
        # Add semantic embeddings to distinguish corner types
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # Top-left corner
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # Bottom-right corner
        
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Convert input masks into dense embeddings.
        
        This method processes dense mask prompts through a convolutional encoder
        to create spatial embeddings that capture detailed pixel-wise information.
        
        Args:
            masks (torch.Tensor): Input masks with shape [batch, 1, H, W]
                where H and W match self.mask_input_size
                
        Returns:
            torch.Tensor: Dense mask embeddings with shape 
                [batch, embed_dim, embedding_H, embedding_W]
                
        Implementation Details:
        - Uses convolutional downsampling to match image embedding spatial resolution
        - Preserves fine-grained spatial information through progressive downsampling
        - Layer normalization and activation functions ensure stable training
        - Output spatial dimensions match image embeddings for proper fusion
        """
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Determine the batch size from the provided prompts.
        
        Since prompts are optional and can come in different combinations,
        this method examines the available inputs to determine the batch size
        for creating appropriately sized output tensors.
        
        Args:
            points: Optional tuple of (coordinates, labels) tensors
            boxes: Optional bounding box tensor
            masks: Optional mask tensor
            
        Returns:
            int: The batch size determined from available prompts
            
        Priority order: points > boxes > masks > default (1)
        This ensures consistent batch size extraction across different prompt types.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        """
        Get the device of the model parameters.
        
        Returns:
            torch.device: The device where model parameters are located
            
        This ensures that newly created tensors are placed on the same device
        as the model parameters, maintaining device consistency.
        """
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multiple types of prompts into sparse and dense embeddings.
        
        This is the main forward pass that processes all available prompts and
        generates embeddings suitable for the mask decoder. The method handles
        the complex logic of combining different prompt types while maintaining
        proper tensor dimensions and device placement.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or None): Tuple containing:
            - coordinates: Point coordinates with shape [batch, num_points, 2] 
            - labels: Point labels with shape [batch, num_points]
              where 0=negative, 1=positive
          boxes (torch.Tensor or None): Bounding boxes with shape [batch, num_boxes, 4]
            in format (x1, y1, x2, y2)
          masks (torch.Tensor or None): Input masks with shape [batch, 1, H, W]

        Returns:
          Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - sparse_embeddings: Embeddings for points and boxes with shape
              [batch, num_sparse_prompts, embed_dim]. These represent discrete
              spatial locations and their associated semantics.
            - dense_embeddings: Embeddings for masks with shape
              [batch, embed_dim, embed_H, embed_W]. These capture dense spatial
              information across the entire image region.
              
        Implementation Flow:
        1. Determine batch size from available prompts
        2. Initialize empty sparse embeddings tensor
        3. Process point prompts if available and concatenate to sparse embeddings
        4. Process box prompts if available and concatenate to sparse embeddings  
        5. Process mask prompts if available, otherwise use default mask embedding
        6. Return both sparse and dense embeddings for decoder processing
        
        Design Rationale:
        - Sparse embeddings capture discrete spatial interactions (clicks, boxes)
        - Dense embeddings capture continuous spatial information (masks)
        - This dual representation allows flexible prompt combination
        - Consistent tensor shapes enable efficient batch processing
        """
        # Determine batch size from available prompts
        bs = self._get_batch_size(points, boxes, masks)
        
        # Initialize sparse embeddings tensor for points and boxes
        # Start with empty tensor and concatenate prompt embeddings as available
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        
        # Process point prompts if provided
        if points is not None:
            coords, labels = points
            # Encode points with padding if no boxes are provided
            # This ensures consistent tensor dimensions across different prompt combinations
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
            
        # Process box prompts if provided
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # Process mask prompts or use default mask embedding
        if masks is not None:
            # Use provided masks for dense spatial information
            dense_embeddings = self._embed_masks(masks)
        else:
            # Create default "no mask" embedding when masks are not provided
            # This maintains consistent output format and provides baseline spatial context
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
