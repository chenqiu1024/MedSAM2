# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Image Encoder for SAM2 - Vision Transformer Backbone with Feature Pyramid Network

This module implements the core image encoding pipeline for SAM2, combining the power
of Vision Transformers (ViT) for feature extraction with Feature Pyramid Networks (FPN)
for multi-scale representation. The design is heavily influenced by foundational work:

1. Vision Transformer (ViT) - "An Image Is Worth 16x16 Words" (arXiv:2010.11929)
   - Revolutionized computer vision by applying transformers to image patches
   - Treats images as sequences of patches processed through self-attention
   - Achieves superior performance through global receptive fields
   - SAM2 uses hierarchical ViT variants for dense prediction tasks

2. Feature Pyramid Networks - Hierarchical feature representation
   - Multi-scale feature extraction essential for object detection/segmentation
   - Combines high-resolution spatial details with high-level semantic information
   - Enables handling objects at different scales within the same image
   - Critical for SAM2's ability to segment objects of varying sizes

3. Masked Autoencoders (MAE) influence - "Masked Autoencoders Are Scalable Vision Learners" (arXiv:2111.06377)
   - Asymmetric encoder-decoder design philosophy
   - Heavy encoder (ViT backbone) for rich feature extraction
   - Efficient processing through hierarchical feature selection
   - SAM2 adopts similar principles for computational efficiency

Architectural Components:

**Trunk (ViT Backbone)**: 
- Processes input images through patch embedding and transformer blocks
- Generates hierarchical features at multiple resolutions (stride 4, 8, 16, 32)
- Uses self-attention to capture global spatial relationships
- Provides rich semantic representations for downstream processing

**Neck (Feature Pyramid Network)**:
- Combines multi-scale features through lateral connections and top-down pathways
- Projects backbone features to consistent channel dimensions
- Adds positional encodings for spatial awareness
- Enables efficient multi-scale processing in SAM decoder

**Integration Benefits**:
- Global context from ViT attention mechanisms
- Multi-scale spatial understanding from FPN design
- Efficient feature representation for real-time video processing
- Robust feature extraction across diverse image content and scales

This encoder serves as the foundation for SAM2's visual understanding, providing
the rich feature representations needed for accurate segmentation and temporal tracking.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_track_anything.modeling.efficienttam_utils import LayerNorm2d


class ImageEncoder(nn.Module):
    """
    Hierarchical image encoder combining Vision Transformer backbone with Feature Pyramid Network.
    
    This class orchestrates the complete image encoding pipeline for SAM2, consisting of:
    
    1. **ViT Trunk**: Hierarchical Vision Transformer for feature extraction
       - Processes images through patch embedding and transformer blocks
       - Generates multi-scale features through hierarchical attention
       - Provides global spatial understanding through self-attention mechanisms
    
    2. **FPN Neck**: Feature Pyramid Network for multi-scale feature fusion
       - Combines features across different scales through lateral connections
       - Adds top-down pathways for feature enhancement
       - Projects features to consistent dimensionality for downstream processing
    
    3. **Position Encoding**: Spatial position information injection
       - Provides spatial awareness to transformer-based downstream modules
       - Enables relative position understanding crucial for segmentation
    
    The encoder follows the hierarchical design principle from ViT while incorporating
    FPN's multi-scale processing capabilities, resulting in rich feature representations
    suitable for both image and video segmentation tasks.
    """
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        """
        Initialize the hierarchical image encoder with Vision Transformer backbone.
        
        Args:
            trunk (nn.Module): Vision Transformer backbone for feature extraction.
                             Typically a hierarchical ViT (e.g., Hiera, ViT-Det) that
                             generates multi-scale features through transformer blocks.
                             Following ViT design from arXiv:2010.11929 with adaptations
                             for dense prediction tasks.
                             
            neck (nn.Module): Feature Pyramid Network for multi-scale feature fusion.
                            Combines trunk features through lateral connections and
                            top-down pathways. Projects features to consistent dimensions
                            and adds positional encodings for spatial awareness.
                            
            scalp (int): Number of lowest-resolution feature levels to discard.
                       Used to reduce computational cost by removing the coarsest
                       features that may not be needed for the specific task.
                       Default 0 keeps all feature levels.
        """
        super().__init__()
        
        # Vision Transformer backbone following ViT architecture principles
        # Processes input images through patch embedding and transformer blocks
        # Generates hierarchical features at multiple scales for comprehensive understanding
        self.trunk = trunk
        
        # Feature Pyramid Network for multi-scale feature integration
        # Combines features across scales through lateral and top-down connections
        # Essential for handling objects of different sizes within the same image
        self.neck = neck
        
        # Feature level selection for computational efficiency
        # Allows discarding the coarsest features when not needed
        self.scalp = scalp
        
        # Validate consistency between trunk output and neck input dimensions
        # Ensures proper feature flow through the encoder pipeline
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor, debug_name: str = None):
        # Debug capture for input image
        if debug_name:
            from sam2.debug_utils import capture_debug_state, is_debug_enabled
            if is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "image_encoder",
                    state_name="input_image",
                    data=sample,
                    metadata={'component_type': 'image_encoder', 'stage': 'input'}
                )

        # Forward through backbone
        trunk_features = self.trunk(sample)
        
        # Debug capture for trunk features
        if debug_name:
            from sam2.debug_utils import capture_debug_state, is_debug_enabled
            if is_debug_enabled():
                if isinstance(trunk_features, (list, tuple)):
                    for i, feat in enumerate(trunk_features):
                        capture_debug_state(
                            component_name=debug_name or "image_encoder",
                            state_name=f"trunk_features_level_{i}",
                            data=feat,
                            metadata={'component_type': 'image_encoder', 'stage': 'trunk_output', 'level': i}
                        )
                else:
                    capture_debug_state(
                        component_name=debug_name or "image_encoder",
                        state_name="trunk_features",
                        data=trunk_features,
                        metadata={'component_type': 'image_encoder', 'stage': 'trunk_output'}
                    )
        
        features, pos = self.neck(trunk_features, debug_name=debug_name)
        
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        
        # Debug capture for final output features
        if debug_name:
            from sam2.debug_utils import capture_debug_state, is_debug_enabled
            if is_debug_enabled():
                # Capture final vision features
                capture_debug_state(
                    component_name=debug_name or "image_encoder",
                    state_name="vision_features_final",
                    data=src,
                    metadata={'component_type': 'image_encoder', 'stage': 'final_output'}
                )
                
                # Capture FPN features at all levels
                for i, feat in enumerate(features):
                    capture_debug_state(
                        component_name=debug_name or "image_encoder",
                        state_name=f"fpn_features_level_{i}",
                        data=feat,
                        metadata={'component_type': 'image_encoder', 'stage': 'fpn_output', 'level': i}
                    )
                
                # Capture position encodings
                for i, p in enumerate(pos):
                    capture_debug_state(
                        component_name=debug_name or "image_encoder",
                        state_name=f"position_encoding_level_{i}",
                        data=p,
                        metadata={'component_type': 'image_encoder', 'stage': 'position_encoding', 'level': i}
                    )

        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) Neck for Multi-Scale Feature Integration in SAM2
    
    This class implements a modified Feature Pyramid Network that serves as the "neck"
    component connecting the Vision Transformer backbone to the downstream SAM decoder.
    The FPN design is crucial for SAM2's ability to segment objects at different scales
    within the same image.
    
    **Theoretical Foundation:**
    Feature Pyramid Networks address the fundamental challenge in computer vision of
    handling objects at multiple scales. The key insight is that different network
    layers capture different levels of abstraction:
    - Early layers: High-resolution, low-level features (edges, textures)
    - Late layers: Low-resolution, high-level features (semantic content)
    
    **FPN Architecture:**
    1. **Lateral Connections**: Project each backbone level to common dimension
    2. **Top-Down Pathway**: Propagate high-level semantics to high-resolution levels
    3. **Feature Fusion**: Combine lateral and top-down features for rich representations
    
    **SAM2-Specific Modifications:**
    - Removes output convolutions for efficiency (features fed directly to transformer)
    - Uses bilinear interpolation consistent with ViT position embedding handling
    - Integrates position encoding generation for spatial awareness
    - Supports flexible feature level selection for computational optimization
    
    **Benefits for SAM2:**
    - Enables segmentation of objects at vastly different scales
    - Provides rich multi-scale features for the two-way transformer
    - Maintains spatial resolution needed for precise mask boundaries
    - Balances semantic understanding with spatial precision
    
    This FPN variant is specifically optimized for SAM2's transformer-based architecture,
    providing the multi-scale visual representations essential for accurate segmentation
    across diverse object sizes and image content.
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor], debug_name: str = None):
        from sam2.debug_utils import capture_debug_state, is_debug_enabled

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        
        # Debug capture for input features
        if debug_name and is_debug_enabled():
            for i, x in enumerate(xs):
                capture_debug_state(
                    component_name=debug_name or "fpn_neck",
                    state_name=f"input_features_level_{i}",
                    data=x,
                    metadata={'component_type': 'fpn_neck', 'stage': 'input', 'level': i}
                )
        
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            
            # Debug capture for lateral features
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "fpn_neck",
                    state_name=f"lateral_features_level_{i}",
                    data=lateral_features,
                    metadata={'component_type': 'fpn_neck', 'stage': 'lateral', 'level': i}
                )
            
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                
                # Debug capture for top-down features
                if debug_name and is_debug_enabled():
                    capture_debug_state(
                        component_name=debug_name or "fpn_neck",
                        state_name=f"top_down_features_level_{i}",
                        data=top_down_features,
                        metadata={'component_type': 'fpn_neck', 'stage': 'top_down', 'level': i}
                    )
                
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            
            x_out = prev_features
            out[i] = x_out
            
            # Generate position encoding with debug support
            if hasattr(self.position_encoding, 'forward') and 'debug_name' in self.position_encoding.forward.__code__.co_varnames:
                pos[i] = self.position_encoding(x_out, debug_name=f"{debug_name or 'fpn_neck'}_pos_enc_level_{i}").to(x_out.dtype)
            else:
                pos[i] = self.position_encoding(x_out).to(x_out.dtype)
            
            # Debug capture for output features and position encodings
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "fpn_neck",
                    state_name=f"output_features_level_{i}",
                    data=x_out,
                    metadata={'component_type': 'fpn_neck', 'stage': 'output', 'level': i}
                )
                capture_debug_state(
                    component_name=debug_name or "fpn_neck",
                    state_name=f"position_encoding_level_{i}",
                    data=pos[i],
                    metadata={'component_type': 'fpn_neck', 'stage': 'position_encoding', 'level': i}
                )

        return out, pos


class ViTDetNeck(nn.Module):
    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        neck_norm=None,
    ):
        """Initialize the neck

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.backbone_channel_list = backbone_channel_list
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.d_model = d_model
        use_bias = neck_norm is None
        for dim in self.backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_0", LayerNorm2d(d_model))
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_1", LayerNorm2d(d_model))
            self.convs.append(current)

    def forward(self, xs: List[torch.Tensor], debug_name: str = None):
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)

        x = xs[0]
        
        # Debug capture for input
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "vitdet_neck",
                state_name="input_features",
                data=x,
                metadata={'component_type': 'vitdet_neck', 'stage': 'input'}
            )
        
        x_out = self.convs[0](x)
        out[0] = x_out
        
        # Generate position encoding with debug support
        if hasattr(self.position_encoding, 'forward') and 'debug_name' in self.position_encoding.forward.__code__.co_varnames:
            pos[0] = self.position_encoding(x_out, debug_name=f"{debug_name or 'vitdet_neck'}_pos_enc").to(x_out.dtype)
        else:
            pos[0] = self.position_encoding(x_out).to(x_out.dtype)
        
        # Debug capture for output
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "vitdet_neck",
                state_name="output_features",
                data=x_out,
                metadata={'component_type': 'vitdet_neck', 'stage': 'output'}
            )
            capture_debug_state(
                component_name=debug_name or "vitdet_neck",
                state_name="position_encoding",
                data=pos[0],
                metadata={'component_type': 'vitdet_neck', 'stage': 'position_encoding'}
            )

        return out, pos


