# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_track_anything.modeling.efficienttam_utils import LayerNorm2d



class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
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
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
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


