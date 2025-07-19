# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM2 (Segment Anything Model 2) Library Initialization

SAM2 is an advanced computer vision model that extends the original Segment Anything Model (SAM)
to handle both images and videos with sophisticated memory mechanisms. This library provides
unified APIs for image segmentation and video object tracking/segmentation.

Key Features:
- Zero-shot and few-shot segmentation capabilities
- Video object tracking with temporal consistency
- Memory-based architecture for multi-frame understanding
- Support for various prompt types (points, boxes, masks)
- Efficient inference with memory attention mechanisms

The library uses Hydra for configuration management, enabling flexible model instantiation
and parameter tuning through YAML configuration files.

Main Components:
- SAM2Base: Core model architecture with memory mechanisms
- SAM2ImagePredictor: Interface for single image segmentation
- SAM2VideoPredictor: Interface for video sequence processing
- Various backbone encoders (Hiera, ViT, etc.)
- Memory attention and encoding modules
- Transformer-based mask decoder

Usage:
    from sam2 import build_sam2, SAM2ImagePredictor
    
    # Build model from config
    sam2_model = build_sam2("sam2_hiera_tiny.yaml", "sam2_hiera_tiny.pt")
    
    # Create predictor
    predictor = SAM2ImagePredictor(sam2_model)
"""

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

# Initialize Hydra configuration system for SAM2
# This enables automatic loading of configuration files from the sam2/configs directory
# and provides a unified interface for model instantiation with different parameters
if not GlobalHydra.instance().is_initialized():
    # Initialize with SAM2's configuration module, using Hydra version 1.2 as base
    # This allows the library to automatically discover and load YAML config files
    # for different model variants (tiny, small, base, large) and training settings
    initialize_config_module("sam2", version_base="1.2")
