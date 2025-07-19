# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM2 Modeling Package - Core Neural Network Architecture Components

This package contains the complete neural network architecture implementation for SAM2
(Segment Anything Model 2), organized into logical modules that handle different aspects
of the segmentation and video tracking pipeline.

Package Organization:

**Core Architecture (sam2_base.py, efficienttam_base.py)**:
- SAM2Base: Main model class with memory mechanisms for video understanding
- EfficientTAMBase: Optimized variant for efficient inference and deployment
- Integration of image encoding, memory attention, and mask decoding components

**SAM Components (sam/ directory)**:
- mask_decoder.py: Transformer-based mask generation with hypernetwork architecture  
- prompt_encoder.py: Multi-modal prompt processing (points, boxes, masks)
- transformer.py: Two-way attention mechanism for prompt-image interaction

**Memory Systems (memory_*.py)**:
- memory_attention.py: Temporal attention mechanisms for video consistency
- memory_encoder.py: Compression and encoding of frame information for memory bank

**Backbone Networks (backbones/ directory)**:
- image_encoder.py: Vision transformer variants for image feature extraction
- hieradet.py: Hierarchical attention backbone for multi-scale processing
- vitdet.py: Vision transformer with detection-specific modifications

**Position Encoding (position_encoding.py)**:
- Sinusoidal, random Fourier, and rotary position encoding implementations
- Spatial awareness for transformer-based architectures
- Support for 2D vision tasks and relative position modeling

**Utilities (sam2_utils.py, efficienttam_utils.py)**:
- Common neural network components and helper functions
- Activation functions, normalization layers, and architectural primitives
- Optimization utilities and computational efficiency helpers

Design Philosophy:

The modeling package follows key architectural principles:

1. **Modularity**: Clear separation of concerns with well-defined interfaces
2. **Reusability**: Shared components across different model variants  
3. **Efficiency**: Optimized implementations for both training and inference
4. **Flexibility**: Configurable architectures supporting diverse use cases
5. **Scalability**: Support for different model sizes and computational budgets

Key Features:

- **Memory-Augmented Architecture**: Novel temporal modeling for video understanding
- **Multi-Scale Processing**: Hierarchical feature extraction and attention mechanisms
- **Interactive Design**: Optimized for real-time user interaction and feedback
- **Quality Prediction**: Integrated confidence estimation for mask outputs
- **Multi-Object Support**: Simultaneous tracking and segmentation of multiple objects

The package enables SAM2's core capabilities:
- Zero-shot and interactive image segmentation
- Robust video object tracking with temporal consistency  
- Multi-modal prompt understanding (clicks, boxes, masks)
- Real-time inference with memory-efficient processing
- Scalable architecture supporting various deployment scenarios

Usage:
    The modeling package is typically used through the high-level predictor interfaces,
    but individual components can be imported for research, customization, or integration
    into other systems.
    
    Example:
        from sam2.modeling.sam2_base import SAM2Base
        from sam2.modeling.sam.mask_decoder import MaskDecoder
        from sam2.modeling.backbones.image_encoder import ImageEncoder
"""
