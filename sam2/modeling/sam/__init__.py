# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM Core Components - Segment Anything Model Architecture

This module contains the core architectural components that implement the original
Segment Anything Model (SAM) functionality within the SAM2 framework. These components
handle the fundamental segmentation capabilities that form the foundation of SAM2's
extended video and interactive features.

Core Components:

**mask_decoder.py** - Mask Generation Engine:
- Transformer-based decoder that converts embeddings to segmentation masks
- Hypernetwork architecture for dynamic mask generation
- Multi-mask output strategy for handling ambiguous prompts
- Integrated quality prediction (IoU) for mask confidence estimation
- Progressive upsampling for high-resolution mask generation

**prompt_encoder.py** - User Interaction Interface:
- Unified encoding of diverse prompt types (points, boxes, masks)
- Position-aware embeddings for spatial understanding
- Separate pathways for sparse (point/box) and dense (mask) prompts
- Flexible interface supporting iterative refinement workflows

**transformer.py** - Bidirectional Attention Mechanism:
- Two-way transformer enabling prompt-image feature interaction
- Bidirectional attention flow for optimal feature alignment
- RoPE (Rotary Position Encoding) support for spatial awareness
- Efficient attention computation with downsampling options
- Flash Attention optimization for improved performance

Architecture Philosophy:

The SAM components implement a unified framework for interactive segmentation:

1. **User Intent Capture**: Prompt encoder translates user interactions into embeddings
2. **Feature Interaction**: Two-way transformer aligns prompts with image features  
3. **Mask Generation**: Decoder produces high-quality masks with confidence scores
4. **Quality Assessment**: Integrated prediction helps users understand mask quality

Key Design Principles:

- **Interactive First**: Optimized for real-time user interaction and feedback
- **Quality Aware**: Built-in mechanisms for mask quality assessment and selection
- **Flexible Prompting**: Support for diverse interaction modalities
- **Efficient Processing**: Optimized attention mechanisms and feature representations
- **Extensible Design**: Clean interfaces enabling integration with video components

Integration with SAM2:

These core SAM components integrate seamlessly with SAM2's extended capabilities:
- Memory mechanisms for temporal consistency in video
- Multi-object tracking through enhanced attention systems
- Advanced position encoding for spatial-temporal understanding
- Efficient inference optimizations for real-time applications

The components maintain backward compatibility with original SAM while providing
enhanced functionality and performance optimizations for the SAM2 ecosystem.
"""
