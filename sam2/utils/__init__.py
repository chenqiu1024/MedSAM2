# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM2 Utilities Package - Supporting Infrastructure and Helper Functions

This package provides essential utility functions, data processing tools, and helper
components that support the core SAM2 functionality. The utilities are organized into
logical modules that handle different aspects of the system infrastructure.

Package Contents:

**amg.py** - Automatic Mask Generation Utilities:
- Batch processing functions for generating masks across entire images
- Grid-based point sampling strategies for comprehensive coverage
- Post-processing utilities for mask refinement and filtering
- Stability scoring and mask quality assessment tools
- Efficient algorithms for handling large-scale mask generation

**misc.py** - Miscellaneous Utilities and Helper Functions:
- General-purpose functions used across the SAM2 codebase
- Data type conversions and tensor manipulation utilities
- Coordinate transformation and normalization functions
- I/O operations for loading and saving various data formats
- Common mathematical operations and statistical computations

**transforms.py** - Data Transformation and Preprocessing:
- Image preprocessing pipelines for model input preparation
- Coordinate space transformations between different reference frames
- Mask post-processing including hole filling and noise removal
- Normalization and scaling operations for consistent data handling
- Augmentation utilities for training and data preparation

Key Functionality Areas:

**Data Processing**:
- Efficient tensor operations and batch processing
- Multi-format data loading and conversion utilities
- Memory-optimized operations for large-scale processing

**Image and Mask Operations**:
- Geometric transformations and coordinate mappings
- Mask post-processing for improved quality and consistency
- Image normalization and preprocessing for model compatibility

**Performance Optimization**:
- Vectorized operations for improved computational efficiency
- Memory management utilities for handling large datasets
- Parallel processing support for multi-threaded operations

**Quality Assessment**:
- Mask stability scoring and confidence estimation
- Statistical analysis tools for performance evaluation
- Debugging and visualization helper functions

Design Principles:

1. **Efficiency**: Optimized implementations using vectorized operations
2. **Modularity**: Well-defined functions with clear responsibilities
3. **Reusability**: Generic utilities usable across different components
4. **Robustness**: Error handling and edge case management
5. **Performance**: Memory-efficient operations for large-scale processing

Usage Patterns:

The utilities package supports common workflows in SAM2:
- Preprocessing image data for model input
- Post-processing model outputs for user consumption
- Batch processing for large-scale applications
- Quality assessment and performance evaluation
- Data format conversions and I/O operations

Integration:

These utilities integrate seamlessly with:
- SAM2 core modeling components for data pipeline support
- Predictor interfaces for preprocessing and post-processing
- Training scripts for data augmentation and preparation
- Evaluation tools for performance assessment and analysis

The package provides the foundation for robust, efficient, and scalable
SAM2 applications across diverse use cases and deployment scenarios.
"""
