# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM2 Model Builder and Factory Functions

This module provides factory functions to construct SAM2 models with different configurations
and for different use cases (image segmentation vs video tracking). It handles model instantiation,
checkpoint loading, device placement, and configuration management through Hydra.

Key Functions:
- build_sam2(): Creates base SAM2 model for image segmentation
- build_sam2_video_predictor(): Creates SAM2 video predictor for tracking
- build_sam2_video_predictor_npz(): Creates SAM2 video predictor with NPZ data support
- build_sam2_hf(): Creates SAM2 from Hugging Face model hub
- Utility functions for device detection and checkpoint loading

The module supports various SAM2 model variants:
- Tiny: Fastest inference, smallest memory footprint
- Small: Balanced speed and accuracy
- Base Plus: Higher accuracy with moderate compute
- Large: Best accuracy with highest compute requirements

Configuration is managed through YAML files that specify:
- Model architecture (backbone, decoder, memory components)
- Training hyperparameters
- Inference settings and post-processing options
"""

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Mapping of Hugging Face model IDs to their corresponding config files and checkpoint names
# This enables easy model loading from the Hugging Face model hub with automatic
# configuration and checkpoint selection for different model sizes and versions
HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def get_best_available_device():
    """
    Automatically detect and return the best available compute device.
    
    Checks for available compute devices in order of preference:
    1. CUDA (NVIDIA GPUs) - Best performance for large models
    2. MPS (Apple Silicon) - Good performance on Apple M1/M2 chips  
    3. CPU - Universal fallback, slower but always available
    
    Returns:
        str: Device string compatible with torch.device()
        
    Note:
        This function prioritizes CUDA over MPS as CUDA typically offers
        better performance for transformer-based models like SAM2.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_sam2(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    """
    Build a SAM2 model for image segmentation.
    
    This function creates a SAM2Base model instance configured for single image
    segmentation tasks. It loads the specified configuration, applies any overrides,
    loads pretrained weights, and sets up the model for inference or training.
    
    Args:
        config_file (str): Path to YAML configuration file (relative to configs/ directory)
                          Examples: "sam2_hiera_t.yaml", "sam2_hiera_l.yaml"
        ckpt_path (str, optional): Path to checkpoint file. If None, model uses random weights.
        device (str, optional): Target device ('cuda', 'mps', 'cpu'). Auto-detected if None.
        mode (str): Model mode, either 'eval' for inference or 'train' for training.
        hydra_overrides_extra (list): Additional Hydra configuration overrides.
        apply_postprocessing (bool): Whether to enable post-processing features like
                                   dynamic multi-mask selection based on stability.
        **kwargs: Additional arguments (currently unused).
        
    Returns:
        SAM2Base: Configured SAM2 model ready for image segmentation.
        
    Note:
        When apply_postprocessing=True, the model will automatically fall back to
        multi-mask output if the single mask prediction is unstable, improving
        robustness for ambiguous prompts.
    """
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    if apply_postprocessing:
        # Create a copy to avoid modifying the input list
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # Enable dynamic multi-mask prediction: if single mask is unstable,
            # automatically switch to multi-mask output for better robustness
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            # Stability delta threshold: minimum difference between best and second-best mask
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            # Stability threshold: minimum stability score to use single mask output
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    
    # Load configuration using Hydra and resolve any variable references
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    
    # Instantiate the model from configuration with recursive instantiation of subcomponents
    model = instantiate(cfg.model, _recursive_=True)
    
    # Load pretrained weights if checkpoint path is provided
    _load_checkpoint(model, ckpt_path)
    
    # Move model to target device
    model = model.to(device)
    
    # Set model mode (eval disables dropout, batch norm updates, etc.)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    """
    Build a SAM2 video predictor for video object tracking and segmentation.
    
    This function creates a SAM2VideoPredictor instance that extends the base SAM2
    model with video-specific functionality including memory management, temporal
    consistency, and efficient multi-frame processing.
    
    Args:
        config_file (str): Path to YAML configuration file.
        ckpt_path (str, optional): Path to checkpoint file.
        device (str, optional): Target device. Auto-detected if None.
        mode (str): Model mode ('eval' or 'train').
        hydra_overrides_extra (list): Additional Hydra configuration overrides.
        apply_postprocessing (bool): Whether to enable video-specific post-processing
                                   including mask binarization and hole filling.
        **kwargs: Additional arguments.
        
    Returns:
        SAM2VideoPredictor: Configured video predictor with memory mechanisms.
        
    Features:
        - Memory bank management for temporal consistency
        - Efficient multi-frame batch processing  
        - Automatic mask propagation across frames
        - Interactive refinement capabilities
        - Support for multiple object tracking
        
    Note:
        Video predictors include additional post-processing steps like hole filling
        and mask binarization to improve temporal consistency and user experience.
    """
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    # Override the target class to use SAM2VideoPredictor instead of SAM2Base
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # Enable dynamic multi-mask selection for stability
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # Binarize mask logits on user-interacted frames to ensure consistency
            # between what users see and what gets encoded in memory
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # Fill small holes in low-resolution masks before upsampling to improve quality
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model with video-specific overrides
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def build_sam2_video_predictor_npz(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    """
    Build a SAM2 video predictor with NPZ data format support.
    
    This variant of the video predictor is specifically designed to work with
    NPZ (NumPy compressed) data formats, commonly used in medical imaging and
    research applications where data is pre-processed and stored in NumPy arrays.
    
    Args:
        config_file (str): Path to YAML configuration file.
        ckpt_path (str, optional): Path to checkpoint file.
        device (str, optional): Target device. Auto-detected if None.
        mode (str): Model mode ('eval' or 'train').
        hydra_overrides_extra (list): Additional Hydra configuration overrides.
        apply_postprocessing (bool): Whether to enable post-processing features.
        **kwargs: Additional arguments.
        
    Returns:
        SAM2VideoPredictorNPZ: Video predictor optimized for NPZ data format.
        
    Use Cases:
        - Medical imaging workflows with pre-processed data
        - Research applications with standardized data formats
        - Batch processing of large datasets stored as NPZ files
        - Scientific computing workflows using NumPy ecosystem
        
    Note:
        The NPZ variant includes the same post-processing features as the standard
        video predictor but with optimized data loading and handling for NPZ formats.
    """
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    # Override target class to use NPZ-specific video predictor
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ",
    ]
    
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # Enable dynamic multi-mask selection for robust predictions
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # Ensure mask consistency in memory encoding for interactive sessions
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # Post-process masks by filling small holes for better visual quality
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model with NPZ-specific settings
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    """
    Download SAM2 model configuration and checkpoint from Hugging Face Hub.
    
    This helper function retrieves the appropriate configuration file and
    checkpoint for a given Hugging Face model ID, enabling easy access to
    pretrained SAM2 models without manual file management.
    
    Args:
        model_id (str): Hugging Face model identifier (e.g., "facebook/sam2-hiera-tiny")
        
    Returns:
        tuple: (config_name, ckpt_path) where:
            - config_name: Path to configuration YAML file
            - ckpt_path: Local path to downloaded checkpoint file
            
    Raises:
        KeyError: If model_id is not found in HF_MODEL_ID_TO_FILENAMES
        
    Note:
        The function uses huggingface_hub to download files and cache them locally
        for subsequent use, reducing download time for repeated model loading.
    """
    from huggingface_hub import hf_hub_download

    # Look up config and checkpoint filenames for the given model ID
    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    
    # Download checkpoint file and return local path
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    """
    Build SAM2 model directly from Hugging Face model hub.
    
    Convenience function that combines model download and instantiation in a single call.
    This is the easiest way to get started with pretrained SAM2 models.
    
    Args:
        model_id (str): Hugging Face model ID (e.g., "facebook/sam2-hiera-tiny")
        **kwargs: Additional arguments passed to build_sam2()
        
    Returns:
        SAM2Base: Loaded and configured SAM2 model.
        
    Example:
        >>> model = build_sam2_hf("facebook/sam2-hiera-tiny")
        >>> predictor = SAM2ImagePredictor(model)
    """
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    """
    Build SAM2 video predictor directly from Hugging Face model hub.
    
    Convenience function for creating video predictors with pretrained weights
    downloaded automatically from Hugging Face.
    
    Args:
        model_id (str): Hugging Face model ID
        **kwargs: Additional arguments passed to build_sam2_video_predictor()
        
    Returns:
        SAM2VideoPredictor: Loaded video predictor ready for tracking.
        
    Example:
        >>> predictor = build_sam2_video_predictor_hf("facebook/sam2-hiera-small")
        >>> predictor.init_state(video_path)
    """
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    """
    Load pretrained weights from checkpoint file into model.
    
    This function handles the loading of SAM2 checkpoints with proper error checking
    for missing or unexpected keys. It ensures that the model state exactly matches
    the checkpoint to prevent silent failures or parameter mismatches.
    
    Args:
        model (torch.nn.Module): The model instance to load weights into
        ckpt_path (str or None): Path to checkpoint file. If None, no loading occurs.
        
    Raises:
        RuntimeError: If there are missing or unexpected keys in the checkpoint,
                     indicating a mismatch between model architecture and saved weights.
                     
    Note:
        The function uses weights_only=True for security, preventing execution of
        arbitrary code that might be embedded in checkpoint files. Checkpoints are
        loaded to CPU first then moved to target device by the calling function.
    """
    if ckpt_path is not None:
        # Load checkpoint with security restrictions (weights only, no code execution)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        
        # Load state dict and check for parameter mismatches
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        
        # Raise errors for any parameter mismatches to ensure model integrity
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
            
        logging.info("Loaded checkpoint sucessfully")
