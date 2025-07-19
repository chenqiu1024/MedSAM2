# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM2 Image Predictor - Interactive Image Segmentation Interface

This module provides the SAM2ImagePredictor class, which offers a user-friendly interface
for interactive image segmentation using the SAM2 model. The predictor enables efficient
mask generation from various prompt types including points, boxes, and previous masks.

Key Features:
- Efficient image embedding computation and caching
- Support for multiple prompt types (points, boxes, masks)
- Batch processing capabilities for multiple images
- Interactive refinement through iterative prompting
- Automatic post-processing for improved mask quality
- Multi-mask output for handling ambiguous prompts

Workflow:
1. Initialize predictor with a SAM2 model
2. Set target image(s) to compute embeddings once
3. Predict masks using various prompts (multiple iterations possible)
4. Optionally refine results with additional prompts

The predictor caches image embeddings, enabling fast inference for multiple
prompt iterations on the same image without recomputing expensive features.

Example Usage:
    # Single image segmentation
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(point_coords=[[100, 200]], 
                                            point_labels=[1])
    
    # Batch processing
    predictor.set_image_batch([img1, img2, img3])
    all_masks, all_scores, all_logits = predictor.predict_batch(
        point_coords_batch=[coords1, coords2, coords3],
        point_labels_batch=[labels1, labels2, labels3]
    )
"""

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor:
    """
    Interactive image segmentation predictor using SAM2.
    
    This class provides a high-level interface for SAM2-based image segmentation.
    It handles image preprocessing, embedding computation, prompt processing,
    and mask post-processing, while maintaining an efficient caching system
    for repeated predictions on the same image.
    
    The predictor supports both single image and batch processing modes,
    with various prompt types including points, bounding boxes, and mask inputs.
    Multi-mask output capability helps handle ambiguous prompts by providing
    multiple segmentation hypotheses ranked by predicted quality scores.
    
    Architecture:
    - Image Encoder: Processes input images to extract dense feature representations
    - Prompt Encoder: Converts user prompts (points, boxes) into embeddings
    - Mask Decoder: Combines image and prompt features to generate masks
    - Post-processing: Applies thresholding, hole filling, and upsampling
    
    Performance Features:
    - Cached embeddings: Compute image features once, use for multiple predictions
    - Batch processing: Efficient handling of multiple images simultaneously
    - Memory optimization: Automatic cleanup and state management
    - Device management: Automatic GPU/CPU placement and tensor handling
    """
    
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Initialize the SAM2 image predictor.
        
        Sets up the predictor with a SAM2 model and configures post-processing
        parameters for mask refinement. The predictor maintains internal state
        for efficient multi-prompt inference on the same image.

        Args:
            sam_model (SAM2Base): The SAM2 model to use for segmentation.
                                 Should be pre-loaded with appropriate weights.
            mask_threshold (float): Threshold for converting mask logits to binary masks.
                                  Higher values produce more conservative masks.
                                  Default 0.0 uses the natural decision boundary.
            max_hole_area (int): Maximum area of holes to fill in low-resolution masks.
                                If > 0, small holes up to this area will be filled
                                to improve mask connectivity. Specified in pixels.
            max_sprinkle_area (int): Maximum area of isolated regions to remove.
                                   If > 0, small disconnected regions up to this area
                                   will be removed to reduce noise. Specified in pixels.
            **kwargs: Additional arguments (currently unused, for future extensibility).
        """
        super().__init__()
        self.model = sam_model
        
        # Initialize transform pipeline for image preprocessing and postprocessing
        # This handles resizing, normalization, coordinate transformations, and mask refinement
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state management
        self._is_image_set = False  # Flag indicating if image embeddings are cached
        self._features = None       # Cached image embeddings and high-res features
        self._orig_hw = None        # Original image dimensions for coordinate transforms
        self._is_batch = False      # Flag indicating batch vs single image mode

        # Predictor configuration
        self.mask_threshold = mask_threshold

        # Pre-compute spatial dimensions for backbone feature maps at different scales
        # SAM2 uses multi-scale features: the image backbone outputs features at multiple
        # resolutions for hierarchical processing and high-resolution mask generation
        hires_size = self.model.image_size // 4  # Base high-resolution size (e.g., 128 for 512 input)
        self._bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
        # This creates feature map sizes like: [[128, 128], [64, 64], [32, 32]]
        # These correspond to different levels of the feature pyramid used by the model

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Create a predictor from a pretrained model on Hugging Face Hub.
        
        This convenience method automatically downloads and loads a pretrained SAM2
        model from the Hugging Face model hub, then wraps it in a predictor interface.
        
        Args:
            model_id (str): Hugging Face model repository ID.
                          Examples: "facebook/sam2-hiera-tiny", "facebook/sam2-hiera-large"
            **kwargs: Additional arguments passed to model constructor and predictor.

        Returns:
            SAM2ImagePredictor: Ready-to-use predictor with loaded pretrained weights.
            
        Example:
            >>> predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
            >>> predictor.set_image(my_image)
            >>> masks, scores, logits = predictor.predict(point_coords=[[100, 200]], 
                                                        point_labels=[1])
        """
        from sam2.build_sam import build_sam2_hf

        # Download and instantiate the model from Hugging Face
        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Compute and cache embeddings for the provided image.
        
        This method processes the input image through the SAM2 image encoder to
        extract dense feature representations. The embeddings are cached for
        efficient repeated inference with different prompts on the same image.
        
        The image is automatically preprocessed (resized, normalized) to match
        the model's expected input format. Multi-scale features are extracted
        to support high-resolution mask generation.

        Args:
            image (np.ndarray or PIL.Image): Input image in RGB format.
                - np.ndarray: Expected shape (H, W, C) with values in [0, 255]
                - PIL.Image: Standard PIL image format
                
        Raises:
            NotImplementedError: If image format is not supported.
            
        Note:
            This method should be called once per image before any predictions.
            Subsequent predict() calls will use the cached embeddings for efficiency.
            The method automatically detects image format and handles preprocessing.
        """
        # Clear any previous state to prepare for new image
        self.reset_predictor()
        
        # Extract and store original image dimensions for coordinate transformations
        # These are needed to map between model input coordinates and original image coordinates
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]  # Store (height, width)
        elif isinstance(image, Image):
            w, h = image.size  # PIL Image.size returns (width, height)
            self._orig_hw = [(h, w)]  # Convert to (height, width) format
        else:
            raise NotImplementedError("Image format not supported")

        # Transform image to model input format (resize, normalize, tensorize)
        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)  # Add batch dimension and move to device

        # Validate input tensor format
        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        
        logging.info("Computing image embeddings for the provided image...")
        
        # Extract image features using the backbone encoder
        backbone_out = self.model.forward_image(input_image)
        
        # Prepare multi-scale features for the SAM decoder
        # This extracts features at different resolutions for hierarchical processing
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        
        # Add no-memory embedding for video training compatibility
        # During video training, a special embedding is added to indicate no previous memory
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # Reorganize features for decoder consumption
        # Features are permuted and reshaped to match expected decoder input format
        # The decoder expects features in decreasing resolution order
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        
        # Cache processed features for efficient repeated inference
        self._features = {
            "image_embed": feats[-1],      # Lowest resolution features for primary processing
            "high_res_feats": feats[:-1]   # Higher resolution features for detail refinement
        }
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Compute and cache embeddings for a batch of images.
        
        This method extends single image processing to handle multiple images
        simultaneously, enabling efficient batch inference. All images in the
        batch are processed together through the encoder for optimal GPU utilization.
        
        Batch processing is particularly useful for:
        - Processing large datasets efficiently
        - Leveraging GPU parallelism for better throughput
        - Maintaining consistent preprocessing across multiple images

        Args:
            image_list (List[np.ndarray]): List of input images in RGB format.
                All images should be np.ndarray with shape (H, W, C) and values in [0, 255].
                Images can have different sizes as they will be resized to model input size.
                
        Note:
            After calling this method, use predict_batch() for generating predictions
            rather than the single-image predict() method. The batch mode maintains
            separate feature caches for each image in the batch.
        """
        # Clear previous state and prepare for batch processing
        self.reset_predictor()
        assert isinstance(image_list, list)
        
        # Store original dimensions for each image in the batch
        # This is essential for proper coordinate transformations during prediction
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
            
        # Transform all images to model input format simultaneously
        # The batch transform handles resizing and normalization for all images
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        
        # Validate batch tensor format
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        
        logging.info("Computing image embeddings for the provided images...")
        
        # Process entire batch through backbone encoder
        backbone_out = self.model.forward_image(img_batch)
        
        # Extract and prepare multi-scale features for all images in batch
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        
        # Add no-memory embedding for video training compatibility
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # Reorganize features for batch decoder processing
        # Each feature level maintains the batch dimension for parallel processing
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        
        # Cache batch features with batch dimension preserved
        self._features = {
            "image_embed": feats[-1],      # Shape: (B, C, H, W)
            "high_res_feats": feats[:-1]   # List of tensors with shape (B, C, H_i, W_i)
        }
        self._is_image_set = True
        self._is_batch = True  # Flag to indicate batch mode for subsequent operations
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Generate mask predictions for a batch of images with corresponding prompts.
        
        This method performs segmentation on multiple images simultaneously, where each
        image can have different prompts. It's optimized for batch processing while
        maintaining per-image prompt flexibility.
        
        The method processes each image-prompt pair sequentially but leverages the
        pre-computed batch embeddings for efficiency. This approach balances
        computational efficiency with prompt flexibility.

        Args:
            point_coords_batch (List[np.ndarray], optional): List of point coordinate arrays.
                Each array has shape (N, 2) with points in (X, Y) format.
                One array per image in the batch.
            point_labels_batch (List[np.ndarray], optional): List of point label arrays.
                Each array has shape (N,) with labels: 1=foreground, 0=background.
                Must match point_coords_batch if provided.
            box_batch (List[np.ndarray], optional): List of bounding box arrays.
                Each array has shape (4,) in XYXY format: [x_min, y_min, x_max, y_max].
                One array per image in the batch.
            mask_input_batch (List[np.ndarray], optional): List of low-resolution mask inputs.
                Each array has shape (1, H, W) from previous predictions.
                Used for iterative refinement.
            multimask_output (bool): Whether to output multiple mask candidates.
                True (default) generates 3 masks for ambiguous prompts.
                False generates 1 mask for unambiguous prompts.
            return_logits (bool): Whether to return raw logits instead of binary masks.
                True returns continuous values for further processing.
                False returns thresholded binary masks.
            normalize_coords (bool): Whether to normalize coordinates to [0,1].
                True expects coordinates relative to image dimensions.
                False expects coordinates in model input space.

        Returns:
            Tuple of three lists, each containing results for all images:
            - masks (List[np.ndarray]): Binary masks or logits for each image.
                Each array has shape (C, H, W) where C is number of masks.
            - iou_predictions (List[np.ndarray]): Quality scores for each mask.
                Each array has shape (C,) with values in [0, 1].
            - low_res_masks (List[np.ndarray]): Low-resolution mask logits.
                Each array has shape (C, 256, 256) for iterative refinement.
                
        Raises:
            AssertionError: If not in batch mode or if image embeddings not set.
            
        Note:
            This method requires prior call to set_image_batch(). The number of
            prompt lists should match the number of images in the batch.
        """
        # Ensure we're in batch mode with cached embeddings
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
            
        # Process each image in the batch with its corresponding prompts
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        
        for img_idx in range(num_images):
            # Extract prompts for current image (handle None cases gracefully)
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            
            # Preprocess prompts for current image
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            
            # Generate predictions for current image using cached embeddings
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            
            # Convert tensors to numpy arrays for output
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            
            # Collect results for current image
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mask predictions for the currently set image using various prompts.
        
        This is the main inference method for single-image segmentation. It accepts
        multiple types of prompts (points, boxes, previous masks) and generates
        high-quality segmentation masks with confidence scores.
        
        The method supports iterative refinement: masks from previous predictions
        can be provided as mask_input for refinement with additional prompts.
        Multi-mask output helps handle ambiguous cases by providing alternatives.

        Args:
            point_coords (np.ndarray, optional): Point prompts as array of shape (N, 2).
                Each point is specified as [X, Y] in image coordinates.
                Positive points indicate foreground, negative indicate background.
            point_labels (np.ndarray, optional): Labels for point prompts, shape (N,).
                Values: 1 = foreground point, 0 = background point.
                Required if point_coords is provided.
            box (np.ndarray, optional): Bounding box prompt, shape (4,).
                Format: [x_min, y_min, x_max, y_max] in image coordinates.
                Provides strong spatial constraints for segmentation.
            mask_input (np.ndarray, optional): Low-resolution mask input, shape (1, H, W).
                Typically from previous prediction iteration (use low_res_masks output).
                Enables iterative refinement and temporal consistency.
            multimask_output (bool): Control number of output masks.
                True: Generate 3 masks for ambiguous prompts (default).
                False: Generate 1 mask for unambiguous prompts.
                Use quality scores to select best mask when True.
            return_logits (bool): Output format control.
                True: Return continuous logit values for further processing.
                False: Return binary masks after thresholding (default).
            normalize_coords (bool): Coordinate space interpretation.
                True: Coordinates relative to original image dimensions (default).
                False: Coordinates in model input space (advanced usage).

        Returns:
            Tuple of three arrays:
            - masks (np.ndarray): Output masks, shape (C, H, W) where C is number of masks.
                Values are binary (0/1) if return_logits=False, continuous if True.
                H, W match original image dimensions.
            - iou_predictions (np.ndarray): Quality scores, shape (C,).
                Values in [0, 1] indicating predicted IoU with ground truth.
                Higher scores indicate better quality masks.
            - low_res_masks (np.ndarray): Low-resolution logits, shape (C, 256, 256).
                Can be used as mask_input for subsequent refinement iterations.
                Provides efficient representation for iterative workflows.
                
        Raises:
            RuntimeError: If image embeddings haven't been computed via set_image().
            
        Example:
            >>> # Simple point-based segmentation
            >>> masks, scores, logits = predictor.predict(
            ...     point_coords=[[100, 200]],
            ...     point_labels=[1],
            ...     multimask_output=True
            ... )
            >>> 
            >>> # Select best mask based on quality score
            >>> best_mask = masks[np.argmax(scores)]
            >>> 
            >>> # Iterative refinement with additional point
            >>> refined_masks, _, new_logits = predictor.predict(
            ...     point_coords=[[100, 200], [150, 250]],
            ...     point_labels=[1, 1],
            ...     mask_input=logits[np.argmax(scores)],
            ...     multimask_output=False
            ... )
        """
        # Ensure image embeddings are available
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Preprocess and validate input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        # Generate predictions using the SAM2 model
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        # Convert output tensors to numpy arrays for user consumption
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        """
        Preprocess and validate input prompts for model consumption.
        
        This internal method handles the conversion of user-provided prompts
        (in various formats) into the standardized tensor format expected by
        the SAM2 model. It performs coordinate transformations, type conversions,
        and format validation.
        
        The method handles coordinate normalization, ensuring that prompts
        specified in original image coordinates are properly transformed to
        the model's input coordinate system. It also manages batch dimensions
        and device placement for optimal performance.

        Args:
            point_coords: Raw point coordinates from user input
            point_labels: Raw point labels from user input  
            box: Raw bounding box from user input
            mask_logits: Raw mask input from user input
            normalize_coords: Whether to apply coordinate normalization
            img_idx: Image index for batch processing (-1 for single image)

        Returns:
            Tuple of processed prompts ready for model consumption:
            - mask_input: Preprocessed mask tensor or None
            - unnorm_coords: Transformed point coordinates tensor or None
            - labels: Point labels tensor or None  
            - unnorm_box: Transformed box coordinates tensor or None
        """
        # Initialize outputs
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        
        # Process point prompts if provided
        if point_coords is not None:
            # Point labels are required when point coordinates are provided
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            
            # Convert to tensors and move to appropriate device
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            
            # Transform coordinates from image space to model input space
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            
            # Convert labels to tensor
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            
            # Ensure batch dimension is present for consistency
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
                
        # Process bounding box prompt if provided
        if box is not None:
            # Convert to tensor and move to device
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            
            # Transform box coordinates and reshape for model consumption
            # The model expects boxes in a specific format: Bx2x2 (batch, corners, xy)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            
        # Process mask input if provided (for iterative refinement)
        if mask_logits is not None:
            # Convert to tensor and ensure proper format
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            
            # Ensure 4D tensor format: (batch, channels, height, width)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
                
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal prediction method that interfaces with the SAM2 model.
        
        This method orchestrates the complete prediction pipeline using the SAM2
        architecture. It combines preprocessed prompts with cached image embeddings
        to generate segmentation masks through the model's encoder-decoder framework.
        
        The method handles prompt encoding, feature fusion, mask decoding, and
        post-processing to produce final segmentation results. It leverages the
        cached image embeddings for efficiency while supporting various prompt
        combinations and output configurations.

        Args:
            point_coords (torch.Tensor, optional): Transformed point coordinates, shape (B, N, 2).
            point_labels (torch.Tensor, optional): Point labels, shape (B, N).
            boxes (torch.Tensor, optional): Transformed bounding boxes, shape (B, 4).
            mask_input (torch.Tensor, optional): Low-res mask input, shape (B, 1, H, W).
            multimask_output (bool): Whether to generate multiple mask hypotheses.
            return_logits (bool): Whether to return continuous logits or binary masks.
            img_idx (int): Image index for batch processing.

        Returns:
            Tuple of prediction tensors:
            - masks (torch.Tensor): Output masks, shape (B, C, H, W).
            - iou_predictions (torch.Tensor): Quality scores, shape (B, C).
            - low_res_masks (torch.Tensor): Low-resolution logits, shape (B, C, 256, 256).
            
        Note:
            This method assumes image embeddings have been precomputed and cached.
            It directly interfaces with the SAM2 model components for maximum efficiency.
        """
        # Ensure image embeddings are available
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Combine point and box prompts into unified format
        # The SAM prompt encoder expects a single "points" input that can include
        # both point clicks and box corners (with appropriate labels)
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Process bounding box prompts by converting them to point format
        # Each box is represented as two corner points with special labels (2, 3)
        if boxes is not None:
            # Reshape box from (B, 4) to (B, 2, 2) for corner points
            box_coords = boxes.reshape(-1, 2, 2)
            # Create labels for box corners: 2 = top-left, 3 = bottom-right
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            
            # Merge box points with existing point prompts
            # Boxes are added at the beginning for consistent processing order
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        # Encode prompts into sparse and dense embeddings
        # The prompt encoder converts discrete prompts into continuous embeddings
        # that can be processed by the transformer decoder
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,  # Combined point and box prompts
            boxes=None,           # Boxes are handled through points
            masks=mask_input,     # Previous mask for iterative refinement
        )

        # Determine if we're in multi-object prediction mode
        # This affects how the decoder processes multiple prompts
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )

        # Extract high-resolution features for current image
        # These features provide fine-grained spatial information for detailed masks
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        # Generate masks using the SAM decoder
        # The decoder combines image embeddings, prompt embeddings, and high-res features
        # to produce accurate segmentation masks with quality predictions
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),  # Positional encoding
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,  # Handle multiple objects on same image
            high_res_features=high_res_features,
        )

        # Upscale masks to original image resolution
        # The decoder outputs low-resolution masks that need upsampling
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        
        # Clamp low-resolution masks to prevent numerical instability
        # This ensures stable gradients and consistent behavior
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        
        # Apply threshold for binary output if requested
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Retrieve the cached image embeddings for the currently set image.
        
        This method provides access to the dense feature representations computed
        by the image encoder. These embeddings capture semantic and spatial
        information about the image and serve as the foundation for mask generation.
        
        The embeddings can be useful for:
        - Advanced applications requiring direct feature access
        - Custom prompt processing workflows
        - Analysis of learned image representations
        - Integration with other vision models

        Returns:
            torch.Tensor: Image embeddings with shape (1, C, H, W) where:
                - C is the embedding dimension (typically 256)
                - H, W are spatial dimensions (typically 64x64 for 1024x1024 input)
                
        Raises:
            RuntimeError: If no image has been set via set_image().
            
        Note:
            These are the low-resolution embeddings used by the mask decoder.
            High-resolution features are also available internally but not exposed
            through this interface to maintain API simplicity.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        """
        Get the device (CPU/GPU) where the model and computations are located.
        
        This property provides convenient access to the model's device for
        ensuring tensor placement consistency and debugging purposes.
        
        Returns:
            torch.device: The device where the SAM2 model is located.
        """
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Clear all cached state and prepare for processing a new image.
        
        This method resets the predictor to its initial state, clearing cached
        image embeddings and resetting all internal flags. It should be called
        before processing a new image or when memory cleanup is needed.
        
        The reset includes:
        - Clearing cached image embeddings and features
        - Resetting image dimensions and preprocessing state  
        - Clearing batch mode flags
        - Preparing for fresh image processing
        
        Note:
            This method is automatically called by set_image() and set_image_batch(),
            so manual calls are typically not necessary unless explicit cleanup is desired.
        """
        self._is_image_set = False  # Clear image state flag
        self._features = None       # Clear cached embeddings  
        self._orig_hw = None        # Clear image dimensions
        self._is_batch = False      # Reset batch mode flag
