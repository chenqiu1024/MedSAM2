# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mask Decoder for SAM2 (Segment Anything Model 2)

This module implements the mask decoder component of SAM2, which takes processed
image embeddings and prompt embeddings from the transformer to generate high-quality
segmentation masks. The decoder employs a sophisticated hypernetwork architecture
that enables dynamic mask generation based on learned prompt representations.

Key Architectural Components:

1. Output Tokens: Learnable embeddings that represent different mask types
   - IoU Token: Predicts mask quality scores
   - Mask Tokens: Generate different mask hypotheses (single + multi-mask)
   - Object Score Token: Predicts object presence probability

2. Transformer Processing: Two-way attention between prompts and image features
   - Bidirectional information flow for optimal alignment
   - Iterative refinement of representations

3. Hypernetwork Architecture: Dynamic mask generation
   - Output tokens generate parameters for mask prediction networks
   - Each mask token produces its own set of network weights
   - Enables flexible and adaptive mask generation

4. Multi-Scale Upsampling: Progressive resolution enhancement
   - Convolutional upsampling to restore mask resolution
   - Optional high-resolution feature integration
   - Maintains fine spatial details in predictions

5. Multi-Mask Strategy: Ambiguity handling through multiple hypotheses
   - Single mask output for unambiguous cases
   - Multiple mask outputs for ambiguous prompts
   - Dynamic selection based on stability scores

Design Philosophy:
- Learnable output tokens encode different mask generation strategies
- Hypernetworks enable adaptive computation based on prompt context
- Multi-mask outputs handle inherent ambiguity in segmentation tasks
- Quality prediction guides mask selection and user feedback
"""

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    """
    Transformer-based mask decoder that generates segmentation masks from image and prompt embeddings.
    
    This decoder is the final component of the SAM2 pipeline, responsible for converting
    the refined embeddings from the two-way transformer into pixel-accurate segmentation
    masks. It employs a hypernetwork architecture where learnable output tokens generate
    the parameters for dynamic mask prediction networks.
    
    Key Features:
    - Learnable output tokens for different mask types and quality prediction
    - Hypernetwork-based mask generation for adaptive computation
    - Multi-mask output strategy to handle ambiguous prompts
    - Progressive upsampling for high-resolution mask generation
    - Integrated quality prediction for mask selection and user feedback
    
    The decoder supports both single-mask and multi-mask modes:
    - Single-mask: Best for unambiguous prompts with clear segmentation intent
    - Multi-mask: Useful for ambiguous prompts where multiple valid interpretations exist
    
    Quality prediction helps users understand mask confidence and guides automatic
    mask selection in multi-mask scenarios.
    """
    
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Initialize the MaskDecoder with comprehensive configuration options.

        Args:
            transformer_dim (int): Channel dimension of the transformer embeddings.
                Must match the output dimension of the two-way transformer.
            transformer (nn.Module): Two-way transformer for processing image and
                prompt embeddings. Typically a TwoWayTransformer instance.
            num_multimask_outputs (int): Number of mask predictions to generate
                in multi-mask mode. More masks provide better ambiguity handling
                but increase computational cost.
            activation (nn.Module): Activation function for upsampling networks.
                GELU provides smoother gradients than ReLU for better mask quality.
            iou_head_depth (int): Number of layers in the IoU prediction MLP.
                Deeper networks can capture more complex quality patterns.
            iou_head_hidden_dim (int): Hidden dimension for IoU prediction MLP.
                Larger dimensions allow more sophisticated quality modeling.
            use_high_res_features (bool): Whether to integrate high-resolution
                features from the image encoder for finer mask details.
            iou_prediction_use_sigmoid (bool): Whether to apply sigmoid to IoU
                predictions. Useful for constraining outputs to [0,1] range.
            dynamic_multimask_via_stability (bool): Enable dynamic switching
                between single and multi-mask outputs based on stability scores.
            dynamic_multimask_stability_delta (float): Threshold delta for
                stability score computation. Smaller values = stricter stability.
            dynamic_multimask_stability_thresh (float): Threshold for stability-
                based mask selection. Higher values prefer more stable masks.
            pred_obj_scores (bool): Whether to predict object presence scores
                in addition to mask quality. Useful for detection tasks.
            pred_obj_scores_mlp (bool): Use MLP instead of linear layer for
                object score prediction. Provides more modeling capacity.
            use_multimask_token_for_obj_ptr (bool): Use multi-mask tokens for
                object pointer generation in tracking scenarios.
        """
        super().__init__()
        
        # Store core architecture parameters
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # Multi-mask configuration
        self.num_multimask_outputs = num_multimask_outputs

        # Learnable output tokens that guide mask generation
        # IoU token: generates features for mask quality prediction
        self.iou_token = nn.Embedding(1, transformer_dim)
        
        # Mask tokens: each generates a different mask hypothesis
        # Total tokens = multi-mask outputs + 1 single-mask output
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Object score prediction setup (optional)
        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            # Additional token for object presence prediction
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        # Progressive upsampling network for mask resolution enhancement
        # Transforms low-resolution transformer output to high-resolution masks
        self.output_upscaling = nn.Sequential(
            # First upsampling stage: 2x spatial increase
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),  # Stabilize training
            activation(),
            # Second upsampling stage: another 2x spatial increase (4x total)
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        
        # High-resolution feature integration (optional)
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            # Projection layers for integrating multi-scale features
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        # Hypernetwork: each mask token generates its own prediction network
        # This enables adaptive computation based on the specific mask token
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # IoU (mask quality) prediction network
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,  # Predict quality for each mask
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        
        # Object score prediction network (optional)
        if self.pred_obj_scores:
            if pred_obj_scores_mlp:
                # Use MLP for more sophisticated object score modeling
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)
            else:
                # Simple linear projection for object scores
                self.pred_obj_score_head = nn.Linear(transformer_dim, 1)

        # Dynamic multi-mask selection based on stability scores
        # This feature automatically chooses between single and multi-mask outputs
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        debug_name: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate segmentation masks from image and prompt embeddings.
        
        This is the main forward pass that orchestrates the entire mask prediction
        process, from transformer processing to final mask generation and quality
        assessment.

        Args:
            image_embeddings (torch.Tensor): Dense image features from image encoder
                with shape [B, C, H, W] representing visual content.
            image_pe (torch.Tensor): Positional encoding for image features with
                the same shape as image_embeddings.
            sparse_prompt_embeddings (torch.Tensor): Processed prompt embeddings
                (points, boxes) from the prompt encoder with shape [B, N, C].
            dense_prompt_embeddings (torch.Tensor): Dense mask embeddings from
                the prompt encoder with shape [B, C, H, W].
            multimask_output (bool): Whether to return multiple mask hypotheses
                (True) or a single best mask (False).
            repeat_image (bool): Whether to repeat image embeddings to match
                batch size. Used when processing multiple prompts per image.
            high_res_features (Optional[List[torch.Tensor]]): Multi-scale features
                from image encoder for enhanced resolution (if enabled).
            debug_name (str, optional): Name for debug state capture.

        Returns:
            Tuple containing:
                - masks: Predicted segmentation masks [B, N_masks, H, W]
                - iou_predictions: Mask quality scores [B, N_masks]
                - sam_tokens: Processed mask tokens for tracking [B, N_tokens, C]
                - object_scores: Object presence scores [B, 1] (if enabled)
                
        Processing Pipeline:
        1. Prepare output tokens (IoU, mask, optional object score)
        2. Combine tokens with prompt embeddings
        3. Process through two-way transformer
        4. Generate masks using hypernetwork architecture
        5. Predict mask quality and object scores
        6. Select appropriate masks based on output mode
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input embeddings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="input_image_embeddings",
                data=image_embeddings,
                metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': 'image_embeddings'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="input_image_pe",
                data=image_pe,
                metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': 'image_pe'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="input_sparse_prompt_embeddings",
                data=sparse_prompt_embeddings,
                metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': 'sparse_prompt_embeddings'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="input_dense_prompt_embeddings",
                data=dense_prompt_embeddings,
                metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': 'dense_prompt_embeddings'}
            )
            if high_res_features:
                for i, feat in enumerate(high_res_features):
                    capture_debug_state(
                        component_name=debug_name or "mask_decoder",
                        state_name=f"input_high_res_features_{i}",
                        data=feat,
                        metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': f'high_res_features_{i}'}
                    )
        
        # Generate masks and predictions using the core prediction pipeline
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            debug_name=f"{debug_name or 'mask_decoder'}_predict_masks" if debug_name else None,
        )

        # Debug capture for raw prediction outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="raw_masks_output",
                data=masks,
                metadata={'component_type': 'mask_decoder', 'stage': 'raw_output', 'tensor_type': 'masks'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="raw_iou_predictions",
                data=iou_pred,
                metadata={'component_type': 'mask_decoder', 'stage': 'raw_output', 'tensor_type': 'iou_predictions'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="raw_mask_tokens_output",
                data=mask_tokens_out,
                metadata={'component_type': 'mask_decoder', 'stage': 'raw_output', 'tensor_type': 'mask_tokens'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="raw_object_score_logits",
                data=object_score_logits,
                metadata={'component_type': 'mask_decoder', 'stage': 'raw_output', 'tensor_type': 'object_scores'}
            )

        # Select appropriate masks based on output mode and dynamic selection
        if multimask_output:
            # Multi-mask mode: return masks 1-3 (exclude single-mask output)
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            # Dynamic selection: choose between single and multi-mask based on stability
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            # Single-mask mode: return only the first mask
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        # Select appropriate mask tokens for output (used in tracking scenarios)
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            # Use multi-mask tokens for object pointer generation
            sam_tokens_out = mask_tokens_out[:, 1:]  # [B, 3, C] shape
        else:
            # Use single-mask token for object memory
            # This maintains consistency during training where single-mask tokens
            # are used as object memory representations
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [B, 1, C] shape

        # Debug capture for final outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="final_masks_output",
                data=masks,
                metadata={'component_type': 'mask_decoder', 'stage': 'final_output', 'tensor_type': 'masks', 'multimask_output': multimask_output}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="final_iou_predictions",
                data=iou_pred,
                metadata={'component_type': 'mask_decoder', 'stage': 'final_output', 'tensor_type': 'iou_predictions'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="final_sam_tokens_output",
                data=sam_tokens_out,
                metadata={'component_type': 'mask_decoder', 'stage': 'final_output', 'tensor_type': 'sam_tokens'}
            )

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        debug_name: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core mask prediction pipeline using hypernetwork architecture.
        
        This method implements the heart of the mask decoder, where learnable output
        tokens are processed through the transformer and then used to generate
        dynamic mask prediction networks via hypernetworks.

        Args:
            image_embeddings: Image features [B, C, H, W]
            image_pe: Image positional encoding [B, C, H, W]  
            sparse_prompt_embeddings: Prompt features [B, N_prompts, C]
            dense_prompt_embeddings: Dense prompt features [B, C, H, W]
            repeat_image: Whether to repeat image for batch processing
            high_res_features: Multi-scale features for enhanced resolution

        Returns:
            Tuple containing:
                - masks: Raw mask logits [B, N_masks, H_out, W_out]
                - iou_predictions: Quality scores [B, N_masks]
                - mask_tokens: Processed mask token features [B, N_masks, C]
                - object_scores: Object presence logits [B, 1]
                
        Hypernetwork Architecture:
        1. Output tokens encode different mask generation strategies
        2. Tokens are processed through transformer with image context
        3. Each mask token generates parameters for a mask prediction network
        4. Networks are applied to upsampled image features
        5. Multiple masks enable ambiguity handling and quality comparison
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Prepare learnable output tokens for transformer processing
        s = 0  # Offset for indexing tokens
        
        if self.pred_obj_scores:
            # Include object score token if object prediction is enabled
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,  # Object presence token
                    self.iou_token.weight,        # Mask quality token
                    self.mask_tokens.weight,      # Mask generation tokens
                ],
                dim=0,
            )
            s = 1  # Adjust indexing offset
        else:
            # Standard configuration: IoU + mask tokens only
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
            
        # Debug capture for output tokens
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="output_tokens",
                data=output_tokens,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'output_tokens', 'pred_obj_scores': self.pred_obj_scores}
            )
            if self.pred_obj_scores:
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name="obj_score_token",
                    data=self.obj_score_token.weight,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'obj_score_token'}
                )
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="iou_token",
                data=self.iou_token.weight,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'iou_token'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="mask_tokens",
                data=self.mask_tokens.weight,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'mask_tokens'}
            )
            
        # Expand tokens to match batch size and combine with prompt embeddings
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        # Concatenate output tokens with prompt embeddings for joint processing
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Debug capture for combined tokens
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="combined_tokens",
                data=tokens,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'combined_tokens'}
            )

        # Prepare image embeddings for transformer processing
        if repeat_image:
            # Repeat image embeddings to match the number of prompt sets
            # This is used when processing multiple prompts for the same image
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            # Direct assignment when batch sizes match
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
            
        # Integrate dense prompt embeddings (masks) with image features
        src = src + dense_prompt_embeddings
        
        # Debug capture for prepared source features
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="prepared_src_features",
                data=src,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'preparation', 'tensor_type': 'src_features', 'repeat_image': repeat_image}
            )
        
        # Ensure positional encoding has correct batch dimension
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        
        # Repeat positional encoding to match batch size
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Debug capture for positional encoding
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="positional_encoding_repeated",
                data=pos_src,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'preparation', 'tensor_type': 'pos_src'}
            )

        # Process tokens and image features through two-way transformer
        # This enables bidirectional attention between tokens and image features
        hs, src = self.transformer(src, pos_src, tokens, debug_name=f"{debug_name or 'mask_decoder_predict'}_transformer" if debug_name else None)
        
        # Extract processed tokens after transformer processing
        iou_token_out = hs[:, s, :]  # IoU prediction token
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]  # Mask tokens

        # Debug capture for processed tokens
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="transformer_output_all_tokens",
                data=hs,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'transformer_output', 'tensor_type': 'all_tokens'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="iou_token_processed",
                data=iou_token_out,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'transformer_output', 'tensor_type': 'iou_token'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="mask_tokens_processed",
                data=mask_tokens_out,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'transformer_output', 'tensor_type': 'mask_tokens'}
            )

        # Prepare image features for mask generation
        # Reshape from transformer format back to spatial format
        src = src.transpose(1, 2).view(b, c, h, w)
        
        # Debug capture for reshaped source
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="src_reshaped_spatial",
                data=src,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'reshape', 'tensor_type': 'src_spatial'}
            )
        
        # Apply progressive upsampling to restore mask resolution
        if not self.use_high_res_features:
            # Standard upsampling without high-resolution feature integration
            upscaled_embedding = self.output_upscaling(src)
        else:
            # Enhanced upsampling with multi-scale feature integration
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            
            # Debug capture for high-res features
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name="high_res_feat_s0",
                    data=feat_s0,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'upsampling', 'tensor_type': 'high_res_feat_s0'}
                )
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name="high_res_feat_s1",
                    data=feat_s1,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'upsampling', 'tensor_type': 'high_res_feat_s1'}
                )
            
            # First upsampling stage with high-res feature integration
            stage1_output = dc1(src) + feat_s1
            upscaled_embedding = act1(ln1(stage1_output))
            
            # Debug capture for first upsampling stage
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name="upsampling_stage1_output",
                    data=upscaled_embedding,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'upsampling', 'tensor_type': 'stage1_output'}
                )
            
            # Second upsampling stage with finest feature integration  
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # Debug capture for final upscaled embedding
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="upscaled_embedding_final",
                data=upscaled_embedding,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'upsampling', 'tensor_type': 'upscaled_embedding', 'use_high_res': self.use_high_res_features}
            )

        # Hypernetwork: Generate mask prediction networks from mask tokens
        # Each mask token produces parameters for its own prediction network
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            # Each MLP generates network parameters for mask prediction
            hyper_out = self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            hyper_in_list.append(hyper_out)
            
            # Debug capture for individual hypernetwork outputs
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name=f"hypernetwork_output_{i}",
                    data=hyper_out,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'hypernetwork', 'tensor_type': f'hyper_out_{i}', 'mask_token_idx': i}
                )
                
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, N_masks, C_hyper]
        
        # Debug capture for stacked hypernetwork outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="hypernetwork_parameters_stacked",
                data=hyper_in,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'hypernetwork', 'tensor_type': 'hyper_in_stacked'}
            )
        
        # Apply hypernetwork-generated parameters to upscaled features
        # This performs dynamic convolution based on the mask token context
        b, c, h, w = upscaled_embedding.shape
        # Matrix multiplication: [B, N_masks, C_hyper] @ [B, C_hyper, H*W] -> [B, N_masks, H*W]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Debug capture for mask generation
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="masks_raw_logits",
                data=masks,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'mask_generation', 'tensor_type': 'mask_logits'}
            )

        # Generate mask quality predictions from IoU token
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        # Debug capture for IoU predictions
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="iou_predictions",
                data=iou_pred,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'prediction', 'tensor_type': 'iou_predictions'}
            )
        
        # Generate object score predictions (if enabled)
        if self.pred_obj_scores:
            assert s == 1  # Ensure object token was included
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Default object scores: high confidence (assuming object is present)
            # sigmoid(10.0) ≈ 1.0, indicating strong object presence
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        # Debug capture for object score predictions
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="object_score_logits",
                data=object_score_logits,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'prediction', 'tensor_type': 'object_scores', 'pred_obj_scores': self.pred_obj_scores}
            )

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores for dynamic mask selection based on IoU between thresholds.
        
        Stability scores measure how consistent a mask is across different binarization
        thresholds. More stable masks have similar shapes when thresholded at different
        values, indicating higher confidence in the prediction.
        
        Args:
            mask_logits: Raw mask predictions [B, N_masks, H, W]
            
        Returns:
            Stability scores [B, N_masks] in range [0, 1]
            
        Algorithm:
        1. Flatten spatial dimensions for efficient computation
        2. Compute areas above upper and lower thresholds
        3. Calculate IoU between threshold-based masks
        4. Higher IoU indicates more stable predictions
        
        This metric helps identify masks that are robust to threshold selection,
        which correlates with prediction confidence and accuracy.
        """
        # Flatten spatial dimensions for area computation
        mask_logits = mask_logits.flatten(-2)
        
        # Define stability threshold delta
        stability_delta = self.dynamic_multimask_stability_delta
        
        # Compute areas above different thresholds
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()   # Upper threshold
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()  # Lower threshold
        
        # Calculate stability as IoU between threshold-based masks
        # Higher values indicate more stable predictions across thresholds
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        Dynamically select between single-mask and multi-mask outputs based on stability.
        
        This method implements an intelligent selection strategy that chooses the most
        appropriate mask output mode based on prediction stability. When the single-mask
        output is unstable (low stability score), it falls back to the best multi-mask
        output, ensuring robust mask selection.
        
        Args:
            all_mask_logits: All mask predictions [B, N_masks, H, W]
                where N_masks = 1 (single) + N (multi-mask outputs)
            all_iou_scores: Quality scores for all masks [B, N_masks]
            
        Returns:
            Tuple of selected masks and IoU scores [B, 1, H, W], [B, 1]
            
        Selection Strategy:
        1. Evaluate single-mask stability (mask 0)
        2. Find best multi-mask candidate (masks 1-3) based on IoU
        3. Choose single-mask if stable, otherwise use best multi-mask
        
        This approach ensures that users always receive a high-quality mask,
        whether from the focused single-mask output or the more exploratory
        multi-mask outputs.
        """
        # Extract multi-mask predictions and their quality scores
        multimask_logits = all_mask_logits[:, 1:, :, :]      # Masks 1-3
        multimask_iou_scores = all_iou_scores[:, 1:]         # Corresponding IoU scores
        
        # Find the best multi-mask candidate based on predicted IoU
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        
        # Extract best multi-mask predictions
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # Evaluate single-mask output (mask 0) and its stability
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        
        # Determine which masks meet the stability threshold
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamic selection: stable single-mask vs. best multi-mask
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,      # Use single-mask if stable
            best_multimask_logits,  # Otherwise use best multi-mask
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,      # Corresponding IoU scores
            best_multimask_iou_scores,
        )
        
        return mask_logits_out, iou_scores_out
