# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Debug Utilities for SAM2 Internal State Visualization

This module provides comprehensive tools for capturing, analyzing, and visualizing
internal states of SAM2 components during inference and training. It enables
intuitive insights into model behavior and facilitates debugging of complex
transformer-based architectures.

Key Features:

1. **Non-intrusive Capture**: Debug hooks that don't affect model performance
2. **On-demand Activation**: Easy enable/disable functionality for production use
3. **Comprehensive Coverage**: Visualization for all major model components
4. **Interactive Ready**: Designed for future integration with interactive UIs
5. **Performance Aware**: Minimal overhead when debug mode is disabled

Supported Components:
- Position Encodings (sinusoidal, random Fourier features)
- Image Embeddings and their spatial relationships
- Attention matrices and attention patterns
- Transformer layer activations
- Mask decoder intermediate representations
- Training dynamics and gradient flows

Usage:
    # Enable debug mode in model
    predictor = SAM2ImagePredictor(model, debug_mode=True)
    
    # Run inference with debug capture
    masks, scores, logits, debug_states = predictor.predict(
        point_coords=points, point_labels=labels, return_debug_states=True
    )
    
    # Visualize internal states
    from sam2.debug_utils import visualize_debug_states
    visualize_debug_states(debug_states, save_path="debug_output/")
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path


class DebugStateCapture:
    """
    Central registry for capturing and managing debug states across SAM2 components.
    
    This class provides a unified interface for collecting internal states from
    different model components without affecting the original model behavior.
    """
    
    def __init__(self):
        self.states = defaultdict(dict)
        self.enabled = False
        self.capture_attention = True
        self.capture_embeddings = True
        self.capture_position_encoding = True
        self.capture_gradients = False
        
    def enable(self, capture_attention=True, capture_embeddings=True, 
               capture_position_encoding=True, capture_gradients=False):
        """Enable debug capture with specified components."""
        self.enabled = True
        self.capture_attention = capture_attention
        self.capture_embeddings = capture_embeddings
        self.capture_position_encoding = capture_position_encoding
        self.capture_gradients = capture_gradients
        
    def disable(self):
        """Disable debug capture and clear stored states."""
        self.enabled = False
        self.clear()
        
    def clear(self):
        """Clear all captured states."""
        self.states.clear()
        
    def capture(self, component_name: str, state_name: str, data: torch.Tensor, 
                metadata: Optional[Dict] = None):
        """
        Capture a tensor state from a model component.
        
        Args:
            component_name: Name of the component (e.g., 'image_encoder', 'attention_block_0')
            state_name: Name of the specific state (e.g., 'attention_weights', 'embeddings')
            data: Tensor data to capture
            metadata: Additional metadata about the captured state
        """
        if not self.enabled:
            return
            
        # Convert to CPU and detach for safe storage
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
            
        self.states[component_name][state_name] = {
            'data': data,
            'shape': data.shape if isinstance(data, torch.Tensor) else None,
            'dtype': data.dtype if isinstance(data, torch.Tensor) else None,
            'metadata': metadata or {}
        }
        
    def get_state(self, component_name: str, state_name: str = None):
        """Retrieve captured state(s) for a component."""
        if state_name is None:
            return self.states.get(component_name, {})
        return self.states.get(component_name, {}).get(state_name)
        
    def get_all_states(self):
        """Retrieve all captured states."""
        return dict(self.states)


# Global debug capture instance
_debug_capture = DebugStateCapture()


def enable_debug_mode(capture_attention=True, capture_embeddings=True, 
                     capture_position_encoding=True, capture_gradients=False):
    """Enable global debug mode for SAM2 components."""
    _debug_capture.enable(capture_attention, capture_embeddings, 
                         capture_position_encoding, capture_gradients)


def disable_debug_mode():
    """Disable global debug mode."""
    _debug_capture.disable()


def capture_debug_state(component_name: str, state_name: str, data: torch.Tensor, 
                       metadata: Optional[Dict] = None):
    """Capture debug state using the global capture instance."""
    _debug_capture.capture(component_name, state_name, data, metadata)


def get_debug_states():
    """Get all captured debug states."""
    return _debug_capture.get_all_states()


def clear_debug_states():
    """Clear all captured debug states."""
    _debug_capture.clear()


def is_debug_enabled():
    """Check if debug mode is currently enabled."""
    return _debug_capture.enabled


# Visualization Functions
class SAM2Visualizer:
    """
    Comprehensive visualization suite for SAM2 internal states.
    """
    
    def __init__(self, figsize_base=(12, 8), dpi=100):
        self.figsize_base = figsize_base
        self.dpi = dpi
        
        # Custom color maps
        self.attention_cmap = LinearSegmentedColormap.from_list(
            'attention', ['white', 'red'], N=256
        )
        self.feature_cmap = 'viridis'
        
    def visualize_position_encoding(self, pos_enc_data: torch.Tensor, 
                                   encoding_type: str = "sine",
                                   save_path: Optional[str] = None,
                                   show_individual_dims: bool = True):
        """
        Visualize position encoding patterns.
        
        Args:
            pos_enc_data: Position encoding tensor (B, C, H, W) or (B, H*W, C)
            encoding_type: Type of encoding ("sine", "random", "rope")
            save_path: Path to save the visualization
            show_individual_dims: Whether to show individual encoding dimensions
        """
        if pos_enc_data.dim() == 4:
            # (B, C, H, W) format
            B, C, H, W = pos_enc_data.shape
            pos_enc_data = pos_enc_data[0]  # Take first batch
        elif pos_enc_data.dim() == 3:
            # (B, H*W, C) format - reshape to spatial
            B, HW, C = pos_enc_data.shape
            H = W = int(np.sqrt(HW))
            pos_enc_data = pos_enc_data[0].transpose(0, 1).reshape(C, H, W)
        else:
            raise ValueError(f"Unsupported position encoding shape: {pos_enc_data.shape}")
            
        # Create comprehensive visualization
        if show_individual_dims:
            n_dims_to_show = min(8, pos_enc_data.shape[0])
            fig, axes = plt.subplots(2, max(4, n_dims_to_show//2), 
                                   figsize=(20, 8), dpi=self.dpi)
            axes = axes.flatten()
            
            for i in range(n_dims_to_show):
                im = axes[i].imshow(pos_enc_data[i].numpy(), cmap=self.feature_cmap)
                axes[i].set_title(f'{encoding_type.title()} PE Dim {i}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046)
                
            # Turn off unused axes
            for i in range(n_dims_to_show, len(axes)):
                axes[i].axis('off')
        else:
            # Show summary statistics
            fig, axes = plt.subplots(2, 2, figsize=self.figsize_base, dpi=self.dpi)
            
            # Mean across all dimensions
            axes[0, 0].imshow(pos_enc_data.mean(0).numpy(), cmap=self.feature_cmap)
            axes[0, 0].set_title(f'{encoding_type.title()} PE - Mean')
            axes[0, 0].axis('off')
            
            # Standard deviation
            axes[0, 1].imshow(pos_enc_data.std(0).numpy(), cmap=self.feature_cmap)
            axes[0, 1].set_title(f'{encoding_type.title()} PE - Std')
            axes[0, 1].axis('off')
            
            # Range (max - min)
            pe_range = pos_enc_data.max(0)[0] - pos_enc_data.min(0)[0]
            axes[1, 0].imshow(pe_range.numpy(), cmap=self.feature_cmap)
            axes[1, 0].set_title(f'{encoding_type.title()} PE - Range')
            axes[1, 0].axis('off')
            
            # Magnitude
            magnitude = torch.norm(pos_enc_data, dim=0)
            axes[1, 1].imshow(magnitude.numpy(), cmap=self.feature_cmap)
            axes[1, 1].set_title(f'{encoding_type.title()} PE - Magnitude')
            axes[1, 1].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/position_encoding_{encoding_type}.png", 
                       bbox_inches='tight', dpi=self.dpi)
        plt.show()
        
    def visualize_attention_patterns(self, attention_weights: torch.Tensor,
                                   layer_name: str = "",
                                   head_idx: Optional[int] = None,
                                   save_path: Optional[str] = None,
                                   max_heads_to_show: int = 8):
        """
        Visualize attention weight patterns.
        
        Args:
            attention_weights: Attention weights tensor (B, H, N, N) or (B, N, N)
            layer_name: Name of the attention layer
            head_idx: Specific head to visualize (if None, show multiple heads)
            save_path: Path to save the visualization
            max_heads_to_show: Maximum number of attention heads to display
        """
        if attention_weights.dim() == 4:
            # Multi-head attention (B, H, N, N)
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            attention_weights = attention_weights[0]  # Take first batch
        elif attention_weights.dim() == 3:
            # Single head or already batched (B, N, N)
            if attention_weights.shape[0] == 1:
                attention_weights = attention_weights[0]
                num_heads = 1
            else:
                num_heads = attention_weights.shape[0]
        else:
            raise ValueError(f"Unsupported attention shape: {attention_weights.shape}")
            
        if head_idx is not None:
            # Show specific head
            fig, ax = plt.subplots(1, 1, figsize=self.figsize_base, dpi=self.dpi)
            im = ax.imshow(attention_weights[head_idx].numpy(), cmap=self.attention_cmap)
            ax.set_title(f'{layer_name} - Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            plt.colorbar(im, ax=ax)
        else:
            # Show multiple heads
            heads_to_show = min(max_heads_to_show, num_heads)
            cols = min(4, heads_to_show)
            rows = (heads_to_show + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), dpi=self.dpi)
            if heads_to_show == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
                
            for i in range(heads_to_show):
                ax = axes[i] if heads_to_show > 1 else axes[0]
                im = ax.imshow(attention_weights[i].numpy(), cmap=self.attention_cmap)
                ax.set_title(f'{layer_name} - Head {i}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                plt.colorbar(im, ax=ax, fraction=0.046)
                
            # Turn off unused axes
            if heads_to_show > 1:
                for i in range(heads_to_show, len(axes)):
                    axes[i].axis('off')
                    
        plt.tight_layout()
        if save_path:
            head_suffix = f"_head_{head_idx}" if head_idx is not None else ""
            plt.savefig(f"{save_path}/attention_{layer_name}{head_suffix}.png", 
                       bbox_inches='tight', dpi=self.dpi)
        plt.show()
        
    def visualize_image_embeddings(self, embeddings: torch.Tensor,
                                 original_image: Optional[torch.Tensor] = None,
                                 save_path: Optional[str] = None,
                                 n_components_to_show: int = 8):
        """
        Visualize image embeddings and their spatial patterns.
        
        Args:
            embeddings: Image embeddings tensor (B, C, H, W)
            original_image: Original input image for context
            save_path: Path to save the visualization
            n_components_to_show: Number of embedding dimensions to visualize
        """
        if embeddings.dim() != 4:
            raise ValueError(f"Expected 4D embeddings (B, C, H, W), got {embeddings.shape}")
            
        embeddings = embeddings[0]  # Take first batch
        C, H, W = embeddings.shape
        
        # Create comprehensive visualization
        n_components = min(n_components_to_show, C)
        cols = 4
        rows = (n_components + cols - 1) // cols + (1 if original_image is not None else 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), dpi=self.dpi)
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        plot_idx = 0
        
        # Show original image if provided
        if original_image is not None:
            if original_image.dim() == 4:
                original_image = original_image[0]
            if original_image.shape[0] == 3:  # RGB
                img_np = original_image.permute(1, 2, 0).numpy()
                # Normalize to [0, 1] if needed
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                axes[plot_idx].imshow(img_np)
            else:
                axes[plot_idx].imshow(original_image[0].numpy(), cmap='gray')
            axes[plot_idx].set_title('Original Image')
            axes[plot_idx].axis('off')
            plot_idx += 1
            
        # Show embedding components
        for i in range(n_components):
            if plot_idx < len(axes):
                im = axes[plot_idx].imshow(embeddings[i].numpy(), cmap=self.feature_cmap)
                axes[plot_idx].set_title(f'Embedding Dim {i}')
                axes[plot_idx].axis('off')
                plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
                plot_idx += 1
                
        # Turn off unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/image_embeddings.png", 
                       bbox_inches='tight', dpi=self.dpi)
        plt.show()
        
        # Additional analysis: PCA visualization
        self._visualize_embedding_pca(embeddings, save_path)
        
    def _visualize_embedding_pca(self, embeddings: torch.Tensor, save_path: Optional[str] = None):
        """Visualize PCA components of embeddings."""
        C, H, W = embeddings.shape
        # Reshape to (H*W, C) for PCA
        embeddings_flat = embeddings.view(C, -1).transpose(0, 1).numpy()
        
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca_components = pca.fit_transform(embeddings_flat)
            pca_components = pca_components.reshape(H, W, 3)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=self.dpi)
            
            # RGB visualization of first 3 PCA components
            pca_rgb = (pca_components - pca_components.min()) / (pca_components.max() - pca_components.min())
            axes[0].imshow(pca_rgb)
            axes[0].set_title('PCA RGB (PC1=R, PC2=G, PC3=B)')
            axes[0].axis('off')
            
            # Individual PCA components
            for i in range(3):
                im = axes[i+1].imshow(pca_components[:, :, i], cmap=self.feature_cmap)
                axes[i+1].set_title(f'PCA Component {i+1}\n(Var: {pca.explained_variance_ratio_[i]:.3f})')
                axes[i+1].axis('off')
                plt.colorbar(im, ax=axes[i+1], fraction=0.046)
                
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/embedding_pca.png", 
                           bbox_inches='tight', dpi=self.dpi)
            plt.show()
            
        except ImportError:
            warnings.warn("scikit-learn not available, skipping PCA visualization")
            
    def visualize_mask_decoder_states(self, decoder_states: Dict[str, torch.Tensor],
                                    save_path: Optional[str] = None):
        """
        Visualize mask decoder internal states.
        
        Args:
            decoder_states: Dictionary containing decoder states
            save_path: Path to save the visualization
        """
        n_states = len(decoder_states)
        cols = min(3, n_states)
        rows = (n_states + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), dpi=self.dpi)
        if n_states == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        plot_idx = 0
        for state_name, state_data in decoder_states.items():
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            if state_data.dim() == 4:  # (B, C, H, W)
                # Show mean across channels
                data_to_plot = state_data[0].mean(0).numpy()
            elif state_data.dim() == 3:  # (B, N, C)
                # Show as heatmap
                data_to_plot = state_data[0].numpy()
            elif state_data.dim() == 2:  # (N, C)
                data_to_plot = state_data.numpy()
            else:
                # For other dimensions, flatten and show as 1D plot
                ax.plot(state_data.flatten().numpy())
                ax.set_title(f'{state_name} (1D)')
                plot_idx += 1
                continue
                
            im = ax.imshow(data_to_plot, cmap=self.feature_cmap)
            ax.set_title(f'{state_name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            plot_idx += 1
            
        # Turn off unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/mask_decoder_states.png", 
                       bbox_inches='tight', dpi=self.dpi)
        plt.show()


def visualize_debug_states(debug_states: Optional[Dict] = None, 
                          save_path: Optional[str] = None,
                          create_summary: bool = True):
    """
    Comprehensive visualization of all captured debug states.
    
    Args:
        debug_states: Debug states dictionary (if None, use global states)
        save_path: Directory to save visualizations
        create_summary: Whether to create a summary report
    """
    if debug_states is None:
        debug_states = get_debug_states()
        
    if not debug_states:
        print("No debug states captured. Enable debug mode first.")
        return
        
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    visualizer = SAM2Visualizer()
    
    # Visualize each component's states
    for component_name, component_states in debug_states.items():
        print(f"\nVisualizing {component_name}...")
        
        for state_name, state_info in component_states.items():
            data = state_info['data']
            metadata = state_info.get('metadata', {})
            
            try:
                if 'position_encoding' in state_name.lower():
                    encoding_type = metadata.get('encoding_type', 'unknown')
                    visualizer.visualize_position_encoding(
                        data, encoding_type=encoding_type, save_path=save_path
                    )
                elif 'attention' in state_name.lower():
                    visualizer.visualize_attention_patterns(
                        data, layer_name=f"{component_name}_{state_name}", save_path=save_path
                    )
                elif 'embedding' in state_name.lower() or 'features' in state_name.lower():
                    if data.dim() == 4:  # Image-like features
                        visualizer.visualize_image_embeddings(
                            data, save_path=save_path
                        )
                elif 'decoder' in component_name.lower():
                    # Collect all decoder states for combined visualization
                    decoder_states = {name: info['data'] for name, info in component_states.items()}
                    visualizer.visualize_mask_decoder_states(decoder_states, save_path=save_path)
                    break  # Don't visualize individual decoder states again
                    
            except Exception as e:
                print(f"Failed to visualize {component_name}.{state_name}: {e}")
                
    if create_summary and save_path:
        _create_debug_summary(debug_states, save_path)
        
        
def _create_debug_summary(debug_states: Dict, save_path: str):
    """Create a text summary of captured debug states."""
    summary_path = os.path.join(save_path, "debug_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("SAM2 Debug States Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for component_name, component_states in debug_states.items():
            f.write(f"Component: {component_name}\n")
            f.write("-" * 30 + "\n")
            
            for state_name, state_info in component_states.items():
                data = state_info['data']
                metadata = state_info.get('metadata', {})
                
                f.write(f"  State: {state_name}\n")
                f.write(f"    Shape: {state_info['shape']}\n")
                f.write(f"    Dtype: {state_info['dtype']}\n")
                
                if isinstance(data, torch.Tensor):
                    f.write(f"    Min: {data.min().item():.6f}\n")
                    f.write(f"    Max: {data.max().item():.6f}\n")
                    f.write(f"    Mean: {data.mean().item():.6f}\n")
                    f.write(f"    Std: {data.std().item():.6f}\n")
                    
                if metadata:
                    f.write(f"    Metadata: {metadata}\n")
                f.write("\n")
            f.write("\n")
            
    print(f"Debug summary saved to: {summary_path}") 