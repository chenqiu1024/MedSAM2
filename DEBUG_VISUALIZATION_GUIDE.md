# SAM2 Debug Visualization Guide

This guide explains how to use the comprehensive debug visualization system added to MedSAM2. The system allows you to capture and visualize internal states of the model during inference, providing insights into position encodings, attention patterns, embeddings, and other intermediate representations.

## Overview

The debug visualization system provides:

- **Non-intrusive capture**: Debug hooks that don't affect model performance
- **On-demand activation**: Easy enable/disable functionality 
- **Comprehensive coverage**: Visualization for all major model components
- **Interactive ready**: Designed for future integration with interactive UIs
- **Performance aware**: Minimal overhead when debug mode is disabled

## Quick Start

### Basic Usage

```python
import torch
import numpy as np
from sam2 import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.debug_utils import visualize_debug_states

# Load your model
checkpoint = "path/to/sam2_checkpoint.pt"
model_cfg = "sam2_hiera_b.yaml"
sam2_model = build_sam2(model_cfg, checkpoint)

# Initialize predictor with debug mode enabled
predictor = SAM2ImagePredictor(sam2_model, debug_mode=True)

# Set your image
image = ... # Your input image (numpy array or PIL Image)
predictor.set_image(image)

# Run prediction with debug state capture
masks, scores, logits, debug_states = predictor.predict(
    point_coords=[[100, 200]],
    point_labels=[1],
    return_debug_states=True
)

# Visualize all captured internal states
visualize_debug_states(debug_states, save_path="debug_output/")
```

### Advanced Usage with Selective Visualization

```python
from sam2.debug_utils import (
    enable_debug_mode, 
    disable_debug_mode, 
    get_debug_states,
    clear_debug_states,
    SAM2Visualizer
)

# Enable debug mode with specific components
enable_debug_mode(
    capture_attention=True,
    capture_embeddings=True, 
    capture_position_encoding=True,
    capture_gradients=False  # Disable gradient capture for faster inference
)

# Initialize predictor (debug_mode=True not needed if globally enabled)
predictor = SAM2ImagePredictor(sam2_model)

# Run multiple predictions
for i, (image, points) in enumerate(dataset):
    clear_debug_states()  # Clear previous states
    
    predictor.set_image(image)
    masks, scores, logits, debug_states = predictor.predict(
        point_coords=points,
        point_labels=[1]*len(points),
        return_debug_states=True
    )
    
    # Create custom visualizations
    visualizer = SAM2Visualizer(figsize_base=(15, 10))
    
    # Visualize specific components
    if 'sam_mask_decoder' in debug_states:
        decoder_states = debug_states['sam_mask_decoder']
        visualizer.visualize_mask_decoder_states(
            {name: info['data'] for name, info in decoder_states.items()},
            save_path=f"debug_output/sample_{i}/"
        )
    
    # Visualize attention patterns
    for component_name, states in debug_states.items():
        for state_name, state_info in states.items():
            if 'attention_weights' in state_name:
                visualizer.visualize_attention_patterns(
                    state_info['data'],
                    layer_name=f"{component_name}_{state_name}",
                    save_path=f"debug_output/sample_{i}/"
                )

# Disable debug mode when done
disable_debug_mode()
```

## Captured Internal States

### Position Encodings

The system captures position encoding at multiple stages:

```python
# Access position encoding data
pos_enc_states = debug_states['position_encoding_sine']
# or
pos_enc_states = debug_states['position_encoding_random']

# Available states:
# - 'coordinate_grids_x': Raw x-coordinate grids
# - 'coordinate_grids_y': Raw y-coordinate grids  
# - 'coordinate_grids_x_normalized': Normalized x-coordinates
# - 'coordinate_grids_y_normalized': Normalized y-coordinates
# - 'frequency_dimensions': Sinusoidal frequency components
# - 'sinusoidal_encoding_x': X-coordinate sinusoidal encoding
# - 'sinusoidal_encoding_y': Y-coordinate sinusoidal encoding
# - 'position_embeddings_final': Final position embeddings
```

**Visualization:**
- Shows spatial patterns in position encodings
- Compares different encoding types (sine vs random)
- Analyzes frequency components and their spatial distribution

### Image Encoder States

```python
# Access image encoder states
encoder_states = debug_states['image_encoder']

# Available states:
# - 'input_image': Original input image
# - 'trunk_features_level_*': Backbone features at different scales
# - 'vision_features_final': Final vision features
# - 'fpn_features_level_*': FPN features at different levels
# - 'position_encoding_level_*': Position encodings for each level
```

**Visualization:**
- Multi-scale feature representations
- Feature pyramid network (FPN) outputs
- Spatial feature distributions
- Principal Component Analysis (PCA) of embeddings

### Attention Patterns

```python
# Access attention states from transformer components
attention_states = debug_states['two_way_transformer_layer_0']

# Available states:
# - 'attention_weights': Attention weight matrices
# - 'attention_scores': Pre-softmax attention scores
# - 'multihead_queries': Multi-head query representations
# - 'multihead_keys': Multi-head key representations
# - 'multihead_values': Multi-head value representations
```

**Visualization:**
- Attention weight heatmaps for each head
- Cross-attention between prompts and image features
- Self-attention patterns within prompts
- Evolution of attention across transformer layers

### Mask Decoder States

```python
# Access mask decoder states
decoder_states = debug_states['sam_mask_decoder']

# Available states:
# - 'input_image_embeddings': Input image features
# - 'input_sparse_prompt_embeddings': Prompt embeddings
# - 'raw_masks_output': Raw mask logits before post-processing
# - 'iou_predictions': Mask quality predictions
# - 'hypernetwork_output_*': Hypernetwork parameters for each mask
# - 'upscaled_embedding_final': Upsampled features for mask generation
```

**Visualization:**
- Hypernetwork parameter distributions
- Mask generation process
- IoU prediction patterns
- Intermediate upsampling stages

## Visualization Functions

### Built-in Visualizers

#### Position Encoding Visualization

```python
from sam2.debug_utils import SAM2Visualizer

visualizer = SAM2Visualizer()

# Visualize sinusoidal position encoding
pos_enc_data = debug_states['position_encoding_sine']['position_embeddings_final']['data']
visualizer.visualize_position_encoding(
    pos_enc_data,
    encoding_type="sine",
    save_path="debug_output/",
    show_individual_dims=True  # Show individual encoding dimensions
)
```

#### Attention Pattern Visualization

```python
# Visualize attention weights
attention_weights = debug_states['two_way_transformer_layer_0']['attention_weights']['data']
visualizer.visualize_attention_patterns(
    attention_weights,
    layer_name="two_way_transformer_layer_0",
    save_path="debug_output/",
    max_heads_to_show=8
)
```

#### Image Embedding Visualization

```python
# Visualize image embeddings with PCA analysis
embeddings = debug_states['image_encoder']['vision_features_final']['data']
original_image = debug_states['image_encoder']['input_image']['data']

visualizer.visualize_image_embeddings(
    embeddings,
    original_image=original_image,
    save_path="debug_output/",
    n_components_to_show=8
)
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt

# Create custom visualization for specific analysis
def visualize_attention_evolution(debug_states, save_path=None):
    """Visualize how attention patterns evolve across transformer layers."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    layer_names = [name for name in debug_states.keys() if 'two_way_transformer_layer' in name]
    
    for i, layer_name in enumerate(layer_names[:6]):
        if 'attention_weights' in debug_states[layer_name]:
            attn_weights = debug_states[layer_name]['attention_weights']['data']
            # Average across heads and batch
            attn_avg = attn_weights.mean(dim=(0, 1)).numpy()
            
            im = axes[i].imshow(attn_avg, cmap='viridis')
            axes[i].set_title(f'{layer_name}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Turn off unused axes
    for i in range(len(layer_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/attention_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()

# Use custom visualization
visualize_attention_evolution(debug_states, save_path="debug_output/")
```

## Advanced Features

### Memory and Performance Optimization

```python
# For long videos or large batch processing
from sam2.debug_utils import DebugStateCapture

# Create custom debug capture with memory limits
debug_capture = DebugStateCapture()
debug_capture.enable(
    capture_attention=True,
    capture_embeddings=False,  # Disable to save memory
    capture_position_encoding=True,
    capture_gradients=False
)

# Process video frames
for frame_idx, frame in enumerate(video_frames):
    if frame_idx % 10 == 0:  # Capture debug states every 10 frames
        debug_capture.clear()  # Clear previous states
        
        predictor.set_image(frame)
        masks, scores, logits, debug_states = predictor.predict(
            point_coords=tracked_points[frame_idx],
            point_labels=[1] * len(tracked_points[frame_idx]),
            return_debug_states=True
        )
        
        # Save debug visualization for key frames
        visualize_debug_states(
            debug_states, 
            save_path=f"debug_output/frame_{frame_idx}/",
            create_summary=True
        )
```

### Integration with Training

```python
# Monitor training dynamics (if training enabled)
if model.training:
    enable_debug_mode(capture_gradients=True)
    
    for epoch in range(num_epochs):
        for batch_idx, (images, targets) in enumerate(dataloader):
            clear_debug_states()
            
            # Forward pass with debug capture
            outputs = model(images, return_debug_states=True)
            loss = criterion(outputs, targets)
            
            # Capture gradients
            loss.backward()
            
            # Analyze training dynamics every N batches
            if batch_idx % log_interval == 0:
                debug_states = get_debug_states()
                
                # Custom analysis for training
                analyze_gradient_flow(debug_states)
                visualize_activation_statistics(debug_states)
                
            optimizer.step()
            optimizer.zero_grad()
```

## Debugging Common Issues

### Performance Issues

```python
# Profile debug overhead
import time

# Without debug mode
start_time = time.time()
masks, scores, logits = predictor.predict(
    point_coords=[[100, 200]], 
    point_labels=[1]
)
time_without_debug = time.time() - start_time

# With debug mode
predictor.debug_mode = True
enable_debug_mode()

start_time = time.time()
masks, scores, logits, debug_states = predictor.predict(
    point_coords=[[100, 200]], 
    point_labels=[1],
    return_debug_states=True
)
time_with_debug = time.time() - start_time

print(f"Debug overhead: {(time_with_debug - time_without_debug) / time_without_debug * 100:.2f}%")
```

### Memory Issues

```python
# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Before debug capture
mem_before = get_memory_usage()

# Run with debug
masks, scores, logits, debug_states = predictor.predict(
    point_coords=[[100, 200]], 
    point_labels=[1],
    return_debug_states=True
)

mem_after = get_memory_usage()
print(f"Memory usage: {mem_after - mem_before:.2f} MB")

# Clean up
clear_debug_states()
disable_debug_mode()
```

### State Analysis

```python
def analyze_debug_states(debug_states):
    """Analyze captured debug states for potential issues."""
    
    for component_name, component_states in debug_states.items():
        print(f"\n=== {component_name} ===")
        
        for state_name, state_info in component_states.items():
            data = state_info['data']
            metadata = state_info.get('metadata', {})
            
            if isinstance(data, torch.Tensor):
                print(f"{state_name}:")
                print(f"  Shape: {data.shape}")
                print(f"  Range: [{data.min().item():.6f}, {data.max().item():.6f}]")
                print(f"  Mean: {data.mean().item():.6f}")
                print(f"  Std: {data.std().item():.6f}")
                
                # Check for potential issues
                if torch.isnan(data).any():
                    print(f"  ⚠️  Contains NaN values!")
                if torch.isinf(data).any():
                    print(f"  ⚠️  Contains infinite values!")
                if data.std() < 1e-6:
                    print(f"  ⚠️  Very low variance - potential dead neurons!")

# Run analysis
analyze_debug_states(debug_states)
```

## Best Practices

### 1. Selective Debug Capture

Only capture what you need to minimize performance impact:

```python
# For attention analysis
enable_debug_mode(
    capture_attention=True,
    capture_embeddings=False,
    capture_position_encoding=False
)

# For embedding analysis  
enable_debug_mode(
    capture_attention=False,
    capture_embeddings=True,
    capture_position_encoding=True
)
```

### 2. Regular Cleanup

Clear debug states regularly to prevent memory leaks:

```python
# Process in batches
for batch in batches:
    clear_debug_states()  # Clear before each batch
    
    # Process batch...
    debug_states = get_debug_states()
    
    # Save important results
    if should_save_debug:
        visualize_debug_states(debug_states, save_path=f"debug_batch_{batch_id}/")
```

### 3. Meaningful Names

Use descriptive debug names for complex workflows:

```python
# For multi-stage processing
predictor.set_image(image)

# First stage
masks_stage1, _, _, debug_states_1 = predictor.predict(
    point_coords=initial_points,
    point_labels=[1] * len(initial_points),
    return_debug_states=True
)

# Save first stage
visualize_debug_states(debug_states_1, save_path="debug_output/stage1/")

# Second stage with refinement
clear_debug_states()
masks_stage2, _, _, debug_states_2 = predictor.predict(
    point_coords=refined_points,
    point_labels=[1] * len(refined_points),
    mask_input=logits_stage1,
    return_debug_states=True
)

# Save second stage
visualize_debug_states(debug_states_2, save_path="debug_output/stage2/")
```

## Integration with Jupyter Notebooks

```python
# Notebook-friendly visualization
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage

def notebook_visualize_debug_states(debug_states, max_components=3):
    """Notebook-friendly debug visualization."""
    
    visualizer = SAM2Visualizer(figsize_base=(10, 6))
    
    # Show key components
    component_names = list(debug_states.keys())[:max_components]
    
    for component_name in component_names:
        print(f"\n📊 {component_name}")
        print("=" * 50)
        
        component_states = debug_states[component_name]
        
        # Show first few interesting states
        for state_name, state_info in list(component_states.items())[:3]:
            data = state_info['data']
            metadata = state_info['metadata']
            
            print(f"\n🔍 {state_name}")
            print(f"Shape: {data.shape}, Type: {metadata.get('tensor_type', 'unknown')}")
            
            # Create appropriate visualization
            if 'attention' in state_name.lower() and data.dim() >= 3:
                visualizer.visualize_attention_patterns(data, layer_name=state_name)
            elif 'embedding' in state_name.lower() and data.dim() == 4:
                visualizer.visualize_image_embeddings(data)
            elif 'position' in state_name.lower():
                encoding_type = metadata.get('encoding_type', 'unknown')
                visualizer.visualize_position_encoding(data, encoding_type=encoding_type)

# Use in notebook
notebook_visualize_debug_states(debug_states)
```

## Troubleshooting

### Common Issues and Solutions

1. **No debug states captured**
   ```python
   # Check if debug mode is enabled
   from sam2.debug_utils import is_debug_enabled
   print(f"Debug enabled: {is_debug_enabled()}")
   
   # Enable if needed
   enable_debug_mode()
   ```

2. **Memory errors with large images**
   ```python
   # Use selective capture
   enable_debug_mode(
       capture_attention=True,
       capture_embeddings=False,  # Disable large embeddings
       capture_position_encoding=False
   )
   ```

3. **Visualization errors**
   ```python
   # Check data shapes and types
   for component_name, states in debug_states.items():
       for state_name, state_info in states.items():
           data = state_info['data']
           print(f"{component_name}.{state_name}: {data.shape}, {data.dtype}")
   ```

4. **Performance degradation**
   ```python
   # Disable debug mode for production
   disable_debug_mode()
   
   # Or use minimal capture
   enable_debug_mode(
       capture_attention=False,
       capture_embeddings=False,
       capture_position_encoding=True
   )
   ```

## Extending the Debug System

### Adding Custom Debug Hooks

```python
from sam2.debug_utils import capture_debug_state, is_debug_enabled

class CustomModule(nn.Module):
    def forward(self, x, debug_name=None):
        # Your computation
        output = self.process(x)
        
        # Add debug capture
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name,
                state_name="custom_output",
                data=output,
                metadata={'custom_info': 'example'}
            )
        
        return output
```

### Custom Visualizations

```python
def create_custom_visualizer(debug_states, analysis_type="attention_flow"):
    """Create custom visualizations for specific analysis needs."""
    
    if analysis_type == "attention_flow":
        # Analyze attention flow across layers
        attention_data = []
        for component_name, states in debug_states.items():
            if 'attention_weights' in states:
                attn = states['attention_weights']['data']
                attention_data.append((component_name, attn))
        
        # Create flow visualization
        fig, axes = plt.subplots(1, len(attention_data), figsize=(20, 5))
        for i, (name, attn) in enumerate(attention_data):
            # Average across heads and batch
            attn_avg = attn.mean(dim=(0, 1))
            im = axes[i].imshow(attn_avg, cmap='viridis')
            axes[i].set_title(name)
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
        
    elif analysis_type == "embedding_similarity":
        # Analyze embedding similarities
        pass  # Implement your custom analysis

# Use custom visualizer
create_custom_visualizer(debug_states, "attention_flow")
```

This debug visualization system provides powerful tools for understanding and debugging SAM2's internal behavior, enabling researchers and developers to gain deep insights into the model's decision-making process. 