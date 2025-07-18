# MedSAM2 CT Lesion Inference - Colab Setup Guide

This document explains the modifications made to `MedSAM2_inference_CT_Lesion.ipynb` to make it compatible with Google Colab environment following the established RL-CC-SAM project structure.

## 🔧 Key Modifications Made

### 1. **Environment Setup Cell Added**
- **Cell 1-2**: Added Colab environment detection and setup
- Automatically mounts Google Drive
- Sets up project directory structure (`/content/drive/MyDrive/RL-CC-SAM`)
- Uses prebuilt Python environment at `{DRIVE_ROOT}/colab_envs/llms/`
- Configures all necessary directory paths

### 2. **Path Configuration Updated**
- **Cell 4-5**: Replaced hardcoded paths with dynamic directory structure
- `checkpoint`: `./checkpoints/MedSAM2_latest.pt` → `{PRETRAINED_DIR}/MedSAM2_latest.pt`
- `imgs_path`: `./data` → `{DATASETS_DIR}/CT_DeepLesion/images`
- `model_cfg`: `configs/sam2.1_hiera_t512.yaml` → `{MEDSAM2_DIR}/sam2/configs/sam2.1_hiera_t512.yaml`
- `pred_save_dir`: `./DeeLesion_results` → `{PROJECT_ROOT}/results/CT_DeepLesion_results`
- `path_DL_info`: `CT_DeepLesion/DeepLesion_Dataset_Info.csv` → `{DATASETS_DIR}/CT_DeepLesion/DeepLesion_Dataset_Info.csv`

### 3. **Robust Error Handling**
- **Cell 5**: Added file existence verification for all critical paths
- **Cell 7**: Added error handling for missing dataset files
- **Cell 8**: Added fallback parameters when dataset info is unavailable

### 4. **Improved Device Handling**
- **Cell 8**: Replaced hardcoded `.cuda()` with dynamic device detection
- Supports both GPU and CPU execution
- Uses `torch.autocast(device.type)` for cross-platform compatibility

### 5. **Enhanced Logging and Feedback**
- Added progress indicators (🧠, ✅, 📄, 🚀, etc.)
- Better error messages with actionable suggestions
- Informative output about file processing and results

## 📁 Expected Directory Structure

```
/content/drive/MyDrive/RL-CC-SAM/
├── datasets/
│   └── CT_DeepLesion/
│       ├── images/                 # NIfTI volume files (.nii.gz)
│       └── DeepLesion_Dataset_Info.csv
├── pretrained/
│   └── MedSAM2_latest.pt          # MedSAM2 checkpoint
├── MedSAM2/                       # Git submodule
│   ├── sam2/
│   │   └── configs/
│   │       └── sam2.1_hiera_t512.yaml
│   └── notebooks/
│       └── MedSAM2_inference_CT_Lesion.ipynb
└── results/
    └── CT_DeepLesion_results/     # Output directory (auto-created)
```

## 🚀 Usage Instructions

### Prerequisites
1. **Environment**: Ensure the prebuilt 'llms' environment is available at `{DRIVE_ROOT}/colab_envs/llms/`
2. **Checkpoints**: MedSAM2 checkpoint files should be in `./pretrained/` directory
3. **Dataset**: CT_DeepLesion dataset should be in `./datasets/CT_DeepLesion/` directory

### Running the Notebook
1. Open the notebook in Google Colab
2. Run all cells sequentially
3. The notebook will:
   - Automatically detect Colab environment
   - Mount Google Drive
   - Set up all paths
   - Load the prebuilt environment
   - Process CT lesion segmentation
   - Save results to the results directory

### Expected Output
- **Segmented volumes**: Saved as NIfTI files in `results/CT_DeepLesion_results/`
- **Segmentation info**: Saved as `seg_info_colab.csv`
- **Visualization**: 3D slice visualization showing original, ground truth, and prediction

## 🔍 Key Benefits

1. **Zero Installation**: Uses prebuilt environment, no package installation needed
2. **Robust Path Handling**: Works regardless of local vs Colab environment
3. **Error Recovery**: Graceful handling of missing files or dataset info
4. **Cross-Platform**: Compatible with GPU, CPU, and MPS devices
5. **Comprehensive Logging**: Clear feedback about processing status

## 🛠️ Troubleshooting

### Common Issues:

1. **"MedSAM2 directory not found"**
   - Ensure MedSAM2 submodule is properly initialized
   - Run `setup_colab.ipynb` first to set up git submodules

2. **"Checkpoint not found"**
   - Verify MedSAM2_latest.pt is in `./pretrained/` directory
   - Run the download script to get checkpoints

3. **"No files found in images directory"**
   - Ensure CT_DeepLesion dataset is downloaded to `./datasets/CT_DeepLesion/images/`
   - Check that files have `.nii.gz` extension

4. **"Dataset info not available"**
   - This is a warning, not an error
   - The notebook will use default parameters and continue processing

### Fallback Mode
If dataset info (`DeepLesion_Dataset_Info.csv`) is not available, the notebook automatically:
- Uses middle slice as key slice
- Applies default CT windowing (-200, 200 HU)
- Uses default bounding box (64, 64, 192, 192)
- Continues processing with these parameters

## 📝 Notes

- This modified notebook maintains full compatibility with the original MedSAM2 functionality
- All original processing logic is preserved
- Changes are minimal and focused on path configuration and environment setup
- Results should be identical to the original notebook when run with the same data 