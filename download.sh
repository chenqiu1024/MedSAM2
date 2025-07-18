#!/usr/bin/env bash
# Script to download MedSAM2 model checkpoints
# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Function to check if file exists and skip download if it does
check_and_download() {
    local file_name="$1"
    local url="$2"
    local file_path="checkpoints/${file_name}"
    
    if [ -f "$file_path" ]; then
        echo "✓ ${file_name} already exists, skipping download."
        return 0
    fi
    
    echo "Downloading ${file_name}..."
    
    if [ -n "$CURL" ]; then
        $CMD "$file_path" "$url" || { echo "Failed to download checkpoint from $url"; exit 1; }
    else
        $CMD "$url" || { echo "Failed to download checkpoint from $url"; exit 1; }
    fi
    
    echo "✓ ${file_name} downloaded successfully."
}

# Use either wget or curl to download the checkpoints
if command -v wget > /dev/null 2>&1; then
    CMD="wget -P checkpoints"
elif command -v curl > /dev/null 2>&1; then
    CMD="curl -L -o"
    CURL=1
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the base URL for MedSAM2 models on Hugging Face
HF_BASE_URL="https://huggingface.co/wanglab/MedSAM2/resolve/main"

# Define the model checkpoint files (as separate variables instead of an array)
MODEL1="MedSAM2_2411.pt"
MODEL2="MedSAM2_US_Heart.pt"
MODEL3="MedSAM2_MRI_LiverLesion.pt"
MODEL4="MedSAM2_CTLesion.pt"
MODEL5="MedSAM2_latest.pt"

echo "=== Downloading MedSAM2 model checkpoints ==="
# Download each checkpoint
for model in $MODEL1 $MODEL2 $MODEL3 $MODEL4 $MODEL5; do
    model_url="${HF_BASE_URL}/${model}"
    check_and_download "$model" "$model_url"
done

echo "=== Downloading Efficient Track Anything checkpoints ==="
# Download the Efficient Track Anything checkpoint
ETA_BASE_URL="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main"
ETA_MODELS=("efficienttam_s_512x512.pt" "efficienttam_ti_512x512.pt")

for ETA_MODEL in "${ETA_MODELS[@]}"; do
    eta_model_url="${ETA_BASE_URL}/${ETA_MODEL}"
    check_and_download "$ETA_MODEL" "$eta_model_url"
done

echo "=== Downloading SAM2 model checkpoints ==="
# download SAM2 model
SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_MODELs=("sam2.1_hiera_tiny.pt" "sam2.1_hiera_large.pt")
for SAM2_MODEL in "${SAM2_MODELs[@]}"; do
    sam2_model_url="${SAM2_BASE_URL}/${SAM2_MODEL}"
    check_and_download "$SAM2_MODEL" "$sam2_model_url"
done

echo ""
echo "🎉 All checkpoint downloads completed successfully!"
echo "📁 Check the 'checkpoints' directory for all downloaded model files."


