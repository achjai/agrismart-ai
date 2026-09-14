#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install python dependencies
pip install -r requirements.txt

# Create weights directory if it doesn't exist
mkdir -p model/weights

# Download the main ResNet50 model weights from Google Drive using gdown
# The file ID is passed from Render Environment Variables as MODEL_DRIVE_ID
echo "Downloading model weights from Google Drive..."
gdown $MODEL_DRIVE_ID -O model/weights/RESNET50_FINETUNED.weights.h5

echo "Build complete."
