#!/bin/bash

# Test script for the video upscaling validator

echo "=== Video Upscaling Validation Test ==="
echo

# Check if dependencies are installed
python3 -c "import cv2, skimage, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements_validation.txt
fi

# Check if test videos exist
if [ ! -f "upscale_test.mp4" ]; then
    echo "ERROR: upscale_test.mp4 not found"
    exit 1
fi

# Create a test upscaled video (using Lanczos as fallback)
echo "Creating test upscaled video..."
python3 upscale_video.py upscale_test.mp4 --scale 2 --preset ultrafast --crf 23 --disable-runtime-guardrail --force --output test_upscaled_2x.mp4

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create upscaled video"
    exit 1
fi

# Run validation
echo "Running validation..."
python3 validate_upscale.py \
    --input upscale_test.mp4 \
    --output test_upscaled_2x.mp4 \
    --scale 2 \
    --outdir validation_report \
    --sample 3  # Sample every 3rd frame for speed

echo
echo "Validation complete! Check validation_report/ for results."
