#!/bin/bash
# Download YOLOv7 model weights
echo "Downloading YOLOv7 model weights..."
wget -O yolov7.pt "https://your_cloud_storage_link/yolov7.pt"

# Download U-Net model weights for 6 brain regions
echo "Downloading U-Net model weights for 6 brain regions..."
wget -O unet_brain_regions.h5 "https://your_cloud_storage_link/unet_brain_regions.h5"

echo "Download complete."
