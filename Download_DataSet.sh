#!/bin/bash
# Download YOLOv7 and 3D U-Net DataSets and save in AllinOne_dataset directory
echo "Downloading YOLOv7 DataSet..."

# Use gdown to download YOLOv7 DataSet
gdown "https://drive.google.com/uc?id=1BjwxoGo3Eey1Qj-OhpAroZlwYcF0qzGv" -O YOLOv7_dataset.zip

# Download 3D U-Net DataSet and save in AllinOne_dataset directory
echo "Downloading 3D U-Net DataSet..."

# Use gdown to download 3D U-Net DataSet
gdown "https://drive.google.com/uc?id=10tcU5n4yi5cgUs2M3GGCRx5WArDGCIJS" -O 3D_UNet_dataset.zip

echo "All datasets have been downloaded."