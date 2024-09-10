#!/bin/bash
# Need to change the download link to your own cloud storage link
# Download YOLOv7 model weights and save in AllinOne_trained directory
echo "Downloading YOLOv7 model weights..."
mkdir -p AllinOne_trained
wget -O AllinOne_trained/best.pt "https://your_cloud_storage_link/yolov7.pt"

# Download 3D U-Net model weights for 6 brain regions and save in corresponding directories
echo "Downloading 3D U-Net model weights for 6 brain regions..."

# AL best model
mkdir -p "已經完成訓練的3D_UNet模型/AL最佳模型"
wget -O "已經完成訓練的3D_UNet模型/AL最佳模型/best_model.h5" "https://your_cloud_storage_link/al_best_model.h5"

# CAL best model
mkdir -p "已經完成訓練的3D_UNet模型/CAL最佳模型"
wget -O "已經完成訓練的3D_UNet模型/CAL最佳模型/best_model.h5" "https://your_cloud_storage_link/cal_best_model.h5"

# MB best model
mkdir -p "已經完成訓練的3D_UNet模型/MB最佳模型"
wget -O "已經完成訓練的3D_UNet模型/MB最佳模型/best_model.h5" "https://your_cloud_storage_link/mb_best_model.h5"

# EB best model
mkdir -p "已經完成訓練的3D_UNet模型/EB最佳模型"
wget -O "已經完成訓練的3D_UNet模型/EB最佳模型/best_model.h5" "https://your_cloud_storage_link/eb_best_model.h5"

# FB best model
mkdir -p "已經完成訓練的3D_UNet模型/FB最佳模型"
wget -O "已經完成訓練的3D_UNet模型/FB最佳模型/best_model.h5" "https://your_cloud_storage_link/fb_best_model.h5"

# PB best model
mkdir -p "已經完成訓練的3D_UNet模型/PB最佳模型"
wget -O "已經完成訓練的3D_UNet模型/PB最佳模型/best_model.h5" "https://your_cloud_storage_link/pb_best_model.h5"

echo "All models have been downloaded and saved to the appropriate directories."
