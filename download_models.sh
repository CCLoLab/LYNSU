#!/bin/bash
# Download YOLOv7 model weights and save in AllinOne_trained directory
echo "Downloading YOLOv7 model weights..."
mkdir -p AllinOne_trained

# Use gdown to download YOLOv7 model weights
gdown "https://drive.google.com/uc?id=1x123NlNDSGicUZde-gwyYSSh_9ANPmkN" -O AllinOne_trained/best.pt

# Download 3D U-Net model weights for 6 brain regions and save in corresponding directories
echo "Downloading 3D U-Net model weights for 6 brain regions..."

# AL best model
mkdir -p "已經完成訓練的3D_UNet模型/AL最佳模型"
gdown "https://drive.google.com/uc?id=1Pf2fPgz7lJtLiGd9jVKp7ACmMnwHhmtT" -O "已經完成訓練的3D_UNet模型/AL最佳模型/best_model.h5"

# CAL best model
mkdir -p "已經完成訓練的3D_UNet模型/CAL最佳模型"
gdown "https://drive.google.com/uc?id=1azSMJNJtKhNZ2dvS0SrYtbRgA7wV61BY" -O "已經完成訓練的3D_UNet模型/CAL最佳模型/best_model.h5"

# MB best model
mkdir -p "已經完成訓練的3D_UNet模型/MB最佳模型"
gdown "https://drive.google.com/uc?id=1teb3Escc-2s1_uIHielTCYFhHp0TTaGo" -O "已經完成訓練的3D_UNet模型/MB最佳模型/best_model.h5"

# EB best model
mkdir -p "已經完成訓練的3D_UNet模型/EB最佳模型"
gdown "https://drive.google.com/uc?id=1t98eaJbP5y0_Qg61ASk9dhg0M8y3EkGd" -O "已經完成訓練的3D_UNet模型/EB最佳模型/best_model.h5"

# FB best model
mkdir -p "已經完成訓練的3D_UNet模型/FB最佳模型"
gdown "https://drive.google.com/uc?id=1u8nyQZ9KtrSWYOajJS2FI-FmTdPz526s" -O "已經完成訓練的3D_UNet模型/FB最佳模型/best_model.h5"

# PB best model
mkdir -p "已經完成訓練的3D_UNet模型/PB最佳模型"
gdown "https://drive.google.com/uc?id=1S4l20LuFR1zkDkXOEAxHIonYeUn5ttcd" -O "已經完成訓練的3D_UNet模型/PB最佳模型/best_model.h5"

echo "All models have been downloaded and saved to the appropriate directories."
