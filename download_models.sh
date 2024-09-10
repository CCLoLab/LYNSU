#!/bin/bash
# Need to change the download link to your own cloud storage link
# Download YOLOv7 model weights and save in AllinOne_trained directory
echo "Downloading YOLOv7 model weights..."
mkdir -p AllinOne_trained
wget -O AllinOne_trained/best.pt "https://drive.google.com/file/d/1x123NlNDSGicUZde-gwyYSSh_9ANPmkN/view?usp=sharing"

# Download 3D U-Net model weights for 6 brain regions and save in corresponding directories
echo "Downloading 3D U-Net model weights for 6 brain regions..."

# AL best model
mkdir -p "已經完成訓練的3D_UNet模型/AL最佳模型"
wget -O "已經完成訓練的3D_UNet模型/AL最佳模型/best_model.h5" "https://drive.google.com/file/d/1Pf2fPgz7lJtLiGd9jVKp7ACmMnwHhmtT/view?usp=sharing"

# CAL best model
mkdir -p "已經完成訓練的3D_UNet模型/CAL最佳模型"
wget -O "已經完成訓練的3D_UNet模型/CAL最佳模型/best_model.h5" "https://drive.google.com/file/d/1azSMJNJtKhNZ2dvS0SrYtbRgA7wV61BY/view?usp=sharing"

# MB best model
mkdir -p "已經完成訓練的3D_UNet模型/MB最佳模型"
wget -O "已經完成訓練的3D_UNet模型/MB最佳模型/best_model.h5" "https://drive.google.com/file/d/1teb3Escc-2s1_uIHielTCYFhHp0TTaGo/view?usp=sharing"

# EB best model
mkdir -p "已經完成訓練的3D_UNet模型/EB最佳模型"
wget -O "已經完成訓練的3D_UNet模型/EB最佳模型/best_model.h5" "https://drive.google.com/file/d/1t98eaJbP5y0_Qg61ASk9dhg0M8y3EkGd/view?usp=sharing"

# FB best model
mkdir -p "已經完成訓練的3D_UNet模型/FB最佳模型"
wget -O "已經完成訓練的3D_UNet模型/FB最佳模型/best_model.h5" "https://drive.google.com/file/d/1u8nyQZ9KtrSWYOajJS2FI-FmTdPz526s/view?usp=sharing"

# PB best model
mkdir -p "已經完成訓練的3D_UNet模型/PB最佳模型"
wget -O "已經完成訓練的3D_UNet模型/PB最佳模型/best_model.h5" "https://drive.google.com/file/d/1S4l20LuFR1zkDkXOEAxHIonYeUn5ttcd/view?usp=sharing"

echo "All models have been downloaded and saved to the appropriate directories."
