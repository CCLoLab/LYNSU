# LYNSU
**LYNSU**: Automated neuropil segmentation of fluorescent images for Drosophila brains.

The code for this project is hosted on GitHub. We will provide the training and inference code upon the publication of our paper. Visit our repository here: [LYNSU GitHub Repository](https://github.com/CCLoLab/LYNSU)

**兩種建立環境方法:**

**第一種: Docker (最簡單無須額外安裝套件)**
*拉取映像檔: docker pull kaiyihsu/lynsu_image_with_packages*
*掛載持久化資料夾路徑: export STORAGE_LOCATION=/home/brc/TensorFlow_Dev && \
mkdir -p $STORAGE_LOCATION && \
touch "$STORAGE_LOCATION/.env" (STORAGE_LOCATION 路徑請更改成自己的路徑)*
*啟動容器: docker run -p 13826:13826 -it --gpus all \
-v ${STORAGE_LOCATION}:/workspace \
-v ${STORAGE_LOCATION}/.env:/workspace/.env \
-e STORAGE_DIR="/workspace" \
--name my_tensorflow_container kaiyihsu/lynsu_image_with_packages*




*拉取映像檔: docker pull kaiyihsu/lynsu_image_with_packages >>安裝torch(可能只需安裝torchvision==0.15.2 就會自動安裝對應版本的torch 2.0.1) >> 安裝requirements.txt >> 下載模型*
**第一步先安裝Use_pip_install_pytorch.txt，第二步安裝requirements.txt，第三步執行bash download_models.sh 下載所有模型權重，即可python GUI.py啟動程序**
p.s Win系統需要手動複製 download_models.sh內容指令貼到終端機執行
