# LYNSU
**LYNSU**: Automated neuropil segmentation of fluorescent images for Drosophila brains.

The code for this project is hosted on GitHub. We will provide the training and inference code upon the publication of our paper. Visit our repository here: [LYNSU GitHub Repository](https://github.com/CCLoLab/LYNSU)

**兩種建立環境方法:**

**第一種: Docker (最簡單無須額外安裝套件)**

*拉取映像檔: docker pull kaiyihsu/lynsu_image_with_packages*

掛載持久化資料夾路徑: export STORAGE_LOCATION=/home/brc/TensorFlow_Dev && \
mkdir -p $STORAGE_LOCATION && \
touch "$STORAGE_LOCATION/.env" (STORAGE_LOCATION 路徑請更改成自己的路徑)

啟動容器: docker run -p 13826:13826 -it --gpus all \
-v ${STORAGE_LOCATION}:/workspace \
-v ${STORAGE_LOCATION}/.env:/workspace/.env \
-e STORAGE_DIR="/workspace" \
--name my_tensorflow_container kaiyihsu/lynsu_image_with_packages

*執行bash download_models.sh 下載所有模型權重*

*python GUI.py啟動程序*

**第二種: 自行安裝pip套件 (可能遭遇cuda版本衝突)**

*安裝torch(先安裝 pip install torchvision==0.15.2 就會自動安裝對應版本的torch 2.0.1)*

*安裝requirements.txt pip install -r requirements.txt*

*執行bash download_models.sh 下載所有模型權重*
p.s Win系統需要手動複製 download_models.sh內容指令貼到終端機執行
*python GUI.py啟動程序*

**建議使用Docker Image建立環境，可以無需額外安裝任何套件即可直接使用LYNSU 網頁應用程序**
