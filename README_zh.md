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

*執行bash Download_DataSet.sh 下載YOLOv7 & 3D U-Net所需數據集*

*執行Unzip.py 將數據集壓縮檔解壓縮。*

**第二種: 自行安裝pip套件 (可能遭遇cuda版本衝突)**

*安裝torch(先安裝 pip install torchvision==0.15.2 就會自動安裝對應版本的torch 2.0.1)*

*安裝requirements.txt pip install -r requirements.txt*

*執行bash Download_DataSet.sh 下載YOLOv7 & 3D U-Net所需數據集*
p.s Win系統需要手動複製 download_models.sh內容指令貼到終端機執行

*執行Unzip.py 將數據集壓縮檔解壓縮。*

**建議使用Docker Image建立環境，可以無需額外安裝任何套件即可直接使用LYNSU 網頁應用程序**

訓練模型說明：
1. 執行YOLOv7訓練前需要先 git clone https://github.com/WongKinYiu/yolov7.git
2. 依序執行YOLOv7_Step1資料預處理的ipynb檔案
3. 確定訓練資料準備好接續執行YOLOv7_Step2訓練模型
4. 依照特定腦區執行3D_UNET_Step1資料預處理
5. 接續執行3D_UNET_Step2訓練3D U-Net模型

批量推理說明：
1. 公母果蠅分開進行
2. 先執行批量推理_Step1（其中有平行化批量將大腦影像投影成2D，供應給YOLO模型）
3. 以此類推執行批量推理_Step2

