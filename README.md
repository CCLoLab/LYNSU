# LYNSU
**LYNSU**: Automated neuropil segmentation of fluorescent images for Drosophila brains.

The code for this project is hosted on GitHub. We will provide the training and inference code upon the publication of our paper. Visit our repository here: [LYNSU GitHub Repository](https://github.com/CCLoLab/LYNSU)

**第一步先安裝Use_pip_install_pytorch.txt，第二步安裝requirements.txt，第三步執行bash Download_DataSet.sh 下載YOLOv7 & 3D U-Net所需數據集，第四步執行Unzip.py 將數據集壓縮檔解壓縮。**

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

p.s Win系統需要手動複製 Download_DataSet.sh內容指令貼到終端機執行
