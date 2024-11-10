# LYNSU
**LYNSU**: Automated neuropil segmentation of fluorescent images for Drosophila brains.

**中文版說明**: [README_zh.md](https://github.com/CCLoLab/LYNSU/blob/Training_and_Inference/README_zh.md)

The code for this project is hosted on GitHub. We will provide the training and inference code upon the publication of our paper. Visit our repository here: [LYNSU GitHub Repository](https://github.com/CCLoLab/LYNSU)

**Two Methods to Set Up the Environment:**

**First Method: Docker (Recommended - No Additional Package Installation Needed)**

1. **Pull the Docker Image**: 
   ```
   docker pull kaiyihsu/lynsu_image_with_packages
   ```

2. **Mount Persistent Storage Folder Path**: 
   ```
   export STORAGE_LOCATION=/home/brc/TensorFlow_Dev && \
   mkdir -p $STORAGE_LOCATION && \
   touch "$STORAGE_LOCATION/.env"
   ```
   *(Please modify the STORAGE_LOCATION path to your own directory)*

3. **Start the Docker Container**:
   ```
   docker run -p 13826:13826 -it --gpus all \
   -v ${STORAGE_LOCATION}:/workspace \
   -v ${STORAGE_LOCATION}/.env:/workspace/.env \
   -e STORAGE_DIR="/workspace" \
   --name my_tensorflow_container kaiyihsu/lynsu_image_with_packages
   ```

4. **Run `Download_DataSet.sh` to Download the Required Datasets for YOLOv7 & 3D U-Net**

5. **Run `Unzip.py` to Extract the Dataset Files**

**Second Method: Install Pip Packages Manually (Potential CUDA Version Conflicts)**

1. **Install Torch**:
   
   First, install `torchvision`, which will automatically install the corresponding version of `torch` (2.0.1):
   ```
   pip install torchvision==0.15.2
   ```

2. **Install Requirements**:
   ```
   pip install -r requirements.txt
   ```

3. **Run `Download_DataSet.sh` to Download the Required Datasets for YOLOv7 & 3D U-Net**

   *Note: For Windows, manually copy the commands from `download_models.sh` and paste them into the terminal to execute.*

4. **Run `Unzip.py` to Extract the Dataset Files**

**Recommendation**: It is highly recommended to use the Docker image for setting up the environment, as it allows you to use the LYNSU web application without the need for additional package installations.

### Model Training Instructions

1. **YOLOv7 Training**:
   - Before starting, clone the YOLOv7 repository:
     ```
     git clone https://github.com/WongKinYiu/yolov7.git
     ```
   - Execute the YOLOv7_Step1 data preprocessing `.ipynb` file.
   - Once the training data is ready, proceed to execute YOLOv7_Step2 for model training.

2. **3D U-Net Training**:
   - Perform data preprocessing for specific brain regions by executing the `3D_UNET_Step1` file.
   - Continue with `3D_UNET_Step2` to train the 3D U-Net model.

### Batch Inference Instructions

1. **Inference is Conducted Separately for Male and Female Drosophila**.
2. **Batch Inference Step 1**: Parallel projection of brain images to 2D for the YOLO model.
3. **Continue with Batch Inference Step 2**.
