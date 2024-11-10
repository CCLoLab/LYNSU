# LYNSU
**LYNSU**: Automated neuropil segmentation of fluorescent images for Drosophila brains.

**中文版說明**: [README_zh.md](https://github.com/CCLoLab/LYNSU/blob/Web_GUI/README_zh.md)

The code for this project is hosted on GitHub. We will provide the training and inference code upon the publication of our paper. Visit our repository here: [LYNSU GitHub Repository](https://github.com/CCLoLab/LYNSU)

**Two Methods to Set Up the Environment:**

**Method 1: Docker (Recommended - No Additional Package Installation Needed)**

1. **Pull the Docker image:**
   ```sh
   docker pull kaiyihsu/lynsu_image_with_packages
   ```

2. **Mount a persistent folder path:**
   ```sh
   export STORAGE_LOCATION=/home/brc/TensorFlow_Dev && \
   mkdir -p $STORAGE_LOCATION && \
   touch "$STORAGE_LOCATION/.env"
   ```
   *(Please modify the STORAGE_LOCATION path to your own desired path)*

3. **Start the container:**
   ```sh
   docker run -p 13826:13826 -it --gpus all \
   -v ${STORAGE_LOCATION}:/workspace \
   -v ${STORAGE_LOCATION}/.env:/workspace/.env \
   -e STORAGE_DIR="/workspace" \
   --name my_tensorflow_container kaiyihsu/lynsu_image_with_packages
   ```

4. **Download all model weights:**
   ```sh
   bash download_models.sh
   ```

5. **Run the application:**
   ```sh
   python GUI.py
   ```

**Method 2: Manual Installation with pip (May Encounter CUDA Version Conflicts)**

1. **Install PyTorch:**
   ```sh
   pip install torchvision==0.15.2
   ```
   *(This will automatically install the corresponding version of torch 2.0.1)*

2. **Install dependencies from requirements.txt:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download all model weights:**
   ```sh
   bash download_models.sh
   ```
   *Note: For Windows users, manually copy the commands from `download_models.sh` and run them in the terminal.*

4. **Run the application:**
   ```sh
   python GUI.py
   ```

**Recommendation:**

We recommend using the Docker image to set up the environment, as it allows you to use the LYNSU web application without needing to install additional packages.
