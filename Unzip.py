# 目的： 解壓縮兩個zip檔案

import zipfile

def unzip_file(zip_path):
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('.')


unzip_file('3D_UNet_dataset.zip')
unzip_file('YOLOv7_dataset.zip')