#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# 目的：讀取tif檔案投影成2D，提供給YOLO作為輸入
# 使用者上傳的Tif檔案會放在: User_Input_DLG_tif
# 會輸出2D 投影圖到: DEMO_For_YOLO_as_Input

# 這是第一步驟，在此先將之前的" 網頁應用程式暫存資料夾 六個資料夾重置 "


# %%


import os
import shutil
# ==========================設定所有會儲存使用的路徑==========================
# 設定儲存母資料夾路徑: 網頁應用程式暫存資料夾 # 這個通常不會動
nas_path = '網頁應用程式暫存資料夾_AL專用/'
# 清空並重建 'nas_path' 資料夾
folder = nas_path
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)
os.mkdir(folder+"使用者程序_第五階段存檔內容_完成放回原始影像/")
# ===
nas_path = '網頁應用程式暫存資料夾_MB專用/'
# 清空並重建 'nas_path' 資料夾
folder = nas_path
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)
os.mkdir(folder+"使用者程序_第五階段存檔內容_完成放回原始影像/")
# ===
nas_path = '網頁應用程式暫存資料夾_CAL專用/'
# 清空並重建 'nas_path' 資料夾
folder = nas_path
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)
os.mkdir(folder+"使用者程序_第五階段存檔內容_完成放回原始影像/")
# ===
nas_path = '網頁應用程式暫存資料夾_FB專用/'
# 清空並重建 'nas_path' 資料夾
folder = nas_path
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)
os.mkdir(folder+"使用者程序_第五階段存檔內容_完成放回原始影像/")
# ===
nas_path = '網頁應用程式暫存資料夾_EB專用/'
# 清空並重建 'nas_path' 資料夾
folder = nas_path
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)
os.mkdir(folder+"使用者程序_第五階段存檔內容_完成放回原始影像/")
# ===
nas_path = '網頁應用程式暫存資料夾_PB專用/'
# 清空並重建 'nas_path' 資料夾
folder = nas_path
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)
os.mkdir(folder+"使用者程序_第五階段存檔內容_完成放回原始影像/")


# %%


import os
import shutil
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# Step 1: 檢查 'User_Input_DLG_tif' 資料夾是否有檔案
DLG_files = [file for file in os.listdir('User_Input_DLG_tif') if file.endswith('.tif') and not file.startswith('.')]
if not DLG_files:
    print("No image files uploaded in 'User_Input_DLG_tif'.")
else:
    print(f"{len(DLG_files)} image(s) found in 'User_Input_DLG_tif'.")

    # 排序檔案
    DLG_files.sort()

    # Step 2: 清空並重建 'DEMO_For_YOLO_as_Input' 資料夾
    folder = 'DEMO_For_YOLO_as_Input'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    print(f"Folder '{folder}' is ready.")

    # Step 3: 批量處理 DLG 檔案
    start_time = time.time()
    for file in DLG_files:
        DLG = tif.imread(os.path.join('User_Input_DLG_tif', file))
        print(f'Processing {file}, shape: {DLG.shape}')

#         # 顯示第 Z_num 切片的三視圖
#         Z_num = 70
#         plt.figure(figsize=(10, 20))
#         plt.imshow(DLG[Z_num], cmap='gray')
#         plt.title(file, fontsize=24)
#         plt.axis('off')
#         plt.show()

        # 計算投影
        projection = np.sum(DLG, axis=0)  # 投影成2D圖 (取加總值)

#         # 顯示投影結果
#         plt.figure(figsize=(10, 10))
#         plt.imshow(projection, cmap='gray')
#         plt.title(f'{file}_to_2D', fontsize=24)
#         plt.axis('off')
#         plt.show()

        # Step 4: 輸出投影圖至 YOLO 輸入資料夾
        projection = (projection / np.max(projection)) * 255  # 正規化並轉換至 0-255
        im = Image.fromarray(projection.astype('uint8'))
        im.save(os.path.join(folder, file + '.png'))

    # 記錄並顯示執行時間
    end_time_to_2D = time.time()
    print(f'Time to complete 2D projection: {end_time_to_2D - start_time:.2f} seconds.')


# %%




