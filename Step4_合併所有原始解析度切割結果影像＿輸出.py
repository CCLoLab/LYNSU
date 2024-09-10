#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# 目的：將本次所有切割的腦區原始解析度影像進行編號後，輸出成tif(最後改成只輸出投影圖)
# 不區分前後順序的展示，目的是展示切割腦區數量是否正確


# %%


# import os
# import numpy as np
# import tifffile as tif
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 定義腦區的資料夾與對應的標籤
# directories = {
#     "AL": "網頁應用程式暫存資料夾_AL專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "MB": "網頁應用程式暫存資料夾_MB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "CAL": "網頁應用程式暫存資料夾_CAL專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "FB": "網頁應用程式暫存資料夾_FB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "EB": "網頁應用程式暫存資料夾_EB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "PB": "網頁應用程式暫存資料夾_PB專用/使用者程序_第五階段存檔內容_完成放回原始影像"
# }

# # 定義腦區標籤號碼
# labels = {
#     "AL": 1,
#     "MB": 2,
#     "CAL": 3,
#     "FB": 4,
#     "EB": 5,
#     "PB": 6
# }

# # 初始化一個空的3D圖像空間
# combined_image = None

# # 讀取每個資料夾中的 Seg 開頭的 tif 檔案，並將不同腦區合併
# for region, directory in directories.items():
#     for filename in os.listdir(directory):
#         if filename.startswith("Seg") and filename.endswith(".tif"):
#             file_path = os.path.join(directory, filename)
#             print(f"Reading: {file_path}")
#             image_data = tif.imread(file_path)
            
#             # 如果還沒有初始化，創建一個與第一個影像相同大小的空間
#             if combined_image is None:
#                 combined_image = np.zeros_like(image_data)
            
#             # 將讀取到的影像數據標記為對應的腦區號碼
#             combined_image[image_data > 0] = labels[region]

# # 顯示3D影像
# def plot_3d_segmentation(segmented_image):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # 定義顏色對應，每個腦區一種顏色
#     colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow', 5: 'cyan', 6: 'magenta'}

#     # 獲取非零點的索引
#     z, y, x = np.nonzero(segmented_image)

#     # 遍歷不同的腦區，繪製對應的3D點
#     for label, color in colors.items():
#         region_indices = np.where(segmented_image == label)
#         ax.scatter(region_indices[2], region_indices[1], region_indices[0], c=color, label=f'Region {label}', s=0.1)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.legend()
#     plt.show()

# # 調用繪圖函數
# plot_3d_segmentation(combined_image)


# %%


# import os
# import numpy as np
# import tifffile as tif
# import matplotlib.pyplot as plt

# # 定義腦區的資料夾與對應的標籤
# directories = {
#     "AL": "網頁應用程式暫存資料夾_AL專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "MB": "網頁應用程式暫存資料夾_MB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "CAL": "網頁應用程式暫存資料夾_CAL專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "FB": "網頁應用程式暫存資料夾_FB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "EB": "網頁應用程式暫存資料夾_EB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "PB": "網頁應用程式暫存資料夾_PB專用/使用者程序_第五階段存檔內容_完成放回原始影像"
# }

# # 定義腦區標籤號碼
# labels = {
#     "AL": 1,
#     "MB": 2,
#     "CAL": 3,
#     "FB": 4,
#     "EB": 5,
#     "PB": 6
# }

# # 初始化一個空的3D圖像空間
# combined_image = None

# # 讀取每個資料夾中的 Seg 開頭的 tif 檔案，並將不同腦區合併
# for region, directory in directories.items():
#     for filename in os.listdir(directory):
#         if filename.startswith("Seg") and filename.endswith(".tif"):
#             file_path = os.path.join(directory, filename)
#             print(f"Reading: {file_path}")
#             image_data = tif.imread(file_path)
            
#             # 如果還沒有初始化，創建一個與第一個影像相同大小的空間
#             if combined_image is None:
#                 combined_image = np.zeros_like(image_data)
            
#             # 將讀取到的影像數據標記為對應的腦區號碼
#             combined_image[image_data > 0] = labels[region]

# # 沿Z軸投影，將3D影像轉換為2D影像
# # 可以用np.max()沿著Z軸疊加
# projected_image = np.max(combined_image, axis=0)

# # 定義顏色對應，每個腦區一種顏色
# colors = {
#     0: [0, 0, 0],        # 背景黑色
#     1: [1, 0, 0],        # AL - 紅色
#     2: [0, 1, 0],        # MB - 綠色
#     3: [0, 0, 1],        # CAL - 藍色
#     4: [1, 1, 0],        # FB - 黃色
#     5: [0, 1, 1],        # EB - 青色
#     6: [1, 0, 1]         # PB - 洋紅色
# }

# # 創建一個RGB圖像以顯示不同腦區的顏色
# rgb_image = np.zeros((*projected_image.shape, 3))

# for label, color in colors.items():
#     rgb_image[projected_image == label] = color

# # 顯示2D投影影像
# plt.figure(figsize=(8, 8))
# plt.imshow(rgb_image)
# plt.title("Z-axis Projected 2D Brain Region Segmentation")
# plt.axis('off')  # 關閉軸線
# plt.show()


# %%

import shutil
import os
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# 清空並重建 '最後合併腦區結果圖' 資料夾
folder = '最後合併腦區結果圖'
if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)
# 定義腦區的資料夾與對應的標籤
directories = {
    "AL": "網頁應用程式暫存資料夾_AL專用/使用者程序_第五階段存檔內容_完成放回原始影像",
    "MB": "網頁應用程式暫存資料夾_MB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
    "CAL": "網頁應用程式暫存資料夾_CAL專用/使用者程序_第五階段存檔內容_完成放回原始影像",
    "FB": "網頁應用程式暫存資料夾_FB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
    "EB": "網頁應用程式暫存資料夾_EB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
    "PB": "網頁應用程式暫存資料夾_PB專用/使用者程序_第五階段存檔內容_完成放回原始影像"
}

# 定義腦區標籤號碼
labels = {
    "AL": 1,
    "MB": 2,
    "CAL": 3,
    "FB": 4,
    "EB": 5,
    "PB": 6
}

# 初始化一個空的3D圖像空間
combined_image = None

# # 讀取每個資料夾中的 Seg 開頭的 tif 檔案，並將不同腦區合併
# for region, directory in directories.items():
#     for filename in os.listdir(directory):
#         if filename.startswith("Seg") and filename.endswith(".tif"):
#             file_path = os.path.join(directory, filename)
#             print(f"Reading: {file_path}")
#             image_data = tif.imread(file_path)
            
#             # 如果還沒有初始化，創建一個與第一個影像相同大小的空間
#             if combined_image is None:
#                 combined_image = np.zeros_like(image_data)
            
#             # 將讀取到的影像數據標記為對應的腦區號碼
#             combined_image[image_data > 0] = labels[region]
# 使用 os.path.exists() 函數來先檢查資料夾是否存在，這樣就可以避免出現 FileNotFoundError
# 讀取每個資料夾中的 Seg 開頭的 tif 檔案，並將不同腦區合併
for region, directory in directories.items():
    if os.path.exists(directory):  # 先檢查資料夾是否存在
        for filename in os.listdir(directory):
            if filename.startswith("Seg") and filename.endswith(".tif"):
                file_path = os.path.join(directory, filename)
                print(f"Reading: {file_path}")
                image_data = tif.imread(file_path)
                
                # 如果還沒有初始化，創建一個與第一個影像相同大小的空間
                if combined_image is None:
                    combined_image = np.zeros_like(image_data)
                
                # 將讀取到的影像數據標記為對應的腦區號碼
                combined_image[image_data > 0] = labels[region]
    else:
        print(f"Directory not found: {directory}")

# 沿Z軸投影，將3D影像轉換為2D影像
projected_image = np.max(combined_image, axis=0)

# 定義顏色對應，每個腦區一種顏色
colors = {
    0: [0, 0, 0],        # 背景黑色
    1: [1, 0, 0],        # AL - 紅色
    2: [0, 1, 1],        # MB - 青色
    3: [0, 1, 0],        # CAL - 綠色
    4: [0, 0, 1],        # FB - 藍色
    5: [1, 0.5, 1],        # EB - 粉紅色
    6: [1, 1, 0]         # PB - 黃色
}

# 創建一個RGB圖像以顯示不同腦區的顏色
rgb_image = np.zeros((*projected_image.shape, 3))

for label, color in colors.items():
    rgb_image[projected_image == label] = color

# 顯示2D投影影像
plt.figure(figsize=(6, 6))
plt.imshow(rgb_image)
plt.title("Z-axis Projected 2D Brain Region Segmentation")
plt.axis('off')  # 關閉軸線

# 使用 RGB 值來指定顏色
legend_labels = {
    'AL': (1, 0, 0),
    'MB': (0, 1, 1),
    'CAL': (0, 1, 0),
    'FB': (0, 0, 1),
    'EB': (1, 0.5, 1),  # 使用 RGB 值來指定亮粉紅色
    'PB': (1, 1, 0)
}

# 使用 matplotlib.patches.Patch 創建圖例對應的顏色塊
patches = [mpatches.Patch(color=color, label=label) for label, color in legend_labels.items()]

# 添加圖例
plt.legend(handles=patches, loc='upper right', title="Brain Regions")
# 可選：保存圖片
plt.savefig("最後合併腦區結果圖/brain_regions_projection.png", bbox_inches='tight')
# 顯示圖片
# plt.show()


# %%


# import os
# import numpy as np
# import tifffile as tif
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # 定義腦區的資料夾與對應的標籤
# directories = {
#     "AL": "網頁應用程式暫存資料夾_AL專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "MB": "網頁應用程式暫存資料夾_MB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "CAL": "網頁應用程式暫存資料夾_CAL專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "FB": "網頁應用程式暫存資料夾_FB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "EB": "網頁應用程式暫存資料夾_EB專用/使用者程序_第五階段存檔內容_完成放回原始影像",
#     "PB": "網頁應用程式暫存資料夾_PB專用/使用者程序_第五階段存檔內容_完成放回原始影像"
# }

# # 定義腦區標籤號碼
# labels = {
#     "AL": 1,
#     "MB": 2,
#     "CAL": 3,
#     "FB": 4,
#     "EB": 5,
#     "PB": 6
# }

# # 初始化一個空的3D圖像空間
# combined_image = None

# # 讀取每個資料夾中的 Seg 開頭的 tif 檔案，並將不同腦區合併
# for region, directory in directories.items():
#     if not os.path.exists(directory):
#         print(f"資料夾不存在: {directory}")
#         continue
#     for filename in os.listdir(directory):
#         if filename.startswith("Seg") and filename.endswith(".tif"):
#             file_path = os.path.join(directory, filename)
#             print(f"Reading: {file_path}")
#             try:
#                 image_data = tif.imread(file_path)
#             except Exception as e:
#                 print(f"無法讀取檔案 {file_path}: {e}")
#                 continue
            
#             # 如果還沒有初始化，創建一個與第一個影像相同大小的空間
#             if combined_image is None:
#                 combined_image = np.zeros_like(image_data)
            
#             # 將讀取到的影像數據與現有的 combined_image 取最大值
#             combined_image = np.maximum(combined_image, labels[region] * (image_data > 0))

# # 沿Z軸投影，將3D影像轉換為2D影像
# projected_image = np.max(combined_image, axis=0)

# # 定義顏色對應，每個腦區一種顏色
# colors = {
#     0: [0, 0, 0],        # 背景黑色
#     1: [1, 0, 0],        # AL - 紅色
#     2: [0, 1, 1],        # MB - 青色
#     3: [0, 1, 0],        # CAL - 綠色
#     4: [0, 0, 1],        # FB - 藍色
#     5: [1, 0.5, 1],      # EB - 粉紅色
#     6: [1, 1, 0]         # PB - 黃色
# }

# # 創建一個RGB圖像以顯示不同腦區的顏色
# rgb_image = np.zeros((*projected_image.shape, 3))

# for label, color in colors.items():
#     rgb_image[projected_image == label] = color

# # 顯示2D投影影像
# plt.figure(figsize=(8, 8))
# plt.imshow(rgb_image)
# plt.title("Z-axis Projected 2D Brain Region Segmentation")
# plt.axis('off')  # 關閉軸線

# # 使用 RGB 值來指定顏色
# legend_labels = {
#     'AL': (1, 0, 0),
#     'MB': (0, 1, 1),
#     'CAL': (0, 1, 0),
#     'FB': (0, 0, 1),
#     'EB': (1, 0.5, 1),  # 使用 RGB 值來指定亮粉紅色
#     'PB': (1, 1, 0)
# }

# # 使用 matplotlib.patches.Patch 創建圖例對應的顏色塊
# patches = [mpatches.Patch(color=color, label=label) for label, color in legend_labels.items()]

# # 添加圖例，並將其放置在圖像外部以避免遮擋
# plt.legend(handles=patches, loc='upper right', title="Brain Regions", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# # 顯示圖片
# plt.show()

# # 可選：保存圖片
# plt.savefig("brain_regions_projection.png", bbox_inches='tight')


# %%




