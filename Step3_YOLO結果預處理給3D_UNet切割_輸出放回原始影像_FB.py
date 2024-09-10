#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# 目的：讀取YOLO切割結果並提取3D影像進行滑動切割(預處理)
# 接續使用3D U-Net進行腦區切割，並輸出3D腦區切割結果

# 此代碼專門切割特定腦區
# Label標籤: AL=0 MB=1 Central_complex=2

# 需要修改的位置: 
# 1. nas_path = '網頁應用程式暫存資料夾_MB專用/'
# 2. seg_target
# 3. int(class_idx) == 1
# 4. DLG[DLG>0]=2 # 重疊腦區的位置都成最終腦區標籤

# 未來可以修改: DLG tif輸出可以放在其他代碼中，才不會每一個腦區資料夾都輸出一次重複的DLG.tif


# %%


# Step1 設定路徑跟導入套件
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import tifffile as tif
import time
from skimage import measure
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import gc  # 引入垃圾回收模組
from patchify import patchify, unpatchify
from segmentation_models_3D import get_preprocessing
from typing import Tuple, Union, cast
# ==========================設定所有會儲存使用的路徑==========================
# 設定儲存母資料夾路徑: 網頁應用程式暫存資料夾 # 這個通常不會動
nas_path = '網頁應用程式暫存資料夾_FB專用/'
# 已經在最初Step1 清空所有暫存資料夾
# 清空並重建 'nas_path' 資料夾
# folder = nas_path
# if os.path.exists(folder):
#     shutil.rmtree(folder)
# os.mkdir(folder)
# print(f"Folder '{folder}' is ready.")
# 定義第一階段輸出資料夾
Step1_output_path = nas_path+'使用者程序_第一階段存檔內容_YOLO提取3D_DLG'

# 定義第二階段輸出資料夾
Step2_output_path = nas_path+'使用者程序_第二階段存檔內容_完成滑動切割'

# 定義第三階段輸出資料夾
Step3_output_path = nas_path+'使用者程序_第三階段存檔內容_完成推理'

# 定義第四階段輸出資料夾
Step4_output_path = nas_path+'使用者程序_第四階段存檔內容_完成還原Z拓展後尺寸'

# 定義第五階段輸出資料夾
Step5_output_path = nas_path+'使用者程序_第五階段存檔內容_完成放回原始影像'

# 定義 original_info 儲存名稱
original_info_name = nas_path+'original_info'

# Step1 設定大腦區域和相關變數
seg_target = 'FB'
# nii_gz_src = "/mnt/nas_1/LoLab/kaiyi_hsu/Ouput_from_Amira/npy_files/"


# %%


# Step2 讀取用戶上傳的原始DLG檔案名稱
DLG_files = [file for file in os.listdir('User_Input_DLG_tif') if file.endswith('.tif') and not file.startswith('.')]
if not DLG_files:
    print("No image files uploaded in 'User_Input_DLG_tif'.")
else:
    print(f"{len(DLG_files)} image(s) found in 'User_Input_DLG_tif'.")

    # 排序檔案
    DLG_files.sort()


# %%


# Step3讀取 用戶上傳影像資料夾內的大腦，並提取範圍內3D影像，再加以拓展進行滑動切割立方體輸出。
start_time_crop_resize = time.time()  # 記錄開始時間

def resize_and_pad(whole_brain, x_start, x_end, y_start, y_end, target_size=168, z_min_size=64, step_of_slide=20):
    """
    對特定區域的大腦進行調整大小和填充
    """
    brain_region = [whole_brain[i, y_start:y_end, x_start:x_end] for i in range(len(whole_brain))]
    brain_region = np.array(brain_region)
    
    # 確定需要縮放還是放大
    if brain_region.shape[1] > target_size:
        resized_brain = np.array([cv2.resize(slice, (target_size, target_size), interpolation=cv2.INTER_AREA) for slice in brain_region])
    else:
        resized_brain = np.array([cv2.resize(slice, (target_size, target_size), interpolation=cv2.INTER_CUBIC) for slice in brain_region])

    # 填充 Z 軸以達到最小尺寸
    num_layers_to_add = 0
    while len(resized_brain) < z_min_size or (len(resized_brain) - z_min_size) % step_of_slide != 0:
        resized_brain = np.append(resized_brain, np.zeros((1, target_size, target_size)), axis=0)
        num_layers_to_add += 1
    print(f'擴展前層數: {brain_region.shape}')
    return resized_brain, num_layers_to_add

def process_brain_region(seg_target, num_of_yolo_box, x1, x2, y1, y2, whole_brain,max_dim=1023):
    """
    處理大腦區域
    """
    original_info = []
    for box_num in range(1, num_of_yolo_box + 1):
        print(f'處理第 {box_num} 個腦區')
        X_start, X_end = x1[box_num - 1], x2[box_num - 1]
        Y_start, Y_end = y1[box_num - 1], y2[box_num - 1]
        X_length, Y_length = X_end - X_start, Y_end - Y_start

        # 按需要擴展區域 ===============================這裡確保 X_start 和 Y_start 不小於 0 &  X_end 和 Y_end 不超過最大維度（在這個例子中是 1023）
        if X_length > Y_length:
            Y_start = max(Y_start - (X_length - Y_length) // 2, 0)
            Y_end = min(Y_end + (X_length - Y_length) - (X_length - Y_length) // 2, max_dim)
        elif Y_length > X_length:
            X_start = max(X_start - (Y_length - X_length) // 2, 0)
            X_end = min(X_end + (Y_length - X_length) - (Y_length - X_length) // 2, max_dim)
        # 調整大小和填充
        resized_brain, num_layers_added = resize_and_pad(whole_brain, X_start, X_end, Y_start, Y_end)
        print(f'額外添加層數: {num_layers_added}')
#         print(f'擴展前層數: {len(resized_brain) - num_layers_added}')
        print(f'擴展後層數: {resized_brain.shape}')

        # 計算未來滑動切割生成的 cubes 數量
        future_cubes = ((resized_brain.shape[0] - 64) / Step_of_slide + 1) * ((resized_brain.shape[1] - Size_of_cube) / Step_of_slide + 1) ** 2
        original_info.append([f'{file}_No_{box_num}_DLG', X_start, X_end, Y_start, Y_end, len(resized_brain) - num_layers_added, int(future_cubes)])

        # 保存處理後的大腦區域
        np.save(f'{Step1_output_path}/{file}_No_{box_num}_DLG', resized_brain)

    return original_info

# 主程序
original_info = []
Step_of_slide, Size_of_cube = 20, 128
# 清空並創建 使用者程序_第一階段存檔內容_YOLO提取DLG 資料夾
if not os.path.exists(Step1_output_path):
    os.mkdir(Step1_output_path)
else:
    for file in os.listdir(Step1_output_path):
        os.remove(Step1_output_path+'/' + file)

for file in DLG_files:
    DLG = tif.imread(os.path.join('User_Input_DLG_tif', file))
    print(f'Processing {file}, shape: {DLG.shape}')
    WholeBrain_DLG_Amira = DLG.copy()

    # 讀取 YOLO 檢測結果
    # Label標籤: AL=0 MB=1 Central_complex=2
    txt_path = os.path.join('DEMO_YOLO_Inference/exp/labels', f'{file}.txt')
    try:
        with open(txt_path, 'r') as source_file:
            x1, y1, x2, y2 = [], [], [], []
            for line in source_file:
                class_idx, *bbox = map(float, line.split())
                # 只提取標籤為 2 (Central_complex) 的座標#######################################
                if int(class_idx) == 2:
                    x_center, y_center, w, h = bbox[0] * DLG.shape[2], bbox[1] * DLG.shape[1], bbox[2] * DLG.shape[2], bbox[3] * DLG.shape[1]
                    x1.append(round(x_center - w / 2))
                    y1.append(round(y_center - h / 2))
                    x2.append(round(x_center + w / 2))
                    y2.append(round(y_center + h / 2))
#==============這部分可以省略，主要是檢查是否讀取到正確標籤( 注意: 不一定每一個腦區都會偵測到兩個)
#             # 目前是單純做圖檢查!!!!
#             # 作圖展示txt讀取出得特徵框是否正確(右腦)
#             X_start = x1[0]
#             Y_start = y1[0]
#             X_end = x2[0]
#             Y_end = y2[0]
#             X_length = X_end-X_start
#             Y_length = Y_end-Y_start
#             # 使用投影後的最大值2D圖來人工檢查
#             projection = np.max(DLG, axis=0) # 取最大值
#             img_gray = projection
#             #img_gray = img
#             fig, ax = plt.subplots(figsize=(20,20))
#             #plt.axis(False)
#             ax.imshow(img_gray)
#             for i in range(1):
#               # X第一條線（上方橫線）
#               x_1 = np.arange(X_start , X_start+X_length)
#               ax.plot(x_1, np.full_like(x_1,Y_start) , linewidth=2, alpha=1,color='y')

#               # X第二條線(下方橫線)
#               ax.plot(x_1, np.full_like(x_1,Y_start+Y_length) , linewidth=2, alpha=1,color='y')

#               # Y第一條線
#               y_1 = np.arange(Y_start, Y_start+Y_length)
#               ax.plot(np.full_like(y_1,X_start), y_1 , linewidth=2, alpha=1,color='y')

#               ax.plot(np.full_like(y_1,X_start+X_length), y_1 , linewidth=2, alpha=1,color='y')
#             # 判斷是否有兩個特徵框
#             if len(x1)<2:
#                 print('只有一個ROI')
#             else:

#                 X_start = x1[1]
#                 Y_start = y1[1]
#                 X_end = x2[1]
#                 Y_end = y2[1]
#                 X_length = X_end-X_start
#                 Y_length = Y_end-Y_start
#                 for i in range(1):
#                   # X第一條線（上方橫線）
#                   x_1 = np.arange(X_start , X_start+X_length)
#                   ax.plot(x_1, np.full_like(x_1,Y_start) , linewidth=2, alpha=1,color='y')

#                   # X第二條線(下方橫線)
#                   ax.plot(x_1, np.full_like(x_1,Y_start+Y_length) , linewidth=2, alpha=1,color='y')

#                   # Y第一條線
#                   y_1 = np.arange(Y_start, Y_start+Y_length)
#                   ax.plot(np.full_like(y_1,X_start), y_1 , linewidth=2, alpha=1,color='y')

#                   ax.plot(np.full_like(y_1,X_start+X_length), y_1 , linewidth=2, alpha=1,color='y')
#             plt.axis('off')
#             plt.show()
#==============
        # 處理每個檢測到的大腦區域
            # 處理每個檢測到的大腦區域
            if len(x1) > 0:
                # 2024/03/05 新增判斷式: 因為DLG會有可能出現 1024x1700 or 1700x1024的特殊尺寸
                # 所以要選擇大的邊作為 max_dim
                if DLG.shape[1]>=DLG.shape[2]:
                    
                    original_info += process_brain_region(seg_target, len(x1), x1, x2, y1, y2, WholeBrain_DLG_Amira,max_dim=DLG.shape[1]-1)
                    print('此大腦有',len(x1),'個腦區')
                else:
                    original_info += process_brain_region(seg_target, len(x1), x1, x2, y1, y2, WholeBrain_DLG_Amira,max_dim=DLG.shape[2]-1)
                    print('此大腦有',len(x1),'個腦區')
            else:
                print(f'此大腦沒有 {seg_target} 腦區')
    except FileNotFoundError:
        print(f'沒有找到 {txt_path}')
end_time_crop_resize = time.time()  # 記錄結束時間
# 儲存 original_info
np.save(original_info_name,original_info)
# 輸出推理時間
print("批量提取腦區3D區域完成，總耗時: {:.2f} 秒".format(end_time_crop_resize - start_time_crop_resize))


# %%


# Step4 將提取完成的小3D DLG影像進行數據增強(滑動切割) 
DLG_src_files = os.listdir(Step1_output_path)
DLG_src_files = [i for i in DLG_src_files if '_DLG.npy' in i]
print('大腦數量: ',len(DLG_src_files)/2)
# 將列表重新排列
DLG_src_files.sort()

# 清空並創建 使用者程序_第二階段存檔內容_完成滑動切割 資料夾
if not os.path.exists(Step2_output_path):
    os.mkdir(Step2_output_path)
else:
    for file in os.listdir(Step2_output_path):
        os.remove(Step2_output_path+'/' + file)
        
        
def normalize_image(img, lower_bound=0, upper_bound=255):
    """ 將圖像標準化到指定的亮度範圍內 """
    if np.min(img) < lower_bound or np.max(img) > upper_bound:
        img = img - np.min(img)
        img = img / np.max(img) * upper_bound
    return img

def process_and_patch(npy_file, size_of_cube, step_of_slide):
    """ 對單個 NPY 文件進行處理和滑動切割 """
    npy_file = normalize_image(npy_file)
    npy_file_patches = patchify(npy_file, (64, size_of_cube, size_of_cube), step=step_of_slide)
    return np.reshape(npy_file_patches, (-1, npy_file_patches.shape[3], npy_file_patches.shape[4], npy_file_patches.shape[5]))
# 主程序
Size_of_cube = 128  # 長寬尺寸
Step_of_slide = 20  # 滑動步數

output_dir = Step2_output_path
os.makedirs(output_dir, exist_ok=True)

start_time_crop_resize = time.time()

for i, file_name in enumerate(DLG_src_files):
    npy_file = np.load(f'{Step1_output_path}/{file_name}')
    print(file_name)
    input_npy_file_patches = process_and_patch(npy_file, Size_of_cube, Step_of_slide)
    # 為每個文件保存獨立的切片
    np.save(f'{output_dir}/{file_name}', input_npy_file_patches)
    print(input_npy_file_patches.shape)
    del input_npy_file_patches
    gc.collect()

end_time_crop_resize = time.time()

# 輸出推理時間
print("批量滑動切割 cubes 完成，總耗時: {:.2f} 秒".format(end_time_crop_resize - start_time_crop_resize))


# %%


# Step5 載入滑動切割後的檔案，開始進行3D U-Net 腦區切割
# 先導入之前儲存的 original_info
original_info = np.load(original_info_name+'.npy')
# 轉換函數
def convert_array_elements(arr):
    # 將第一個元素保留為字符串，其餘轉換為整數
    return (arr[0],) + tuple(int(x) for x in arr[1:])

# 應用轉換
converted_info = [convert_array_elements(row) for row in original_info]
original_info = converted_info
# 檢查轉換後的結果和類型
type(original_info[0][1])
# 設定 GPU 和模型
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BACKBONE = 'resnet50'
preprocess_input = get_preprocessing(BACKBONE)
my_model = keras.models.load_model('已經完成訓練的3D_UNet模型/'+seg_target+'最佳模型/best_model.h5', compile=False)


# 批量處理 npy 文件
input_dir = Step2_output_path
# 清空並創建 使用者程序_第三階段存檔內容_完成推理 資料夾
if not os.path.exists(Step3_output_path):
    os.mkdir(Step3_output_path)
else:
    for file in os.listdir(Step3_output_path):
        os.remove(Step3_output_path+'/' + file)    
output_dir = Step3_output_path

start_time_seg = time.time()

for file_info in original_info:
    file_name = file_info[0] + '.npy'
    test_img = np.load(os.path.join(input_dir, file_name))
    test_img = np.stack((test_img,) * 1, axis=-1)
    test_img_input = preprocess_input(test_img)

    test_pred = my_model.predict(test_img_input, batch_size=1)
    test_pred = np.argmax(test_pred, axis=4)

    predicted_reshaped = np.reshape(test_pred, (int(int(file_info[-1]) / 9), 3, 3, 64, 128, 128))
    print(predicted_reshaped.shape)
    np.save(os.path.join(output_dir, file_name.replace('.npy', '_predicted')), predicted_reshaped)

    del test_img, test_img_input, test_pred, predicted_reshaped
    gc.collect()

end_time_seg = time.time()
print("3D腦區切割推理完成，總耗時: {:.2f} 秒".format(end_time_seg - start_time_seg))


# %%


# Step6 定義 _unpatchify3d 函數
Imsize = Union[Tuple[int, int], Tuple[int, int, int]]
def _unpatchify3d(  # pylint: disable=too-many-locals
    patches: np.ndarray, imsize: Tuple[int, int, int]
) -> np.ndarray:
    assert len(patches.shape) == 6
    i_h, i_w, i_c = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    n_h, n_w, n_c, p_h, p_w, p_c = patches.shape
    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)
    s_c = 0 if n_c <= 1 else (i_c - p_c) / (n_c - 1)
    # The step size should be same for all patches, otherwise the patches are unable
    # to reconstruct into a image
    if int(s_w) != s_w:
        raise NonUniformStepSizeError(i_w, n_w, p_w, s_w)
    if int(s_h) != s_h:
        raise NonUniformStepSizeError(i_h, n_h, p_h, s_h)
    if int(s_c) != s_c:
        raise NonUniformStepSizeError(i_c, n_c, p_c, s_c)
    s_w = int(s_w)
    s_h = int(s_h)
    s_c = int(s_c)

    i, j, k = 0, 0, 0
    while True:
        i_o, j_o, k_o = i * s_h, j * s_w, k * s_c
        # 原本合併(直接賦予數值)
        #image[i_o : i_o + p_h, j_o : j_o + p_w, k_o : k_o + p_c] = patches[i, j, k]
        # 修改成累加(投票)
        image[i_o : i_o + p_h, j_o : j_o + p_w, k_o : k_o + p_c] = image[i_o : i_o + p_h, j_o : j_o + p_w, k_o : k_o + p_c] + patches[i, j, k]

        if k < n_c - 1:
            k = min((k_o + p_c) // s_c, n_c - 1)
        elif j < n_w - 1 and k >= n_c - 1:
            j = min((j_o + p_w) // s_w, n_w - 1)
            k = 0
        elif i < n_h - 1 and j >= n_w - 1 and k >= n_c - 1:
            i = min((i_o + p_h) // s_h, n_h - 1)
            j = 0
            k = 0
        elif i >= n_h - 1 and j >= n_w - 1 and k >= n_c - 1:
            # Finished
            break
        else:
            raise RuntimeError("Unreachable")
    return image


# %%


# Step7 使用unpatchify對預測結果還原成Z軸拓展後的 尺寸(124,168,168) or (144,168,168)
start_time_seg = time.time()
# 清空並創建 使用者程序_第四階段存檔內容_完成還原Z拓展後尺寸_母果蠅 資料夾
if not os.path.exists(Step4_output_path):
    os.mkdir(Step4_output_path)
else:
    for file in os.listdir(Step4_output_path):
        os.remove(Step4_output_path+'/' + file)  
        
def get_reconstructed_size(original_z, target_size=168, step_of_slide=20, z_min_size=64):
    """ 計算還原後的尺寸 """
    num_layers_to_add = 0
    while original_z + num_layers_to_add < z_min_size or (original_z + num_layers_to_add - z_min_size) % step_of_slide != 0:
        num_layers_to_add += 1
    return original_z + num_layers_to_add
def save_reconstructed_images(original_info, output_dir, prefix=Step3_output_path+'/'):
    """ 批量保存重建的影像 """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, info in enumerate(original_info):
        file_name = info[0] + '_predicted.npy'
        predicted_reshaped = np.load(prefix + file_name)

        reconstructed_size = get_reconstructed_size(info[5])
        reconstructed = _unpatchify3d(predicted_reshaped, (reconstructed_size, 168, 168))
        reconstructed[reconstructed > 0] = 1
        # ==============================2024/02/05 新增去除雜訊==============================
        labels = measure.label(reconstructed, connectivity=1)# 這是最嚴苛，只有在同一個平面有在四個方向有連接才算同一個物體
        regions = measure.regionprops(labels)
        # 計算封閉區域的數量
        num_regions = len(regions)
        print(info[0])
        # 計算每個區域的像素數量
        for region in regions:
            print("Region area: ", region.area)
        print("Number of closed areas: ", num_regions)
        # 找到最大的Region area
        # =================2024/02/20 ==========新增判斷式(因為YOLO會偵測到3個腦區，3D UNet 推理後會是空矩陣)
        # 跳過 np.max(reconstructed)<1的檔案
        if np.max(reconstructed)<1:
            # 保存重建的影像
            np.save(os.path.join(output_dir, info[0] + '_reconstructed.npy'), reconstructed)
            continue
        else:

            max_area_index = np.argmax([region.area for region in regions])
            # 設置除了最大的Region area之外的所有區域為0
            for i, region in enumerate(regions):
                if i != max_area_index:
                    reconstructed[tuple(region.coords.T)] = 0
            # 將大於0的位置設為1
            reconstructed[reconstructed>0] =1
            # ==========================================================================================
            # 保存重建的影像
            np.save(os.path.join(output_dir, info[0] + '_reconstructed.npy'), reconstructed)
        # =======================================================
# 使用函数保存重建的影像
output_dir = Step4_output_path
save_reconstructed_images(original_info, output_dir)
end_time_seg = time.time()
print("預測結果完成還原Z拓展，總耗時: {:.2f} 秒".format(end_time_seg - start_time_seg))


# %%


# Step8 將模型切割的腦區縮放回原始解析度
start_time_seg = time.time()
def resize_reconstructed(reconstructed, original_X_width, original_Y_width, target_size=168):
    """根據原始寬度調整重建影像的大小"""
    resized = []
    for slice in reconstructed:
        # 如果原始寬度小於目標大小，則使用INTER_AREA進行縮小 (INTER_AREA 效果很好)
        if original_X_width < target_size and original_Y_width < target_size:
            resized_slice = cv2.resize(slice, (original_X_width, original_Y_width), interpolation=cv2.INTER_AREA)
        # 如果原始寬度大於目標大小，則使用INTER_CUBIC進行放大
        elif original_X_width > target_size and original_Y_width > target_size:
            resized_slice = cv2.resize(slice, (original_X_width, original_Y_width), interpolation=cv2.INTER_CUBIC)
        # 如果原始寬度等於目標大小，則不需要調整大小
        else:
            # 如果原始尺寸跟切割完的相比有大有小，優先選擇縮小演算法去還原回 original_X_width
            resized_slice = cv2.resize(slice, (original_X_width, original_Y_width), interpolation=cv2.INTER_AREA)
        resized.append(resized_slice)
    return np.array(resized)


def process_and_save_reconstructed_images(input_dir, original_info):
    """大量處理並保存重建的影像"""
    for info in original_info:
        file_name = info[0] + '_reconstructed.npy'
        reconstructed = np.load(os.path.join(input_dir, file_name))
        #============注意!! 因 DLG[:,info[3]:info[4],info[1]:info[2]] 不一定每一個腦區區域都是正方形
        #============DLG是 Z Y X 順序
        #============所以寬不能統一只用一個數值
        original_X_width = abs(info[1] - info[2])
        original_Y_width = abs(info[3] - info[4])
        print('這個檔案名稱:',file_name)
        print('X寬:',original_X_width)
        print('Y寬:',original_Y_width)

        # 删除额外添加的 Z 轴层
        need_del_num = abs(info[5] - len(reconstructed))
        if need_del_num != 0:
            reconstructed = np.delete(reconstructed, slice(-need_del_num, None), axis=0)
            
        # 將 reconstructed 轉換數據格式 成 uint8
        reconstructed = reconstructed.astype('uint8')
        # 调整大小
        reconstructed_resized = resize_reconstructed(reconstructed, original_X_width,original_Y_width)
        print('完成還原原始解析度尺寸:',reconstructed_resized.shape)
        # 保存處理後的影像
        np.save(os.path.join(input_dir, file_name), reconstructed_resized)

# 使用函数處理並保存重建的影像
input_dir = Step4_output_path
process_and_save_reconstructed_images(input_dir, original_info)
end_time_seg = time.time()
print("完成還原原始解析度，總耗時: {:.2f} 秒".format(end_time_seg - start_time_seg))


# %%


# # 投影成2D圖
# input_dir = Step4_output_path
# reconstructed = np.load(os.path.join(input_dir, "DLG_G0239_F_000005.tif_No_1_DLG_reconstructed.npy"))
# print(reconstructed.shape)
# projection = np.sum(reconstructed, axis=0)  # 取加總值
# # 顯示2D圖像
# plt.figure(figsize=(10, 10))
# plt.imshow(projection, cmap='gray')
# plt.axis('off')
# plt.show()


# %%


# Step9 將原始解析度的腦區結果放回原始畫布
start_time_seg = time.time()
# 清空並創建 使用者程序_第五階段存檔內容_完成放回原始影像_母果蠅 資料夾
if not os.path.exists(Step5_output_path):
    os.mkdir(Step5_output_path)
else:
    for file in os.listdir(Step5_output_path):
        os.remove(Step5_output_path+'/' + file)  
        
        
nii_gz_src = "/mnt/nas_1/LoLab/kaiyi_hsu/Ouput_from_Amira/npy_files/"

input_dir = Step4_output_path

# 用雙層迴圈處理: 外層是資料夾內原始大腦數量，內層是已經完成切割的腦區數量
for ii in DLG_files:
    # ii == 本次處理的大腦名稱
    DLG = tif.imread(os.path.join('User_Input_DLG_tif', ii))
    # 也同時輸出原始DLG的tif檔案(這樣在Avizo視覺化才會匹配位置)
    tif.imsave(Step5_output_path+'/'+ii[:-4]+'.tif',DLG.astype('uint16'))

    # 清空DLG，只保留尺寸
    DLG = DLG*0
    
    # 這個迴圈是要先檢查目前處理的大腦名稱 ii 對應的切割結果
    for info in original_info:
        file_name = info[0] + '_reconstructed.npy'
        # 檢查腦區變數名稱和原始大腦名稱是否匹配，若匹配則將腦區加入此大腦畫布
        if ii in file_name:
            reconstructed = np.load(os.path.join(input_dir, file_name))
            print('目前處理: ',file_name)
            print('腦區放回前最大值: ',np.max(DLG))
            # 開始將腦區結果放回 !! 注意: DLG是 Z Y X 順序 修改成空白DLG和腦區結果相加
            # 因為如果直接賦予數值會因為Bounding Box重疊而發生腦區被切掉，用相加在用二值化即可
            DLG[:,info[3]:info[4],info[1]:info[2]] = DLG[:,info[3]:info[4],info[1]:info[2]] + reconstructed
            print('腦區放回後最大值: ',np.max(DLG))
#* 輸出腦區標籤號碼: 
# AL=1
# MB=2
# CAL=3
# FB=4
# EB=5
# PB=6
    DLG[DLG>0]=4 # 重疊腦區的位置都成最終腦區標籤
    print('----------------查看--------------')
    tif.imsave(Step5_output_path+'/Seg_'+seg_target+'_'+ii[:-4]+'.tif',DLG.astype('uint16'))
    # 最後將DLG & Seg 結果都輸出到 特殊果蠅腦_視覺化資料夾，以便後續和其他腦區合併
    
end_time_seg = time.time()
print("完成輸出大腦DLG和Seg結果tif檔案，總耗時: {:.2f} 秒".format(end_time_seg - start_time_seg))
    
    
    

