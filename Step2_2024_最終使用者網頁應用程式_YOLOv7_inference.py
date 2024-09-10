# -*- coding: utf-8 -*-
# +
# # !python detect_custom_color.py --conf-thres 0.5 --weights ./AllinOne_trained/best.pt --source DEMO_For_YOLO_as_Input --img-size 640 --save-txt --project DEMO_For_YOLO_as_Input --class-colors "0 0 255 240 176 0 0 255 0"

# +
# Label標籤: AL=0 MB=1 Central_complex=2
# 目的：
# 此py檔案專門用推理3個 腦區ROI區域
# 需要條件：
# 1. 已經是2D圖片
# 2. 需要指定來源和輸出資料夾(來源資料夾: DEMO_For_YOLO_as_Input   輸出資料夾：DEMO_YOLO_Inference) *測試時來源：測試用_人工驗證
# 3. 固定使用權重: ./AllinOne_trained/train/exp5/weights/best.pt

# +
import subprocess
import os
import shutil
# 設定儲存母資料夾路徑: 網頁應用程式暫存資料夾 # 這個通常不會動
folder = 'DEMO_YOLO_Inference/'
# 清空並重建 'folder' 資料夾
if os.path.exists(folder):
    shutil.rmtree(folder)
    
def run_yolo_inference():
    # 定義命令
    command = [
        'python', 'detect_custom_color.py', 
        '--conf-thres', '0.5', 
        '--weights', './AllinOne_trained/best.pt', 
        '--source', 'DEMO_For_YOLO_as_Input', 
        '--img-size', '640', 
        '--save-txt', 
        '--project', 'DEMO_YOLO_Inference', 
        '--class-colors', '0 0 255 240 176 0 0 255 0',
        '--no-trace'
    ]

    # 執行命令
    result = subprocess.run(command, capture_output=True, text=True)
    
    # 打印輸出結果（如果需要）
    print("標準輸出:", result.stdout)
#     print("標準錯誤:", result.stderr)

if __name__ == "__main__":
    run_yolo_inference()

# -


