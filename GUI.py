#!/usr/bin/env python
# coding: utf-8
# %%
# # 初版: 上傳並完成YOLO，展示圖片版本
# test
# import base64
# import os
# from flask import Flask, send_from_directory
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc
# import subprocess
# from werkzeug.utils import secure_filename
# import tifffile as tif

# # 設定目錄
# UPLOAD_FOLDER = 'User_Input_DLG_tif/'
# OUTPUT_FOLDER = 'DEMO_YOLO_Inference/exp/'
# ALLOWED_EXTENSIONS = {'tif'}
# MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# # 檢查上傳的檔案格式
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # 建立 Flask 伺服器
# server = Flask(__name__)
# server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# server.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # 建立 Dash 應用程式
# app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # UI Layout
# app.layout = html.Div([
#     html.H1("腦區選擇與 YOLOv7 影像處理"),
#     dcc.Upload(
#         id='upload-image',
#         children=html.Button('上傳 TIF 檔案', className='btn btn-primary'),
#         multiple=False
#     ),
#     html.Div(id='upload-output'),
#     dbc.Checklist(
#         id='region-switches',
#         options=[
#             {'label': 'AL', 'value': 'AL'},
#             {'label': 'MB', 'value': 'MB'},
#             {'label': 'CAL', 'value': 'CAL'},
#             {'label': 'FB', 'value': 'FB'},
#             {'label': 'EB', 'value': 'EB'},
#             {'label': 'PB', 'value': 'PB'},
#         ],
#         switch=True,
#         inline=True
#     ),
#     html.Button('SUBMIT', id='submit-button', n_clicks=0, className='btn btn-success'),
#     html.Div(id='output-image')
# ])

# # 上傳檔案處理
# @app.callback(
#     Output('upload-output', 'children'),
#     [Input('upload-image', 'filename'), Input('upload-image', 'contents')]
# )
# # 檢查檔案是否為正確的 TIF 並可讀取
# def handle_upload(filename, contents):
#     if filename is not None and allowed_file(filename):
#         # 將 Base64 編碼的內容解碼成二進位格式
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)

#         filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

#         # 儲存上傳的 TIF 檔案
#         with open(filepath, 'wb') as f:
#             f.write(decoded)  # 寫入二進位格式

#         # 讀取並檢查 TIF 檔案
#         try:
#             DLG = tif.imread(filepath)
#             print(f'Processing {filename}, shape: {DLG.shape}')
#             return html.Div(f"已成功上傳: {filename}")
#         except Exception as e:
#             print(f"無法讀取 TIF 檔案: {e}")
#             return html.Div("檔案上傳失敗，請上傳正確的 TIF 檔案")
    
#     return html.Div("請上傳 TIF 檔案")

# # 提交按鈕處理
# @app.callback(
#     Output('output-image', 'children'),
#     [Input('submit-button', 'n_clicks')],
#     [State('region-switches', 'value'), State('upload-image', 'filename')]
# )

# # 修改後的 process_image 函數，動態抓取 PNG 檔案
# def process_image(n_clicks, selected_regions, filename):
#     if n_clicks > 0 and filename is not None:
#         # 執行後端程式 Step1 和 Step2
#         tif_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
#         subprocess.run(['python', 'Step1_讀取DLG_tif檔案_輸出成2D投影圖.py', tif_filepath])
#         subprocess.run(['python', 'Step2_2024_最終使用者網頁應用程式_YOLOv7_inference.py', '--regions', ','.join(selected_regions)])
        
#         # 動態檢查 PNG 檔案名稱
#         png_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.png')]
        
#         if png_files:
#             # 取出資料夾內的第一個 PNG 檔案顯示
#             png_image_path = os.path.join(OUTPUT_FOLDER, png_files[0])
#             return html.Img(src=f'/output/{secure_filename(png_files[0])}')
#         return html.Div("尚未生成影像，請稍後再試")
    
#     return ""

# # 伺服器靜態檔案路由
# @server.route('/output/<filename>')
# def send_image(filename):
#     return send_from_directory(OUTPUT_FOLDER, filename)

# # 啟動伺服器
# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run_server(debug=True, port=16362)


# %%
# # 第二版本: 已經可以完成3D U-Net切割並展示切割後的投影圖(尚未新增防呆(YOLO偵測不到ROI，阻擋下一步)功能
# import os
# import base64
# import subprocess
# from flask import Flask, send_from_directory
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc
# import tifffile as tif
# from werkzeug.utils import secure_filename


# # 設定目錄
# UPLOAD_FOLDER = 'User_Input_DLG_tif/'
# YOLO_OUTPUT_FOLDER = 'DEMO_YOLO_Inference/exp/'
# ALLOWED_EXTENSIONS = {'tif'}
# MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# # 檢查上傳的檔案格式
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # 建立 Flask 伺服器
# server = Flask(__name__)
# server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# server.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # 建立 Dash 應用程式
# app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # UI Layout
# app.layout = html.Div([
#     html.H1("腦區選擇與 YOLOv7 影像處理"),
#     dcc.Upload(
#         id='upload-image',
#         children=html.Button('上傳 TIF 檔案', className='btn btn-primary'),
#         multiple=False
#     ),
#     html.Div(id='upload-output'),
#     dbc.Checklist(
#         id='region-switches',
#         options=[
#             {'label': 'AL', 'value': 'AL'},
#             {'label': 'MB', 'value': 'MB'},
#             {'label': 'CAL', 'value': 'CAL'},
#             {'label': 'FB', 'value': 'FB'},
#             {'label': 'EB', 'value': 'EB'},
#             {'label': 'PB', 'value': 'PB'},
#         ],
#         switch=True,
#         inline=True
#     ),
#     html.Button('SUBMIT', id='submit-button', n_clicks=0, className='btn btn-success'),
#     html.Div(id='output-image'),
#     html.Button('NEXT', id='next-button', n_clicks=0, className='btn btn-warning', style={'display': 'none'}),
#     html.Div(id='processing-status')
# ])

# # 上傳檔案處理
# @app.callback(
#     Output('upload-output', 'children'),
#     [Input('upload-image', 'filename'), Input('upload-image', 'contents')]
# )
# def handle_upload(filename, contents):
#     if filename is not None and allowed_file(filename):
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)

#         filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

#         # 儲存上傳的 TIF 檔案
#         with open(filepath, 'wb') as f:
#             f.write(decoded)

#         try:
#             DLG = tif.imread(filepath)
#             print(f'Processing {filename}, shape: {DLG.shape}')
#             return html.Div(f"已成功上傳: {filename}")
#         except Exception as e:
#             print(f"無法讀取 TIF 檔案: {e}")
#             return html.Div("檔案上傳失敗，請上傳正確的 TIF 檔案")
    
#     return html.Div("請上傳 TIF 檔案")

# # 提交按鈕處理
# @app.callback(
#     [Output('output-image', 'children'), Output('next-button', 'style')],
#     [Input('submit-button', 'n_clicks')],
#     [State('region-switches', 'value'), State('upload-image', 'filename')]
# )
# def process_image(n_clicks, selected_regions, filename):
#     if n_clicks > 0 and filename is not None:
#         tif_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
#         # 改良：無需提供其他資訊
#         subprocess.run(['python', 'Step1_讀取DLG_tif檔案_輸出成2D投影圖.py'])
#         # 改良：無需提供其他資訊
#         subprocess.run(['python', 'Step2_2024_最終使用者網頁應用程式_YOLOv7_inference.py'])

#         png_files = [f for f in os.listdir(YOLO_OUTPUT_FOLDER) if f.endswith('.png')]
        
#         if png_files:
#             png_image_path = os.path.join(YOLO_OUTPUT_FOLDER, png_files[0])
#             return html.Img(src=f'/output/yolo/{secure_filename(png_files[0])}'), {'display': 'inline'}
#         return html.Div("尚未生成影像，請稍後再試"), {'display': 'none'}
    
#     return "", {'display': 'none'}


# # NEXT 按鈕處理，依次執行 Step3
# @app.callback(
#     Output('processing-status', 'children'),
#     [Input('next-button', 'n_clicks')],
#     [State('region-switches', 'value')]
# )
# def process_next(n_clicks, selected_regions):
#     if n_clicks > 0 and selected_regions:
#         status_messages = []
#         for region in selected_regions:
#             status_message = f"正在切割 {region} 腦區..."
#             status_messages.append(html.Div(status_message))
#             script_name = f"Step3_YOLO結果預處理給3D_UNet切割_輸出放回原始影像_{region}.py"
#             subprocess.run(['python', script_name])

#         # 執行 Step4
#         subprocess.run(['python', 'Step4_合併所有原始解析度切割結果影像＿輸出.py'])

#         # 最終結果影像
#         status_messages.append(html.Div("所有腦區切割已完成，正在展示最終合併影像..."))
#         status_messages.append(html.Img(src='/output/brain/brain_regions_projection.png'))

#         return status_messages
    
#     return html.Div("請至少選擇一個腦區進行切割")

# # 更新 OUTPUT_FOLDER 設置
# BRAIN_REGIONS_FOLDER = '最後合併腦區結果圖/'  # 新增這個路徑

# # 伺服器靜態檔案路由 - 提供 YOLO 推理的圖片
# @server.route('/output/yolo/<filename>')
# def send_yolo_image(filename):
#     return send_from_directory(YOLO_OUTPUT_FOLDER, filename)

# # 伺服器靜態檔案路由 - 提供 brain_regions_projection.png
# @server.route('/output/brain/<filename>')
# def send_brain_image(filename):
#     return send_from_directory(BRAIN_REGIONS_FOLDER, filename)

# # ====
# # 啟動伺服器
# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run_server(debug=True, port=16362)



# %%
# # 第三版本: 新增完成3D U-Net切割後的下載tif按鈕(尚未新增防呆(YOLO偵測不到ROI，阻擋下一步)功能
# import shutil
# import os
# import base64
# import subprocess
# from flask import Flask, send_from_directory
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc
# import tifffile as tif
# from werkzeug.utils import secure_filename

# # 設定目錄
# UPLOAD_FOLDER = 'User_Input_DLG_tif/'
# YOLO_OUTPUT_FOLDER = 'DEMO_YOLO_Inference/exp/'
# ALLOWED_EXTENSIONS = {'tif'}
# MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# # 用於腦區的下載路徑前綴
# BRAIN_REGION_PATHS = {
#     'AL': '網頁應用程式暫存資料夾_AL專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'MB': '網頁應用程式暫存資料夾_MB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'CAL': '網頁應用程式暫存資料夾_CAL專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'FB': '網頁應用程式暫存資料夾_FB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'EB': '網頁應用程式暫存資料夾_EB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'PB': '網頁應用程式暫存資料夾_PB專用/使用者程序_第五階段存檔內容_完成放回原始影像/'
# }

# # 檢查上傳的檔案格式
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # 建立 Flask 伺服器
# server = Flask(__name__)
# server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# server.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # 建立 Dash 應用程式
# app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # UI Layout
# app.layout = html.Div([
#     html.H1("LYNSU: 自動化腦區切割"),
#     dcc.Upload(
#         id='upload-image',
#         children=html.Button('上傳 TIF 檔案', className='btn btn-primary'),
#         multiple=False
#     ),
#     html.Div(id='upload-output'),
#     dbc.Checklist(
#         id='region-switches',
#         options=[
#             {'label': 'AL', 'value': 'AL'},
#             {'label': 'MB', 'value': 'MB'},
#             {'label': 'CAL', 'value': 'CAL'},
#             {'label': 'FB', 'value': 'FB'},
#             {'label': 'EB', 'value': 'EB'},
#             {'label': 'PB', 'value': 'PB'},
#         ],
#         switch=True,
#         inline=True
#     ),
#     html.Button('SUBMIT', id='submit-button', n_clicks=0, className='btn btn-success'),
#     html.Div(id='output-image'),
#     html.Button('NEXT', id='next-button', n_clicks=0, className='btn btn-warning', style={'display': 'none'}),
#     html.Div(id='processing-status'),
#     html.Div(id='download-buttons')  # 用於顯示下載按鈕
# ])

# # 上傳檔案處理
# @app.callback(
#     Output('upload-output', 'children'),
#     [Input('upload-image', 'filename'), Input('upload-image', 'contents')]
# )

# def handle_upload(filename, contents):
#     if filename is not None and allowed_file(filename):
#         # 清空 User_Input_DLG_tif 資料夾
#         if os.path.exists(UPLOAD_FOLDER):
#             shutil.rmtree(UPLOAD_FOLDER)  # 刪除整個資料夾及其內容
#         os.makedirs(UPLOAD_FOLDER)  # 重新創建空資料夾
#         # 保守起見: 也先刪除 YOLO_OUTPUT_FOLDER，因為防呆: 有人會上傳未完成時按下SUBMIT，導致YOLO沒跑就先呈現之前圖片
#         # 清空 User_Input_DLG_tif 資料夾
#         if os.path.exists(YOLO_OUTPUT_FOLDER):
#             shutil.rmtree(YOLO_OUTPUT_FOLDER)  # 刪除整個資料夾及其內容
#         os.makedirs(YOLO_OUTPUT_FOLDER)  # 重新創建空資料夾
        
        
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)

#         filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

#         # 儲存上傳的 TIF 檔案
#         with open(filepath, 'wb') as f:
#             f.write(decoded)

#         try:
#             DLG = tif.imread(filepath)
#             print(f'Processing {filename}, shape: {DLG.shape}')
#             return html.Div(f"已成功上傳: {filename}")
#         except Exception as e:
#             print(f"無法讀取 TIF 檔案: {e}")
#             return html.Div("檔案上傳失敗，請上傳正確的 TIF 檔案")
    
#     return html.Div("*單檔上傳果蠅大腦DLG.tif影像，果蠅大腦方向請勿上下顛倒，允許Anterior & Posterior 任一方向!")

# # 提交按鈕處理
# @app.callback(
#     [Output('output-image', 'children'), Output('next-button', 'style')],
#     [Input('submit-button', 'n_clicks')],
#     [State('region-switches', 'value'), State('upload-image', 'filename')]
# )
# def process_image(n_clicks, selected_regions, filename):
#     if n_clicks > 0 and filename is not None:
#         tif_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
#         subprocess.run(['python', 'Step1_讀取DLG_tif檔案_輸出成2D投影圖.py'])
#         subprocess.run(['python', 'Step2_2024_最終使用者網頁應用程式_YOLOv7_inference.py'])

#         png_files = [f for f in os.listdir(YOLO_OUTPUT_FOLDER) if f.endswith('.png')]
        
#         if png_files:
#             png_image_path = os.path.join(YOLO_OUTPUT_FOLDER, png_files[0])
#             return [html.Img(src=f'/output/yolo/{secure_filename(png_files[0])}'),html.Div("請點擊NEXT按鈕開始切割3D腦區!")], {'display': 'inline'}
#         return html.Div("尚未生成影像，請稍後再試"), {'display': 'none'}

#     elif n_clicks > 0:
#         # 目的: 如果還沒上傳檔案按下SUBMIT，顯示請先上傳檔案
#         return html.Div("請先上傳檔案再點擊SUBMIT按鈕!"), {'display': 'none'}
#     return "", {'display': 'none'}

# # NEXT 按鈕處理，依次執行 Step3
# @app.callback(
#     [Output('processing-status', 'children'), Output('download-buttons', 'children')],
#     [Input('next-button', 'n_clicks')],
#     [State('region-switches', 'value')]
# )
# def process_next(n_clicks, selected_regions):
#     if n_clicks > 0 and selected_regions:
#         status_messages = []
#         download_buttons = []
        
#         for region in selected_regions:
#             status_message = f"正在切割 {region} 腦區..."
#             status_messages.append(html.Div(status_message))
#             script_name = f"Step3_YOLO結果預處理給3D_UNet切割_輸出放回原始影像_{region}.py"
#             subprocess.run(['python', script_name])
            
#             # 動態生成下載按鈕的路徑
#             brain_region_folder = BRAIN_REGION_PATHS[region]
#             tif_file = [f for f in os.listdir(brain_region_folder) if f.startswith('Seg') and f.endswith('.tif')]
#             if tif_file:
#                 download_buttons.append(html.A(f'下載 {region}.tif', href=f'/download/{region}/{tif_file[0]}', download=f'{region}.tif', className='btn btn-secondary'))
#         # 執行 Step4
#         subprocess.run(['python', 'Step4_合併所有原始解析度切割結果影像＿輸出.py'])

#         # 最終結果影像
#         Final_status_messages = []
#         Final_status_messages.append(html.Div("所有腦區切割已完成，正在展示最終合併影像..."))
#         Final_status_messages.append(html.Div("*注意: 下方圖像是不區分前後順序的展示，目的是展示切割腦區【數量】是否正確"))
#         Final_status_messages.append(html.Img(src='/output/brain/brain_regions_projection.png'))

#         return Final_status_messages, download_buttons
#     elif n_clicks > 0:
#         return html.Div("請至少選擇一個腦區進行切割!"), []
#     return html.Div(""), []

# # 更新 Flask 路由，處理下載
# @server.route('/download/<region>/<filename>')
# def download_file(region, filename):
#     brain_region_folder = BRAIN_REGION_PATHS[region]
#     return send_from_directory(brain_region_folder, filename)

# # 提供 YOLO 推理的圖片
# @server.route('/output/yolo/<filename>')
# def send_yolo_image(filename):
#     return send_from_directory(YOLO_OUTPUT_FOLDER, filename)

# # 提供 brain_regions_projection.png
# @server.route('/output/brain/<filename>')
# def send_brain_image(filename):
#     return send_from_directory('最後合併腦區結果圖/', filename)

# # 啟動伺服器
# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run_server(debug=True, port=16362)


# %%
# # Step4 第四版本: 切割和下載都完成，新增3D 切割進度文字說明。
# # 剩下排版需要加強(以及YOLO後的防呆)
# import shutil
# import os
# import base64
# import subprocess
# from flask import Flask, send_from_directory
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc
# import tifffile as tif
# from werkzeug.utils import secure_filename

# # 設定目錄
# UPLOAD_FOLDER = 'User_Input_DLG_tif/'
# YOLO_OUTPUT_FOLDER = 'DEMO_YOLO_Inference/exp/'
# ALLOWED_EXTENSIONS = {'tif'}
# MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# # 用於腦區的下載路徑前綴
# BRAIN_REGION_PATHS = {
#     'AL': '網頁應用程式暫存資料夾_AL專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'MB': '網頁應用程式暫存資料夾_MB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'CAL': '網頁應用程式暫存資料夾_CAL專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'FB': '網頁應用程式暫存資料夾_FB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'EB': '網頁應用程式暫存資料夾_EB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'PB': '網頁應用程式暫存資料夾_PB專用/使用者程序_第五階段存檔內容_完成放回原始影像/'
# }

# # 檢查上傳的檔案格式
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # 建立 Flask 伺服器
# server = Flask(__name__)
# server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# server.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # 建立 Dash 應用程式
# app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # UI Layout
# app.layout = html.Div([
#     html.H1("LYNSU: 自動化腦區切割"),
#     dcc.Upload(
#         id='upload-image',
#         children=html.Button('上傳 TIF 檔案', className='btn btn-primary'),
#         multiple=False
#     ),
#     html.Div(id='upload-output'),
#     dbc.Checklist(
#         id='region-switches',
#         options=[
#             {'label': 'AL', 'value': 'AL'},
#             {'label': 'MB', 'value': 'MB'},
#             {'label': 'CAL', 'value': 'CAL'},
#             {'label': 'FB', 'value': 'FB'},
#             {'label': 'EB', 'value': 'EB'},
#             {'label': 'PB', 'value': 'PB'},
#         ],
#         switch=True,
#         inline=True
#     ),
#     html.Button('SUBMIT', id='submit-button', n_clicks=0, className='btn btn-success'),
#     html.Div(id='output-image'),
#     html.Button('NEXT', id='next-button', n_clicks=0, className='btn btn-warning', style={'display': 'none'}),
#     html.Button('檢查當前3D腦區切割進度', id='check-status-button', n_clicks=0, className='btn btn-info', style={'display': 'none'}),
#     html.Div(id='processing-status'),
#     html.Div(id='download-buttons'),  # 用於顯示下載按鈕
#     html.Div(id='progress-status')  # 用于显示检查进度结果
# ])

# # 上傳檔案處理
# @app.callback(
#     Output('upload-output', 'children'),
#     [Input('upload-image', 'filename'), Input('upload-image', 'contents')]
# )
# def handle_upload(filename, contents):
#     if filename is not None and allowed_file(filename):
#         # 清空 User_Input_DLG_tif 資料夾
#         if os.path.exists(UPLOAD_FOLDER):
#             shutil.rmtree(UPLOAD_FOLDER)  # 刪除整個資料夾及其內容
#         os.makedirs(UPLOAD_FOLDER)  # 重新創建空資料夾
#         # 保守起見: 也先刪除 YOLO_OUTPUT_FOLDER，因為防呆: 有人會上傳未完成時按下SUBMIT，導致YOLO沒跑就先呈現之前圖片
#         if os.path.exists(YOLO_OUTPUT_FOLDER):
#             shutil.rmtree(YOLO_OUTPUT_FOLDER)  # 刪除整個資料夾及其內容
#         os.makedirs(YOLO_OUTPUT_FOLDER)  # 重新創建空資料夾
        
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)

#         filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

#         # 儲存上傳的 TIF 檔案
#         with open(filepath, 'wb') as f:
#             f.write(decoded)

#         try:
#             DLG = tif.imread(filepath)
#             print(f'Processing {filename}, shape: {DLG.shape}')
#             return html.Div(f"已成功上傳: {filename}")
#         except Exception as e:
#             print(f"無法讀取 TIF 檔案: {e}")
#             return html.Div("檔案上傳失敗，請上傳正確的 TIF 檔案")
    
#     return html.Div("*單檔上傳果蠅大腦DLG.tif影像，果蠅大腦方向請勿上下顛倒，允許Anterior & Posterior 任一方向!")

# # 提交按鈕處理
# @app.callback(
#     [Output('output-image', 'children'), Output('next-button', 'style')],
#     [Input('submit-button', 'n_clicks')],
#     [State('region-switches', 'value'), State('upload-image', 'filename')]
# )
# def process_image(n_clicks, selected_regions, filename):
#     if n_clicks > 0 and filename is not None:
#         tif_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
#         subprocess.run(['python', 'Step1_讀取DLG_tif檔案_輸出成2D投影圖.py'])
#         subprocess.run(['python', 'Step2_2024_最終使用者網頁應用程式_YOLOv7_inference.py'])

#         png_files = [f for f in os.listdir(YOLO_OUTPUT_FOLDER) if f.endswith('.png')]
        
#         if png_files:
#             png_image_path = os.path.join(YOLO_OUTPUT_FOLDER, png_files[0])
#             return [html.Img(src=f'/output/yolo/{secure_filename(png_files[0])}'), html.Div("請點擊NEXT按鈕開始切割3D腦區!")], {'display': 'inline'}
#         return html.Div("尚未生成影像，請稍後再試"), {'display': 'none'}

#     elif n_clicks > 0:
#         return html.Div("請先上傳檔案再點擊SUBMIT按鈕!"), {'display': 'none'}
#     return "", {'display': 'none'}

# # NEXT 按鈕處理，依次執行 Step3 並顯示進度檢查按鈕
# @app.callback(
#     [Output('processing-status', 'children'), Output('download-buttons', 'children'), Output('check-status-button', 'style')],
#     [Input('next-button', 'n_clicks')],
#     [State('region-switches', 'value')]
# )
# def process_next(n_clicks, selected_regions):
#     if n_clicks > 0 and selected_regions:
#         status_messages = []
#         download_buttons = []
        
#         for region in selected_regions:
#             script_name = f"Step3_YOLO結果預處理給3D_UNet切割_輸出放回原始影像_{region}.py"
#             subprocess.run(['python', script_name])
            
#             # 動態生成下載按鈕的路徑
#             brain_region_folder = BRAIN_REGION_PATHS[region]
#             tif_file = [f for f in os.listdir(brain_region_folder) if f.startswith('Seg') and f.endswith('.tif')]
#             if tif_file:
#                 download_buttons.append(html.A(f'下載 {region}.tif', href=f'/download/{region}/{tif_file[0]}', download=f'{region}.tif', className='btn btn-secondary'))
        
#         # 執行 Step4
#         subprocess.run(['python', 'Step4_合併所有原始解析度切割結果影像＿輸出.py'])

#         # 最終結果影像
#         status_messages.append(html.Div("所有腦區切割已完成，正在展示最終合併影像..."))
#         status_messages.append(html.Div("*注意: 下方圖像是不區分前後順序的展示，目的是展示切割腦區【數量】是否正確"))
#         status_messages.append(html.Img(src='/output/brain/brain_regions_projection.png'))

#         # 顯示檢查進度按鈕
#         return status_messages, download_buttons, {'display': 'inline'}
#     elif n_clicks > 0:
#         return html.Div("請至少選擇一個腦區進行切割!"), [], {'display': 'inline'}
#     return html.Div(""), [], {'display': 'none'}

# # "檢查當前3D腦區切割進度" 按鈕處理
# @app.callback(
#     Output('progress-status', 'children'),
#     [Input('check-status-button', 'n_clicks')],
#     [State('region-switches', 'value')]
# )
# def check_progress(n_clicks, selected_regions):
#     if n_clicks > 0 and selected_regions:
#         progress_messages = []
#         for region in selected_regions:
#             brain_region_folder = BRAIN_REGION_PATHS[region]
#             tif_file = [f for f in os.listdir(brain_region_folder) if f.startswith('Seg') and f.endswith('.tif')]
#             if tif_file:
#                 progress_messages.append(html.Div(f"===== {region} 腦區已完成切割 ====="))
#             else:
#                 progress_messages.append(html.Div(f"===== {region} 腦區尚未完成切割 ====="))

#         progress_messages.append(html.Div("*可重新按鈕刷新切割進度*"))
#         return progress_messages
    
#     return ""

# # 更新 Flask 路由，處理下載
# @server.route('/download/<region>/<filename>')
# def download_file(region, filename):
#     brain_region_folder = BRAIN_REGION_PATHS[region]
#     return send_from_directory(brain_region_folder, filename)

# # 提供 YOLO 推理的圖片
# @server.route('/output/yolo/<filename>')
# def send_yolo_image(filename):
#     return send_from_directory(YOLO_OUTPUT_FOLDER, filename)

# # 提供 brain_regions_projection.png
# @server.route('/output/brain/<filename>')
# def send_brain_image(filename):
#     return send_from_directory('最後合併腦區結果圖/', filename)

# # 啟動伺服器
# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run_server(debug=True, port=16362)


# %%
# # Step5 第四版本: 切割和下載、進度資訊檢查都完成( 需要新增讓SUBMIT消失，在按下NEXT成功後)
# # 新增SUBMIT按鈕因 NEXT觸發消失（或是當YOLO圖片出現後消失）
# # 剩下排版需要加強(以及YOLO後的防呆)
# import shutil
# import os
# import base64
# import subprocess
# from flask import Flask, send_from_directory
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc
# import tifffile as tif
# from werkzeug.utils import secure_filename

# # 設定目錄
# UPLOAD_FOLDER = 'User_Input_DLG_tif/'
# YOLO_OUTPUT_FOLDER = 'DEMO_YOLO_Inference/exp/'
# ALLOWED_EXTENSIONS = {'tif'}
# MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# # 用於腦區的下載路徑前綴
# BRAIN_REGION_PATHS = {
#     'AL': '網頁應用程式暫存資料夾_AL專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'MB': '網頁應用程式暫存資料夾_MB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'CAL': '網頁應用程式暫存資料夾_CAL專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'FB': '網頁應用程式暫存資料夾_FB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'EB': '網頁應用程式暫存資料夾_EB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
#     'PB': '網頁應用程式暫存資料夾_PB專用/使用者程序_第五階段存檔內容_完成放回原始影像/'
# }

# # 檢查上傳的檔案格式
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # 建立 Flask 伺服器
# server = Flask(__name__)
# server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# server.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # 建立 Dash 應用程式
# app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.DARKLY])  # 使用暗色主题

# # UI Layout
# app.layout = html.Div(
#     className='bg-dark text-white',  # 设置暗色背景和白色文字
#     children=[
#         html.Div([
#             html.H1("LYNSU: 自動化腦區切割", className="text-center mt-4"),  # 居中标题
#             html.Hr(),
#             html.P(
#                 "本系統支持單檔上傳果蠅大腦DLG.tif影像，請確保影像方向無誤 (Anterior & Posterior 任一方向)，且檔案大小不超過200MB。"
#                 "切割完成後的腦區將標示如下：",
#                 className="text-center"
#             ),
#             html.Ul(
#                 className="list-unstyled text-center",
#                 children=[
#                     html.Li("AL = 1"),
#                     html.Li("MB = 2"),
#                     html.Li("CAL = 3"),
#                     html.Li("FB = 4"),
#                     html.Li("EB = 5"),
#                     html.Li("PB = 6")
#                 ]
#             ),
#         ], className="mb-4"),
#         dcc.Upload(
#             id='upload-image',
#             children=html.Button('上傳 TIF 檔案', className='btn btn-primary'),
#             multiple=False,
#             className="d-flex justify-content-center mb-4"  # 上傳按钮居中
#         ),
#         html.Div(id='upload-output', className="text-center mb-4"),
#         dbc.Checklist(
#             id='region-switches',
#             options=[
#                 {'label': 'AL', 'value': 'AL'},
#                 {'label': 'MB', 'value': 'MB'},
#                 {'label': 'CAL', 'value': 'CAL'},
#                 {'label': 'FB', 'value': 'FB'},
#                 {'label': 'EB', 'value': 'EB'},
#                 {'label': 'PB', 'value': 'PB'},
#             ],
#             switch=True,
#             inline=True,
#             className="d-flex justify-content-center mb-4"  # Checklist 居中
#         ),
#         html.Div([
#             html.Button('SUBMIT', id='submit-button', n_clicks=0, className='btn btn-success mb-4'),  # 提交按钮
#             html.Div(id='output-image', className="text-center mb-4"),
#             html.Button('NEXT', id='next-button', n_clicks=0, className='btn btn-warning mb-4', style={'display': 'none'}),
#             html.Button('檢查當前3D腦區切割進度', id='check-status-button', n_clicks=0, className='btn btn-info mb-4', style={'display': 'none'}),
#         ], className="d-flex flex-column align-items-center"),  # Buttons 居中
#         html.Div(id='processing-status', className="text-center mb-4"),
#         html.Div(id='download-buttons', className="d-flex justify-content-center flex-wrap mb-4"),  # 下载按钮
#         html.Div(id='progress-status', className="text-center mb-4"),  # 进度状态
#     ]
# )

# # 上傳檔案處理
# @app.callback(
#     Output('upload-output', 'children'),
#     [Input('upload-image', 'filename'), Input('upload-image', 'contents')]
# )
# def handle_upload(filename, contents):
#     if filename is not None and allowed_file(filename):
#         # 清空 User_Input_DLG_tif 資料夾
#         if os.path.exists(UPLOAD_FOLDER):
#             shutil.rmtree(UPLOAD_FOLDER)
#         os.makedirs(UPLOAD_FOLDER)
#         # 刪除 YOLO_OUTPUT_FOLDER
#         if os.path.exists(YOLO_OUTPUT_FOLDER):
#             shutil.rmtree(YOLO_OUTPUT_FOLDER)
#         os.makedirs(YOLO_OUTPUT_FOLDER)
        
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)

#         filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

#         # 儲存上傳的 TIF 檔案
#         with open(filepath, 'wb') as f:
#             f.write(decoded)

#         try:
#             DLG = tif.imread(filepath)
#             print(f'Processing {filename}, shape: {DLG.shape}')
#             return html.Div(f"已成功上傳: {filename}")
#         except Exception as e:
#             print(f"無法讀取 TIF 檔案: {e}")
#             return html.Div("檔案上傳失敗，請上傳正確的 TIF 檔案")
    
#     return html.Div("*單檔上傳果蠅大腦DLG.tif影像，果蠅大腦方向請勿上下顛倒，允許Anterior & Posterior 任一方向!")

# # 提交按鈕處理
# @app.callback(
#     [Output('output-image', 'children'), Output('next-button', 'style')],
#     [Input('submit-button', 'n_clicks')],
#     [State('region-switches', 'value'), State('upload-image', 'filename')]
# )
# def process_image(n_clicks, selected_regions, filename):
#     if n_clicks > 0 and filename is not None:
#         tif_filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
#         subprocess.run(['python', 'Step1_讀取DLG_tif檔案_輸出成2D投影圖.py'])
#         subprocess.run(['python', 'Step2_2024_最終使用者網頁應用程式_YOLOv7_inference.py'])

#         png_files = [f for f in os.listdir(YOLO_OUTPUT_FOLDER) if f.endswith('.png')]
        
#         if png_files:
#             png_image_path = os.path.join(YOLO_OUTPUT_FOLDER, png_files[0])
#             return [html.Img(src=f'/output/yolo/{secure_filename(png_files[0])}', style={'max-width': '100%', 'height': 'auto'}), html.Div("請點擊NEXT按鈕開始切割3D腦區!")], {'display': 'inline'}
#         return html.Div("尚未生成影像，請稍後再試"), {'display': 'none'}

#     elif n_clicks > 0:
#         return html.Div("請先上傳檔案再點擊SUBMIT按鈕!"), {'display': 'none'}
#     return "", {'display': 'none'}

# # NEXT 按鈕處理，依次執行 Step3 並顯示進度檢查按鈕
# @app.callback(
#     [Output('processing-status', 'children'), Output('download-buttons', 'children'), Output('check-status-button', 'style')],
#     [Input('next-button', 'n_clicks')],
#     [State('region-switches', 'value')]
# )
# def process_next(n_clicks, selected_regions):
#     if n_clicks > 0 and selected_regions:
#         status_messages = []
#         download_buttons = []
        
#         for region in selected_regions:
#             script_name = f"Step3_YOLO結果預處理給3D_UNet切割_輸出放回原始影像_{region}.py"
#             subprocess.run(['python', script_name])
            
#             brain_region_folder = BRAIN_REGION_PATHS[region]
#             tif_file = [f for f in os.listdir(brain_region_folder) if f.startswith('Seg') and f.endswith('.tif')]
#             if tif_file:
#                 download_buttons.append(html.A(f'下載 {region}.tif', href=f'/download/{region}/{tif_file[0]}', download=f'{region}.tif', className='btn btn-secondary m-2'))
        
#         subprocess.run(['python', 'Step4_合併所有原始解析度切割結果影像＿輸出.py'])

#         status_messages.append(html.Div("所有腦區切割已完成，正在展示最終合併影像..."))
#         status_messages.append(html.Div("*注意: 下方圖像是不區分前後順序的展示，目的是展示切割腦區【數量】是否正確"))
#         status_messages.append(html.Img(src='/output/brain/brain_regions_projection.png', style={'max-width': '100%', 'height': 'auto'}))

#         return status_messages, download_buttons, {'display': 'inline'}
#     elif n_clicks > 0:
#         return html.Div("請至少選擇一個腦區進行切割!"), [], {'display': 'inline'}
#     return html.Div(""), [], {'display': 'none'}

# # 檢查3D腦區進度
# @app.callback(
#     Output('progress-status', 'children'),
#     [Input('check-status-button', 'n_clicks')],
#     [State('region-switches', 'value')]
# )
# def check_progress(n_clicks, selected_regions):
#     if n_clicks > 0 and selected_regions:
#         progress_messages = []
#         for region in selected_regions:
#             brain_region_folder = BRAIN_REGION_PATHS[region]
#             tif_file = [f for f in os.listdir(brain_region_folder) if f.startswith('Seg') and f.endswith('.tif')]
#             if tif_file:
#                 progress_messages.append(html.Div(f"===== {region} 腦區已完成切割 ====="))
#             else:
#                 progress_messages.append(html.Div(f"===== {region} 腦區尚未完成切割 ====="))

#         progress_messages.append(html.Div("*可重新按鈕刷新切割進度*"))
#         return progress_messages
    
#     return ""



# # 更新 Flask 路由，處理下載
# @server.route('/download/<region>/<filename>')
# def download_file(region, filename):
#     brain_region_folder = BRAIN_REGION_PATHS[region]
#     return send_from_directory(brain_region_folder, filename)

# # 提供 YOLO 推理的圖片
# @server.route('/output/yolo/<filename>')
# def send_yolo_image(filename):
#     return send_from_directory(YOLO_OUTPUT_FOLDER, filename)

# # 提供 brain_regions_projection.png
# @server.route('/output/brain/<filename>')
# def send_brain_image(filename):
#     return send_from_directory('最後合併腦區結果圖/', filename)

# # 啟動伺服器
# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run_server(debug=True, port=13826)


# %%
# Step6 第五版本: 暫時新增上傳進度條顯示(目前SUBMIT後有問題)
import shutil
import os
import base64
import subprocess
from flask import Flask, send_from_directory
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import tifffile as tif
from werkzeug.utils import secure_filename
import dash_uploader as du  # 引入 dash_uploader
import sys
# 設定目錄
UPLOAD_FOLDER = 'User_Input_DLG_tif/'
YOLO_OUTPUT_FOLDER = 'DEMO_YOLO_Inference/exp/'
ALLOWED_EXTENSIONS = {'tif'}
MAX_FILE_SIZE = 400 * 1024 * 1024  # 200 MB

# 用於腦區的下載路徑前綴
BRAIN_REGION_PATHS = {
    'AL': '網頁應用程式暫存資料夾_AL專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
    'MB': '網頁應用程式暫存資料夾_MB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
    'CAL': '網頁應用程式暫存資料夾_CAL專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
    'FB': '網頁應用程式暫存資料夾_FB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
    'EB': '網頁應用程式暫存資料夾_EB專用/使用者程序_第五階段存檔內容_完成放回原始影像/',
    'PB': '網頁應用程式暫存資料夾_PB專用/使用者程序_第五階段存檔內容_完成放回原始影像/'
}

# 建立 Flask 伺服器
server = Flask(__name__)
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
server.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 建立 Dash 應用程式
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.DARKLY])  # 使用暗色主题

# 配置 dash_uploader 的上传路径
du.configure_upload(app, UPLOAD_FOLDER)

# UI Layout
app.layout = html.Div(
    className='bg-dark text-white',  # 设置暗色背景和白色文字
    children=[
        html.Div([
            html.H1("LYNSU: 自動化腦區切割", className="text-center mt-4"),  # 居中标题
            html.Hr(),
            html.P(
                "本系統支持單檔上傳果蠅大腦DLG.tif影像，請確保影像方向無誤 (Anterior & Posterior 任一方向)，且檔案大小不超過400MB。"
                "切割完成後的腦區將標示如下：",
                className="text-center"
            ),
            html.Ul(
                className="list-unstyled text-center",
                children=[
                    html.Li("AL = 1"),
                    html.Li("MB = 2"),
                    html.Li("CAL = 3"),
                    html.Li("FB = 4"),
                    html.Li("EB = 5"),
                    html.Li("PB = 6")
                ]
            ),
            html.P(
                "*注意: 重新整理網頁可以重置目前進度*",
                className="text-center"
            ),
        ], className="mb-4"),
        html.Div([
            du.Upload(
                id='upload-image',
                text='拖放或點擊上傳 TIF 檔案',
                max_file_size=MAX_FILE_SIZE / (1024 * 1024),  # 转换为 MB
                filetypes=['tif'],
                pause_button=True,  # 上传时暂停按钮
                cancel_button=True  # 上传时取消按钮
            ),
            html.Div(id='upload-progress', className="text-center mb-4")  # 显示上传进度
        ], className="d-flex justify-content-center mb-4"),
        html.Div([
            html.Button('第一步: YOLOv7偵測腦區', id='submit-button', n_clicks=0, className='btn btn-success mb-4'),
            html.Div(id='output-image', className="text-center mb-4"),
            html.Button('第二步: 3D腦區切割', id='next-button', n_clicks=0, className='btn btn-warning mb-4', style={'display': 'none'})        ], className="d-flex flex-column align-items-center"),
        html.Div([
            html.P("請先等待YOLOv7腦區偵測結果!",
                className="text-center"
            )
        ], className="mb-4"),
        html.Div([
            dbc.Checklist(
                id='region-switches',
                options=[
                    {'label': 'AL', 'value': 'AL'},
                    {'label': 'MB', 'value': 'MB'},
                    {'label': 'CAL', 'value': 'CAL'},
                    {'label': 'FB', 'value': 'FB'},
                    {'label': 'EB', 'value': 'EB'},
                    {'label': 'PB', 'value': 'PB'},
                ],
                switch=True,
                inline=True,
                className="d-flex justify-content-center mb-4",
                style={'display': 'none'}# 初始隐藏
            )
        ], className="d-flex justify-content-center mb-4",style={'display': 'none'}),
        html.Div(id='processing-status', className="text-center mb-4"),
        html.Div([
            html.Button('檢查當前3D腦區切割進度', id='check-status-button', n_clicks=0, className='btn btn-info mb-4', style={'display': 'none'}),
        ], className="d-flex flex-column align-items-center"),
        html.Div(id='progress-status', className="text-center mb-4"),
        html.Div(id='download-buttons', className="d-flex justify-content-center flex-wrap mb-4")
    ]
)
# 上传之前手动清空文件夹
def clear_upload_folder():
    # 清空 User_Input_DLG_tif 資料夾
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)
    
    
# 上传完成后的处理
@du.callback(
    output=Output('upload-progress', 'children'),
    id='upload-image'
)
def callback_on_complete(filenames):
#     print('這裡是檢查檔案上傳狀態:',filenames )

    if filenames is not None:
        # 刪除 YOLO_OUTPUT_FOLDER
        if os.path.exists(YOLO_OUTPUT_FOLDER):
            shutil.rmtree(YOLO_OUTPUT_FOLDER)
        os.makedirs(YOLO_OUTPUT_FOLDER)
        # 将文件移动到指定目录并删除随机子目录
        for file_path in filenames:
            file_name = os.path.basename(file_path)  # 获取文件名
            new_file_path = os.path.join(UPLOAD_FOLDER, file_name)  # 目标文件路径

            # 移动文件到指定的 UPLOAD_FOLDER 目录下
            shutil.move(file_path, new_file_path)
        return ""
    return html.Div("尚未上傳任何檔案", className="text-danger")
# ================

# ================
# 提交按鈕處理
@app.callback(
    [Output('output-image', 'children'), Output('next-button', 'style'), Output('region-switches', 'style')],
    [Input('submit-button', 'n_clicks')],
    [State('region-switches', 'value'), State('upload-progress', 'children')]  # 获取上传完成后的状态
)
def process_image(n_clicks, selected_regions, upload_progress):
    # 处理上传文件路径
#     print(upload_progress)
    if n_clicks > 0 and upload_progress is not None:
        subprocess.run([sys.executable, 'Step1_讀取DLG_tif檔案_輸出成2D投影圖.py'])
        subprocess.run([sys.executable, 'Step2_2024_最終使用者網頁應用程式_YOLOv7_inference.py'])

        png_files = [f for f in os.listdir(YOLO_OUTPUT_FOLDER) if f.endswith('.png')]

        if png_files:
            png_image_path = os.path.join(YOLO_OUTPUT_FOLDER, png_files[0])
            return [html.Img(src=f'/output/yolo/{secure_filename(png_files[0])}', style={'max-width': '100%', 'height': 'auto'}), html.Div("請點擊【3D腦區切割】按鈕開始切割3D腦區!")], {'display': 'inline'}, {'display': 'inline'}
        return html.Div("尚未生成影像，請稍後再試"), {'display': 'none'}, {'display': 'none'}

    elif n_clicks > 0:
#         print('還沒有檔案',upload_progress)
        # 在上传之前先清空上传文件夹
        clear_upload_folder()
        return html.Div("請先上傳檔案再點擊【偵測腦區】按鈕!"), {'display': 'none'}, {'display': 'none'}
    return "", {'display': 'none'}, {'display': 'none'}


# NEXT 按鈕處理，依次執行 Step3 並顯示進度檢查按鈕
@app.callback(
    [Output('processing-status', 'children'), Output('download-buttons', 'children'), Output('check-status-button', 'style')],
    [Input('next-button', 'n_clicks')],
    [State('region-switches', 'value')]
)
def process_next(n_clicks, selected_regions):
    if n_clicks > 0 and selected_regions:
        status_messages = []
        download_buttons = []
        
        for region in selected_regions:
            script_name = f"Step3_YOLO結果預處理給3D_UNet切割_輸出放回原始影像_{region}.py"
            subprocess.run([sys.executable, script_name])
            
            brain_region_folder = BRAIN_REGION_PATHS[region]
            tif_file = [f for f in os.listdir(brain_region_folder) if f.startswith('Seg') and f.endswith('.tif')]
            if tif_file:
                download_buttons.append(html.A(f'下載 {region}.tif', href=f'/download/{region}/{tif_file[0]}', download=f'{region}.tif', className='btn btn-secondary m-2'))
        
        subprocess.run([sys.executable, 'Step4_合併所有原始解析度切割結果影像＿輸出.py'])

        status_messages.append(html.Div("所有腦區切割已完成，正在展示最終合併影像..."))
        status_messages.append(html.Div("*注意: 下方圖像是不區分前後順序的展示，目的是展示切割腦區【數量】是否正確"))
        status_messages.append(html.Img(src='/output/brain/brain_regions_projection.png', style={'max-width': '100%', 'height': 'auto'}))
        # 清空上传文件夹
        clear_upload_folder()
        return status_messages, download_buttons, {'display': 'inline'}
    elif n_clicks > 0:
        return html.Div("請至少選擇一個腦區進行切割!"), [], {'display': 'inline'}
    return html.Div(""), [], {'display': 'inline'}

# 檢查3D腦區進度
@app.callback(
    Output('progress-status', 'children'),
    [Input('check-status-button', 'n_clicks')],
    [State('region-switches', 'value')]
)
def check_progress(n_clicks, selected_regions):
    if n_clicks > 0 and selected_regions:
        progress_messages = []
        for region in selected_regions:
            brain_region_folder = BRAIN_REGION_PATHS[region]
            tif_file = [f for f in os.listdir(brain_region_folder) if f.startswith('Seg') and f.endswith('.tif')]
            if tif_file:
                progress_messages.append(html.Div(f"===== {region} 腦區已完成切割 ====="))
            else:
                progress_messages.append(html.Div(f"===== {region} 腦區尚未完成切割 ====="))

        progress_messages.append(html.Div("*可重新按鈕刷新切割進度*"))
        return progress_messages
    if n_clicks > 0:
        return html.Div("*您尚未選擇欲切割的腦區*")
    return ""

# 更新 Flask 路由，處理下載
@server.route('/download/<region>/<filename>')
def download_file(region, filename):
    brain_region_folder = BRAIN_REGION_PATHS[region]
    return send_from_directory(brain_region_folder, filename)

# 提供 YOLO 推理的圖片
@server.route('/output/yolo/<filename>')
def send_yolo_image(filename):
    return send_from_directory(YOLO_OUTPUT_FOLDER, filename)

# 提供 brain_regions_projection.png
@server.route('/output/brain/<filename>')
def send_brain_image(filename):
    return send_from_directory('最後合併腦區結果圖/', filename)

# 啟動伺服器
if __name__ == '__main__':
    # 清空 User_Input_DLG_tif 資料夾
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)
    # app.run_server(debug=True, port=13826)
    app.run_server(debug=True, host='0.0.0.0', port=13826)



