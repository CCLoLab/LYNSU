# -*- coding: utf-8 -*-
import os
import re
import subprocess

# 指定專案目錄（預設當前目錄）
project_directory = '.'

# 用來識別 import 語句的正則表達式模式
import_pattern = re.compile(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))')

# 用於存儲獨特的套件名稱
packages = set()

# 掃描專案目錄中的所有 .py 檔案
for root, dirs, files in os.walk(project_directory):
    for file in files:
        if file.endswith('.py'):
            print(f"正在掃描 {file} 檔案")
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 匹配 import 語句
                    match = import_pattern.match(line.strip())
                    if match:
                        package = match.group(1) or match.group(2)
                        if package and not package.startswith('.'):  # 忽略相對匯入
                            packages.add(package.split('.')[0])  # 只保留頂層的套件名稱

# 使用 pip freeze 來捕捉套件的版本並生成 requirements.txt
with open('requirements.txt', 'w') as req_file:
    for package in packages:
        try:
            # 使用 pip show 來獲取套件版本
            result = subprocess.run(['pip', 'show', package], stdout=subprocess.PIPE, text=True)
            for line in result.stdout.splitlines():
                if line.startswith('Version:'):
                    version = line.split()[-1]
                    req_file.write(f"{package}=={version}\n")
                    break
        except Exception as e:
            print(f"處理套件 {package} 時出錯: {e}")

print("requirements.txt 已成功生成。")
