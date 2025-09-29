import os
import shutil

def organize_files(source_folder, target_folder1, target_folder2, specified_number):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    for file in files:
        parts = file.split("_")
        if len(parts) < 4:
            continue

        subfolder1 = os.path.join(target_folder1, parts[0])
        if not os.path.exists(subfolder1):
            os.makedirs(subfolder1)
        
        # ファイルをフォルダ1に移動
        file_path = os.path.join(subfolder1, file)
        shutil.copy(os.path.join(source_folder, file), file_path)

        # parts[2]の拡張子を除去して比較
        base_name = parts[2][:-4]  # .mp4の部分を取り除く

        if base_name == specified_number:
            subfolder2 = os.path.join(target_folder2, parts[0])
            if not os.path.exists(subfolder2):
                os.makedirs(subfolder2)
            
            # ファイルをフォルダ2に移動（フォルダ1からフォルダ2へ）
            shutil.move(file_path, os.path.join(subfolder2, file))

# 使用例
source_folder = './cut_image'
target_folder1 = './sorted_dataset/dataset'
target_folder2 = './sorted_dataset/test_dataset'
specified_number = '3'


# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(target_folder1):
    os.makedirs(target_folder1)

if not os.path.exists(target_folder2):
    os.makedirs(target_folder2)

organize_files(source_folder, target_folder1, target_folder2, specified_number)
