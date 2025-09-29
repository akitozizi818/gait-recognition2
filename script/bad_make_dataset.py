import os
import shutil

def organize_files(source_folder, target_folder1, target_folder2, specified_range):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    for file in files:
        parts = file.split("_")
        if len(parts) < 2:
            print(f"Skipping file (not enough parts): {file}")
            continue

        # ファイル名から数字部分を取得 (例: '44' in 'yamano_90_3_cut_44.png')
        number_part = parts[-1].split('.')[0] if '.' in parts[-1] else parts[-1]
        print(f"Processing file: {file}, number part: {number_part}")

        if number_part.isdigit() and int(number_part) in specified_range:
            target_folder = target_folder2
            action = "Keeping"
        else:
            target_folder = target_folder1
            action = "Moving"
        
        subfolder = os.path.join(target_folder, parts[0])
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        shutil.move(os.path.join(source_folder, file), os.path.join(subfolder, file))
        print(f"{action} file to {target_folder}: {file}")

# 使用例
source_folder = './cut_image'
target_folder1 = './bad_r_l_merged/dataset'
target_folder2 = './bad_r_l_merged/test_dataset'
specified_range = range(10, 30)

organize_files(source_folder, target_folder1, target_folder2, specified_range)
