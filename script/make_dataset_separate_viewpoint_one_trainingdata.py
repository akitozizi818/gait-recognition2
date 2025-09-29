import os
import shutil

viewpoints = [45,90,225,270]

for viewpoint in viewpoints:
    # 使用例
    source_folder = '../data/cut_3Dskeleton_scale_only_image2'
    target_folder1 = f'../data/sorted_dataset_{viewpoint}/dataset'
    target_folder2 = f'../data/sorted_dataset_{viewpoint}/test_dataset'
    specified_viewpoint =viewpoint
    specified_number = '3'


    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(target_folder1):
        os.makedirs(target_folder1)

    if not os.path.exists(target_folder2):
        os.makedirs(target_folder2)

    def organize_files(source_folder, target_folder1, target_folder2,specified_viewpoint, specified_number):
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
            number = parts[2][:-4]

            if parts[1].isdigit() and int(parts[1]) != specified_viewpoint:
                subfolder2 = os.path.join(target_folder2, parts[0])
                if not os.path.exists(subfolder2):
                    os.makedirs(subfolder2)
                # ファイルをフォルダ2に移動（フォルダ1からフォルダ2へ）
                shutil.move(file_path, os.path.join(subfolder2, file))
            else:
                if number == specified_number:
                    subfolder2 = os.path.join(target_folder2, parts[0])
                    if not os.path.exists(subfolder2):
                        os.makedirs(subfolder2)
                    # ファイルをフォルダ2に移動（フォルダ1からフォルダ2へ）
                    shutil.move(file_path, os.path.join(subfolder2, file))

    organize_files(source_folder, target_folder1, target_folder2, specified_viewpoint,specified_number)
