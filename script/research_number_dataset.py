# import os

# def count_files_in_subdirectories(directory):
#     total_files = 0
#     # 指定したディレクトリ内のサブディレクトリを取得
#     for subdir in os.listdir(directory):
#         subdir_path = os.path.join(directory, subdir)
#         # サブディレクトリのみを対象にする
#         if os.path.isdir(subdir_path):
#             # サブディレクトリ内のファイル数をカウント
#             file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
#             total_files += file_count  # 合計に加算
#     return total_files

# # 使用例
# directory_path = './sorted_dataset/dataset'  # 調べたいディレクトリのパスを指定
# total = count_files_in_subdirectories(directory_path)
# print(f"dataset: {total}")

# directory_path = './sorted_dataset/test_dataset'  # 調べたいディレクトリのパスを指定
# total = count_files_in_subdirectories(directory_path)
# print(f"test_dataset: {total}")

import os

def count_files_by_angle(directory, angles):
    # 各角度のカウントを初期化
    angle_counts = {angle: 0 for angle in angles}

    # 指定したディレクトリ内のサブディレクトリを取得
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        # サブディレクトリのみを対象にする
        if os.path.isdir(subdir_path):
            # サブディレクトリ内のファイルを処理
            for f in os.listdir(subdir_path):
                if os.path.isfile(os.path.join(subdir_path, f)):
                    # ファイル名を分割して角度を抽出
                    parts = f.split('_')
                    if len(parts) > 2:  # 安全にインデックスを参照するためのチェック
                        angle = parts[1]  # 2つ目のワード（角度）
                        if angle in angle_counts:
                            angle_counts[angle] += 1  # 該当する角度をカウント

    # 合計を計算
    total_files = sum(angle_counts.values())
    return angle_counts, total_files

# 使用例
angles = ["45", "90", "225", "270"]  # カウント対象の角度
directory_path = './sorted_dataset/dataset'

# datasetディレクトリ内のファイルをカウント
dataset_counts, dataset_total = count_files_by_angle(directory_path, angles)
print("Dataset counts by angle:", dataset_counts)
print("Dataset total:", dataset_total)

# test_datasetディレクトリ内のファイルをカウント
directory_path = './sorted_dataset/test_dataset'
test_dataset_counts, test_dataset_total = count_files_by_angle(directory_path, angles)
print("Test Dataset counts by angle:", test_dataset_counts)
print("Test Dataset total:", test_dataset_total)
