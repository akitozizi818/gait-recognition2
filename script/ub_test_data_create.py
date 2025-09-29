import numpy as np
import os
import cv2

# 新しいサイズにリサイズして余白にランダムなグレースケール値を埋める関数
def resize_and_crop_with_random_padding(img):
    # 画像をまず(36, 36)にリサイズ
    img = cv2.resize(img, (42, 42))

    # 右側12ピクセルを切り落とす
    cropped_img = img[:, :24]

    # 新しいサイズで空の画像を作成
    new_img = np.zeros((42, 42), dtype=img.dtype)

    # 左詰めで切り落とされた画像を配置
    new_img[:, :24] = cropped_img

    # 右側の余白にランダムなグレースケール値を埋める
    new_img[:, 24:] = np.random.randint(0, 256, (42, 18), dtype=img.dtype)

    return new_img


# テストデータディレクトリの定義
test_data_dir = "./shift/shift_test_dataset"

# リサイズされたテストデータを保存するディレクトリ
resized_test_data_dir = "./shift/ub_test_data"
os.makedirs(resized_test_data_dir, exist_ok=True)

# テストデータをリサイズして新しいディレクトリに保存
for folder in os.listdir(test_data_dir):
    folder_path = os.path.join(test_data_dir, folder)
    resized_folder_path = os.path.join(resized_test_data_dir, folder)
    os.makedirs(resized_folder_path, exist_ok=True)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_img = resize_and_crop_with_random_padding(img)
        cv2.imwrite(os.path.join(resized_folder_path, img_name), resized_img)
