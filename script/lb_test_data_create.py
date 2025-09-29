import numpy as np
import os
import cv2

def resize_and_crop(img):
    # 画像をまず(42, 42)にリサイズ
    img = cv2.resize(img, (42, 42))

    # 左側の24ピクセルを切り落とす
    cropped_img = img[:, 24:]

    return cropped_img

def image_already_exists(cropped_img, existing_images):
    for img in existing_images:
        if np.array_equal(cropped_img, img):
            return True
    return False

# 切り落とした後の画像を保持するリスト
cropped_images = []

# テストデータディレクトリの定義
test_data_dir = "./shift/shift_test_dataset"

# リサイズされたテストデータを保存するディレクトリ
resized_test_data_dir = "./shift/lb_test_data"
os.makedirs(resized_test_data_dir, exist_ok=True)

# テストデータをリサイズして新しいディレクトリに保存
for folder in os.listdir(test_data_dir):
    folder_path = os.path.join(test_data_dir, folder)
    resized_folder_path = os.path.join(resized_test_data_dir, folder)
    os.makedirs(resized_folder_path, exist_ok=True)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cropped_img = resize_and_crop(img)

        # 切り落とした後の画像が既に存在するかどうかをチェック
        if not image_already_exists(cropped_img, cropped_images):
            # ここで新しい画像にランダムな値を追加する処理を行います
            new_img = np.zeros((42, 42), dtype=img.dtype)
            new_img[:, :24] = np.random.randint(0, 256, (42, 24), dtype=img.dtype)
            new_img[:, 24:] = cropped_img
            cv2.imwrite(os.path.join(resized_folder_path, img_name), new_img)
            cropped_images.append(cropped_img)
        else:
            print(f"Duplicate cropped image detected: {img_name}")
