from PIL import Image
import os

# フォルダのパスを設定
folder1 = '../old_nocut_image'  # 左側に置く画像が入っているフォルダ
folder2 = '../new_nocut_image'  # 右側に置く画像が入っているフォルダ
merged_folder = './merge_nocut_image'  # マージ済みの画像が入るフォルダ

# フォルダ内の全画像ファイル名を取得
images1 = set(os.listdir(folder1))
images2 = set(os.listdir(folder2))

# 両フォルダに存在する画像ファイルのみを対象にする
common_images = images1.intersection(images2)

for image_name in common_images:
    # 画像を開く
    image_path1 = os.path.join(folder1, image_name)
    image_path2 = os.path.join(folder2, image_name)
    with Image.open(image_path1) as img1, Image.open(image_path2) as img2:
        # 画像を横に並べる
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        new_img = Image.new('RGB', (total_width, max_height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))

        # 新しい画像を保存
        new_img.save(os.path.join(merged_folder, image_name))

print("画像のマージが完了しました。")
