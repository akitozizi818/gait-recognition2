# -*- coding: utf-8 -*-
import sys
from PIL import Image
import os

# トリミング前の画像の格納先
ORIGINAL_FILE_DIR = "./nocut_confidence/"
# トリミング後の画像の格納先
TRIMMED_FILE_DIR = "../data/cut_confidence_image/"

# もしトリミング後の画像の格納先が存在しなければ作る
if not os.path.isdir(TRIMMED_FILE_DIR):
    os.makedirs(TRIMMED_FILE_DIR)

# 画像パスと、左上座標、右下座標を指定して、トリミングされたimageオブジェクトを返す。
def trim(path, left, top, right, bottom):
    im = Image.open(path)
    im_trimmed = im.crop((left, top, right, bottom))
    return im_trimmed

if __name__ == "__main__":

    # トリミングする高さと重なり、最終Y座標を設定
    trim_height = 48  # トリミングする縦のサイズ（高さ）
    overlap = trim_height - 1      # 重なりのサイズ

    # 画像ファイル名を取得
    files = os.listdir(ORIGINAL_FILE_DIR)
    # 特定の拡張子のファイルだけを採用
    files = [name for name in files if name.lower().endswith(".png")]

    for val in files:
        # オリジナル画像へのパス
        path = ORIGINAL_FILE_DIR + val
        # オリジナル画像を開く
        with Image.open(path) as im:
            current_top = 0
            while current_top + trim_height <= im.size[1]:
                new_bottom = current_top + trim_height
                # トリミングされたimageオブジェクトを取得
                im_trimmed = trim(path, 0, current_top, im.size[0], new_bottom)
                # トリミング後のディレクトリに保存
                save_path = f"{TRIMMED_FILE_DIR}{val.replace('.png', '')}_cut_{current_top}.png"
                im_trimmed.save(save_path, quality=95)
                #print(f"トリミングされた画像が保存されました: {save_path}")

                current_top += trim_height - overlap  # 重なりを考慮して次のトリミング位置を設定