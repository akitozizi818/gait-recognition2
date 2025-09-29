import os
import json
import numpy as np
from PIL import Image

# JSONファイルが保存されているディレクトリ
json_dir = './json_2D'
# 生成した画像を保存するディレクトリ
output_dir = './nocut_confidence'

# ラベル順に信頼度を取得するインデックス
ordered_indices = [
    12, 14, 16, 11, 13, 15, 19, 18, 0, 17, 5, 7, 9, 6, 8, 10
]

# ラベルの定義
labels = {
    0: "Nose", 1: "LEye", 2: "REye", 3: "LEar", 4: "REar", 
    5: "LShoulder", 6: "RShoulder", 7: "LElbow", 8: "RElbow", 
    9: "LWrist", 10: "RWrist", 11: "LHip", 12: "RHip", 
    13: "LKnee", 14: "Rknee", 15: "LAnkle", 16: "RAnkle", 
    17: "Head", 18: "Neck", 19: "Hip", 20: "LBigToe", 
    21: "RBigToe", 22: "LSmallToe", 23: "RSmallToe", 24: "LHeel", 
    25: "RHeel"
}

# JSONファイルを読み込む
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

# ディレクトリが存在しない場合、作成する
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 特定の信頼度を抽出する関数
def filter_keypoints_by_ordered_confidence(keypoints, ordered_indices):
    """指定された順番で信頼度を抽出し、ラベル付けする"""
    filtered_keypoints = []
    for i in ordered_indices:
        # 各keypointsは[x, y, confidence]の順で格納されている
        # x = keypoints[i * 3]   # x座標
        # y = keypoints[i * 3 + 1]  # y座標
        confidence = keypoints[i * 3 + 2]  # 信頼度
        
        # 信頼度に基づいてフィルタリング
        filtered_keypoints.extend([confidence,confidence,confidence])
    
    return filtered_keypoints

# 各JSONファイルを処理する
for json_file in json_files:
    with open(os.path.join(json_dir, json_file), 'r') as f:
        data = json.load(f)
    # 横幅は信頼度の数に応じて決定（3pxずつ並べるので3 * 16)
    width = 3 * len(ordered_indices)  # 横幅は3px * 信頼度の数
    
    # 高さはJSON内のimage_idの数に基づいて決定
    height = len(data)  # 画像の高さはjson内のエントリ数に応じて決定
    
    # 抽出した信頼度を画像に描画する
    x_offset = 0  # 横に並べるためのオフセット
    confidence_map = []

    for entry in data:
        # image_id = entry['image_id']
        keypoints = entry['keypoints']
        
        # 指定された順番で信頼度を抽出し、ラベル付けする
        filtered_keypoints = filter_keypoints_by_ordered_confidence(keypoints, ordered_indices)
        confidence_map.append(filtered_keypoints)

    # 信頼度に基づいてグレースケール値を計算
    grayscale_value = np.array(confidence_map) * 255

    # 画像を作成（グレースケール画像）
    img = Image.fromarray(grayscale_value.astype(np.uint8))  # uint8型に変換

    # ファイル名の設定（.jsonを削除して_image.pngを追加）
    output_filename = os.path.splitext(json_file)[0] + "_image.png"
    img.save(os.path.join(output_dir, output_filename))

print("画像の保存が完了しました。")
