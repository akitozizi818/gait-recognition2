import pandas as pd
import json
import matplotlib.pyplot as plt
import os

viewpoints = [45,90,225,270]
video_numbers = [1,2,3,4,5]
labels = [0,1,2,3,4,5,6,7,8,9]

# JSONファイルを読み込む
with open('./outresult/confidence_sorted_dataset_unified_viewpoint_excluding_90/all_predictions_file_name.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ディレクトリを作成
output_dir = './outresult/confidence_sorted_dataset_unified_viewpoint_excluding_90'
os.makedirs(output_dir, exist_ok=True)  # ディレクトリが存在しない場合のみ作成

# データを整理して DataFrame に変換
rows = []
for entry in data:
    # img_name から name, viewpoint, number, time を抽出
    img_path, true_label = entry["img_name"]
    parts = img_path.split('\\')[-1].split('_')
    name = parts[0]
    viewpoint = int(parts[1])
    video_number = int(parts[2].split('.')[0])
    time = int(parts[-1].split('.')[0])
    predicted_label = entry["predicted_label"]
    
    # correct 列を計算
    correct = true_label == predicted_label
    
    # 行データを作成
    row = {
        "name": name,
        "viewpoint": viewpoint,
        "video_number": video_number,
        "time": time,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": correct
    }
    rows.append(row)

# DataFrame に変換
df = pd.DataFrame(rows)

#--------------------------------------------------------------------
# フレーム数 ごとに correct が True である割合を計算
time_correct_ratio = df.groupby('time')['correct'].mean() * 100  # 百分率に変換

# 棒グラフを描画
plt.figure(figsize=(10, 6))
time_correct_ratio.plot(kind='bar', color='skyblue', edgecolor='black')

# x軸の目盛りを20単位で設定
time_values = time_correct_ratio.index  # time の値
plt.xticks(range(0, len(time_values), 20), time_values[::20], rotation=45)  # 20単位で目盛りを設定

# グラフの装飾
plt.title('Correct Ratio by Time')
plt.xlabel('Time')
plt.ylabel('Correct Ratio (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # y軸にグリッドを表示
plt.tight_layout()  # レイアウトを調整

# グラフを保存
output_path = os.path.join(output_dir, 'correct_ratio_by_time.png')
plt.savefig(output_path)
plt.clf()

#------------------------------------------------------------------------------
# フレーム数 ごとのデータ数を計算
time_counts = df['time'].value_counts().sort_index()  # time ごとのデータ数を計算し、インデックスでソート

# 棒グラフを描画
plt.figure(figsize=(10, 6))
time_counts.plot(kind='bar', color='lightgreen', edgecolor='black')

# x軸の目盛りを20単位で設定
time_values = time_counts.index  # time の値
plt.xticks(range(0, len(time_values), 20), time_values[::20], rotation=45)  # 20単位で目盛りを設定

# グラフの装飾
plt.title('Number of Data Points by Time')
plt.xlabel('Time')
plt.ylabel('Number of Data Points')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # y軸にグリッドを表示
plt.tight_layout()  # レイアウトを調整

# グラフを保存
output_path = os.path.join(output_dir, 'data_points_by_time.png')
plt.savefig(output_path)
plt.clf()

#------------------------------------------------------------------------------
#全フレームにおける予測ラベルの集計
# 結果を保存するための辞書
viewpoint_results = {}

for viewpoint in viewpoints:
    true_count = 0
    false_count = 0
    for video_number in video_numbers:
        for true_label in labels:
            filtered_df = df[(df['viewpoint'] == viewpoint) & (df['video_number'] == video_number) & (df['true_label'] == true_label)]
            if filtered_df.empty:
                continue
            # predicted_label ごとのデータ数を計算
            predicted_label_counts = filtered_df['predicted_label'].value_counts().sort_index()
            predicted_label_counts = predicted_label_counts.reindex(range(10), fill_value=0)
            # 最も値が大きいラベルを取得
            most_frequent_predicted_label = predicted_label_counts.idxmax()
            # true_label と比較
            if most_frequent_predicted_label == true_label:
                true_count += 1
            else:
                false_count += 1

            # 棒グラフを描画
            plt.figure(figsize=(10, 6))
            predicted_label_counts.plot(kind='bar', color='skyblue', edgecolor='black')

            # グラフの装飾
            plt.title(f'Number of Data Points by Predicted Label (Viewpoint={viewpoint}, Video Number={video_number}, True Label={true_label})')
            plt.xlabel('Predicted Label')
            plt.ylabel('Number of Data Points')
            plt.xticks(range(10), range(10), rotation=0)  # 横軸を0から9に設定
            plt.grid(axis='y', linestyle='--', alpha=0.7)  # y軸にグリッドを表示
            plt.tight_layout()  # レイアウトを調整

            # グラフを保存
            output_dir_predict_per_all_frames = os.path.join(output_dir,'predict_per_all_frames')
            os.makedirs(output_dir_predict_per_all_frames, exist_ok=True)  # ディレクトリが存在しない場合のみ作成

            output_path = os.path.join(output_dir_predict_per_all_frames, f'data_points_by_predicted_label_viewpoint_{viewpoint}_video_{video_number}_true_{true_label}.png')

            plt.savefig(output_path)
            plt.close()  # 図を閉じる
    
    # viewpointごとのtrueとfalseの数を記録
    viewpoint_results[viewpoint] = {
        'true_count': true_count,
        'false_count': false_count,
        'true_ratio': true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0.0
    }

# 結果をテキストファイルに保存
output_text_path = os.path.join(output_dir_predict_per_all_frames, 'viewpoint_true_false_counts.txt')
with open(output_text_path, 'w') as f:
    for viewpoint, result in viewpoint_results.items():
        f.write(f'Viewpoint {viewpoint}:\n')
        f.write(f'  True Count: {result["true_count"]}\n')
        f.write(f'  False Count: {result["false_count"]}\n')
        f.write(f'  True Ratio: {result["true_ratio"]:.2f}\n\n')
