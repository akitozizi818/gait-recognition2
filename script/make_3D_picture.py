import shutil
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd
import numpy as np
import glob
import csv
import os
import cv2

#フォルダ名を入力
# subject = input("subject番号を入力してください:")
# os.mkdir("../output_image/"+subject)

# フォルダ内に存在するcsvファイルの一覧をcsv_filesに代入
csv_files = glob.glob("./json_3D_scale_only/*.csv")
#print(csv_files)
# json_file = glob.glob("./output_json_dir/*.json")

# csvファイルを一つずつ選ぶ
for i in range(len(csv_files)):
    csv_file = pd.read_csv(csv_files[i], skipinitialspace=True)
    # print(csv_file)
    colum_name_list = csv_file.columns.values

    # # スキップしたい列名のリスト
    # skip_columns = ["0", "2","3","5","6"]

    # # スキップしたい列名を部分一致で除外
    # colum_name_list = [col for col in colum_name_list if not any(skip in col for skip in skip_columns)]

    norm_df = pd.DataFrame()

    # head_heightを取得（32列目）
    head_height = csv_file.iloc[:, 31].values  # 32列目はインデックス31

    for j in range(len(colum_name_list)):
        colum_name = colum_name_list[j]
        colum_data = csv_file[colum_name]

        # NaNをゼロに置き換える
        colum_data = colum_data.fillna(0)

        # ratioの計算 すでにinfer_wild_transformation.pyで処理済みのためなし
        ratio = -0.95 / head_height

         # 各座標にratioを掛ける
        adjusted_colum_data = colum_data * ratio

        # # 列名に応じて正規化の最大値と最小値を設定
        # if "_x" in colum_name:
        #     max_val, min_val = 1920, 0
        # elif "_y" in colum_name:
        #     max_val, min_val = 1080, 0
        # else:
        #     continue  # 他の列は無視
        
        # 列データを正規化 -1 < colum_data < 1 なので　0 <= norm_colum_data < 256
        # norm_colum_data = 128 * colum_data + 128
        norm_colum_data = 128 * adjusted_colum_data + 128
        # 要素が0の場所にはランダムな値を代入
        # norm_colum_data_np = np.where(colum_data == 0, np.random.randint(0, 256, size=len(colum_data)), norm_colum_data)
        # norm_colum_data_np = norm_colum_data_np.clip(0, 255).astype(np.uint8)  # 値を0から255の範囲にクリップして型変換

        norm_colum_data_np = norm_colum_data.clip(0, 255).astype(np.uint8)  # 値を0から255の範囲にクリップして型変換

        # # 正規化後のデータを表示
        # print(f"Column: {colum_name}, Normalized Data: {norm_colum_data_np}")

        # numpy.ndarray を pandas.Series に変換
        norm_colum_data_series = pd.Series(norm_colum_data_np)

        # DataFrameに追加する際に Series として結合
        norm_df = pd.concat([norm_df, norm_colum_data_series], axis=1)

    # 続きの画像保存処理など
    # ...
    # 最初の3列を削除　ミドルヒップのx,y,z座標は固定なので削除
    norm_df = norm_df.iloc[:, 3:]

    norm_nd = norm_df.values
    # print(norm_nd)
    # 画像ファイルの保存先
    output_dir = "./nocut_3Dskeleton_scale_only_image2"

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_filename = os.path.basename(csv_files[i]).replace(".csv", "")
    cv2.imwrite(os.path.join(output_dir, csv_filename + "_image.png"), norm_nd)    # os.remove(csv_files[i])
    # #print(json_file)
    # shutil.rmtree(json_file[i])