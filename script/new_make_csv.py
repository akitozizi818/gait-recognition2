import json
import pandas as pd
import numpy as np
import glob
import csv
import os

dir_path = os.getcwd()

#JSONファイルパス
json_dir = "./output_json_dir"

#CSVファイルの保存先
output_dir = "./output_csv_dir"

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def getFileName(path):
    filelist = glob.glob(path + "/*")
     # ファイル名を整数でソート
    filelist.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    return filelist

# 信頼度の閾値を変数で定義、信頼度が閾値未満の場合、XとYを0に設定
confidence_threshold = 0.9


def getSpecificData(filelist, currentFilename):
    for i in range(len(filelist)):
        try:
            with open(filelist[i], encoding='utf-8') as f:
                data = json.load(f)
                keypoint_data = data.get("people", [{}])[
                    0].get("pose_keypoints_2d", [])
                if not keypoint_data:
                    raise IndexError(
                        "Keypoint data not found in the expected structure.")
                data = np.array(keypoint_data).reshape(-1, 3)
        except (IndexError, KeyError) as e:
            # IndexErrorやKeyErrorが発生した場合、エラーメッセージを表示してスキップします。
            print(f"ファイル {filelist[i]} の処理中にエラーが発生しました: {e}")
            continue

        # 信頼度が閾値未満の場合、XとYを0に設定
        for j in range(data.shape[0]):
            if data[j][2] < confidence_threshold:  # 信頼度Pが0.9未満のとき
                data[j][0] = 0  # Xを0に
                data[j][1] = 0  # Yを0に

        df = pd.DataFrame(
            data,
            columns=["X", "Y", "P"],
            index=[
                "nose",  # 0
                "eye(L)",  # 1
                "eye(R)",  # 2
                "ear(L)",  # 3
                "ear(R)",  # 4
                "shoulder(L)",  # 5
                "shoulder(R)",  # 6
                "elbow(L)",  # 7
                "elbow(R)",  # 8
                "wrist(L)",  # 9
                "wrist(R)",  # 10
                "hip(L)",  # 11
                "hip(R)",  # 12
                "knee(L)",  # 13
                "knee(R)",  # 14
                "ankle(L)",  # 15x
                "ankle(R)",  # 16
                # "Background"
            ],
        )
        # 自分の必要なデータを取り出す
        # writeCSV(
        #     [
        #     float(df.at["nose", "X"]),
        #     float(df.at["nose", "Y"]),
        #     float(df.at["eye(L)", "X"]),
        #     float(df.at["eye(L)", "Y"]),
        #     float(df.at["eye(R)", "X"]),
        #     float(df.at["eye(R)", "Y"]),
        #     float(df.at["ear(L)", "X"]),
        #      float(df.at["ear(L)", "Y"]),
        #      float(df.at["ear(R)", "X"]),
        #      float(df.at["ear(R)", "Y"]),
        #      float(df.at["shoulder(L)", "X"]),
        #      float(df.at["shoulder(L)", "Y"]),
        #      float(df.at["shoulder(R)", "X"]),
        #      float(df.at["shoulder(R)", "Y"]),
        #      float(df.at["elbow(L)", "X"]),
        #      float(df.at["elbow(L)", "Y"]),
        #      float(df.at["elbow(R)", "X"]),
        #      float(df.at["elbow(R)", "Y"]),
        #      float(df.at["wrist(L)", "X"]),
        #      float(df.at["wrist(L)", "Y"]),
        #      float(df.at["wrist(R)", "X"]),
        #      float(df.at["wrist(R)", "Y"]),
        #      float(df.at["hip(L)", "X"]),
        #      float(df.at["hip(L)", "Y"]),
        #      float(df.at["hip(R)", "X"]),
        #      float(df.at["hip(R)", "Y"]),
        #      float(df.at["knee(L)", "X"]),
        #      float(df.at["knee(L)", "Y"]),
        #      float(df.at["knee(R)", "X"]),
        #      float(df.at["knee(R)", "Y"]),
        #      float(df.at["ankle(L)", "X"]),
        #      float(df.at["ankle(L)", "Y"]),
        #      float(df.at["ankle(R)", "X"]),
        #      float(df.at["ankle(R)", "Y"]),
        #      ], currentFilename
        # keypointの並べ方を変更する（顔から順番に足先まで並べる方法）
        writeCSV(
            [
            float(df.at["eye(R)", "X"]),
            float(df.at["eye(R)", "Y"]),
            float(df.at["eye(L)", "X"]),
            float(df.at["eye(L)", "Y"]),
            float(df.at["ear(R)", "X"]),
            float(df.at["ear(R)", "Y"]),
            float(df.at["ear(L)", "X"]),
            float(df.at["ear(L)", "Y"]),
            float(df.at["nose", "X"]),
            float(df.at["nose", "Y"]),
            float(df.at["shoulder(R)", "X"]),
            float(df.at["shoulder(R)", "Y"]),     
            float(df.at["shoulder(L)", "X"]),
            float(df.at["shoulder(L)", "Y"]),
            float(df.at["elbow(R)", "X"]),
            float(df.at["elbow(R)", "Y"]), 
            float(df.at["elbow(L)", "X"]),
            float(df.at["elbow(L)", "Y"]),
            float(df.at["wrist(R)", "X"]),
            float(df.at["wrist(R)", "Y"]),
            float(df.at["wrist(L)", "X"]),
            float(df.at["wrist(L)", "Y"]),
            float(df.at["hip(R)", "X"]),
            float(df.at["hip(R)", "Y"]),
            float(df.at["hip(L)", "X"]),
            float(df.at["hip(L)", "Y"]),
            float(df.at["knee(R)", "X"]),
            float(df.at["knee(R)", "Y"]),
            float(df.at["knee(L)", "X"]),
            float(df.at["knee(L)", "Y"]),
            float(df.at["ankle(R)", "X"]),
            float(df.at["ankle(R)", "Y"]),
            float(df.at["ankle(L)", "X"]),
            float(df.at["ankle(L)", "Y"]), 
             ], currentFilename
        )


def writeCSV(data, currentFilename):
    with open(os.path.join(output_dir,currentFilename + ".csv"), "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(data)


def main():
    files_dir = [
        f for f in os.listdir(json_dir) if os.path.isdir(os.path.join(json_dir, f))
    ]
    for i in range(len(files_dir)):
        currentFilename = files_dir[i]
        print(currentFilename)
        filelist = getFileName(os.path.join(json_dir,currentFilename))
        with open(os.path.join(output_dir,currentFilename + ".csv"), "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            # 自分の必要なデータの列の名前を用意。上のデータと同じだけの列数を揃える。
            writer.writerow(
                [
                    "nose_x",
                    "nose_y",
                    "eye(L)_x",
                    "eye(L)_y",
                    "eye(R)_x",
                    "eye(R)_y",
                    "ear(L)_x",
                    "ear(L)_y",
                    "ear(R)_x",
                    "ear(R)_y",
                    "shoulder(L)_x",
                    "shoulder(L)_y",
                    "shoulder(R)_x",
                    "shoulder(R)_y",
                    "elbow(L)_x",
                    "elbow(L)_y",
                    "elbow(R)_x",
                    "elbow(R)_y",
                    "wrist(L)_x",
                    "wrist(L)_y",
                    "wrist(R)_x",
                    "wrist(R)_y",
                    "hip(L)_x",
                    "hip(L)_y",
                    "hip(R)_x",
                    "hip(R)_y",
                    "knee(L)_x",
                    "knee(L)_y",
                    "knee(R)_x",
                    "knee(R)_y",
                    "ankle(L)_x",
                    "ankle(L)_y",
                    "ankle(R)_x",
                    "ankle(R)_y",
                ]
            )
        getSpecificData(filelist, currentFilename)


if __name__ == "__main__":
    main()
