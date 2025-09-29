import json
import pandas as pd
import numpy as np
import glob
import csv
import os

dir_path = os.getcwd()

def getFileName(path):
    filelist = glob.glob(path + "/*")
    return filelist


def getSpecificData(filelist,currentFilename):
    for i in range(len(filelist)):
        try:
            with open(filelist[i], encoding='utf-8') as f:
                data = json.load(f)
                keypoint_data = data.get("people", [{}])[0].get("pose_keypoints_2d", [])
                if not keypoint_data:
                    raise IndexError("Keypoint data not found in the expected structure.")
                data = np.array(keypoint_data).reshape(-1, 3)
        except (IndexError, KeyError) as e:
            # IndexErrorやKeyErrorが発生した場合、エラーメッセージを表示してスキップします。
            print(f"ファイル {filelist[i]} の処理中にエラーが発生しました: {e}")
            continue
        
        df = pd.DataFrame(
            data,
            columns=["X", "Y", "P"],
            index=[
                "Nose",
                "Neck",
                "RShoulder",
                "RElbow",
                "RWrist",
                "LShoulder",
                "LElbow",
                "LWrist",
                "MidHip",
                "RHip",
                "RKnee",
                "RAnkle",
                "LHip",
                "LKnee",
                "LAnkle",
                "REye",
                "LEye",
                "REar",
                "LEar",
                "LBigToe",
                "LSmallToe",
                "LHeel",
                "RBigToe",
                "RSmallToe",
                "RHeel"
                #"Background"
            ],
        )
        # 自分の必要なデータを取り出す
        writeCSV(
            [   
                float(df.at["Nose", "X"]),
                float(df.at["Nose", "Y"]),
                float(df.at["Neck", "X"]),
                float(df.at["Neck", "Y"]),
                float(df.at["REye", "X"]),
                float(df.at["REye", "Y"]),
                float(df.at["REar", "X"]),
                float(df.at["REar", "Y"]),
                float(df.at["RShoulder", "X"]),
                float(df.at["RShoulder", "Y"]),
                float(df.at["RElbow", "X"]),
                float(df.at["RElbow", "Y"]),
                float(df.at["RWrist", "X"]),
                float(df.at["RWrist", "Y"]),
                float(df.at["LEye", "X"]),
                float(df.at["LEye", "Y"]),
                float(df.at["LEar", "X"]),
                float(df.at["LEar", "Y"]),
                float(df.at["LShoulder", "X"]),
                float(df.at["LShoulder", "Y"]),
                float(df.at["LElbow", "X"]),
                float(df.at["LElbow", "Y"]),
                float(df.at["LWrist", "X"]),
                float(df.at["LWrist", "Y"]),
                float(df.at["MidHip", "X"]),
                float(df.at["MidHip", "Y"]),
                float(df.at["RHip", "X"]),
                float(df.at["RHip", "Y"]),
                float(df.at["RKnee", "X"]),
                float(df.at["RKnee", "Y"]),
                float(df.at["RAnkle", "X"]),
                float(df.at["RAnkle", "Y"]),
                float(df.at["RBigToe", "X"]),
                float(df.at["RBigToe", "Y"]),
                float(df.at["LHip", "X"]),
                float(df.at["LHip", "Y"]),
                float(df.at["LKnee", "X"]),
                float(df.at["LKnee", "Y"]),
                float(df.at["LAnkle", "X"]),
                float(df.at["LAnkle", "Y"]),
                float(df.at["LBigToe", "X"]),
                float(df.at["LBigToe", "Y"]),
            ],currentFilename
        )


def writeCSV(data,currentFilename):
    with open(currentFilename + "_output.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(data)


def main():
    files_dir = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
    ]
    for i in range(len(files_dir)):
        currentFilename = files_dir[i]
        filelist = getFileName(currentFilename) 
        #print(currentFilename)
        with open(currentFilename + "_output.csv", "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            # 自分の必要なデータの列の名前を用意。上のデータと同じだけの列数を揃える。
            writer.writerow(
                [
                    "Nose_x",
                    "Nose_y",
                    "Neck_x",
                    "Neck_y",
                    "REye_x",
                    "REye_y",
                    "REar_x",
                    "REar_y",
                    "RShoulder_x",
                    "RShoulder_y",
                    "RElbow_x",
                    "RElbow_y",
                    "RWrist_x",
                    "RWrist_y",
                    "LEye_x",
                    "LEye_y",
                    "LEar_x",
                    "LEar_y",
                    "LShoulder_x",
                    "LShoulder_y",
                    "LElbow_x",
                    "LElbow_y",
                    "LWrist_x",
                    "LWrist_y",
                    "MidHip_x",
                    "MidHip_y",
                    "RHip_x",
                    "RHip_y",
                    "RKnee_x",
                    "RKnee_y",
                    "RAnkle_x",
                    "RAnkle_y",
                    "RBigToe_x",
                    "RBigToe_y",
             "LHip_x",
                    "LHip_y",
                    "LKnee_x",
                    "LKnee_y",
                    "LAnkle_x",
                    "LAnkle_y",
                    "LBigToe_x",
                    "LBigToe_y",
                ]
            )
        getSpecificData(filelist,currentFilename)

if __name__ == "__main__":
    main()
