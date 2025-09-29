import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageDraw, ImageFont
import json
from playsound import playsound


output_dir = "./out_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# コンフィグレーションの設定と保存
config = {
    "dataset_root": "./transfer_dataset/train",
    "test_data_dir": "./transfer_dataset/test",
    # "img_size": (84, 84),
    "img_size": (51, 51),
    "batch_size": 8,
    "num_classes": 2,
    "validation_split": 0.2,
    "iterations": 2,#もともと30
    "epochs": 20,#もともと20
    "best_model_name": "bad_r_l_merged_1_14_model.h5",
    "report_file_name": "bad_r_l_merged_classification_report.txt",
    "incorrect_predictions_file_name": "bad_r_l_merged_incorrect_predictions.txt",
    "model_summary_file_name": "bad_r_l_merged_model_summary.txt",
}

# コンフィグレーションをファイルに保存
config_file_name = "bad_r_l_merged_config.json"
with open(os.path.join(output_dir,config_file_name), "w") as file:
    json.dump(config, file, indent=4)

# コンフィグレーションの読み込み
with open(os.path.join(output_dir,config_file_name)) as file:
    config = json.load(file)

num_classes = config["num_classes"]
test_data_dir = config["test_data_dir"]


# 開始時間
start = time.time()

# データ拡張
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=config["validation_split"]
)

# 訓練データ
train_generator = train_datagen.flow_from_directory(
    config["dataset_root"],
    target_size=config["img_size"],
    batch_size=config["batch_size"],
    class_mode="categorical",
    color_mode="grayscale",
    subset="training"
)

# 検証データ
validation_generator = train_datagen.flow_from_directory(
    config["dataset_root"],
    target_size=config["img_size"],
    batch_size=config["batch_size"],
    class_mode="categorical",
    color_mode="grayscale",
    subset="validation"
)

# テストデータ
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    config["test_data_dir"],
    target_size=config["img_size"],
    batch_size=config["batch_size"],
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

# クラスとディレクトリの対応を取得して反転
class_directories = {v: k for k, v in train_generator.class_indices.items()}

# 最高のテスト精度とそのモデルを保存するための変数
best_test_acc = 0.0
best_model = None
best_history = None

# 学習の繰り返し
for i in range(config["iterations"]):
    print(f"Starting iteration {i+1}/{config['iterations']}")

    # CNNモデルの構築
    model = models.Sequential([
        # 畳み込み層とバッチノーマライゼーション
        # layers.Conv2D(256, (5, 5), activation="swish", input_shape=(84, 84, 1)),
        layers.Conv2D(256, (5, 5), activation="swish", input_shape=(51, 51, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # 畳み込み層、バッチノーマライゼーション、ドロップアウト
        layers.Conv2D(128, (3, 3), activation="swish"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # 畳み込み層、バッチノーマライゼーション
        layers.Conv2D(256, (2, 2), activation="swish"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # 全結合層
        layers.Flatten(),
        layers.Dense(64, activation="swish"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(config["num_classes"], activation="softmax")
    ])

    # モデルのコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.000392),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # モデルの訓練
    history = model.fit(
        train_generator,
        epochs=config["epochs"],
        validation_data=validation_generator
    )

    # テストデータでの評価
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Iteration {i+1}: Test Accuracy: {test_acc}")

    # 最高のテスト精度を更新した場合、モデルと履歴を保存
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model = model
        best_history = history
        model.save(os.path.join(output_dir, config["best_model_name"]))

# 最高のテスト精度を持つモデルでの結果の表示
print(f"Best Test Accuracy: {best_test_acc}")

# playsound('.\mocturne-op9-2.wav')

# モデルの構成をテキストファイルに保存（ベストモデルの保存に変更）
model_summary = []
best_model.summary(print_fn=lambda x: model_summary.append(x))
model_summary_str = "\n".join(model_summary)
with open(os.path.join(output_dir, config["model_summary_file_name"]), "w", encoding="utf-8") as file:
    file.write(model_summary_str)


# テストデータでの予測
predictions = best_model.predict(test_generator)

# 予測結果から最も確率の高いクラスを取得
predicted_classes = np.argmax(predictions, axis=1)

# 正解ラベルを取得
true_classes = test_generator.classes

# 予測が間違っていた画像の名前とその本来のラベルを出力
incorrect_predictions = np.where(predicted_classes != true_classes)[0]
incorrect_images_with_labels = [(test_generator.filenames[i], true_classes[i]) for i in incorrect_predictions]

# ラベル別に分類された不正確な予測のリストを作成
label_wise_incorrect_images = {}
for img_name, true_label in incorrect_images_with_labels:
    if true_label not in label_wise_incorrect_images:
        label_wise_incorrect_images[true_label] = []
    label_wise_incorrect_images[true_label].append(img_name)

# 不正確な予測のテキストを作成
incorrect_predictions_text = '\n'.join([f"Image: {img_name}, True Label: {label}" for img_name, label in incorrect_images_with_labels])

# 不正確な予測をテキストファイルに保存
with open(os.path.join(output_dir, config["incorrect_predictions_file_name"]), "w") as file:
    file.write(incorrect_predictions_text)

print(f"Incorrect predictions saved to {config['incorrect_predictions_file_name']}")



# 分類レポートの生成
classification_report_str = classification_report(true_classes, predicted_classes, digits=2)

# 分類レポートをテキストファイルに保存
with open(os.path.join(output_dir, config["report_file_name"]), "w") as file:
    file.write(classification_report_str)

print(f'分類レポートが {config["report_file_name"]} に保存されました。')

# 分類レポートの表示
print(classification_report(true_classes, predicted_classes))

# 混同行列の表示
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# クラスの目盛り（ラベル）を設定
classes = [str(i + 1) for i in range(num_classes)]

# 混同行列をクラスの目盛りと詳細なラベルをつけてプロット
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# ラベル付きで混同行列を表示
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center")

# 図を保存
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
# 前の図をクリア
plt.clf()

# クラスごとのテストデータの精度を計算
class_accuracy = []
for i in range(num_classes):
    class_true = true_classes == i
    class_pred = predicted_classes == i
    class_acc = np.sum(class_true & class_pred) / np.sum(class_true)
    class_accuracy.append(class_acc)

# クラスごとのテストデータの精度をプロット
plt.bar(classes, class_accuracy, color="blue")
plt.title("Class-wise Test Accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy")
# 図を保存
plt.savefig(os.path.join(output_dir, "class_accuracy.png"))

# 前の図をクリア
plt.clf()

# 精度と損失のプロット
train_acc = best_history.history["accuracy"]
val_acc = best_history.history["val_accuracy"]
train_loss = best_history.history["loss"]
val_loss = best_history.history["val_loss"]

# テストデータに対する評価
test_loss, test_acc = best_model.evaluate(test_generator)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# 終了時間 - 開始時間でかかった時間を計測
end = time.time() - start

print(f"{end}秒かかりました！")

# グラフのプロット
epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_acc, "b", label="Training Accuracy")
plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_loss, "b", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.bar(
    ["Training", "Validation", "Test"],
    [np.max(train_acc), np.max(val_acc), test_acc],
    color=["blue", "red", "green"],
)
plt.title("Best Accuracy")
# # 図を保存
# plt.savefig(os.path.join(output_dir, "training_validation_test_accuracy.png"))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "overall_plot.png"))

