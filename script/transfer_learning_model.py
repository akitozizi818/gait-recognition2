# import os
# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

# # 出力ディレクトリの作成
# output_dir = "./transfer_model_out_results"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# #入力ディレクトリ
# input_dir = "./out_results"
# # コンフィグレーションの読み込み
# config_file_name = os.path.join(input_dir, "bad_r_l_merged_config.json")
# with open(config_file_name) as file:
#     config = json.load(file)

# # モデルの読み込みと層の凍結
# base_model = load_model(os.path.join(input_dir, config["best_model_name"]))
# for layer in base_model.layers:
#     layer.trainable = False  # 既存の層は学習させない

# # 新しい出力層を追加したモデルの構築
# num_new_classes = 5  # 新しいデータセットのクラス数に合わせて変更
# model = Sequential(base_model.layers[:-1])  # 既存モデルの最後の層を除外
# model.add(Dense(num_new_classes, activation="softmax"))

# # モデルのコンパイル
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # 新しいデータセット用のデータジェネレータ
# new_train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     validation_split=0.2
# )

# # 訓練データと検証データの準備
# new_train_generator = new_train_datagen.flow_from_directory(
#     "transfer_dataset/train",
#     target_size=config["img_size"],
#     batch_size=config["batch_size"],
#     class_mode="categorical",
#     color_mode="grayscale",
#     subset="training"
# )

# new_validation_generator = new_train_datagen.flow_from_directory(
#     "transfer_dataset/train",
#     target_size=config["img_size"],
#     batch_size=config["batch_size"],
#     class_mode="categorical",
#     color_mode="grayscale",
#     subset="validation"
# )

# # 転移学習の実行
# history = model.fit(
#     new_train_generator,
#     epochs=10,
#     validation_data=new_validation_generator
# )

# # モデルの保存
# transferred_model_path = os.path.join(output_dir, "transferred_model.h5")
# model.save(transferred_model_path)
# print(f"転移学習済みモデルが {transferred_model_path} に保存されました。")

# # テストデータの準備と評価
# new_test_generator = new_train_datagen.flow_from_directory(
#     "transfer_dataset/test",
#     target_size=config["img_size"],
#     batch_size=config["batch_size"],
#     class_mode="categorical",
#     color_mode="grayscale",
#     shuffle=False
# )

# test_loss, test_acc = model.evaluate(new_test_generator)
# print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

# # 訓練・検証精度と損失のプロット
# epochs = range(1, len(history.history["accuracy"]) + 1)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
# plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
# plt.title("Training and Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, history.history["loss"], label="Training Loss")
# plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
# plt.title("Training and Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# # グラフを保存
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "transfer_learning_plots.png"))
# plt.show()
# import os
# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

# # 出力ディレクトリの作成
# output_dir = "./transfer_model_out_results"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 入力ディレクトリとコンフィグファイルの読み込み
# input_dir = "./out_results"
# config_file_name = os.path.join(input_dir, "bad_r_l_merged_config.json")
# with open(config_file_name) as file:
#     config = json.load(file)

# # モデルの読み込みと層の凍結
# base_model = load_model(os.path.join(input_dir, config["best_model_name"]))
# for layer in base_model.layers:
#     layer.trainable = True  # 一旦すべての層を学習可能に]

# # 最初の8層を凍結（基本的な特徴抽出層を固定）
# for layer in base_model.layers[:12]:
#     layer.trainable = False

# # 新しい出力層を追加したモデルの構築
# num_new_classes = 5  # 新しいデータセットのクラス数に合わせて変更
# model = Sequential(base_model.layers[:-1])  # 最後の層を除外
# model.add(Dense(num_new_classes, activation="softmax"))

# # モデルのコンパイル
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # 新しいデータセット用のデータジェネレータ
# new_train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     validation_split=0.2
# )

# # 訓練データと検証データの準備
# new_train_generator = new_train_datagen.flow_from_directory(
#     "transfer_dataset/train",
#     target_size=config["img_size"],
#     batch_size=config["batch_size"],
#     class_mode="categorical",
#     color_mode="grayscale",
#     subset="training"
# )

# new_validation_generator = new_train_datagen.flow_from_directory(
#     "transfer_dataset/train",
#     target_size=config["img_size"],
#     batch_size=config["batch_size"],
#     class_mode="categorical",
#     color_mode="grayscale",
#     subset="validation"
# )

# # 転移学習の実行
# history = model.fit(
#     new_train_generator,
#     epochs=40,
#     validation_data=new_validation_generator
# )

# # モデルの保存
# transferred_model_path = os.path.join(output_dir, "transferred_model.h5")
# model.save(transferred_model_path)
# print(f"転移学習済みモデルが {transferred_model_path} に保存されました。")

# # テストデータの準備と評価
# new_test_generator = new_train_datagen.flow_from_directory(
#     "transfer_dataset/test",
#     target_size=config["img_size"],
#     batch_size=config["batch_size"],
#     class_mode="categorical",
#     color_mode="grayscale",
#     shuffle=False
# )

# test_loss, test_acc = model.evaluate(new_test_generator)
# print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

# # 訓練・検証精度と損失のプロット
# epochs = range(1, len(history.history["accuracy"]) + 1)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
# plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
# plt.title("Training and Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, history.history["loss"], label="Training Loss")
# plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
# plt.title("Training and Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# # グラフを保存
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "transfer_learning_plots.png"))
# plt.show()
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 出力ディレクトリの作成
output_dir = "./transfer_model_out_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# コンフィグファイルの読み込み
input_dir = "./out_results"
config_file_name = os.path.join(input_dir, "bad_r_l_merged_config.json")
with open(config_file_name) as file:
    config = json.load(file)

# モデルの読み込みと層の凍結
base_model = load_model(os.path.join(input_dir, config["best_model_name"]))
for layer in base_model.layers:
    layer.trainable = True  # 全ての層を学習可能に

# 最初の8層を凍結（基本的な特徴抽出層を固定）
for layer in base_model.layers[:8]:
    layer.trainable = False

# 新しい出力層を構築
num_new_classes = 5  # 新しいデータセットに合わせる
model = Sequential(base_model.layers[:-1])  # 最後の層を除去
model.add(Dense(num_new_classes, activation="softmax"))  # 新しい出力層

# モデルのコンパイル
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# データジェネレータの準備
new_train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# 訓練データと検証データの準備
new_train_generator = new_train_datagen.flow_from_directory(
    "transfer_dataset/train",
    target_size=config["img_size"],
    batch_size=config["batch_size"],
    class_mode="categorical",
    color_mode="grayscale",
    subset="training"
)

new_validation_generator = new_train_datagen.flow_from_directory(
    "transfer_dataset/train",
    target_size=config["img_size"],
    batch_size=config["batch_size"],
    class_mode="categorical",
    color_mode="grayscale",
    subset="validation"
)

# **コールバック**: EarlyStoppingとReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

# 転移学習の実行
history = model.fit(
    new_train_generator,
    epochs=40,
    validation_data=new_validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# モデルの保存
transferred_model_path = os.path.join(output_dir, "transferred_model.h5")
model.save(transferred_model_path)
print(f"転移学習済みモデルが {transferred_model_path} に保存されました。")

# テストデータの準備と評価
new_test_generator = new_train_datagen.flow_from_directory(
    "transfer_dataset/test",
    target_size=config["img_size"],
    batch_size=config["batch_size"],
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

# モデルのテスト評価
test_loss, test_acc = model.evaluate(new_test_generator)
print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

# **混同行列と分類レポートの生成**
y_true = new_test_generator.classes
y_pred = np.argmax(model.predict(new_test_generator), axis=1)

conf_mat = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_mat)

class_report = classification_report(y_true, y_pred, target_names=new_test_generator.class_indices.keys())
print("Classification Report:\n", class_report)

# **学習結果の可視化**
epochs = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history["loss"], label="Training Loss")
plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# グラフの保存
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "transfer_learning_plots.png"))
plt.show()

# モデルの最終結果表示
print(f"最終テスト精度: {test_acc}, テスト損失: {test_loss}")
