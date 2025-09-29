import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import json

# 新しいテストデータの設定
new_config = {
    "new_test_data_dir": "./three_r_l_merged/lb_test_data/",
    "new_img_size": (84, 84),
    "batch_size": 8,
    "num_classes": 10,
    "best_model_name": "three_r_l_merged_1_14_model.h5",
    "report_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
}

# モデルのロード
loaded_model = tf.keras.models.load_model(new_config["best_model_name"])

# テストデータジェネレータの作成
new_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
new_test_generator = new_test_datagen.flow_from_directory(
    new_config["new_test_data_dir"],
    target_size=new_config["new_img_size"],
    batch_size=new_config["batch_size"],
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# モデルの評価
steps = np.ceil(new_test_generator.samples / new_config["batch_size"])
test_loss, test_acc = loaded_model.evaluate(new_test_generator, steps=steps)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# テストデータでの予測
predictions = loaded_model.predict(new_test_generator, steps=steps)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = new_test_generator.classes

# 混同行列の表示
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(new_config["num_classes"])
plt.xticks(tick_marks, range(1, new_config["num_classes"] + 1), rotation=45)
plt.yticks(tick_marks, range(1, new_config["num_classes"] + 1))
plt.xlabel('Predicted Label',fontsize=16)
plt.ylabel('True Label',fontsize=16)

fontsize = 14  # 文字サイズを設定


# 背景色に応じてテキストの色を選択するためのしきい値を定義
threshold = conf_matrix.max() / 2

for i in range(new_config["num_classes"]):
    for j in range(new_config["num_classes"]):
        color = "white" if conf_matrix[i, j] > threshold else "black"
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', fontsize=fontsize, color=color)

plt.tight_layout()
plt.show()
