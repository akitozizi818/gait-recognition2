import os
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageDraw, ImageFont
import json
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

confidence_rates = [0.1,0.3,0.95]

for confidence_rate in confidence_rates:
    output_dir = f"./out_results/confidence_{confidence_rate}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # コンフィグレーションの設定と保存
    config = {
        "dataset_root": "./sorted_dataset_unified_viewpoint_excluding_270_2/dataset",
        "test_data_dir": "./sorted_dataset_unified_viewpoint_excluding_270_2/test_dataset",
        "confidence_map_dir":"./cut_confidence_image",
        "img_size": (48, 48),
        "batch_size": 8,
        "num_classes": 10,
        "validation_split": 0.2,
        "iterations": 5,
        "epochs": 30,
        "best_model_name": "bad_r_l_merged_1_14_model.h5",
        "report_file_name": "bad_r_l_merged_classification_report.txt",
        "incorrect_predictions_file_name": "bad_r_l_merged_incorrect_predictions.txt",
        "model_summary_file_name": "bad_r_l_merged_model_summary.txt",
        "history_file_name":"best_history.json",
        "test_acc_report_file_name":"test_acc_summary"
    }

    # コンフィグレーションをファイルに保存
    config_file_name = "bad_r_l_merged_config.json"
    with open(os.path.join(output_dir, config_file_name), "w") as file:
        json.dump(config, file, indent=4)

    # コンフィグレーションの読み込み
    with open(os.path.join(output_dir, config_file_name)) as file:
        config = json.load(file)

    num_classes = config["num_classes"]
    test_data_dir = config["test_data_dir"]


    # 開始時間
    start = time.time()

    class CustomDataGenerator(tf.keras.utils.Sequence):
        def __init__(self,image_dir, confidence_map_dir, batch_size, img_size, image_filenames=None, shuffle=True,**kwargs):
            super().__init__(**kwargs)
            self.image_dir = image_dir
            self.confidence_map_dir = confidence_map_dir
            self.batch_size = batch_size
            self.img_size = img_size
            self.shuffle = shuffle
            self.class_indices = self._get_class_indices()  # クラスインデックスを生成
            if image_filenames is None:
                self.image_filenames = self._get_image_filenames()  # クラスごとのファイルを取得
            else:
                self.image_filenames = image_filenames  # 既に渡されたファイルリストを使用
            self.on_epoch_end()
            

        def _get_class_indices(self):
            # ディレクトリ名をクラス名とし、インデックスを付与
            class_names = sorted(os.listdir(self.image_dir))
            return {class_name: i for i, class_name in enumerate(class_names)} #{jinbo:0,kawasaki: 1..}

        def _get_image_filenames(self):
            # クラスごとにファイル名を収集し、クラスラベルを付加
            image_filenames = []
            for class_name, class_index in self.class_indices.items():
                class_dir = os.path.join(self.image_dir, class_name)
                if os.path.isdir(class_dir):  # ディレクトリのみを処理
                    for file_name in os.listdir(class_dir):
                        file_path = os.path.join(class_name, file_name)  # クラス名をパスに含める 例：kawasaki/kawasaki_45_1.MP4_image_cut_0.png
                        image_filenames.append((file_path, class_index))  # (ファイルパス, クラスインデックス)のタプル
            return image_filenames

        def __len__(self):
            return int(np.floor(len(self.image_filenames) / self.batch_size))

        def __getitem__(self, index):
            # 現在のバッチ用のファイル名とラベルを取得
            batch_filenames = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
            images = []
            confidence_maps = []
            labels = []

            for (file_path, class_index) in batch_filenames:
                image_path = os.path.join(self.image_dir, file_path)

                _, filename = os.path.split(file_path)
                confidence_map_path = os.path.join(self.confidence_map_dir,filename)
                # print(confidence_map_path)
                # 画像と信頼度マップを読み込む
                image = np.array(Image.open(image_path).resize(self.img_size)) / 255.0
                confidence_map = np.array(Image.open(confidence_map_path).resize(self.img_size)) / 255.0

                images.append(image)
                confidence_maps.append(confidence_map)
                labels.append(class_index)

            # ラベルをワンホットエンコード
            labels = tf.keras.utils.to_categorical(labels, num_classes=len(self.class_indices))

            # 入力データを辞書形式で返す
            inputs = {
                "image_input": np.array(images),
                "confidence_input": np.array(confidence_maps)
            }
            return inputs, np.array(labels)

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.image_filenames)
                print("データをシャッフル^^^^^^^^^^^^^^^^^^^^^^^^")

        def get_classes(self):
            # クラスインデックスのリストを返す
            return [label for _, label in self.image_filenames]
        
        def get_files(self):
            return [file for file, _ in self.image_filenames]

    class ConfidenceDropout(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(ConfidenceDropout, self).__init__(**kwargs)

        def call(self, inputs):
            image, confidence_map = inputs
            # マスクを生成し、float32型にキャスト
            # mask = tf.cast(tf.random.uniform(tf.shape(confidence_map), 0, 0.8) < confidence_map, tf.float32)
            # mask = tf.cast(confidence_map > 0, tf.float32)
            # confidence_map が 0.8 以上の場合はその値を保持し、それ以外は 0 に設定
            mask = tf.where(confidence_map >= confidence_rate, 1.0, 0.0)
            weighted_image = image * mask
            return weighted_image
        
    # データをシャッフルして分割する関数
    def get_train_and_validation_data(image_dir, confidence_map_dir, batch_size, img_size, validation_split=0.2):
        # クラスインデックスを取得
        class_indices = {class_name: i for i, class_name in enumerate(sorted(os.listdir(image_dir)))}

        # クラスごとに画像ファイルを収集
        train_filenames = []
        validation_filenames = []
        
        for class_name, class_index in class_indices.items():
            class_dir = os.path.join(image_dir, class_name)
            if os.path.isdir(class_dir):  # ディレクトリのみを処理
            # クラス内の全画像ファイルを収集
                class_files = [os.path.join(class_name, file_name) for file_name in os.listdir(class_dir)]

                # シャッフル
                np.random.shuffle(class_files)

                # データ分割
                if validation_split > 0:
                    split_index = int(len(class_files) * (1 - validation_split))
                    train_filenames.extend([(file_path, class_index) for file_path in class_files[:split_index]])
                    validation_filenames.extend([(file_path, class_index) for file_path in class_files[split_index:]])
                else:
                    # validation_split = 0 の場合、すべてを訓練データにする
                    train_filenames.extend([(file_path, class_index) for file_path in class_files])


        # 訓練データと検証データをそれぞれのジェネレーターに渡す
        train_generator = CustomDataGenerator(
            image_dir=image_dir,
            confidence_map_dir=confidence_map_dir,
            image_filenames=train_filenames,
            batch_size=batch_size,
            img_size=img_size
        )

        validation_generator = CustomDataGenerator(
            image_dir=image_dir,
            confidence_map_dir=confidence_map_dir,
            image_filenames=validation_filenames,
            batch_size=batch_size,
            img_size=img_size,
            shuffle=False  # 順序を維持
        )

        return train_generator, validation_generator

    # テストデータ
    test_generator = CustomDataGenerator(
        image_dir=config["test_data_dir"],
        confidence_map_dir=config["confidence_map_dir"],
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        shuffle=False  # 順序を維持
    )

    # # クラスとディレクトリの対応を取得して反転
    # class_directories = {v: k for k, v in train_generator.class_indices.items()}#{0:jinbo,1:kawasaki..}

    # 最高のテスト精度とそのモデルを保存するための変数
    best_test_acc = 0.0
    best_model = None
    best_history = None


    test_acc_array =[]
    train_acc_array=[]
    val_acc_array=[]

    # 学習の繰り返し
    for i in range(config["iterations"]):
        print(f"Starting iteration {i+1}/{config['iterations']}")
        train_generator, validation_generator = get_train_and_validation_data(
            image_dir=config["dataset_root"],
            confidence_map_dir=config["confidence_map_dir"],
            batch_size=config["batch_size"],
            img_size=config["img_size"],
            validation_split=config["validation_split"]
        )

        # モデルの作成
        image_input = tf.keras.Input(shape=(48, 48, 1), name="image_input")
        confidence_input = tf.keras.Input(shape=(48, 48, 1), name="confidence_input")

        x = ConfidenceDropout()([image_input, confidence_input])  # 信頼度に基づくドロップアウト

        x = layers.Conv2D(256, (5, 5), activation="swish")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation="swish")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(256, (2, 2), activation="swish")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="swish")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        output = layers.Dense(config["num_classes"], activation="softmax")(x)

        # model = tf.keras.Model(inputs=[input_image, input_confidence_map], outputs=output)
        model = Model(inputs={"image_input": image_input, "confidence_input": confidence_input}, outputs=output)

        # モデルのコンパイル
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.000392),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # 学習率調整用のコールバック
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',    # 検証データの損失を監視
            factor=0.5,            # 学習率を半減
            patience=5,            # エポック改善がない場合に発動
            min_lr=1e-6,           # 最低学習率
            verbose=1              # ログ出力
        )

        # EarlyStoppingのコールバック
        early_stopping = EarlyStopping(
            monitor='val_loss',    # 検証データの損失を監視
            patience=10,            # エポック改善がない場合に終了
            verbose=1,             # ログ出力
            restore_best_weights=True  # 最良モデルの重みを復元
        )

        # コールバックの設定
        callbacks = [
            reduce_lr,
            early_stopping
        ]
        # モデルの訓練
        history = model.fit(
            train_generator,
            epochs=config["epochs"],
            validation_data=validation_generator,
            callbacks=callbacks  # コールバックを指定
        )

        # テストデータでの評価
        test_loss, test_acc = model.evaluate(test_generator)
        print(f"Iteration {i+1}: Test Accuracy: {test_acc}")


        # 訓練データでの精度
        train_loss, train_acc = model.evaluate(train_generator)

        # 検証データでの精度
        val_loss, val_acc = model.evaluate(validation_generator)

        # イテレーションごとの精度を配列に追加
        test_acc_array.append(test_acc)
        train_acc_array.append(train_acc)
        val_acc_array.append(val_acc)

        # 最高のテスト精度を更新した場合、モデルと履歴を保存
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = model
            best_history = history
            with open(os.path.join(output_dir, config["history_file_name"]), "w") as file:
                json.dump(best_history.history, file, indent=4)
            model.save(os.path.join(output_dir, config["best_model_name"]))

    # 最高のテスト精度を持つモデルでの結果の表示
    print(f"Best Test Accuracy: {best_test_acc}")

    # モデルの構成をテキストファイルに保存（ベストモデルの保存に変更）
    model_summary = []
    best_model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_str = "\n".join(model_summary)
    with open(os.path.join(output_dir, config["model_summary_file_name"]), "w", encoding="utf-8") as file:
        file.write(model_summary_str)


    # # 予測を行う際に、stepsの値を整数にキャスト
    # steps = len(test_generator) // test_generator.batch_size  # // 演算子で整数にする
    # predictions = best_model.predict(test_generator, steps=steps)

    predictions = best_model.predict(test_generator)

    # テストデータの正解ラベルを取得
    true_classes = test_generator.get_classes()

    # 予測結果から最も確率の高いクラスを取得
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = true_classes[:len(predicted_classes)]  # 正解ラベルを予測数に合わせて切り詰める


    # サイズ確認とエラー回避
    if len(predicted_classes) != len(true_classes):
        print(f"予測の数: {len(predicted_classes)}")
        print(f"正解ラベルの数: {len(true_classes)}")
        raise ValueError("予測数と正解ラベルの数が一致しません。データセットを確認してください。")

    # 予測が間違っていた画像の名前とその本来のラベルを出力
    incorrect_predictions = np.where(predicted_classes != true_classes)[0]
    incorrect_images_with_labels = [(test_generator.image_filenames[i][0], true_classes[i]) for i in incorrect_predictions]

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


    # 画像名、正解ラベル、予測ラベルをリストにまとめる
    all_predictions_data = []
    for i in range(len(predicted_classes)):
        img_name = test_generator.image_filenames[i]  # 画像名
        true_label = true_classes[i]  # 正解ラベル
        predicted_label = predicted_classes[i]  # 予測ラベル
        all_predictions_data.append({
            "img_name": img_name,
            "true_label": int(true_label),  # JSONではintに変換
            "predicted_label": int(predicted_label)
        })

    # JSONファイルに保存
    with open(os.path.join(output_dir, "all_predictions_file_name"), "w", encoding="utf-8") as file:
        json.dump(all_predictions_data, file, indent=4, ensure_ascii=False)

    test_files = test_generator.get_files()
    test_files = test_files[:len(predicted_classes)]  # 正解ラベルを予測数に合わせて切り詰める

    # 特定の文字列をキーとして管理する辞書
    target_keywords = ["270", "90", "45", "225"]
    results = {}

    # キーワードごとに計算
    for keyword in target_keywords:
        # 該当する画像をフィルタリング
        indices = [
            i for i, file_path in enumerate(test_files)
            if keyword == file_path.split('_')[1]  # 一つ目のアンダースコアで分割した部分を比較
        ]
        
        # 該当するデータがない場合はスキップ
        if not indices:
            results[keyword] = {"match_ratio": 0, "total": 0, "matches": 0}
            continue
        
        # フィルタリングされたデータの正解ラベルと予測ラベル
        filtered_true_classes = [true_classes[i] for i in indices]
        filtered_predicted_classes = [predicted_classes[i] for i in indices]
        
        # 一致する数を計算
        matches = sum(1 for true, pred in zip(filtered_true_classes, filtered_predicted_classes) if true == pred)
        total = len(indices)
        match_ratio = matches / total if total > 0 else 0
        
        # 結果を保存
        results[keyword] = {
            "match_ratio": match_ratio,
            "total": total,
            "matches": matches
        }

    # 結果をテキストファイルに保存
    with open(os.path.join(output_dir, "viewpoints_results.txt"), "w", encoding="utf-8") as file:
        for keyword, result in results.items():
            file.write(f"Keyword: {keyword}\n")
            file.write(f"  Match Ratio: {result['match_ratio']:.2f}\n")
            file.write(f"  Total Samples: {result['total']}\n")
            file.write(f"  Matches: {result['matches']}\n\n")

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
    classes = list(train_generator.class_indices.keys())
    print(classes)

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
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bad_r_l_merged_confusion_matrix.png"))

    # 前の図をクリア
    plt.clf()


    # 学習曲線のプロット
    epochs = range(1, len(best_history.history["accuracy"]) + 1)

    plt.figure(figsize=(12, 6))

    # 精度のプロット
    plt.subplot(1, 2, 1)
    plt.plot(epochs, best_history.history["accuracy"], "b", label="Training Accuracy")
    plt.plot(epochs, best_history.history["val_accuracy"], "r", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # 損失のプロット
    plt.subplot(1, 2, 2)
    plt.plot(epochs, best_history.history["loss"], "b", label="Training Loss")
    plt.plot(epochs, best_history.history["val_loss"], "r", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    # 学習曲線の画像を保存
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.clf()

    # クラスごとの精度を計算
    class_accuracy = []
    for i in range(num_classes):
        class_true = (true_classes == i)
        class_pred = (predicted_classes == i)
        acc = np.sum(class_true & class_pred) / np.sum(class_true) if np.sum(class_true) > 0 else 0
        class_accuracy.append(acc)
    print(class_accuracy)

    # クラスごとの精度をプロット
    plt.bar(classes, class_accuracy, color="blue")
    plt.title("Class-wise Test Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()

    # クラスごとの精度グラフを保存
    plt.savefig(os.path.join(output_dir, "class_accuracy.png"))
    plt.clf()

    #イテレーションごとのテストデータの精度のまとめ
    max_value = max(test_acc_array)
    min_value = min(test_acc_array)
    average = sum(test_acc_array) / len(test_acc_array)
    with open(os.path.join(output_dir, config["test_acc_report_file_name"]), 'w') as file:
        # 配列の内容を書き込む
        file.write("t:\n")
        file.write(str(test_acc_array) + "\n")  # リストを文字列に変換して書き込む

        # 最大値と最小値を書き込む
        file.write(f"Max Value: {max_value}\n")
        file.write(f"Min Value: {min_value}\n")
        file.write(f"Average Value: {average}\n")

    #学習データとテストデータの精度の関係のグラフ
    # 散布図の作成
    plt.scatter(train_acc_array, test_acc_array)

    # x, y軸のラベルを追加
    plt.xlabel('train_acc')
    plt.ylabel('test_acc')

    # グラフのタイトルを追加
    plt.title('Training and Test Accuracy Comparison')

    # 画像として保存
    plt.savefig(os.path.join(output_dir, "Training_and_Test_Accuracy_Comparison.png"))
    plt.clf()
    #学習データとテストデータの精度の関係のグラフ
    # 散布図の作成
    plt.scatter(val_acc_array, test_acc_array)

    # x, y軸のラベルを追加
    plt.xlabel('val_acc')
    plt.ylabel('test_acc')

    # グラフのタイトルを追加
    plt.title('Validation and Test Accuracy Comparison')

    # 画像として保存
    plt.savefig(os.path.join(output_dir, "Validation_and_Test_Accuracy_Comparison.png"))
    plt.clf()
    # 終了時間
    end = time.time()
    print(f"学習と評価が完了しました。経過時間: {end - start}秒")
