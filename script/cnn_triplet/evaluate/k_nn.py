import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import json
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Some visualizations will be simplified.")


# カスタムレイヤーの定義（モデル読み込み用）
class ConfidenceDropout(tf.keras.layers.Layer):
    def __init__(self, confidence_rate=0.4, **kwargs):
        super(ConfidenceDropout, self).__init__(**kwargs)
        self.confidence_rate = confidence_rate

    def call(self, inputs):
        image, confidence_map = inputs
        mask = tf.where(confidence_map >= self.confidence_rate, 1.0, 0.0)
        weighted_image = image * mask
        return weighted_image

    def get_config(self):
        config = super().get_config()
        config.update({"confidence_rate": self.confidence_rate})
        return config


def load_training_data(data_dir, confidence_dir, img_size):
    """
    学習データを読み込む

    Args:
        data_dir (str): 学習画像が格納されたディレクトリ
        confidence_dir (str): 信頼度マップが格納されたディレクトリ
        img_size (tuple): 画像サイズ (width, height)

    Returns:
        tuple: (images, confidence_maps, labels, class_indices, class_names)
    """
    images, conf_maps, labels = [], [], []
    class_indices = {name: i for i, name in enumerate(sorted(os.listdir(data_dir)))}
    class_names = list(class_indices.keys())

    for class_name, class_idx in class_indices.items():
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                conf_path = os.path.join(confidence_dir, filename)

                if os.path.exists(conf_path):
                    img = np.array(Image.open(img_path).resize(img_size)) / 255.0
                    conf = np.array(Image.open(conf_path).resize(img_size)) / 255.0

                    images.append(img)
                    conf_maps.append(conf)
                    labels.append(class_idx)

    return np.array(images), np.array(conf_maps), np.array(labels), class_indices, class_names


def evaluate_training_data_knn(config, k_values=[1, 3, 5, 7, 10], test_size=0.3, random_state=42):
    """
    学習データに対してk-NN評価を実行する

    Args:
        config (dict): 設定辞書
        k_values (list): テストするk値のリスト
        test_size (float): テスト用データの割合
        random_state (int): ランダムシード

    Returns:
        dict: 評価結果
    """
    # モデル読み込み
    model_path = os.path.join(config["output_dir"], config["model_name"])
    print(f"Loading model from: {model_path}")

    custom_objects = {'ConfidenceDropout': ConfidenceDropout}
    embedding_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # 学習データ読み込み
    print("Loading training data...")
    images, conf_maps, labels, class_indices, class_names = load_training_data(
        config["dataset_root"],
        config["confidence_map_dir"],
        config["img_size"]
    )

    print(f"Total samples: {len(images)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    # 埋め込みベクトル生成
    print("Generating embeddings...")
    embeddings = embedding_model.predict([images, conf_maps])

    # 学習データを訓練用と検証用に分割
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    results = {}

    # 異なるk値でk-NN評価
    best_k = None
    best_accuracy = 0

    for k in k_values:
        print(f"\nTesting k={k}...")

        # k-NN分類器を訓練
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # 予測
        y_pred = knn.predict(X_val)

        # 精度計算
        accuracy = accuracy_score(y_val, y_pred)

        # 詳細な分類レポート
        report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)

        # 混同行列
        cm = confusion_matrix(y_val, y_pred)

        results[f"k_{k}"] = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "k_value": k
        }

        print(f"k={k}: accuracy = {accuracy:.4f}")

        # 最高精度の記録
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    # 最高性能のk値で詳細評価
    print(f"\nBest k: {best_k} with accuracy: {best_accuracy:.4f}")

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    best_pred = best_knn.predict(X_val)

    # 詳細な結果保存
    detailed_results = {
        "best_k": best_k,
        "best_accuracy": best_accuracy,
        "k_results": results,
        "dataset_info": {
            "total_samples": len(images),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "num_classes": len(class_names),
            "class_names": class_names,
            "test_size": test_size,
            "random_state": random_state
        }
    }

    # 結果を保存
    output_dir = config["output_dir"]

    # JSON形式で保存（NumPy型を標準型に変換）
    def convert_numpy_types(obj):
        """NumPy型をJSON serializable型に変換"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # NumPy scalar
            return obj.item()
        return obj

    json_safe_results = convert_numpy_types(detailed_results)

    with open(os.path.join(output_dir, "learning_knn_evaluation.json"), "w") as f:
        json.dump(json_safe_results, f, indent=4)

    # テキスト形式の詳細レポート保存
    with open(os.path.join(output_dir, "learning_knn_report.txt"), "w") as f:
        f.write("Learning Data k-NN Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {config['dataset_root']}\n")
        f.write(f"Total samples: {len(images)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")

        f.write("k-NN Results:\n")
        for k in k_values:
            acc = results[f"k_{k}"]["accuracy"]
            f.write(f"k={k}: {acc:.4f}\n")

        f.write(f"\nBest k: {best_k} (accuracy: {best_accuracy:.4f})\n\n")

        f.write("Detailed Classification Report (Best k):\n")
        f.write(classification_report(y_val, best_pred, target_names=class_names))

    # 混同行列の可視化と保存
    plt.figure(figsize=(10, 8))
    best_cm = results[f"k_{best_k}"]["confusion_matrix"]

    if HAS_SEABORN:
        sns.heatmap(np.array(best_cm), annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    else:
        # Simple matplotlib heatmap without seaborn
        im = plt.imshow(np.array(best_cm), cmap='Blues', interpolation='nearest')
        plt.colorbar(im)

        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(best_cm[i][j]), ha='center', va='center')

        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)

    plt.title(f'Confusion Matrix (k={best_k}, accuracy={best_accuracy:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_knn_confusion_matrix.png"))
    plt.close()

    # k値 vs 精度のグラフ
    plt.figure(figsize=(10, 6))
    k_vals = [results[f"k_{k}"]["k_value"] for k in k_values]
    accuracies = [results[f"k_{k}"]["accuracy"] for k in k_values]

    plt.plot(k_vals, accuracies, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.title('k-NN Accuracy vs k value (Learning Data)')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_vals)

    # 最高精度にマーカー
    best_idx = k_vals.index(best_k)
    plt.plot(best_k, accuracies[best_idx], 'ro', markersize=10, label=f'Best k={best_k}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_knn_accuracy_curve.png"))
    plt.close()

    print(f"\nResults saved to: {output_dir}")
    print("Files created:")
    print("- learning_knn_evaluation.json")
    print("- learning_knn_report.txt")
    print("- learning_knn_confusion_matrix.png")
    print("- learning_knn_accuracy_curve.png")

    return detailed_results


if __name__ == "__main__":
    # テスト用のデフォルト設定
    dataset_name = "sorted_dataset_scale_exclude_45"

    config = {
        "dataset_root": f"../../../data/{dataset_name}/dataset",
        "confidence_map_dir": "../../../data/cut_confidence_image",
        "output_dir": f"../../../data/triplet_loss_output/{dataset_name}",
        "img_size": (48, 48),
        "model_name": "embedding_model.h5",
        "confidence_rate": 0.4
    }

    # 評価実行
    results = evaluate_training_data_knn(config)