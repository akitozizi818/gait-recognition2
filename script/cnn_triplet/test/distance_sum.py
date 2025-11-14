import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def load_test_data(data_dir, confidence_dir, img_size):
    """
    Load test data from directories.

    Args:
        data_dir (str): Directory containing test images organized by class
        confidence_dir (str): Directory containing confidence maps
        img_size (tuple): Target image size (width, height)

    Returns:
        tuple: (images, confidence_maps, labels, class_indices, file_paths)
    """
    images, conf_maps, labels, file_paths = [], [], [], []
    class_indices = {name: i for i, name in enumerate(sorted(os.listdir(data_dir)))}

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
                    file_paths.append(filename)

    return np.array(images), np.array(conf_maps), np.array(labels), class_indices, file_paths


def predict_by_distance_sum(test_embedding, train_embeddings, train_labels, num_classes, metric='euclidean'):
    """
    各クラスの学習データとの距離合計が最小のクラスを予測する

    Args:
        test_embedding (np.ndarray): テストデータの埋め込みベクトル (1, embedding_dim)
        train_embeddings (np.ndarray): 学習データの埋め込みベクトル (N, embedding_dim)
        train_labels (np.ndarray): 学習データのラベル (N,)
        num_classes (int): クラス数
        metric (str): 距離メトリック ('euclidean' or 'cosine')

    Returns:
        int: 予測ラベル
        dict: 各クラスとの距離合計
    """
    class_distance_sums = {}

    for class_idx in range(num_classes):
        # このクラスに属する学習データのインデックス
        class_mask = train_labels == class_idx
        class_embeddings = train_embeddings[class_mask]

        if len(class_embeddings) == 0:
            class_distance_sums[class_idx] = float('inf')
            continue

        # 距離計算
        if metric == 'euclidean':
            # ユークリッド距離の合計
            distances = np.linalg.norm(class_embeddings - test_embedding, axis=1)
        elif metric == 'cosine':
            # コサイン距離の合計（1 - コサイン類似度）
            similarities = np.dot(class_embeddings, test_embedding.T).flatten()
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # 距離の合計
        class_distance_sums[class_idx] = np.sum(distances)

    # 距離合計が最小のクラスを予測
    predicted_label = min(class_distance_sums, key=class_distance_sums.get)

    return predicted_label, class_distance_sums


def evaluate_model_distance_sum(config, metric='euclidean', model_path=None):
    """
    距離合計ベースの分類で学習済みモデルを評価

    Args:
        config (dict): Configuration dictionary
        metric (str): 距離メトリック ('euclidean' or 'cosine')
        model_path (str, optional): Path to the saved model

    Returns:
        dict: Evaluation results including accuracy and classification report
    """
    # Load model
    if model_path is None:
        model_path = os.path.join(config["output_dir"], config["model_name"])

    print(f"Loading model from: {model_path}")
    custom_objects = {'ConfidenceDropout': ConfidenceDropout}
    embedding_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # Load training and test data
    print("Loading training data...")
    train_images, train_conf, train_labels, train_class_indices, _ = load_test_data(
        config["dataset_root"],
        config["confidence_map_dir"],
        config["img_size"]
    )

    print("Loading test data...")
    test_images, test_conf, test_labels, test_class_indices, test_files = load_test_data(
        config["test_data_dir"],
        config["confidence_map_dir"],
        config["img_size"]
    )

    # Generate embeddings
    print("Generating embeddings for training data...")
    train_embeddings = embedding_model.predict([train_images, train_conf])

    print("Generating embeddings for test data...")
    test_embeddings = embedding_model.predict([test_images, test_conf])

    # 距離合計による分類
    print(f"Classifying test embeddings using distance sum method (metric: {metric})...")
    num_classes = len(train_class_indices)
    predictions = []
    all_distance_sums = []

    for i, test_emb in enumerate(test_embeddings):
        predicted_label, distance_sums = predict_by_distance_sum(
            test_emb.reshape(1, -1),
            train_embeddings,
            train_labels,
            num_classes,
            metric=metric
        )
        predictions.append(predicted_label)
        all_distance_sums.append(distance_sums)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_embeddings)} samples...")

    predictions = np.array(predictions)

    # Calculate accuracy and generate classification report
    accuracy = accuracy_score(test_labels, predictions)
    class_names = [name for name, _ in sorted(train_class_indices.items(), key=lambda x: x[1])]
    report = classification_report(
        test_labels,
        predictions,
        target_names=class_names,
        output_dict=True
    )

    # 混同行列
    cm = confusion_matrix(test_labels, predictions)

    print(f"\nDistance Sum classification accuracy on test embeddings: {accuracy:.4f}")
    print(f"Metric: {metric}")
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))

    # Save results
    output_dir = config["output_dir"]
    results = {
        "method": "distance_sum",
        "metric": metric,
        "accuracy": accuracy,
        "classification_report": report,
        "num_train_samples": len(train_labels),
        "num_test_samples": len(test_labels),
        "num_classes": len(class_names),
        "confusion_matrix": cm.tolist()
    }

    # NumPy型を標準型に変換
    def convert_numpy_types(obj):
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
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    json_safe_results = convert_numpy_types(results)

    # JSONファイルに保存
    result_filename = f"distance_sum_evaluation_{metric}.json"
    with open(os.path.join(output_dir, result_filename), "w") as f:
        json.dump(json_safe_results, f, indent=4)

    # テキストレポートに保存
    report_filename = f"distance_sum_report_{metric}.txt"
    with open(os.path.join(output_dir, report_filename), "w") as f:
        f.write(f"Distance Sum Classification Report (Metric: {metric})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Classification accuracy: {accuracy:.4f}\n")
        f.write(f"Number of training samples: {len(train_labels)}\n")
        f.write(f"Number of test samples: {len(test_labels)}\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Distance metric: {metric}\n\n")
        f.write("Method: Sum of distances to all training samples per class,\n")
        f.write("        predict the class with minimum distance sum.\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, predictions, target_names=class_names))

    # 混同行列の可視化
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    else:
        im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar(im)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(cm[i][j]), ha='center', va='center')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)

    plt.title(f'Confusion Matrix (Distance Sum, {metric})\nAccuracy: {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    confusion_filename = f"distance_sum_confusion_matrix_{metric}.png"
    plt.savefig(os.path.join(output_dir, confusion_filename))
    plt.close()

    # 視点ごとの正解率を計算
    target_keywords = ["45", "90", "225", "270"]
    viewpoint_results = {}

    for keyword in target_keywords:
        indices = [
            i for i, file_path in enumerate(test_files)
            if keyword == file_path.split('_')[1]
        ]

        if not indices:
            viewpoint_results[keyword] = {"match_ratio": 0, "total": 0, "matches": 0}
            continue

        filtered_true_classes = [test_labels[i] for i in indices]
        filtered_predicted_classes = [predictions[i] for i in indices]

        matches = sum(1 for true, pred in zip(filtered_true_classes, filtered_predicted_classes) if true == pred)
        total = len(indices)
        match_ratio = matches / total if total > 0 else 0

        viewpoint_results[keyword] = {
            "match_ratio": match_ratio,
            "total": total,
            "matches": matches
        }

    # 視点ごとの結果をテキストファイルに保存
    viewpoint_filename = f"distance_sum_viewpoints_{metric}.txt"
    with open(os.path.join(output_dir, viewpoint_filename), "w", encoding="utf-8") as file:
        file.write(f"Viewpoint Results (Distance Sum, {metric})\n")
        file.write("=" * 50 + "\n\n")
        for keyword, result in viewpoint_results.items():
            file.write(f"Viewpoint: {keyword}°\n")
            file.write(f"  Match Ratio: {result['match_ratio']:.4f}\n")
            file.write(f"  Total Samples: {result['total']}\n")
            file.write(f"  Matches: {result['matches']}\n\n")

    print(f"\nEvaluation results saved to: {output_dir}")
    print(f"Files created:")
    print(f"- {result_filename}")
    print(f"- {report_filename}")
    print(f"- {confusion_filename}")
    print(f"- {viewpoint_filename}")

    return results


def compare_metrics(config, metrics=['euclidean', 'cosine']):
    """
    複数の距離メトリックで評価を比較

    Args:
        config (dict): Configuration dictionary
        metrics (list): List of metrics to compare

    Returns:
        dict: Comparison results
    """
    print("=" * 60)
    print("COMPARING DISTANCE METRICS")
    print("=" * 60)

    comparison_results = {}

    for metric in metrics:
        print(f"\n--- Evaluating with {metric} metric ---")
        results = evaluate_model_distance_sum(config, metric=metric)
        comparison_results[metric] = {
            "accuracy": results["accuracy"],
            "classification_report": results["classification_report"]
        }

    # 比較結果を表示
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for metric, result in comparison_results.items():
        print(f"{metric:15s}: accuracy = {result['accuracy']:.4f}")

    # 比較結果を保存
    output_dir = config["output_dir"]
    with open(os.path.join(output_dir, "distance_sum_metric_comparison.json"), "w") as f:
        json.dump(comparison_results, f, indent=4)

    return comparison_results


if __name__ == "__main__":
    # Load configuration from saved file
    dataset_names = [
        # "sorted_dataset_scale_exclude_45",
        "sorted_dataset_scale_exclude_90",
        "sorted_dataset_scale_exclude_225",
        "sorted_dataset_scale_exclude_270"
    ]
    for dataset_name in dataset_names:
        config_path = f"../../../data/triplet_loss_output/{dataset_name}/config.json"

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

            # Run evaluation with both metrics
            print("Evaluating with Euclidean distance...")
            euclidean_results = evaluate_model_distance_sum(config, metric='euclidean')

            print("\n" + "=" * 60 + "\n")

            print("Evaluating with Cosine distance...")
            cosine_results = evaluate_model_distance_sum(config, metric='cosine')

            print("\n" + "=" * 60 + "\n")

            # Compare metrics
            comparison = compare_metrics(config, metrics=['euclidean', 'cosine'])

        else:
            print(f"Configuration file not found: {config_path}")
            print("Please run training first or provide a valid config path.")
