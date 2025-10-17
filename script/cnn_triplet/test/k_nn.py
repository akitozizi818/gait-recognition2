import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import json


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


def evaluate_model(config, model_path=None):
    """
    Evaluate the trained embedding model using k-NN classification.

    Args:
        config (dict): Configuration dictionary
        model_path (str, optional): Path to the saved model. If None, uses config["output_dir"]/config["model_name"]

    Returns:
        dict: Evaluation results including accuracy and classification report
    """
    # Load model
    if model_path is None:
        model_path = os.path.join(config["output_dir"], config["model_name"])

    print(f"Loading model from: {model_path}")
    # カスタムオブジェクトを指定してモデル読み込み
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

    # Use k-NN to classify the test embeddings
    print("Classifying test embeddings using k-NN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings, train_labels)
    predictions = knn.predict(test_embeddings)

    # Calculate accuracy and generate classification report
    accuracy = accuracy_score(test_labels, predictions)
    class_names = [name for name, _ in sorted(train_class_indices.items(), key=lambda x: x[1])]
    report = classification_report(
        test_labels,
        predictions,
        target_names=class_names,
        output_dict=True
    )

    print(f"\nK-NN classification accuracy on test embeddings: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))

    # Save results
    output_dir = config["output_dir"]
    results = {
        "accuracy": accuracy,
        "classification_report": report,
        "num_train_samples": len(train_labels),
        "num_test_samples": len(test_labels),
        "num_classes": len(class_names)
    }

    # Save detailed results（NumPy型を標準型に変換）
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

    json_safe_results = convert_numpy_types(results)

    with open(os.path.join(output_dir, "knn_evaluation_results.json"), "w") as f:
        json.dump(json_safe_results, f, indent=4)

    # Save simple accuracy report
    with open(os.path.join(output_dir, "knn_accuracy_report.txt"), "w") as f:
        f.write(f"K-NN classification accuracy: {accuracy:.4f}\n")
        f.write(f"Number of training samples: {len(train_labels)}\n")
        f.write(f"Number of test samples: {len(test_labels)}\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write("This accuracy measures how well the learned embeddings separate the classes in the test set.\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, predictions, target_names=class_names))

    # 視点ごとの正解率を計算
    target_keywords = ["45", "90", "225", "270"]
    viewpoint_results = {}

    for keyword in target_keywords:
        # ファイル名をアンダースコアで分割した1番目の要素でフィルタリング
        indices = [
            i for i, file_path in enumerate(test_files)
            if keyword == file_path.split('_')[1]
        ]

        if not indices:
            viewpoint_results[keyword] = {"match_ratio": 0, "total": 0, "matches": 0}
            continue

        # フィルタリングされたデータの正解ラベルと予測ラベル
        filtered_true_classes = [test_labels[i] for i in indices]
        filtered_predicted_classes = [predictions[i] for i in indices]

        # 一致する数を計算
        matches = sum(1 for true, pred in zip(filtered_true_classes, filtered_predicted_classes) if true == pred)
        total = len(indices)
        match_ratio = matches / total if total > 0 else 0

        # 結果を保存
        viewpoint_results[keyword] = {
            "match_ratio": match_ratio,
            "total": total,
            "matches": matches
        }

    # 視点ごとの結果をテキストファイルに保存
    with open(os.path.join(output_dir, "viewpoints_results.txt"), "w", encoding="utf-8") as file:
        for keyword, result in viewpoint_results.items():
            file.write(f"Keyword: {keyword}\n")
            file.write(f"  Match Ratio: {result['match_ratio']:.2f}\n")
            file.write(f"  Total Samples: {result['total']}\n")
            file.write(f"  Matches: {result['matches']}\n\n")

    print(f"Evaluation results saved to: {output_dir}")

    return results


def evaluate_with_different_k(config, k_values=[1, 3, 5, 7, 10], model_path=None):
    """
    Evaluate the model with different k values for k-NN.

    Args:
        config (dict): Configuration dictionary
        k_values (list): List of k values to test
        model_path (str, optional): Path to the saved model

    Returns:
        dict: Results for each k value
    """
    # Load model
    if model_path is None:
        model_path = os.path.join(config["output_dir"], config["model_name"])

    print(f"Loading model from: {model_path}")
    # カスタムオブジェクトを指定してモデル読み込み
    custom_objects = {'ConfidenceDropout': ConfidenceDropout}
    embedding_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # Load data
    print("Loading data...")
    train_images, train_conf, train_labels, _, _ = load_test_data(
        config["dataset_root"],
        config["confidence_map_dir"],
        config["img_size"]
    )

    test_images, test_conf, test_labels, _, _ = load_test_data(
        config["test_data_dir"],
        config["confidence_map_dir"],
        config["img_size"]
    )

    # Generate embeddings
    print("Generating embeddings...")
    train_embeddings = embedding_model.predict([train_images, train_conf])
    test_embeddings = embedding_model.predict([test_images, test_conf])

    # Test different k values
    results = {}
    for k in k_values:
        print(f"Testing k={k}...")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_embeddings, train_labels)
        predictions = knn.predict(test_embeddings)
        accuracy = accuracy_score(test_labels, predictions)
        results[f"k_{k}"] = accuracy
        print(f"k={k}: accuracy = {accuracy:.4f}")

    # Save results（NumPy型を標準型に変換）
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

    json_safe_results = convert_numpy_types(results)
    with open(os.path.join(config["output_dir"], "knn_k_comparison.json"), "w") as f:
        json.dump(json_safe_results, f, indent=4)

    return results


if __name__ == "__main__":
    # Load configuration from saved file
    dataset_name = "sorted_dataset_scale_exclude_45"
    config_path = f"../../../data/triplet_loss_output/{dataset_name}/config.json"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Run evaluation
        results = evaluate_model(config)

        # Also test different k values
        k_results = evaluate_with_different_k(config)

    else:
        print(f"Configuration file not found: {config_path}")
        print("Please run training first or provide a valid config path.")