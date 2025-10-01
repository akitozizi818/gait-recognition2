import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import matplotlib.pyplot as plt

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not available. Please install it with: pip install umap-learn")

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


def load_dataset(data_dir, confidence_dir, img_size):
    """
    データセットを読み込む

    Args:
        data_dir (str): 画像が格納されたディレクトリ
        confidence_dir (str): 信頼度マップが格納されたディレクトリ
        img_size (tuple): 画像サイズ (width, height)

    Returns:
        tuple: (images, confidence_maps, labels, class_indices, class_names)
    """
    images, conf_maps, labels = [], [], []
    class_indices = {name: i for i, name in enumerate(sorted(os.listdir(data_dir)))}
    class_names = list(class_indices.keys())

    print(f"Loading dataset from: {data_dir}")
    print(f"Found classes: {class_names}")

    for class_name, class_idx in class_indices.items():
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_count = 0
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                conf_path = os.path.join(confidence_dir, filename)

                if os.path.exists(conf_path):
                    img = np.array(Image.open(img_path).resize(img_size)) / 255.0
                    conf = np.array(Image.open(conf_path).resize(img_size)) / 255.0

                    images.append(img)
                    conf_maps.append(conf)
                    labels.append(class_idx)
                    class_count += 1

            print(f"  {class_name}: {class_count} samples")

    print(f"Total samples loaded: {len(images)}")
    return np.array(images), np.array(conf_maps), np.array(labels), class_indices, class_names


def visualize_embeddings_umap(config, dataset_type="train", umap_params=None):
    """
    学習済みモデルを使って埋め込みベクトルを生成し、UMAPで2次元可視化する

    Args:
        config (dict): 設定辞書
        dataset_type (str): データセットタイプ ("train" または "test")
        umap_params (dict): UMAPのパラメータ

    Returns:
        dict: 可視化結果と統計情報
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is not installed. Please install it with: pip install umap-learn")

    # デフォルトUMAPパラメータ
    default_umap_params = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 42
    }

    if umap_params:
        default_umap_params.update(umap_params)

    print("=" * 60)
    print(f"UMAP Visualization for {dataset_type.upper()} Dataset")
    print("=" * 60)
    print(f"UMAP Parameters: {default_umap_params}")

    # モデル読み込み
    model_path = os.path.join(config["output_dir"], config["model_name"])
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please run training first.")

    custom_objects = {'ConfidenceDropout': ConfidenceDropout}
    embedding_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # データセット読み込み
    if dataset_type == "train":
        data_dir = config["dataset_root"]
    elif dataset_type == "test":
        data_dir = config["test_data_dir"]
    else:
        raise ValueError("dataset_type must be 'train' or 'test'")

    images, conf_maps, labels, class_indices, class_names = load_dataset(
        data_dir,
        config["confidence_map_dir"],
        config["img_size"]
    )

    # 埋め込みベクトル生成
    print("Generating embeddings...")
    embeddings = embedding_model.predict([images, conf_maps])
    print(f"Embedding shape: {embeddings.shape}")

    # UMAP適用
    print("Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(**default_umap_params)
    embedding_2d = reducer.fit_transform(embeddings)

    # 可視化
    plt.figure(figsize=(12, 10))

    # カラーパレットの設定
    if HAS_SEABORN:
        colors = sns.color_palette("husl", len(class_names))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    # クラスごとにプロット
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[colors[i]], label=class_name, alpha=0.7, s=50)

    plt.title(f'UMAP Visualization of {dataset_type.capitalize()} Dataset Embeddings\n'
              f'n_neighbors={default_umap_params["n_neighbors"]}, '
              f'min_dist={default_umap_params["min_dist"]}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 結果保存
    output_dir = config["output_dir"]
    plot_filename = f"umap_visualization_{dataset_type}.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()

    # クラス間距離の統計計算
    print("\nCalculating inter-class and intra-class distances...")

    # 各クラスの中心点を計算
    class_centers = {}
    intra_class_distances = {}

    for i, class_name in enumerate(class_names):
        mask = labels == i
        class_embeddings = embedding_2d[mask]
        center = np.mean(class_embeddings, axis=0)
        class_centers[class_name] = center

        # クラス内距離（中心からの平均距離）
        distances_to_center = np.linalg.norm(class_embeddings - center, axis=1)
        intra_class_distances[class_name] = {
            "mean": float(np.mean(distances_to_center)),
            "std": float(np.std(distances_to_center)),
            "median": float(np.median(distances_to_center))
        }

    # クラス間距離を計算
    inter_class_distances = {}
    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            if i < j:  # 重複を避ける
                distance = np.linalg.norm(class_centers[class1] - class_centers[class2])
                inter_class_distances[f"{class1}_vs_{class2}"] = float(distance)

    # 結果をまとめ
    results = {
        "dataset_type": dataset_type,
        "umap_parameters": default_umap_params,
        "dataset_info": {
            "total_samples": len(images),
            "num_classes": len(class_names),
            "class_names": class_names,
            "samples_per_class": {name: int(np.sum(labels == i)) for i, name in enumerate(class_names)}
        },
        "embedding_info": {
            "original_dimension": embeddings.shape[1],
            "reduced_dimension": embedding_2d.shape[1]
        },
        "distance_analysis": {
            "intra_class_distances": intra_class_distances,
            "inter_class_distances": inter_class_distances,
            "class_centers": {name: center.tolist() for name, center in class_centers.items()}
        },
        "visualization_file": plot_filename
    }

    # JSON形式で結果保存
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

    results_filename = f"umap_analysis_{dataset_type}.json"
    with open(os.path.join(output_dir, results_filename), "w") as f:
        json.dump(json_safe_results, f, indent=4)

    # テキスト形式の詳細レポート
    report_filename = f"umap_report_{dataset_type}.txt"
    with open(os.path.join(output_dir, report_filename), "w") as f:
        f.write(f"UMAP Visualization Report - {dataset_type.capitalize()} Dataset\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Total samples: {len(images)}\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")

        f.write("UMAP Parameters:\n")
        for key, value in default_umap_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Samples per class:\n")
        for name, count in results["dataset_info"]["samples_per_class"].items():
            f.write(f"  {name}: {count}\n")
        f.write("\n")

        f.write("Intra-class distances (mean ± std):\n")
        for class_name, stats in intra_class_distances.items():
            f.write(f"  {class_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        f.write("\n")

        f.write("Inter-class distances:\n")
        for pair, distance in inter_class_distances.items():
            f.write(f"  {pair}: {distance:.4f}\n")

    print(f"\nResults saved to: {output_dir}")
    print("Files created:")
    print(f"- {plot_filename}")
    print(f"- {results_filename}")
    print(f"- {report_filename}")

    # 統計サマリーを表示
    print(f"\nVisualization Summary:")
    print(f"Dataset: {dataset_type}")
    print(f"Total samples: {len(images)}")
    print(f"Classes: {len(class_names)}")
    print(f"Average intra-class distance: {np.mean([stats['mean'] for stats in intra_class_distances.values()]):.4f}")
    print(f"Average inter-class distance: {np.mean(list(inter_class_distances.values())):.4f}")

    return results


if __name__ == "__main__":
    # テスト用のデフォルト設定
    dataset_name = "sorted_dataset_scale_exclude_45"

    config = {
        "dataset_root": f"../../../data/{dataset_name}/dataset",
        "test_data_dir": f"../../../data/{dataset_name}/test_dataset",
        "confidence_map_dir": "../../../data/cut_confidence_image",
        "output_dir": f"../../../data/triplet_loss_output/{dataset_name}",
        "img_size": (48, 48),
        "model_name": "embedding_model.h5",
        "confidence_rate": 0.4
    }

    # UMAPパラメータ
    umap_params = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean"
    }

    try:
        # 学習データセットの可視化
        print("Visualizing training dataset...")
        train_results = visualize_embeddings_umap(config, "train", umap_params)

        # テストデータセットの可視化
        print("\nVisualizing test dataset...")
        test_results = visualize_embeddings_umap(config, "test", umap_params)

    except Exception as e:
        print(f"Error: {e}")