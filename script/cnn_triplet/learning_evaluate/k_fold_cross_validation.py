import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from PIL import Image
import json
import matplotlib.pyplot as plt
from collections import defaultdict

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


def k_fold_cross_validation(config, k_folds=5, k_nn_values=[3, 5, 7], random_state=42):
    """
    学習データに対してk-分割交差検証を実行する

    Args:
        config (dict): 設定辞書
        k_folds (int): 分割数
        k_nn_values (list): テストするk-NN値のリスト
        random_state (int): ランダムシード

    Returns:
        dict: 交差検証結果
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

    # 各クラスのサンプル数を確認
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Samples per class: {[(class_names[i], class_counts[i]) for i in sorted(class_counts.keys())]}")

    # 埋め込みベクトル生成
    print("Generating embeddings...")
    embeddings = embedding_model.predict([images, conf_maps])

    # k-分割交差検証の設定
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    results = {}
    detailed_fold_results = {}

    for k_nn in k_nn_values:
        print(f"\n--- k-NN = {k_nn} ---")

        # 交差検証実行
        knn_classifier = KNeighborsClassifier(n_neighbors=k_nn)

        # sklearn の cross_val_score を使用
        cv_scores = cross_val_score(knn_classifier, embeddings, labels, cv=skf, scoring='accuracy')

        # 詳細な分割ごとの結果を取得
        fold_results = []
        fold_predictions = []
        fold_true_labels = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
            print(f"  Fold {fold_idx + 1}/{k_folds}...")

            X_train_fold, X_val_fold = embeddings[train_idx], embeddings[val_idx]
            y_train_fold, y_val_fold = labels[train_idx], labels[val_idx]

            # k-NN分類器を訓練
            knn_fold = KNeighborsClassifier(n_neighbors=k_nn)
            knn_fold.fit(X_train_fold, y_train_fold)

            # 予測
            y_pred_fold = knn_fold.predict(X_val_fold)

            # 精度計算
            accuracy_fold = accuracy_score(y_val_fold, y_pred_fold)

            # 分類レポート
            report_fold = classification_report(y_val_fold, y_pred_fold, target_names=class_names, output_dict=True)

            # 混同行列
            cm_fold = confusion_matrix(y_val_fold, y_pred_fold)

            fold_result = {
                "fold": fold_idx + 1,
                "accuracy": accuracy_fold,
                "classification_report": report_fold,
                "confusion_matrix": cm_fold.tolist(),
                "train_samples": len(X_train_fold),
                "val_samples": len(X_val_fold)
            }

            fold_results.append(fold_result)
            fold_predictions.extend(y_pred_fold)
            fold_true_labels.extend(y_val_fold)

            print(f"    Accuracy: {accuracy_fold:.4f}")

        # k-NN結果の統計
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)

        # 全体の混同行列（全分割の予測を統合）
        overall_cm = confusion_matrix(fold_true_labels, fold_predictions)

        # 全体の分類レポート
        overall_report = classification_report(fold_true_labels, fold_predictions, target_names=class_names, output_dict=True)

        results[f"k_nn_{k_nn}"] = {
            "k_nn_value": k_nn,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "cv_scores": cv_scores.tolist(),
            "overall_confusion_matrix": overall_cm.tolist(),
            "overall_classification_report": overall_report,
            "fold_results": fold_results
        }

        detailed_fold_results[f"k_nn_{k_nn}"] = fold_results

        print(f"  Cross-validation accuracy: {mean_accuracy:.4f} (+/- {std_accuracy*2:.4f})")

    # 最高精度のk-NN値を特定
    best_k_nn = max(k_nn_values, key=lambda k: results[f"k_nn_{k}"]["mean_accuracy"])
    best_accuracy = results[f"k_nn_{best_k_nn}"]["mean_accuracy"]

    print(f"\nBest k-NN: {best_k_nn} with mean accuracy: {best_accuracy:.4f}")

    # 総合結果
    summary_results = {
        "cross_validation_info": {
            "k_folds": k_folds,
            "k_nn_values": k_nn_values,
            "random_state": random_state,
            "total_samples": len(images),
            "num_classes": len(class_names),
            "class_names": class_names,
            "class_counts": [(class_names[i], int(class_counts[i])) for i in sorted(class_counts.keys())]
        },
        "best_k_nn": best_k_nn,
        "best_mean_accuracy": float(best_accuracy),
        "k_nn_results": results
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

    json_safe_results = convert_numpy_types(summary_results)

    with open(os.path.join(output_dir, "learning_k_fold_cv_results.json"), "w") as f:
        json.dump(json_safe_results, f, indent=4)

    # テキスト形式の詳細レポート保存
    with open(os.path.join(output_dir, "learning_k_fold_cv_report.txt"), "w") as f:
        f.write("Learning Data k-Fold Cross Validation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {config['dataset_root']}\n")
        f.write(f"Total samples: {len(images)}\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"k-folds: {k_folds}\n")
        f.write(f"Random state: {random_state}\n\n")

        f.write("Class distribution:\n")
        for class_name, count in [(class_names[i], class_counts[i]) for i in sorted(class_counts.keys())]:
            f.write(f"  {class_name}: {count} samples\n")
        f.write("\n")

        f.write("Cross-validation Results:\n")
        f.write("-" * 40 + "\n")
        for k_nn in k_nn_values:
            result = results[f"k_nn_{k_nn}"]
            f.write(f"k-NN = {k_nn}:\n")
            f.write(f"  Mean accuracy: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']*2:.4f})\n")
            f.write(f"  Individual fold scores: {[f'{score:.4f}' for score in result['cv_scores']]}\n\n")

        f.write(f"Best k-NN: {best_k_nn} (mean accuracy: {best_accuracy:.4f})\n\n")

        # 最高性能k-NNの詳細レポート
        best_result = results[f"k_nn_{best_k_nn}"]
        f.write(f"Detailed Report for Best k-NN ({best_k_nn}):\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(fold_true_labels, fold_predictions, target_names=class_names))

    # 可視化
    # 1. k-NN値 vs 精度のグラフ
    plt.figure(figsize=(12, 8))

    # サブプロット1: 平均精度とエラーバー
    plt.subplot(2, 2, 1)
    mean_accs = [results[f"k_nn_{k}"]["mean_accuracy"] for k in k_nn_values]
    std_accs = [results[f"k_nn_{k}"]["std_accuracy"] for k in k_nn_values]

    plt.errorbar(k_nn_values, mean_accs, yerr=std_accs, fmt='o-', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('k-NN value')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('k-NN Performance with Error Bars')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_nn_values)

    # サブプロット2: 各分割の精度分布
    plt.subplot(2, 2, 2)
    all_fold_scores = []
    labels_for_box = []
    for k_nn in k_nn_values:
        all_fold_scores.append(results[f"k_nn_{k_nn}"]["cv_scores"])
        labels_for_box.append(f'k={k_nn}')

    plt.boxplot(all_fold_scores, labels=labels_for_box)
    plt.ylabel('Accuracy')
    plt.title('Cross-validation Score Distribution')
    plt.grid(True, alpha=0.3)

    # サブプロット3: 最高性能k-NNの混同行列
    plt.subplot(2, 2, 3)
    best_cm = np.array(results[f"k_nn_{best_k_nn}"]["overall_confusion_matrix"])
    if HAS_SEABORN:
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    else:
        # Simple matplotlib heatmap without seaborn
        im = plt.imshow(best_cm, cmap='Blues', interpolation='nearest')
        plt.colorbar(im)

        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(best_cm[i][j]), ha='center', va='center')

        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
    plt.title(f'Confusion Matrix (k-NN={best_k_nn})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # サブプロット4: 分割ごとの精度推移
    plt.subplot(2, 2, 4)
    for k_nn in k_nn_values:
        fold_accs = [fold["accuracy"] for fold in results[f"k_nn_{k_nn}"]["fold_results"]]
        plt.plot(range(1, k_folds + 1), fold_accs, 'o-', label=f'k-NN={k_nn}')

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, k_folds + 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_k_fold_cv_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 詳細な混同行列（最高性能k-NN）
    plt.figure(figsize=(10, 8))

    if HAS_SEABORN:
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    else:
        # Simple matplotlib heatmap without seaborn
        im = plt.imshow(best_cm, cmap='Blues', interpolation='nearest')
        cbar = plt.colorbar(im)
        cbar.set_label('Count')

        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(best_cm[i][j]), ha='center', va='center')

        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)

    plt.title(f'Detailed Confusion Matrix\nk-NN={best_k_nn}, Mean Accuracy={best_accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_k_fold_cv_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {output_dir}")
    print("Files created:")
    print("- learning_k_fold_cv_results.json")
    print("- learning_k_fold_cv_report.txt")
    print("- learning_k_fold_cv_analysis.png")
    print("- learning_k_fold_cv_confusion_matrix.png")

    return summary_results


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

    # 交差検証実行
    results = k_fold_cross_validation(config, k_folds=5, k_nn_values=[3, 5, 7, 10])