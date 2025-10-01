"""
CNN Triplet Loss Module

This package provides functionality for training and evaluating
CNN models with triplet loss for gait recognition.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# パッケージレベルでの初期化
print("CNN Triplet package loaded")

# 共通設定
DEFAULT_CONFIG = {
    "img_size": (48, 48),
    "batch_size": 8,
    "embedding_dim": 128,
    "confidence_rate": 0.4
}

# サブモジュールから主要な関数をインポート
# これにより外部からシンプルにアクセス可能になる
try:
    from .learning.separate_viewpoint import train_model, create_embedding_model
    from .test.k_nn import evaluate_model, evaluate_with_different_k
    from .learning_evaluate.k_nn import evaluate_training_data_knn
    from .learning_evaluate.k_fold_cross_validation import k_fold_cross_validation

    # パッケージレベルで利用可能な関数のリスト
    __all__ = [
        'train_model',
        'create_embedding_model',
        'evaluate_model',
        'evaluate_with_different_k',
        'evaluate_training_data_knn',
        'k_fold_cross_validation',
        'DEFAULT_CONFIG'
    ]

except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    __all__ = ['DEFAULT_CONFIG']

# 便利なヘルパー関数
def get_version():
    """パッケージのバージョンを返す"""
    return __version__

def create_config(dataset_name="sorted_dataset_scale_exclude_45", **kwargs):
    """デフォルト設定をベースにカスタム設定を作成"""
    config = DEFAULT_CONFIG.copy()
    config.update({
        "dataset_root": f"../../../data/{dataset_name}/dataset",
        "test_data_dir": f"../../../data/{dataset_name}/test_dataset",
        "confidence_map_dir": "../../../data/cut_confidence_image",
        "output_dir": f"../../../data/triplet_loss_output/{dataset_name}",
        "num_classes": 10,
        "epochs": 50,
        "triplet_alpha": 0.2,
        "learning_rate": 0.0001,
        "model_name": "embedding_model.h5",
        "config_file_name": "config.json"
    })
    config.update(kwargs)
    return config