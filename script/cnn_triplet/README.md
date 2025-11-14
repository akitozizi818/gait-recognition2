# CNN Triplet Loss - コード概要ドキュメント

このドキュメントでは、`script/cnn_triplet`ディレクトリ内の各フォルダに含まれるコードの概要を説明します。

## 目次
- [evaluate: 評価スクリプト](#evaluate-評価スクリプト)
- [execute: 実行スクリプト](#execute-実行スクリプト)
- [learning: 学習スクリプト](#learning-学習スクリプト)
- [test: テストスクリプト](#test-テストスクリプト)

---

## evaluate: 評価スクリプト

学習済みモデルの性能評価と可視化を行うスクリプト群です。

### umap.py
**目的**: 学習済みモデルが生成する埋め込みベクトルをUMAPで2次元に次元削減し、可視化します。

**主な機能**:
- `load_dataset()`: 画像と信頼度マップをロードする
- `visualize_embeddings_umap()`: UMAP次元削減と可視化を実行
  - 学習データセットまたはテストデータセットの埋め込みベクトルを生成
  - UMAPで2次元に削減
  - クラスごとに色分けした散布図を生成
  - クラス間距離・クラス内距離の統計分析
  - 結果をPNG、JSON、TXTで保存

**出力ファイル**:
- `umap_visualization_{dataset_type}.png`: 可視化グラフ
- `umap_analysis_{dataset_type}.json`: 数値結果
- `umap_report_{dataset_type}.txt`: テキストレポート

### k_nn.py
**目的**: 学習データを訓練用と検証用に分割し、k-NN分類で性能を評価します。

**主な機能**:
- `load_training_data()`: 学習データをロード
- `evaluate_training_data_knn()`: k-NN評価を実行
  - 学習データを訓練用と検証用に分割（デフォルト7:3）
  - 複数のk値（1, 3, 5, 7, 10）で評価
  - 最適なk値を特定
  - 混同行列と分類レポートを生成

**出力ファイル**:
- `learning_knn_evaluation.json`: 評価結果（JSON形式）
- `learning_knn_report.txt`: テキストレポート
- `learning_knn_confusion_matrix.png`: 混同行列
- `learning_knn_accuracy_curve.png`: k値vs精度のグラフ

### k_fold_cross_validation.py
**目的**: k分割交差検証を用いて、学習データの性能を評価します。

**主な機能**:
- `load_training_data()`: 学習データをロード
- `k_fold_cross_validation()`: k分割交差検証を実行
  - デフォルト5分割の交差検証
  - 複数のk-NN値（3, 5, 7, 10）で評価
  - 各分割の精度と全体の平均精度・標準偏差を計算
  - 分割ごとの詳細結果を記録

**出力ファイル**:
- `learning_k_fold_cv_results.json`: 交差検証結果
- `learning_k_fold_cv_report.txt`: テキストレポート
- `learning_k_fold_cv_analysis.png`: 分析グラフ（4つのサブプロット）
- `learning_k_fold_cv_confusion_matrix.png`: 混同行列

---

## execute: 実行スクリプト

学習・評価・バッチ実験を実行するためのメインスクリプト群です。

### main.py
**目的**: CNN Triplet Lossの学習・テスト・評価を統合的に実行するメインエントリーポイントです。

**主な機能**:
- `create_default_config()`: デフォルト設定を生成
- `load_config_from_file()` / `save_config_to_file()`: 設定ファイルの読み書き
- `run_training()`: 学習プロセスを実行
- `run_evaluation()`: テストデータの評価を実行
- `run_learning_evaluation()`: 学習データの評価を実行（k-NN、k分割交差検証）
- `run_umap_visualization()`: UMAP可視化を実行

**コマンドライン引数**:
```bash
# 学習のみ
python main.py train --dataset sorted_dataset_scale_exclude_45 --epochs 30

# テストのみ
python main.py test --dataset sorted_dataset_scale_exclude_45

# 学習+テスト
python main.py both --dataset sorted_dataset_scale_exclude_45

# 学習データ評価
python main.py learning-eval --dataset sorted_dataset_scale_exclude_45 --eval-type both

# UMAP可視化
python main.py umap --dataset sorted_dataset_scale_exclude_45 --umap-dataset both
```

### batch_experiment.py
**目的**: 複数のデータセットに対して順次実験を自動実行します。

**主な機能**:
- `run_command()`: コマンド実行とログ記録
- `run_training_and_testing()`: 学習とテストを実行
- `run_learning_evaluation()`: 学習データ評価を実行
- `run_single_experiment()`: 単一データセットで完全な実験を実行
  - ステップ1: 学習とテスト
  - ステップ2: 学習データ評価
- `save_experiment_log()`: 実験ログをJSONで保存
- `print_summary()`: 実験結果のサマリーを表示

**デフォルト対象データセット**:
- sorted_dataset_scale_exclude_45
- sorted_dataset_scale_exclude_90
- sorted_dataset_scale_exclude_225
- sorted_dataset_scale_exclude_270

**出力ファイル**:
- `experiment_log_partial_{i}.json`: 中間結果
- `experiment_log_final.json`: 最終結果

### advanced_batch_experiment.py
**目的**: 並列実行機能を備えた高度なバッチ実験スクリプトです。

**主な機能**:
- `run_single_experiment()`: 単一実験を実行
- `run_parallel_experiments()`: 並列実験実行（ThreadPoolExecutor使用）
- `generate_experiment_configs()`: 実験設定リストを生成
- `load_experiment_config()` / `save_experiment_config()`: 設定ファイルの読み書き
- `print_results_summary()`: 結果サマリーを出力

**実験テンプレート**:
- `quick_test`: 10エポック、小規模テスト
- `standard`: 30エポック、標準設定
- `high_quality`: 100エポック、高品質設定
- `comparison_study`: 埋め込み次元の比較研究（64, 128, 256）

**コマンドライン引数**:
```bash
# 標準テンプレートで順次実行
python advanced_batch_experiment.py --template standard

# 並列実行（最大2ワーカー）
python advanced_batch_experiment.py --template standard --parallel --max-workers 2

# カスタム設定ファイルを使用
python advanced_batch_experiment.py --config-file my_config.json

# Dry run（実行せずに設定のみ表示）
python advanced_batch_experiment.py --template comparison_study --dry-run
```

**出力ファイル**:
- `advanced_experiment_results.json`: 実験結果

---

## learning: 学習スクリプト

モデルの学習を行うスクリプトです。

### separate_viewpoint.py
**目的**: Triplet Lossを用いた埋め込みモデルの学習を行います。

**主なクラス・関数**:
- `ConfidenceDropout`: 信頼度マップに基づいて画像をマスクするカスタムレイヤー
- `TripletGenerator`: Triplet（アンカー、正例、負例）を生成するデータジェネレータ
  - 同じクラスからアンカーと正例を選択
  - 異なるクラスから負例を選択
- `create_embedding_model()`: CNNベースの埋め込みモデルを構築
  - 3層のConv2D + BatchNormalization + MaxPooling + Dropout
  - 最終的にL2正規化された埋め込みベクトルを出力
- `triplet_loss()`: Triplet Loss関数
  - `loss = max(d(anchor, positive) - d(anchor, negative) + alpha, 0)`
- `train_model()`: 学習プロセス全体を実行
  - データジェネレータの作成
  - 埋め込みモデルと学習モデルの構築
  - 学習の実行
  - モデルと学習曲線の保存

**モデルアーキテクチャ**:
```
Input (48x48x1) → ConfidenceDropout →
Conv2D(256, 5x5) → BatchNorm → MaxPool → Dropout(0.25) →
Conv2D(128, 3x3) → BatchNorm → MaxPool → Dropout(0.25) →
Conv2D(256, 2x2) → BatchNorm → MaxPool → Dropout(0.25) →
Flatten → Dense(64) → BatchNorm → Dropout(0.25) →
Dense(embedding_dim) → L2 Normalize
```

**出力ファイル**:
- `embedding_model.h5`: 学習済み埋め込みモデル
- `config.json`: 学習設定
- `training_loss_curve.png`: 学習曲線

---

## test: テストスクリプト

テストデータでモデルを評価するスクリプトです。

### k_nn.py
**目的**: 学習済みモデルをテストデータで評価し、k-NN分類による認識精度を測定します。

**主な機能**:
- `load_test_data()`: テストデータをロード
- `evaluate_model()`: k-NN分類でモデルを評価
  - 学習データとテストデータの埋め込みベクトルを生成
  - k-NN（k=5）で分類
  - 精度と分類レポートを計算
  - 視点ごとの正解率を計算（45°, 90°, 225°, 270°）
- `evaluate_with_different_k()`: 異なるk値で評価
  - k=1, 3, 5, 7, 10で評価
  - 各k値の精度を比較

**視点別評価**:
- ファイル名の2番目の要素（アンダースコア区切り）で視点を判別
- 各視点（45, 90, 225, 270）ごとの認識精度を算出

**出力ファイル**:
- `knn_evaluation_results.json`: 評価結果
- `knn_accuracy_report.txt`: テキストレポート
- `viewpoints_results.txt`: 視点ごとの結果
- `knn_k_comparison.json`: k値比較結果

---

## 実験ワークフロー

### 基本的な実験フロー

1. **学習**: `main.py train` または `separate_viewpoint.py`
   - Triplet Lossで埋め込みモデルを学習
   - 出力: 学習済みモデル、設定ファイル、学習曲線

2. **テスト**: `main.py test` または `test/k_nn.py`
   - テストデータでk-NN分類精度を評価
   - 視点ごとの精度も計算

3. **学習データ評価**: `main.py learning-eval`
   - k-NN評価: 学習データを分割して評価
   - k分割交差検証: より堅牢な評価

4. **可視化**: `main.py umap`
   - 埋め込みベクトルを2次元可視化
   - クラス分離の視覚的確認

### バッチ実験

**順次実行**:
```bash
cd script/cnn_triplet/execute
python batch_experiment.py
```

**並列実行**:
```bash
cd script/cnn_triplet/execute
python advanced_batch_experiment.py --template standard --parallel --max-workers 2
```

---

## 設定パラメータ

主要な設定パラメータ:

- `dataset_root`: 学習データディレクトリ
- `test_data_dir`: テストデータディレクトリ
- `confidence_map_dir`: 信頼度マップディレクトリ
- `output_dir`: 出力ディレクトリ
- `img_size`: 画像サイズ（デフォルト: 48x48）
- `batch_size`: バッチサイズ（デフォルト: 8）
- `epochs`: エポック数（デフォルト: 30）
- `embedding_dim`: 埋め込み次元（デフォルト: 128）
- `learning_rate`: 学習率（デフォルト: 0.000392）
- `triplet_alpha`: Triplet Lossのマージン（デフォルト: 0.2）
- `confidence_rate`: 信頼度閾値（デフォルト: 0.4）

---

## 出力ファイル一覧

### 学習時
- `embedding_model.h5`: 学習済みモデル
- `config.json`: 学習設定
- `training_loss_curve.png`: 学習曲線

### テスト時
- `knn_evaluation_results.json`: 評価結果
- `knn_accuracy_report.txt`: テキストレポート
- `viewpoints_results.txt`: 視点ごとの結果
- `knn_k_comparison.json`: k値比較

### 学習データ評価時
- `learning_knn_evaluation.json`: k-NN評価結果
- `learning_knn_report.txt`: k-NNレポート
- `learning_knn_confusion_matrix.png`: 混同行列
- `learning_knn_accuracy_curve.png`: 精度曲線
- `learning_k_fold_cv_results.json`: 交差検証結果
- `learning_k_fold_cv_report.txt`: 交差検証レポート
- `learning_k_fold_cv_analysis.png`: 分析グラフ
- `learning_k_fold_cv_confusion_matrix.png`: 混同行列

### UMAP可視化時
- `umap_visualization_{train|test}.png`: 可視化グラフ
- `umap_analysis_{train|test}.json`: 分析結果
- `umap_report_{train|test}.txt`: テキストレポート

### バッチ実験時
- `experiment_log_final.json`: 実験ログ
- `advanced_experiment_results.json`: 高度実験結果
