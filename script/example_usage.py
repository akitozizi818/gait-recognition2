"""
CNN Triplet Package Usage Examples

This file demonstrates various ways to use the cnn_triplet package
with different levels of __init__.py initialization.
"""

import sys
import os

# Add the script directory to Python path
sys.path.append(os.path.dirname(__file__))

print("=" * 60)
print("CNN TRIPLET PACKAGE USAGE EXAMPLES")
print("=" * 60)

# Example 1: パッケージレベルでの直接インポート
print("\n1. パッケージレベルでの直接インポート:")
try:
    from cnn_triplet import train_model, evaluate_model, create_config, get_version

    print(f"   パッケージバージョン: {get_version()}")
    print("   ✓ 主要関数を直接インポート成功")

    # 設定作成
    config = create_config(
        dataset_name="sorted_dataset_scale_exclude_45",
        epochs=10,  # テスト用に短縮
        batch_size=4
    )
    print(f"   ✓ 設定作成完了 (エポック数: {config['epochs']})")

except ImportError as e:
    print(f"   ✗ インポートエラー: {e}")

# Example 2: サブモジュールからの詳細インポート
print("\n2. サブモジュールからの詳細インポート:")
try:
    from cnn_triplet.learning import TripletGenerator, create_embedding_model
    from cnn_triplet.test import load_test_data

    print("   ✓ 学習モジュールからTripletGenerator, create_embedding_modelをインポート")
    print("   ✓ テストモジュールからload_test_dataをインポート")

except ImportError as e:
    print(f"   ✗ インポートエラー: {e}")

# Example 3: 従来の方法（フルパス）
print("\n3. 従来の方法（フルパス指定）:")
try:
    from cnn_triplet.learning.separate_viewpoint import train_model as train_func
    from cnn_triplet.test.k_nn import evaluate_model as eval_func

    print("   ✓ フルパスでの個別インポート成功")

except ImportError as e:
    print(f"   ✗ インポートエラー: {e}")

# Example 4: __all__の確認
print("\n4. 利用可能な関数・クラスの確認:")
try:
    import cnn_triplet

    if hasattr(cnn_triplet, '__all__'):
        print("   パッケージで公開されている要素:")
        for item in cnn_triplet.__all__:
            print(f"     - {item}")

    print(f"   デフォルト設定:")
    for key, value in cnn_triplet.DEFAULT_CONFIG.items():
        print(f"     {key}: {value}")

except ImportError as e:
    print(f"   ✗ インポートエラー: {e}")

# Example 5: 実用的な使用例
print("\n5. 実用的な使用例:")
print("""
   # シンプルな学習実行
   from cnn_triplet import train_model, create_config

   config = create_config(
       dataset_name="my_dataset",
       epochs=100,
       batch_size=16
   )
   model, history, output_dir = train_model(config)

   # シンプルな評価実行
   from cnn_triplet import evaluate_model
   results = evaluate_model(config)

   # 詳細な操作が必要な場合
   from cnn_triplet.learning import TripletGenerator
   from cnn_triplet.test import load_test_data

   generator = TripletGenerator(...)
   test_data = load_test_data(...)
""")

print("\n" + "=" * 60)
print("INITIALIZATION BENEFITS")
print("=" * 60)

print("""
__init__.pyファイルの初期化により以下の利点があります:

1. 【シンプルなインポート】
   ❌ from cnn_triplet.learning.separate_viewpoint import train_model
   ✅ from cnn_triplet import train_model

2. 【共通設定の提供】
   ✅ cnn_triplet.DEFAULT_CONFIG でデフォルト値にアクセス
   ✅ cnn_triplet.create_config() で簡単設定作成

3. 【バージョン管理】
   ✅ cnn_triplet.get_version() でバージョン確認

4. 【エラーハンドリング】
   ✅ 依存関係エラーを適切にキャッチ

5. 【API統一】
   ✅ __all__ で公開APIを明確化
   ✅ パッケージレベルでのドキュメント提供

6. 【再利用性】
   ✅ 他のプロジェクトでの使用が簡単
   ✅ pip installable なパッケージとして配布可能
""")

print("=" * 60)