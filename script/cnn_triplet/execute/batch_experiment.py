"""
Batch Experiment Script for CNN Triplet Loss

This script automatically runs training, testing, and learning evaluation
for multiple datasets in sequence.
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime


# 実験対象のデータセット
DATASETS = [
    "sorted_dataset_scale_exclude_45",
    "sorted_dataset_scale_exclude_90",
    "sorted_dataset_scale_exclude_225",
    "sorted_dataset_scale_exclude_270"
]

# デフォルト実験設定
DEFAULT_CONFIG = {
    "epochs": 30,
    "batch_size": 8,
    "learning_rate": 0.000392,
    "embedding_dim": 128,
    "confidence_rate": 0.4
}


def log_message(message, level="INFO"):
    """ログメッセージを出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def run_command(command, description):
    """
    コマンドを実行し結果を返す

    Args:
        command (list): 実行するコマンド
        description (str): コマンドの説明

    Returns:
        tuple: (success, stdout, stderr, execution_time)
    """
    log_message(f"実行中: {description}")
    log_message(f"コマンド: {' '.join(command)}")

    start_time = time.time()

    try:
        # リアルタイムでログを表示するため、capture_output=Falseに変更
        result = subprocess.run(
            command,
            text=True,
            timeout=3600*24  # 24時間のタイムアウト
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            log_message(f"成功: {description} ({execution_time:.2f}秒)")
            return True, "", "", execution_time
        else:
            log_message(f"失敗: {description}", "ERROR")
            log_message(f"リターンコード: {result.returncode}", "ERROR")
            return False, "", "", execution_time

    except subprocess.TimeoutExpired:
        log_message(f"タイムアウト: {description}", "ERROR")
        return False, "", "タイムアウトエラー", time.time() - start_time
    except Exception as e:
        log_message(f"例外発生: {str(e)}", "ERROR")
        return False, "", str(e), time.time() - start_time


def run_training_and_testing(dataset_name, config):
    """学習とテストを実行"""
    command = [
        sys.executable, "main.py", "both",
        "--dataset", dataset_name,
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--embedding-dim", str(config["embedding_dim"]),
        "--confidence-rate", str(config["confidence_rate"])
    ]

    return run_command(command, f"学習とテスト ({dataset_name})")


def run_learning_evaluation(dataset_name, eval_type="both"):
    """学習データ評価を実行"""
    command = [
        sys.executable, "main.py", "learning-eval",
        "--dataset", dataset_name,
        "--eval-type", eval_type
    ]

    return run_command(command, f"学習データ評価 ({dataset_name})")


def run_single_experiment(dataset_name, config):
    """
    単一データセットで完全な実験を実行

    Args:
        dataset_name (str): データセット名
        config (dict): 実験設定

    Returns:
        dict: 実験結果の詳細
    """
    log_message(f"{'='*60}")
    log_message(f"実験開始: {dataset_name}")
    log_message(f"{'='*60}")

    experiment_start_time = time.time()
    results = {
        "dataset": dataset_name,
        "config": config,
        "start_time": datetime.now().isoformat(),
        "steps": {},
        "total_time": 0,
        "success": False
    }

    # ステップ1: 学習とテスト
    log_message("ステップ1: 学習とテスト実行中...")
    success, stdout, stderr, exec_time = run_training_and_testing(dataset_name, config)
    results["steps"]["training_testing"] = {
        "success": success,
        "execution_time": exec_time,
        "stdout": "ログは実行時に表示されました",
        "stderr": "ログは実行時に表示されました"
    }

    if not success:
        log_message(f"❌ {dataset_name}: 学習・テストで失敗", "ERROR")
        results["end_time"] = datetime.now().isoformat()
        results["total_time"] = time.time() - experiment_start_time
        return results

    # ステップ2: 学習データ評価
    log_message("ステップ2: 学習データ評価実行中...")
    success, stdout, stderr, exec_time = run_learning_evaluation(dataset_name, "both")
    results["steps"]["learning_evaluation"] = {
        "success": success,
        "execution_time": exec_time,
        "stdout": "ログは実行時に表示されました",
        "stderr": "ログは実行時に表示されました"
    }

    if not success:
        log_message(f"❌ {dataset_name}: 学習データ評価で失敗", "ERROR")
    else:
        log_message(f"✅ {dataset_name}: 全ステップ完了")
        results["success"] = True

    results["end_time"] = datetime.now().isoformat()
    results["total_time"] = time.time() - experiment_start_time

    log_message(f"{dataset_name} 実験完了 (総時間: {results['total_time']:.2f}秒)")
    return results


def save_experiment_log(all_results, log_file="experiment_log.json"):
    """実験ログをJSONファイルに保存"""
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        log_message(f"実験ログを保存: {log_file}")
    except Exception as e:
        log_message(f"ログ保存エラー: {str(e)}", "ERROR")


def print_summary(all_results):
    """実験結果のサマリーを出力"""
    print("\n" + "="*80)
    print("実験結果サマリー")
    print("="*80)

    total_experiments = len(all_results["experiments"])
    successful_experiments = sum(1 for exp in all_results["experiments"] if exp["success"])

    print(f"実行日時: {all_results['start_time']}")
    print(f"総実験数: {total_experiments}")
    print(f"成功: {successful_experiments}")
    print(f"失敗: {total_experiments - successful_experiments}")
    print(f"総実行時間: {all_results['total_execution_time']:.2f}秒")

    print("\n詳細結果:")
    print("-" * 80)

    for exp in all_results["experiments"]:
        status = "✅ 成功" if exp["success"] else "❌ 失敗"
        print(f"{exp['dataset']:<40} {status:>8} ({exp['total_time']:.2f}秒)")

        # ステップ別の結果
        for step_name, step_result in exp["steps"].items():
            step_status = "✅" if step_result["success"] else "❌"
            step_time = step_result["execution_time"]
            print(f"  └─ {step_name:<30} {step_status} ({step_time:.2f}秒)")

    print("\n" + "="*80)


def main():
    """メイン実行関数"""
    log_message("バッチ実験スクリプト開始")
    log_message(f"対象データセット: {', '.join(DATASETS)}")

    # 実験全体の結果を記録
    all_results = {
        "start_time": datetime.now().isoformat(),
        "config": DEFAULT_CONFIG,
        "datasets": DATASETS,
        "experiments": [],
        "total_execution_time": 0
    }

    batch_start_time = time.time()

    # 各データセットで実験実行
    for i, dataset in enumerate(DATASETS, 1):
        log_message(f"\n[{i}/{len(DATASETS)}] {dataset} の実験を開始")

        experiment_result = run_single_experiment(dataset, DEFAULT_CONFIG)
        all_results["experiments"].append(experiment_result)

        # 中間結果保存
        all_results["total_execution_time"] = time.time() - batch_start_time
        save_experiment_log(all_results, f"experiment_log_partial_{i}.json")

    # 最終結果
    all_results["end_time"] = datetime.now().isoformat()
    all_results["total_execution_time"] = time.time() - batch_start_time

    # 最終ログ保存
    save_experiment_log(all_results, "experiment_log_final.json")

    # サマリー出力
    print_summary(all_results)

    log_message("バッチ実験スクリプト完了")


if __name__ == "__main__":
    # 実行前チェック
    if not os.path.exists("main.py"):
        print("エラー: main.pyが見つかりません。script/cnn_triplet/executeディレクトリで実行してください。")
        sys.exit(1)

    try:
        main()
    except KeyboardInterrupt:
        log_message("ユーザーによって中断されました", "WARNING")
        sys.exit(1)
    except Exception as e:
        log_message(f"予期しないエラー: {str(e)}", "ERROR")
        sys.exit(1)