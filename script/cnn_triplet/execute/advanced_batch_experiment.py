"""
Advanced Batch Experiment Script for CNN Triplet Loss

This script provides advanced features for running multiple experiments with
different configurations and datasets.
"""

import subprocess
import sys
import os
import time
import json
import argparse
from datetime import datetime
import concurrent.futures
from threading import Lock


# 実験対象のデータセット
DEFAULT_DATASETS = [
    "sorted_dataset_scale_exclude_90",
    "sorted_dataset_scale_exclude_225",
    "sorted_dataset_scale_exclude_270"
]

# 実験設定テンプレート
EXPERIMENT_TEMPLATES = {
    "quick_test": {
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 0.001,
        "embedding_dim": 64,
        "confidence_rate": 0.4
    },
    "standard": {
        "epochs": 30,
        "batch_size": 8,
        "learning_rate": 0.000392,
        "embedding_dim": 128,
        "confidence_rate": 0.4
    },
    "high_quality": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.00005,
        "embedding_dim": 256,
        "confidence_rate": 0.5
    },
    "comparison_study": [
        {"epochs": 30, "batch_size": 8, "embedding_dim": 64, "learning_rate": 0.000392},
        {"epochs": 30, "batch_size": 8, "embedding_dim": 128, "learning_rate": 0.000392},
        {"epochs": 30, "batch_size": 8, "embedding_dim": 256, "learning_rate": 0.000392}
    ]
}

# ログ出力用のロック
log_lock = Lock()


def log_message(message, level="INFO", worker_id=None):
    """スレッドセーフなログメッセージ出力"""
    with log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worker_prefix = f"[Worker-{worker_id}] " if worker_id else ""
        print(f"[{timestamp}] {level}: {worker_prefix}{message}")


def load_experiment_config(config_file):
    """実験設定ファイルを読み込み"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_message(f"設定ファイル読み込みエラー: {str(e)}", "ERROR")
        return None


def save_experiment_config(config, config_file):
    """実験設定ファイルを保存"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        log_message(f"設定ファイルを保存: {config_file}")
        return True
    except Exception as e:
        log_message(f"設定ファイル保存エラー: {str(e)}", "ERROR")
        return False


def run_command_with_logging(command, description, worker_id=None, timeout=3600):
    """コマンド実行（ログ付き）"""
    log_message(f"実行中: {description}", worker_id=worker_id)

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            log_message(f"成功: {description} ({execution_time:.2f}秒)", worker_id=worker_id)
            return True, result.stdout, result.stderr, execution_time
        else:
            log_message(f"失敗: {description}", "ERROR", worker_id)
            return False, result.stdout, result.stderr, execution_time

    except subprocess.TimeoutExpired:
        log_message(f"タイムアウト: {description}", "ERROR", worker_id)
        return False, "", "タイムアウトエラー", time.time() - start_time
    except Exception as e:
        log_message(f"例外発生: {str(e)}", "ERROR", worker_id)
        return False, "", str(e), time.time() - start_time


def run_single_experiment(experiment_config, worker_id=None):
    """
    単一実験を実行

    Args:
        experiment_config (dict): 実験設定
        worker_id (str, optional): ワーカーID

    Returns:
        dict: 実験結果
    """
    dataset_name = experiment_config["dataset"]
    config = experiment_config["config"]
    experiment_name = experiment_config.get("name", f"{dataset_name}_experiment")

    log_message(f"実験開始: {experiment_name}", worker_id=worker_id)

    experiment_start_time = time.time()
    results = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "config": config,
        "start_time": datetime.now().isoformat(),
        "steps": {},
        "total_time": 0,
        "success": False
    }

    # ステップ1: 学習とテスト
    if experiment_config.get("skip_training", False):
        log_message("学習・テストをスキップ", worker_id=worker_id)
        results["steps"]["training_testing"] = {"success": True, "execution_time": 0, "skipped": True}
    else:
        command = [
            sys.executable, "main.py", "both",
            "--dataset", dataset_name,
            "--epochs", str(config["epochs"]),
            "--batch-size", str(config["batch_size"]),
            "--learning-rate", str(config["learning_rate"]),
            "--embedding-dim", str(config["embedding_dim"]),
            "--confidence-rate", str(config.get("confidence_rate", 0.4))
        ]

        success, stdout, stderr, exec_time = run_command_with_logging(
            command, f"学習とテスト ({experiment_name})", worker_id
        )

        results["steps"]["training_testing"] = {
            "success": success,
            "execution_time": exec_time,
            "stdout": stdout[-1000:] if stdout else "",
            "stderr": stderr[-500:] if stderr else ""
        }

        if not success:
            log_message(f"❌ {experiment_name}: 学習・テストで失敗", "ERROR", worker_id)
            results["end_time"] = datetime.now().isoformat()
            results["total_time"] = time.time() - experiment_start_time
            return results

    # ステップ2: 学習データ評価
    if experiment_config.get("skip_learning_eval", False):
        log_message("学習データ評価をスキップ", worker_id=worker_id)
        results["steps"]["learning_evaluation"] = {"success": True, "execution_time": 0, "skipped": True}
    else:
        eval_type = experiment_config.get("eval_type", "both")
        command = [
            sys.executable, "main.py", "learning-eval",
            "--dataset", dataset_name,
            "--eval-type", eval_type
        ]

        success, stdout, stderr, exec_time = run_command_with_logging(
            command, f"学習データ評価 ({experiment_name})", worker_id
        )

        results["steps"]["learning_evaluation"] = {
            "success": success,
            "execution_time": exec_time,
            "stdout": stdout[-1000:] if stdout else "",
            "stderr": stderr[-500:] if stderr else ""
        }

        if not success:
            log_message(f"❌ {experiment_name}: 学習データ評価で失敗", "ERROR", worker_id)
        else:
            results["success"] = True

    results["end_time"] = datetime.now().isoformat()
    results["total_time"] = time.time() - experiment_start_time

    status = "✅ 成功" if results["success"] else "❌ 失敗"
    log_message(f"{experiment_name} 完了: {status} (総時間: {results['total_time']:.2f}秒)", worker_id=worker_id)

    return results


def run_parallel_experiments(experiment_configs, max_workers=2):
    """並列実験実行"""
    all_results = []

    log_message(f"並列実験開始 (最大{max_workers}ワーカー)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 全実験をワーカーに投入
        future_to_config = {
            executor.submit(run_single_experiment, config, f"W{i+1}"): config
            for i, config in enumerate(experiment_configs)
        }

        # 結果収集
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                log_message(f"並列実験でエラー: {str(e)}", "ERROR")
                # エラー時の結果を作成
                error_result = {
                    "experiment_name": config.get("name", "unknown"),
                    "dataset": config["dataset"],
                    "success": False,
                    "error": str(e),
                    "total_time": 0
                }
                all_results.append(error_result)

    return all_results


def generate_experiment_configs(datasets, template_name_or_configs):
    """実験設定リストを生成"""
    configs = []

    # テンプレート名の場合
    if isinstance(template_name_or_configs, str):
        if template_name_or_configs in EXPERIMENT_TEMPLATES:
            template = EXPERIMENT_TEMPLATES[template_name_or_configs]

            if isinstance(template, list):
                # 複数設定のテンプレート（comparison_study等）
                for dataset in datasets:
                    for i, config in enumerate(template):
                        experiment_config = {
                            "name": f"{dataset}_{template_name_or_configs}_{i+1}",
                            "dataset": dataset,
                            "config": config
                        }
                        configs.append(experiment_config)
            else:
                # 単一設定のテンプレート
                for dataset in datasets:
                    experiment_config = {
                        "name": f"{dataset}_{template_name_or_configs}",
                        "dataset": dataset,
                        "config": template
                    }
                    configs.append(experiment_config)
        else:
            log_message(f"不明なテンプレート: {template_name_or_configs}", "ERROR")
            return []

    # 直接設定が渡された場合
    elif isinstance(template_name_or_configs, dict):
        for dataset in datasets:
            experiment_config = {
                "name": f"{dataset}_custom",
                "dataset": dataset,
                "config": template_name_or_configs
            }
            configs.append(experiment_config)

    return configs


def save_results(all_results, output_file):
    """結果をファイルに保存"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        log_message(f"結果を保存: {output_file}")
        return True
    except Exception as e:
        log_message(f"結果保存エラー: {str(e)}", "ERROR")
        return False


def print_results_summary(all_results):
    """結果サマリーを出力"""
    print("\n" + "="*80)
    print("高度バッチ実験結果サマリー")
    print("="*80)

    total_experiments = len(all_results["experiments"])
    successful_experiments = sum(1 for exp in all_results["experiments"] if exp.get("success", False))

    print(f"実行日時: {all_results['start_time']}")
    print(f"実行モード: {all_results['execution_mode']}")
    print(f"総実験数: {total_experiments}")
    print(f"成功: {successful_experiments}")
    print(f"失敗: {total_experiments - successful_experiments}")
    print(f"総実行時間: {all_results['total_execution_time']:.2f}秒")

    print("\n詳細結果:")
    print("-" * 80)

    for exp in all_results["experiments"]:
        status = "✅ 成功" if exp.get("success", False) else "❌ 失敗"
        total_time = exp.get("total_time", 0)
        print(f"{exp['experiment_name']:<50} {status:>8} ({total_time:.2f}秒)")

    print("\n" + "="*80)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="高度なバッチ実験スクリプト")

    parser.add_argument("--template", type=str, default="standard",
                        choices=list(EXPERIMENT_TEMPLATES.keys()),
                        help="実験テンプレート")

    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="実験対象データセット")

    parser.add_argument("--parallel", action="store_true",
                        help="並列実行を有効化")

    parser.add_argument("--max-workers", type=int, default=2,
                        help="並列実行時の最大ワーカー数")

    parser.add_argument("--config-file", type=str,
                        help="カスタム設定ファイル")

    parser.add_argument("--output", type=str, default="advanced_experiment_results.json",
                        help="結果出力ファイル")

    parser.add_argument("--dry-run", action="store_true",
                        help="実行せずに設定のみ表示")

    args = parser.parse_args()

    # 実験設定生成
    if args.config_file and os.path.exists(args.config_file):
        custom_config = load_experiment_config(args.config_file)
        if custom_config:
            experiment_configs = generate_experiment_configs(args.datasets, custom_config)
        else:
            return 1
    else:
        experiment_configs = generate_experiment_configs(args.datasets, args.template)

    if not experiment_configs:
        log_message("実験設定が生成されませんでした", "ERROR")
        return 1

    # Dry run
    if args.dry_run:
        print("\n" + "="*60)
        print("実験設定プレビュー（Dry Run）")
        print("="*60)
        for i, config in enumerate(experiment_configs, 1):
            print(f"\n実験 {i}: {config['name']}")
            print(f"  データセット: {config['dataset']}")
            print(f"  設定: {config['config']}")
        print(f"\n総実験数: {len(experiment_configs)}")
        return 0

    # 実験実行
    log_message("高度バッチ実験開始")
    batch_start_time = time.time()

    all_results = {
        "start_time": datetime.now().isoformat(),
        "template": args.template,
        "datasets": args.datasets,
        "execution_mode": "parallel" if args.parallel else "sequential",
        "experiments": [],
        "total_execution_time": 0
    }

    if args.parallel:
        results = run_parallel_experiments(experiment_configs, args.max_workers)
    else:
        results = []
        for i, config in enumerate(experiment_configs, 1):
            log_message(f"\n[{i}/{len(experiment_configs)}] 実験実行中")
            result = run_single_experiment(config)
            results.append(result)

    all_results["experiments"] = results
    all_results["end_time"] = datetime.now().isoformat()
    all_results["total_execution_time"] = time.time() - batch_start_time

    # 結果保存と表示
    save_results(all_results, args.output)
    print_results_summary(all_results)

    log_message("高度バッチ実験完了")
    return 0


if __name__ == "__main__":
    if not os.path.exists("main.py"):
        print("エラー: main.pyが見つかりません。script/cnn_triplet/executeディレクトリで実行してください。")
        sys.exit(1)

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log_message("ユーザーによって中断されました", "WARNING")
        sys.exit(1)
    except Exception as e:
        log_message(f"予期しないエラー: {str(e)}", "ERROR")
        sys.exit(1)