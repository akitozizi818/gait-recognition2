import os
import sys
import argparse
import json

# Add parent directories to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'learning'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'test'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluate'))

from separate_viewpoint import train_model
from k_nn import evaluate_model, evaluate_with_different_k

# Import learning evaluation functions with specific module names to avoid conflicts
import sys
import importlib.util

# Function to load module from specific path
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load evaluation modules
eval_dir = os.path.join(os.path.dirname(__file__), '..', 'evaluate')
knn_path = os.path.join(eval_dir, 'k_nn.py')
kfold_path = os.path.join(eval_dir, 'k_fold_cross_validation.py')
umap_path = os.path.join(eval_dir, 'umap.py')

knn_module = load_module_from_path('eval_knn', knn_path)
kfold_module = load_module_from_path('eval_kfold', kfold_path)
umap_module = load_module_from_path('eval_umap', umap_path)

# Extract functions
evaluate_training_data_knn = knn_module.evaluate_training_data_knn
k_fold_cross_validation = kfold_module.k_fold_cross_validation
visualize_embeddings_umap = umap_module.visualize_embeddings_umap


def create_default_config(dataset_name="sorted_dataset_scale_exclude_45", confidence_rate=0.4):
    """
    Create a default configuration dictionary.

    Args:
        dataset_name (str): Name of the dataset directory
        confidence_rate (float): Confidence threshold rate

    Returns:
        dict: Configuration dictionary
    """
    config = {
        "dataset_root": f"../../../data/{dataset_name}/dataset",
        "test_data_dir": f"../../../data/{dataset_name}/test_dataset",
        "confidence_map_dir": "../../../data/cut_confidence_image",
        "output_dir": f"../../../data/triplet_loss_output/{dataset_name}",
        "img_size": (48, 48),
        "batch_size": 8,
        "num_classes": 10,
        "epochs": 30,
        "embedding_dim": 128,
        "triplet_alpha": 0.2,
        "learning_rate": 0.000392,
        "model_name": "embedding_model.h5",
        "config_file_name": "config.json",
        "confidence_rate": confidence_rate
    }
    return config


def load_config_from_file(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config_to_file(config, config_path):
    """
    Save configuration to a JSON file.

    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save the configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_path}")


def run_training(config):
    """
    Run the training process.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (embedding_model, training_history, output_dir)
    """
    print("=" * 50)
    print("STARTING TRAINING PROCESS")
    print("=" * 50)

    # Display configuration
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Run training
    return train_model(config)


def run_evaluation(config):
    """
    Run the evaluation process.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Evaluation results
    """
    print("=" * 50)
    print("STARTING EVALUATION PROCESS")
    print("=" * 50)

    # Check if model exists
    model_path = os.path.join(config["output_dir"], config["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please run training first.")

    # Run evaluation
    results = evaluate_model(config)

    print("\n" + "=" * 30)
    print("TESTING DIFFERENT K VALUES")
    print("=" * 30)

    # Test different k values
    k_results = evaluate_with_different_k(config, k_values=[1, 3, 5, 7, 10])

    return results, k_results


def run_learning_evaluation(config, eval_type="both"):
    """
    Run the learning data evaluation process.

    Args:
        config (dict): Configuration dictionary
        eval_type (str): Type of evaluation ("knn", "kfold", or "both")

    Returns:
        dict: Learning evaluation results
    """
    print("=" * 60)
    print("STARTING LEARNING DATA EVALUATION PROCESS")
    print("=" * 60)

    # Check if model exists
    model_path = os.path.join(config["output_dir"], config["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please run training first.")

    results = {}

    if eval_type in ["knn", "both"]:
        print("\n" + "=" * 40)
        print("RUNNING k-NN EVALUATION")
        print("=" * 40)

        knn_results = evaluate_training_data_knn(
            config,
            k_values=[1, 3, 5, 7, 10],
            test_size=0.3,
            random_state=42
        )
        results["knn_evaluation"] = knn_results

    if eval_type in ["kfold", "both"]:
        print("\n" + "=" * 40)
        print("RUNNING k-FOLD CROSS VALIDATION")
        print("=" * 40)

        kfold_results = k_fold_cross_validation(
            config,
            k_folds=5,
            k_nn_values=[3, 5, 7, 10],
            random_state=42
        )
        results["kfold_evaluation"] = kfold_results

    return results


def run_umap_visualization(config, args):
    """
    Run UMAP visualization process.

    Args:
        config (dict): Configuration dictionary
        args: Command line arguments
    """
    print("=" * 60)
    print("STARTING UMAP VISUALIZATION PROCESS")
    print("=" * 60)

    # Check if model exists
    model_path = os.path.join(config["output_dir"], config["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please run training first.")

    # UMAP parameters from command line arguments
    umap_params = {
        "n_neighbors": args.umap_neighbors,
        "min_dist": args.umap_min_dist,
        "metric": args.umap_metric,
        "n_components": 2,
        "random_state": 42
    }

    # Run visualization for specified datasets
    if args.umap_dataset in ["train", "both"]:
        print("\n" + "=" * 40)
        print("VISUALIZING TRAINING DATASET")
        print("=" * 40)
        train_results = visualize_embeddings_umap(config, "train", umap_params)

    if args.umap_dataset in ["test", "both"]:
        print("\n" + "=" * 40)
        print("VISUALIZING TEST DATASET")
        print("=" * 40)
        test_results = visualize_embeddings_umap(config, "test", umap_params)

    print("\n" + "=" * 40)
    print("UMAP VISUALIZATION COMPLETED")
    print("=" * 40)


def main():
    """
    Main function to handle command line arguments and execute training/testing.
    """
    parser = argparse.ArgumentParser(description="CNN Triplet Loss Training and Evaluation")

    # Main action
    parser.add_argument('action', choices=['train', 'test', 'both', 'learning-eval', 'umap'],
                        help='Action to perform: train, test, both, learning-eval, or umap')

    # Configuration options
    parser.add_argument('--config', type=str,
                        help='Path to configuration JSON file')
    parser.add_argument('--dataset', type=str, default='sorted_dataset_scale_exclude_45',
                        help='Dataset name (default: sorted_dataset_scale_exclude_45)')
    parser.add_argument('--confidence-rate', type=float, default=0.4,
                        help='Confidence rate threshold (default: 0.4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=0.000392,
                        help='Learning rate (default: 0.000392)')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')

    # Learning evaluation options
    parser.add_argument('--eval-type', type=str, choices=['knn', 'kfold', 'both'], default='both',
                        help='Type of learning evaluation: knn, kfold, or both (default: both)')

    # UMAP visualization options
    parser.add_argument('--umap-dataset', type=str, choices=['train', 'test', 'both'], default='both',
                        help='Dataset for UMAP visualization: train, test, or both (default: both)')
    parser.add_argument('--umap-neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                        help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--umap-metric', type=str, default='euclidean',
                        help='UMAP metric parameter (default: euclidean)')

    # Output options
    parser.add_argument('--save-config', type=str,
                        help='Path to save the configuration file')

    args = parser.parse_args()

    try:
        # Load or create configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config = load_config_from_file(args.config)
        else:
            print("Using default configuration with command line overrides...")
            config = create_default_config(args.dataset, args.confidence_rate)

            # Override with command line arguments
            config["epochs"] = args.epochs
            config["batch_size"] = args.batch_size
            config["learning_rate"] = args.learning_rate
            config["embedding_dim"] = args.embedding_dim

        # Save configuration if requested
        if args.save_config:
            save_config_to_file(config, args.save_config)

        # Execute requested action
        if args.action == 'train':
            run_training(config)

        elif args.action == 'test':
            run_evaluation(config)

        elif args.action == 'both':
            # Run training first
            run_training(config)

            print("\n" + "=" * 50)
            print("TRAINING COMPLETED - STARTING EVALUATION")
            print("=" * 50)

            # Then run evaluation
            run_evaluation(config)

        elif args.action == 'learning-eval':
            # Run learning data evaluation
            run_learning_evaluation(config, eval_type=args.eval_type)

        elif args.action == 'umap':
            # Run UMAP visualization
            run_umap_visualization(config, args)

        print("\n" + "=" * 50)
        print("PROCESS COMPLETED SUCCESSFULLY")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()