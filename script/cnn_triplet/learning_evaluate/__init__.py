"""
Learning Evaluate Module

Contains evaluation functionality for training data validation.
"""

from .k_nn import evaluate_training_data_knn
from .k_fold_cross_validation import k_fold_cross_validation

__all__ = ['evaluate_training_data_knn', 'k_fold_cross_validation']