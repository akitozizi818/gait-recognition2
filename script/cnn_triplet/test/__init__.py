"""
Test Module

Contains evaluation and testing functionality for trained models.
"""

from .k_nn import evaluate_model, evaluate_with_different_k, load_test_data

__all__ = ['evaluate_model', 'evaluate_with_different_k', 'load_test_data']