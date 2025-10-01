"""
Simple test for __init__.py functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Test 1: Package level import
print("Testing package level imports...")
try:
    import cnn_triplet
    print(f"Package version: {cnn_triplet.get_version()}")
    print(f"Available functions: {cnn_triplet.__all__}")

    # Test config creation
    config = cnn_triplet.create_config(epochs=10)
    print(f"Config created with epochs: {config['epochs']}")

    print("SUCCESS: Package level imports work!")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Direct function import
print("\nTesting direct function imports...")
try:
    from cnn_triplet import train_model, evaluate_model, create_config
    print("SUCCESS: Direct function imports work!")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: Submodule imports
print("\nTesting submodule imports...")
try:
    from cnn_triplet.learning import TripletGenerator
    from cnn_triplet.test import load_test_data
    print("SUCCESS: Submodule imports work!")
except Exception as e:
    print(f"ERROR: {e}")

print("\nAll tests completed!")