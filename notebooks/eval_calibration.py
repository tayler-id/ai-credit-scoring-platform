import sys
import os
import numpy as np
from sklearn.datasets import make_classification

# Ensure project root is in sys.path for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_models.predictor import calibrate_and_compare_brier

# Generate synthetic imbalanced binary classification data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_clusters_per_class=2,
    weights=[0.9, 0.1],
    flip_y=0.01,
    random_state=42
)

print("Evaluating probability calibration (isotonic):")
result_iso = calibrate_and_compare_brier(X, y, random_state=42, method="isotonic", verbose=True)

print("\nEvaluating probability calibration (Platt/sigmoid):")
result_sig = calibrate_and_compare_brier(X, y, random_state=42, method="sigmoid", verbose=True)

print("\nSummary:")
print("Isotonic:", result_iso)
print("Platt/sigmoid:", result_sig)
