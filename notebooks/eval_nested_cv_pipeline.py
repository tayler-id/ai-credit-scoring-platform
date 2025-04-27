
import sys
import os
import numpy as np
from sklearn.datasets import make_classification

# Ensure project root is in sys.path for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_models.predictor import nested_cv_pipeline

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

# Run nested CV pipeline with Optuna Bayesian HPO
outer_aucs, best_params_list = nested_cv_pipeline(X, y, random_state=42, verbose=True, n_trials=10)

print("\nNested CV with Optuna Results:")
print("Outer fold AUCs:", outer_aucs)
print("Mean AUC: {:.4f}".format(np.mean(outer_aucs)))
print("Std AUC: {:.4f}".format(np.std(outer_aucs)))
print("Best hyperparameters per fold:", best_params_list)
