import sys
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_models.predictor import calibrate_and_compare_brier, optimize_decision_threshold

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

# Split data for threshold optimization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Calibrate model (use Platt/sigmoid as it gave best Brier improvement)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smoteenn", SMOTEENN(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
X_train_trans = pipeline.named_steps["scaler"].transform(
    pipeline.named_steps["smoteenn"].fit_resample(X_train, y_train)[0]
)
y_train_res = pipeline.named_steps["smoteenn"].fit_resample(X_train, y_train)[1]
calibrated = CalibratedClassifierCV(pipeline.named_steps["clf"], method="sigmoid", cv=5)
calibrated.fit(X_train_trans, y_train_res)
X_test_trans = pipeline.named_steps["scaler"].transform(X_test)
y_proba_cal = calibrated.predict_proba(X_test_trans)[:, 1]

# Optimize threshold for profit (profit_tp=1, loss_fp=1)
result_profit = optimize_decision_threshold(y_test, y_proba_cal, metric="profit", profit_tp=1.0, loss_fp=1.0, verbose=True)

# Optimize threshold for F1
result_f1 = optimize_decision_threshold(y_test, y_proba_cal, metric="f1", verbose=True)

print("\nSummary:")
print("Profit-optimal threshold:", result_profit)
print("F1-optimal threshold:", result_f1)
