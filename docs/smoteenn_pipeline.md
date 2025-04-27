# Handling Class Imbalance in Credit Scoring: SMOTEENN Pipeline Integration

## Overview

In emerging markets, credit scoring datasets are often highly imbalanced: the number of "good" (majority) borrowers far exceeds the "bad" (minority) cases. This imbalance can cause ML models to perform poorly on the minority class, which is often the most critical for risk assessment.

To address this, the AI Credit Scoring Platform integrates **SMOTEENN** (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbours) into its ML pipeline. This approach combines over-sampling of the minority class with cleaning of noisy samples, resulting in improved recall and F1 for the minority class without sacrificing overall AUC.

---

## Why SMOTEENN?

- **SMOTE** generates synthetic samples for the minority class, reducing bias toward the majority.
- **ENN** removes ambiguous or noisy samples after over-sampling, cleaning the decision boundary.
- **SMOTEENN** combines both, providing a robust solution for imbalanced credit datasets.

**References:**
- [imblearn SMOTEENN docs](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html)
- [imblearn combine guide](https://imbalanced-learn.org/stable/combine.html)

---

## Pipeline Structure

The ML pipeline is constructed using **imblearn's Pipeline**, which allows samplers like SMOTEENN to be used as steps. The recommended structure is:

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smoteenn", SMOTEENN(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42))
])
```

- **Preprocessing** (StandardScaler) occurs before SMOTEENN to ensure features are on a comparable scale.
- **SMOTEENN** is applied only during training (`fit`), not during prediction.
- **Classifier** (RandomForest by default) is trained on the balanced, cleaned data.

---

## Cross-Validation and Evaluation

To ensure robust evaluation:
- **StratifiedKFold** is used for cross-validation, preserving class ratios in each fold.
- **F1 score for the minority class** is the primary metric, as improving recall for the minority is the goal.
- The pipeline is cross-validated and then fit on the full dataset.

Example (from `src/ml_models/predictor.py`):

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
import numpy as np

# Identify minority class
unique, counts = np.unique(y, return_counts=True)
minority_class = unique[np.argmin(counts)]
f1_minor = make_scorer(f1_score, pos_label=minority_class, average='binary')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring=f1_minor, n_jobs=-1)
```

---

## Comparing Baseline vs. SMOTEENN: Minority-Class F1 Cross-Validation

To rigorously demonstrate the benefit of SMOTEENN, the platform provides a function to compare cross-validated F1 scores for the minority class between:

1. **Baseline pipeline:** StandardScaler + RandomForestClassifier (no resampling)
2. **SMOTEENN pipeline:** StandardScaler + SMOTEENN + RandomForestClassifier

This comparison uses StratifiedKFold and the F1 score for the minority class as the primary metric.

### Example Usage

```python
from src.ml_models.predictor import compare_cv_f1_scores

# X: feature matrix, y: target vector
results = compare_cv_f1_scores(X, y, random_state=42, verbose=True)
```

**Example Output:**
```
Cross-validated F1 scores for minority class:
  Baseline (no resampling): [0.32 0.28 0.30 0.27 0.29]
    Mean: 0.2920 | Std: 0.0180
  SMOTEENN pipeline:        [0.45 0.41 0.43 0.44 0.42]
    Mean: 0.4300 | Std: 0.0141
  Minority class label: 1
```

### Interpretation

- **Mean F1 (SMOTEENN) > Mean F1 (Baseline):** Indicates improved recall and precision for the minority class, validating the use of SMOTEENN.
- **Std:** Shows variability across folds; lower is better for stability.
- **Minority class label:** The class considered "positive" for F1 scoring.

### Reporting

- Include both mean and std F1 scores for baseline and SMOTEENN in documentation and Memory Bank.
- Use these results to justify model selection and pipeline design.

---

## Fairness Testing & Bias Mitigation

### Why Fairness Testing?

Credit scoring models must not only be accurate, but also fair. In emerging markets, it is critical to ensure that the model does not introduce or perpetuate bias against protected groups (e.g., gender, ethnicity). Regulatory and ethical standards require that credit decisions are explainable and equitable.

### Fairness Metrics

The platform now supports two key fairness metrics, computed during cross-validation for each protected group:

- **Demographic Parity:** The rate of positive predictions (e.g., loan approvals) should be similar across groups. Formally, for group *g*, it is the proportion of predicted positives among all group members.
- **Equal Opportunity:** The true positive rate (recall) should be similar across groups. For group *g*, it is the proportion of correctly predicted positives among all actual positives in the group.

Both metrics are reported as mean ± std across cross-validation folds.

### How It Works

The function `compare_cv_f1_scores` in `src/ml_models/predictor.py` now accepts group membership labels and group names:

```python
results = compare_cv_f1_scores(
    X, y,
    group_labels=group_labels,  # e.g., np.array(["F", "M", ...])
    group_names=["F", "M"],     # list of unique group values
    random_state=42,
    verbose=True
)
```

**Example Output:**
```
Cross-validated F1 scores for minority class:
  Baseline (no resampling): [0.32 0.28 0.30 0.27 0.29]
    Mean: 0.2920 | Std: 0.0180
  SMOTEENN pipeline:        [0.45 0.41 0.43 0.44 0.42]
    Mean: 0.4300 | Std: 0.0141
  Minority class label: 1
Fairness metrics (mean ± std across folds):
  Baseline:
    Group 'F': Demographic Parity = 0.52 ± 0.03 | Equal Opportunity = 0.60 ± 0.04
    Group 'M': Demographic Parity = 0.48 ± 0.02 | Equal Opportunity = 0.58 ± 0.03
  SMOTEENN:
    Group 'F': Demographic Parity = 0.55 ± 0.02 | Equal Opportunity = 0.65 ± 0.03
    Group 'M': Demographic Parity = 0.50 ± 0.02 | Equal Opportunity = 0.62 ± 0.03
```

### Interpretation

- **Demographic Parity:** Large differences between groups may indicate bias in approval rates.
- **Equal Opportunity:** Large differences indicate some groups are less likely to be correctly approved.
- **Thresholds:** Regulatory or project-specific thresholds (e.g., max allowed difference = 0.1) should be defined.

### Bias Mitigation

If fairness thresholds are not met, bias mitigation techniques should be applied:
- **Reweighting:** Adjust sample weights during training to balance group outcomes.
- **Post-processing:** Adjust predictions to equalize group rates.
- **Fairness-aware algorithms:** Use models that directly optimize for fairness.

After mitigation, re-run cross-validation and fairness evaluation.

### Documentation & Governance

- All fairness testing and mitigation steps must be documented in the Memory Bank and static documentation.
- Results should be used to inform model selection and pipeline design.

---

## SHAP Explainability: Feature-Level Explanations for Every Credit Decision

### Why Explainability?

Regulatory and business requirements demand that every credit decision be explainable, with ranked feature reasons provided for "adverse-action" notices and transparency. Lenders must be able to justify decisions, and borrowers have a right to know which factors most influenced their score.

### How SHAP Is Integrated

- The platform integrates [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/) to provide feature-level explanations for every prediction.
- SHAP is computed (or stubbed, for rule-based models) for every applicant at prediction time.
- The scoring API (`/api/v1/score`) now returns a `shap_explanation` field in every response, listing features ranked by their impact on the score.
- When a true ML model is integrated, SHAP will provide real, ranked feature importances for each prediction. The API and documentation are already designed to support this transition.

### Example API Response

```json
{
  "applicant_id": "abc123",
  "score": 0.72,
  "risk_level": "medium",
  "status": "completed",
  "message": "Score calculated successfully.",
  "shap_explanation": {
    "shap_values": [
      ["on_time_payment_rate", 0.18],
      ["mm_num_transactions", 0.12],
      ["ec_total_spend", 0.07]
    ],
    "explanation": "SHAP not available for rule-based model."
  }
}
```

- `shap_values` is a list of (feature, value) pairs, ranked by absolute impact.
- For rule-based models, a placeholder is returned; for ML models, real SHAP values will be provided.

### Regulatory and Business Benefits

- **Adverse-Action Explanations:** The top features and their SHAP values can be used to generate compliant "adverse-action" notices, as required by regulators.
- **Transparency:** Lenders and borrowers can see which factors most influenced a decision, supporting trust and auditability.
- **Future-Proof:** The API and documentation are designed to support explainability from day 0, ensuring a smooth transition to ML-based models.

### Documentation & Governance

- All explainability logic and rationale are documented in the Memory Bank and static documentation.
- SHAP explanations are available in API responses and logs, supporting audit, compliance, and future UI integration.

---

## Integration with the Platform

- The pipeline is implemented in `src/ml_models/predictor.py` as `train_ml_pipeline`.
- Feature engineering is performed upstream; the pipeline expects a clean feature matrix and target vector.
- The rule-based model remains for baseline comparison.
- All work is tracked via the Memory Bank protocol and Task Manager MCP.

---

## Documentation & Governance

- This document is part of the platform's static documentation site.
- All major ML workflow changes are documented in the Memory Bank (`memory-bank/activeContext.md`, `memory-bank/progress.md`).
- Task breakdown and approvals are managed via the Task Manager MCP.

---

## Further Reading

- [imblearn Pipeline docs](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
- [StackOverflow: Using SMOTEENN in GridSearchCV Pipeline](https://stackoverflow.com/questions/59516827/using-smoteenn-in-gridsearchcv-pipeline-with-preprocesing)
- [Medium: Strategies for Handling Class Imbalance](https://medium.com/@akash.hiremath25/balancing-act-strategies-for-handling-class-imbalance-in-machine-learning-eaf05dc6225c)

---

_Last updated: 2025-04-27_
