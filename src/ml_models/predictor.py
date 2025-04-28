import logging
import joblib # For loading trained models
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from src.models.applicant_features import ApplicantFeatures

# ML imports for pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

# TODO: Make model path configurable (e.g., via settings)
DEFAULT_MODEL_PATH = "artifacts/model.joblib"

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Loads a trained ML model artifact from the specified path using joblib."""
    logger.info(f"Loading model from: {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {type(model)}")
        # TODO: Add validation checks (e.g., check expected methods like predict_proba)
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        return None

def train_ml_pipeline(X, y, random_state=42):
    """
    Train an ML pipeline with StandardScaler, SMOTEENN, and RandomForestClassifier.

    Args:
        X: Feature matrix (pd.DataFrame or np.ndarray)
        y: Target vector
        random_state: Random seed for reproducibility

    Returns:
        pipeline: Trained imblearn Pipeline
        cv_scores: Cross-validation F1 scores for the minority class
    """
    # Define the pipeline
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smoteenn", SMOTEENN(random_state=random_state)),
        ("clf", RandomForestClassifier(random_state=random_state))
    ])

    # StratifiedKFold for imbalanced data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Cross-validate using F1 for the minority class
    from sklearn.metrics import make_scorer, f1_score
    import numpy as np

    # Identify minority class
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]

    f1_minor = make_scorer(f1_score, pos_label=minority_class, average='binary')

    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring=f1_minor, n_jobs=-1)

    # Fit the pipeline on the full data
    pipeline.fit(X, y)

    return pipeline, cv_scores

def demographic_parity(y_true, y_pred, group_labels, group_value, positive_label=1):
    """
    Compute demographic parity for a specific group.
    Returns the rate of positive predictions for the group.
    """
    import numpy as np
    mask = (group_labels == group_value)
    if np.sum(mask) == 0:
        return None
    return np.mean(y_pred[mask] == positive_label)

def equal_opportunity(y_true, y_pred, group_labels, group_value, positive_label=1):
    """
    Compute equal opportunity (true positive rate) for a specific group.
    """
    import numpy as np
    mask = (group_labels == group_value) & (y_true == positive_label)
    if np.sum(mask) == 0:
        return None
    return np.mean(y_pred[mask] == positive_label)

def compare_cv_f1_scores(X, y, group_labels=None, group_names=None, random_state=42, verbose=True):
    """
    Compare cross-validated F1 scores for the minority class between:
    1. Baseline pipeline (StandardScaler + RandomForestClassifier)
    2. SMOTEENN pipeline (StandardScaler + SMOTEENN + RandomForestClassifier)

    Args:
        X: Feature matrix (pd.DataFrame or np.ndarray)
        y: Target vector
        random_state: Random seed for reproducibility
        verbose: If True, print results

    Returns:
        results: dict with mean/std F1 for both pipelines
    """
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.metrics import make_scorer, f1_score
    import numpy as np

    # Identify minority class
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    f1_minor = make_scorer(f1_score, pos_label=minority_class, average='binary')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Baseline pipeline (no resampling)
    baseline_pipeline = SkPipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=random_state))
    ])
    # SMOTEENN pipeline
    smoteenn_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smoteenn", SMOTEENN(random_state=random_state)),
        ("clf", RandomForestClassifier(random_state=random_state))
    ])

    # Prepare to collect fairness metrics
    baseline_fairness = []
    smoteenn_fairness = []
    baseline_f1_scores = []
    smoteenn_f1_scores = []

    # If group_labels and group_names are provided, compute fairness metrics
    compute_fairness = group_labels is not None and group_names is not None

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if compute_fairness:
            groups_test = group_labels[test_idx]
        # Baseline
        baseline_pipeline.fit(X_train, y_train)
        y_pred_base = baseline_pipeline.predict(X_test)
        baseline_f1_scores.append(f1_score(y_test, y_pred_base, pos_label=minority_class, average='binary'))
        # SMOTEENN
        smoteenn_pipeline.fit(X_train, y_train)
        y_pred_smote = smoteenn_pipeline.predict(X_test)
        smoteenn_f1_scores.append(f1_score(y_test, y_pred_smote, pos_label=minority_class, average='binary'))
        # Fairness metrics
        if compute_fairness:
            base_metrics = {}
            smote_metrics = {}
            for group in group_names:
                base_metrics[group] = {
                    "demographic_parity": demographic_parity(y_test, y_pred_base, groups_test, group, positive_label=minority_class),
                    "equal_opportunity": equal_opportunity(y_test, y_pred_base, groups_test, group, positive_label=minority_class)
                }
                smote_metrics[group] = {
                    "demographic_parity": demographic_parity(y_test, y_pred_smote, groups_test, group, positive_label=minority_class),
                    "equal_opportunity": equal_opportunity(y_test, y_pred_smote, groups_test, group, positive_label=minority_class)
                }
            baseline_fairness.append(base_metrics)
            smoteenn_fairness.append(smote_metrics)

    results = {
        "baseline_mean_f1": np.mean(baseline_f1_scores),
        "baseline_std_f1": np.std(baseline_f1_scores),
        "smoteenn_mean_f1": np.mean(smoteenn_f1_scores),
        "smoteenn_std_f1": np.std(smoteenn_f1_scores),
        "baseline_scores": np.array(baseline_f1_scores),
        "smoteenn_scores": np.array(smoteenn_f1_scores),
        "minority_class": minority_class
    }

    if compute_fairness:
        # Aggregate fairness metrics
        import collections
        def aggregate_fairness(fairness_list):
            agg = {}
            for group in group_names:
                dp = [fold[group]["demographic_parity"] for fold in fairness_list if fold[group]["demographic_parity"] is not None]
                eo = [fold[group]["equal_opportunity"] for fold in fairness_list if fold[group]["equal_opportunity"] is not None]
                agg[group] = {
                    "demographic_parity_mean": np.mean(dp) if dp else None,
                    "demographic_parity_std": np.std(dp) if dp else None,
                    "equal_opportunity_mean": np.mean(eo) if eo else None,
                    "equal_opportunity_std": np.std(eo) if eo else None
                }
            return agg
        results["baseline_fairness"] = aggregate_fairness(baseline_fairness)
        results["smoteenn_fairness"] = aggregate_fairness(smoteenn_fairness)

    if verbose:
        print("Cross-validated F1 scores for minority class:")
        print(f"  Baseline (no resampling): {results['baseline_scores']}")
        print(f"    Mean: {results['baseline_mean_f1']:.4f} | Std: {results['baseline_std_f1']:.4f}")
        print(f"  SMOTEENN pipeline:        {results['smoteenn_scores']}")
        print(f"    Mean: {results['smoteenn_mean_f1']:.4f} | Std: {results['smoteenn_std_f1']:.4f}")
        print(f"  Minority class label: {results['minority_class']}")
        if compute_fairness:
            print("Fairness metrics (mean ± std across folds):")
            for pipeline, fairness in [("Baseline", results["baseline_fairness"]), ("SMOTEENN", results["smoteenn_fairness"])]:
                print(f"  {pipeline}:")
                for group in group_names:
                    dp_mean = fairness[group]["demographic_parity_mean"]
                    dp_std = fairness[group]["demographic_parity_std"]
                    eo_mean = fairness[group]["equal_opportunity_mean"]
                    eo_std = fairness[group]["equal_opportunity_std"]
                    print(f"    Group '{group}': Demographic Parity = {dp_mean:.3f} ± {dp_std:.3f} | Equal Opportunity = {eo_mean:.3f} ± {eo_std:.3f}")

    return results

def get_features_for_applicant(db: Session, applicant_id: str) -> dict:
    """
    Fetch engineered features for an applicant from the applicant_features table.

    Args:
        db: SQLAlchemy database session.
        applicant_id: The identifier of the applicant.

    Returns:
        A dictionary of features, or an empty dict if not found.
    """
    logger.info(f"Fetching features for applicant: {applicant_id}")
    try:
        record = db.query(ApplicantFeatures).filter(ApplicantFeatures.applicant_id == applicant_id).first()
        if record and record.features:
            logger.info(f"Features found for applicant {applicant_id}: {record.features}")
            return record.features
        else:
            logger.warning(f"No features found for applicant: {applicant_id}")
            return {}
    except Exception as e:
        logger.error(f"Error retrieving features for applicant {applicant_id}: {e}", exc_info=True)
        return {}

def compute_shap_values(model, features: dict):
    """
    Compute SHAP values for a given model and feature set.
    For now, returns a placeholder if the model is not a real ML model.
    When a real model is used, this will return actual SHAP values.

    Args:
        model: Trained ML model (must be compatible with SHAP).
        features: Feature dictionary for a single applicant.

    Returns:
        A dict of feature importances (SHAP values), or a placeholder.
    """
    try:
        import shap
        import numpy as np
        import pandas as pd
        # If model is a real ML model, compute SHAP values
        if hasattr(model, "predict") and hasattr(model, "feature_names_in_"):
            X = pd.DataFrame([features])
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            # Get mean absolute SHAP values for ranking
            shap_dict = dict(zip(X.columns, shap_values.values[0]))
            # Sort by absolute value, descending
            ranked = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            return {"shap_values": ranked}
        else:
            # Placeholder for rule-based or dummy model
            return {"shap_values": [("N/A", 0.0)], "explanation": "SHAP not available for rule-based model."}
    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}", exc_info=True)
        return {"shap_values": [("N/A", 0.0)], "explanation": f"SHAP computation error: {e}"}

from sklearn.metrics import roc_auc_score

def nested_cv_pipeline(X, y, random_state=42, verbose=True, n_trials=20):
    """
    Perform nested cross-validation (outer k=5, inner k=3) with Optuna Bayesian HPO in the inner loop.
    Returns list of outer fold AUCs for std(AUC_outer) calculation.

    Args:
        X: Feature matrix (pd.DataFrame or np.ndarray)
        y: Target vector
        random_state: Random seed for reproducibility
        verbose: If True, print results
        n_trials: Number of Optuna trials per outer fold

    Returns:
        outer_aucs: List of AUC scores for each outer fold
        best_params_list: List of best hyperparameters for each outer fold
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.combine import SMOTEENN
    import optuna

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state+1)

    outer_aucs = []
    best_params_list = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        def objective(trial):
            # Hyperparameter search space for RandomForest
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 2, 10)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
            # You can add more hyperparameters as needed

            pipeline = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smoteenn", SMOTEENN(random_state=random_state)),
                ("clf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                ))
            ])

            aucs = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
                y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
                pipeline.fit(X_inner_train, y_inner_train)
                y_pred_proba = pipeline.predict_proba(X_inner_val)[:, 1]
                auc = roc_auc_score(y_inner_val, y_pred_proba)
                aucs.append(auc)
            return np.mean(aucs)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        best_params = study.best_params
        best_params_list.append(best_params)

        # Train pipeline with best hyperparameters on full outer train set
        pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smoteenn", SMOTEENN(random_state=random_state)),
            ("clf", RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                random_state=random_state
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        outer_aucs.append(auc)

        if verbose:
            print(f"Outer Fold {outer_fold+1}: AUC = {auc:.4f} | Best Params: {best_params}")

    if verbose:
        print(f"Outer CV AUCs: {outer_aucs}")
        print(f"Mean AUC: {np.mean(outer_aucs):.4f} | Std AUC: {np.std(outer_aucs):.4f}")

    return outer_aucs, best_params_list

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

def calibrate_and_compare_brier(X, y, random_state=42, method="isotonic", verbose=True):
    """
    Train a pipeline (SMOTEENN + RandomForest), fit uncalibrated and calibrated (isotonic/Platt) models,
    and compare Brier scores.

    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed
        method: "isotonic" or "sigmoid" (Platt scaling)
        verbose: Print results

    Returns:
        dict with uncalibrated and calibrated Brier scores and improvement
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.combine import SMOTEENN
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Split data for calibration evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)

    # Uncalibrated pipeline
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smoteenn", SMOTEENN(random_state=random_state)),
        ("clf", RandomForestClassifier(random_state=random_state))
    ])
    pipeline.fit(X_train, y_train)
    y_pred_proba_uncal = pipeline.predict_proba(X_test)[:, 1]
    brier_uncal = brier_score_loss(y_test, y_pred_proba_uncal)

    # Calibrated pipeline
    calibrated = CalibratedClassifierCV(pipeline.named_steps["clf"], method=method, cv=5)
    # Fit on the same training data (after SMOTEENN + scaling)
    X_train_trans = pipeline.named_steps["scaler"].transform(
        pipeline.named_steps["smoteenn"].fit_resample(X_train, y_train)[0]
    )
    y_train_res = pipeline.named_steps["smoteenn"].fit_resample(X_train, y_train)[1]
    calibrated.fit(X_train_trans, y_train_res)
    X_test_trans = pipeline.named_steps["scaler"].transform(X_test)
    y_pred_proba_cal = calibrated.predict_proba(X_test_trans)[:, 1]
    brier_cal = brier_score_loss(y_test, y_pred_proba_cal)

    if verbose:
        print(f"Brier score (uncalibrated): {brier_uncal:.4f}")
        print(f"Brier score (calibrated, {method}): {brier_cal:.4f}")
        print(f"Improvement: {100 * (brier_uncal - brier_cal) / brier_uncal:.2f}%")

    return {
        "brier_uncalibrated": brier_uncal,
        "brier_calibrated": brier_cal,
        "improvement_pct": 100 * (brier_uncal - brier_cal) / brier_uncal
    }

def optimize_decision_threshold(y_true, y_proba, metric="profit", profit_tp=1.0, loss_fp=1.0, verbose=True):
    """
    Grid-search the decision threshold to maximize expected profit or F1 score.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: "profit" (default) or "f1"
        profit_tp: Profit per true positive (for profit curve)
        loss_fp: Loss per false positive (for profit curve)
        verbose: Print results

    Returns:
        dict with optimal threshold and metric value
    """
    import numpy as np
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.01, 0.99, 99)
    best_metric = -np.inf
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if metric == "profit":
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            profit = tp * profit_tp - fp * loss_fp
            if profit > best_metric:
                best_metric = profit
                best_threshold = thresh
        elif metric == "f1":
            f1 = f1_score(y_true, y_pred)
            if f1 > best_metric:
                best_metric = f1
                best_threshold = thresh

    if verbose:
        print(f"Optimal threshold ({metric}): {best_threshold:.2f} | Value: {best_metric:.4f}")

    return {
        "optimal_threshold": best_threshold,
        "metric": metric,
        "metric_value": best_metric
    }

def predict_score(db: Session, applicant_id: str):
    """
    Predict a credit score for an applicant and provide SHAP explanations.

    Args:
        db: SQLAlchemy database session.
        applicant_id: The identifier of the applicant.

    Returns:
        A tuple: (score, risk_level, shap_explanation)
    """
    logger.info(f"Starting score prediction for applicant: {applicant_id}")

    # 1. Load the trained ML model
    model = load_model() # Uses DEFAULT_MODEL_PATH unless overridden
    if model is None:
        logger.error("Failed to load scoring model. Cannot predict.")
        # Return default/error values
        return 0.0, "error", {"shap_values": [("N/A", 0.0)], "explanation": "Model not loaded."}

    # 2. Get features for the applicant
    try:
        features_dict = get_features_for_applicant(db, applicant_id)
        if not features_dict:
            logger.warning(f"No features found for applicant: {applicant_id}. Cannot predict.")
            return 0.0, "undetermined", {"shap_values": [("N/A", 0.0)], "explanation": "No features found."}
    except Exception as e:
        logger.error(f"Error retrieving features for applicant {applicant_id}: {e}", exc_info=True)
        return 0.0, "error", {"shap_values": [("N/A", 0.0)], "explanation": f"Feature retrieval error: {e}"}

    # 3. Prepare features for the model
    # Assuming the model expects a pandas DataFrame with specific columns
    # The order of columns must match the order used during training
    try:
        # TODO: Get the expected feature order from the loaded model or metadata
        # Example: feature_names = model.feature_names_in_
        # Ensure the columns in features_df match exactly what the model expects.
        # This might involve selecting specific keys from features_dict based on model metadata.
        # For now, assume features_dict keys are the features, but filter out unexpected ones if possible.
        try:
            expected_features = model.feature_names_in_ # Attempt to get expected features
        except AttributeError:
            logger.warning("Model does not have 'feature_names_in_'. Using all available features.")
            expected_features = list(features_dict.keys())

        # Include only expected features, handle missing columns if needed
        features_for_model = {k: features_dict.get(k) for k in expected_features if k in features_dict}
        # TODO: Add robust handling for missing expected features (e.g., imputation based on training)
        features_df = pd.DataFrame([features_for_model], columns=expected_features)
        # Example: Fill missing expected features with 0 (use a better strategy if possible)
        features_df = features_df.fillna(0)

        # Log if supply chain features are present but potentially unused by the current model
        sc_keys = [k for k in features_dict if k.startswith("sc_")]
        if sc_keys and not any(k in expected_features for k in sc_keys):
            logger.warning(f"Supply chain features present ({sc_keys}) but may not be used by the loaded model.")

    except Exception as e:
        logger.error(f"Error preparing features for model prediction: {e}", exc_info=True)
        return 0.0, "error", {"shap_values": [("N/A", 0.0)], "explanation": f"Feature preparation error: {e}"}


    # 4. Predict probability using the loaded ML model
    logger.info(f"Predicting using loaded ML model ({type(model)}) with features: {features_dict}")
    try:
        # Assuming the model's pipeline handles scaling/preprocessing
        # predict_proba usually returns [[prob_class_0, prob_class_1]]
        probability_class_1 = model.predict_proba(features_df)[0, 1]
        score = float(probability_class_1) # Ensure score is a standard float

        # Determine risk level based on probability score
        # TODO: Use the optimized threshold from T-5 if available
        threshold_medium = 0.5 # Example threshold
        threshold_low = 0.8    # Example threshold

        if score >= threshold_low:
            risk_level = "low"
        elif score >= threshold_medium:
            risk_level = "medium"
        else:
            risk_level = "high"

    except AttributeError:
         logger.error(f"Loaded model object does not have 'predict_proba' method. Using default score.")
         score = 0.0
         risk_level = "error"
    except Exception as e:
        logger.error(f"Error during ML model prediction for applicant {applicant_id}: {e}", exc_info=True)
        score = 0.0
        risk_level = "error"
        # Ensure shap_explanation is still generated even if prediction fails
        shap_explanation = {"shap_values": [("N/A", 0.0)], "explanation": f"Prediction error: {e}"}
        logger.info(f"Prediction failed for applicant {applicant_id}. Score: {score}, Risk: {risk_level}, SHAP: {shap_explanation}")
        return score, risk_level, shap_explanation


    # 5. Compute SHAP values using the loaded model and prepared features
    # Pass the dictionary to compute_shap_values
    shap_explanation = compute_shap_values(model, features_for_model) # Pass dictionary here

    logger.info(f"Prediction complete for applicant {applicant_id}. Score: {score:.4f}, Risk: {risk_level}, SHAP: {shap_explanation}")
    return score, risk_level, shap_explanation
