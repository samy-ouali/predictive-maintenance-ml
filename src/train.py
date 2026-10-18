"""Train, compare and evaluate failure-prediction models.

Run from the project root:
    python -m src.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import plots
from src.data import load_data
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, add_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
SEED = 42

# Business assumption: missing a failure (unplanned downtime, scrapped parts)
# costs about 10x more than an unnecessary preventive inspection.
COST_MISSED_FAILURE = 10
COST_FALSE_ALARM = 1


def build_pipeline(model) -> Pipeline:
    """Preprocessing + model in one object, so the same steps run in training and in production."""
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(categories=[["L", "M", "H"]]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def candidate_models() -> dict:
    # Only ~3% of cycles end in a failure, so we re-weight the rare class.
    return {
        "Logistic regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
    }


def best_threshold(y_true, proba) -> float:
    """Pick the probability threshold that minimises the total maintenance cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        pred = proba >= t
        missed = np.sum((pred == 0) & (y_true == 1))
        false_alarms = np.sum((pred == 1) & (y_true == 0))
        costs.append(missed * COST_MISSED_FAILURE + false_alarms * COST_FALSE_ALARM)
    return float(thresholds[int(np.argmin(costs))])


def main():
    df = add_features(load_data())
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].to_numpy()
    print(f"{len(df)} production cycles, failure rate = {y.mean():.2%}")

    plots.plot_failure_rate(df)
    plots.plot_failure_map(df)

    # Keep 20% of the data aside: it is only used once, at the very end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. Compare models with cross-validation on the training set.
    # Accuracy is useless here (predicting "never fails" gives 97%), so we use
    # average precision (area under the precision-recall curve).
    scores = {}
    for name, model in candidate_models().items():
        s = cross_val_score(build_pipeline(model), X_train, y_train, cv=cv,
                            scoring="average_precision")
        scores[name] = {"mean": float(s.mean()), "std": float(s.std())}
        print(f"{name:20s} PR-AUC = {s.mean():.3f} ± {s.std():.3f}")
    plots.plot_model_comparison(scores)

    best_name = max(scores, key=lambda n: scores[n]["mean"])
    print(f"Best model: {best_name}")
    pipeline = build_pipeline(candidate_models()[best_name])

    # 2. Choose the decision threshold on out-of-fold predictions (never on the test set).
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                                  method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof_proba)
    print(f"Cost-optimal threshold: {threshold:.2f}")

    # 3. Final training and a single, honest evaluation on the test set.
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    test_ap = average_precision_score(y_test, proba)
    print(classification_report(y_test, pred, target_names=["no failure", "failure"],
                                zero_division=0))

    precision, recall, _ = precision_recall_curve(y_test, proba)
    plots.plot_precision_recall(precision, recall,
                                report["1"]["precision"], report["1"]["recall"])
    plots.plot_confusion(cm)

    # 4. Explainability: which inputs does the model rely on?
    imp = permutation_importance(pipeline, X_test, y_test, scoring="average_precision",
                                 n_repeats=10, random_state=SEED, n_jobs=-1)
    plots.plot_feature_importance(list(X_test.columns), imp.importances_mean,
                                  imp.importances_std)

    # 5. Save the model and the metrics.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, MODEL_PATH)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    metrics = {
        "best_model": best_name,
        "cv_pr_auc": scores,
        "threshold": threshold,
        "test": {
            "pr_auc": float(test_ap),
            "precision_failure": report["1"]["precision"],
            "recall_failure": report["1"]["recall"],
            "f1_failure": report["1"]["f1-score"],
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "cost_model": fn * COST_MISSED_FAILURE + fp * COST_FALSE_ALARM,
            "cost_no_model": int(y_test.sum()) * COST_MISSED_FAILURE,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH.relative_to(ROOT)} and metrics to "
          f"{METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
