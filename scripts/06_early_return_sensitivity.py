"""Compare baseline features with a version excluding early-return fields.

Run: python scripts/06_early_return_sensitivity.py
Uses the existing five development folds only; holdout and test_public stay unused.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = next((p for p in (ROOT / "data/raw/train_public.csv", ROOT / "data/train_public.csv") if p.is_file()), None)
SPLITS = ROOT / "data/splits/split_indices.json"
OUT = ROOT / "results"
SEED = 2026
EARLY_RETURN = ["early_return", "early_return_amount", "early_return_amount_3mon"]
NUMERIC = [
    "total_loan", "year_of_loan", "interest", "monthly_payment",
    "debt_loan_ratio", "del_in_18month", "scoring_low", "scoring_high",
    "known_outstanding_loan", "known_dero", "pub_dero_bankrup",
    "recircle_b", "recircle_u", "f0", "f1", "f2", "f3", "f4",
    *EARLY_RETURN,
]
CATEGORICAL = [
    "class", "employer_type", "industry", "work_year", "house_exist",
    "censor_status", "use", "region", "initial_list_status", "app_type",
]
THRESHOLD_BY_MODEL = {
    "decision_tree": 0.300,
    "logistic_regression": 0.275,
    "random_forest": 0.275,
}


def make_pipeline(estimator, numeric_columns):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, CATEGORICAL),
        ])),
        ("model", estimator),
    ])


if DATA is None:
    raise FileNotFoundError("找不到 train_public.csv")
with DATA.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
split = json.loads(SPLITS.read_text(encoding="utf-8"))
if digest != split["source_sha256"]:
    raise ValueError("数据 SHA256 与保存切分时不同")
df = pd.read_csv(DATA, encoding="utf-8-sig")
train_ids = set(split["train_indices"])
holdout_ids = set(split["holdout_indices"])
folds = split["cv_validation_indices"]
if len(df) != len(train_ids) + len(holdout_ids):
    raise ValueError("数据行数与切分文件不一致")
if not set(NUMERIC + CATEGORICAL + ["isDefault"]) <= set(df.columns):
    raise ValueError("训练数据缺少预期字段")
assert train_ids.isdisjoint(holdout_ids)
assert set().union(*(set(fold) for fold in folds)) == train_ids
assert sum(map(len, folds)) == len(train_ids)
X = df[NUMERIC + CATEGORICAL].copy()
for col in NUMERIC:
    X[col] = pd.to_numeric(X[col], errors="coerce")
for col in CATEGORICAL:
    X[col] = X[col].astype("string").fillna("MISSING").astype(str)
y = df["isDefault"].astype(int)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=SEED),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        n_jobs=4, random_state=SEED,
    ),
}
scenarios = {
    "all_features": NUMERIC,
    "without_early_return": [col for col in NUMERIC if col not in EARLY_RETURN],
}
rows = []
probability_rows = []
for scenario, numeric_columns in scenarios.items():
    for name, estimator in models.items():
        for fold_number, valid in enumerate(folds, start=1):
            valid_set = set(valid)
            assert valid_set <= train_ids and valid_set.isdisjoint(holdout_ids)
            fit_ids = sorted(train_ids - valid_set)
            pipeline = make_pipeline(estimator, numeric_columns)
            start = time.perf_counter()
            pipeline.fit(X.iloc[fit_ids], y.iloc[fit_ids])
            train_seconds = time.perf_counter() - start
            probabilities = pipeline.predict_proba(X.iloc[valid])[:, 1]
            actual = y.iloc[valid].to_numpy()
            auc = roc_auc_score(actual, probabilities)
            for row_index, label, prob in zip(valid, actual, probabilities):
                probability_rows.append({
                    "scenario": scenario, "model": name, "fold": fold_number,
                    "row_index": row_index, "isDefault": int(label),
                    "probability": float(prob),
                })
            for threshold_name, threshold in [
                ("default_0.5", 0.5),
                ("previous_cv_choice", THRESHOLD_BY_MODEL[name]),
            ]:
                predicted = (probabilities >= threshold).astype(int)
                rows.append({
                    "scenario": scenario, "model": name, "fold": fold_number,
                    "n_train": len(fit_ids), "n_valid": len(valid),
                    "threshold_name": threshold_name, "threshold": threshold,
                    "roc_auc": auc,
                    "f1": f1_score(actual, predicted, zero_division=0),
                    "precision": precision_score(actual, predicted, zero_division=0),
                    "recall": recall_score(actual, predicted, zero_division=0),
                    "train_seconds": train_seconds,
                })
            print(f"{scenario:22s} {name:20s} fold={fold_number} AUC={auc:.4f}", flush=True)

fold_table = pd.DataFrame(rows)
predictions = pd.DataFrame(probability_rows)
assert len(predictions) == 2 * len(models) * len(train_ids)
for _, part in predictions.groupby(["scenario", "model"]):
    assert set(part["row_index"]) == train_ids
    assert not part["row_index"].duplicated().any()
summary = (
    fold_table.groupby(["scenario", "model", "threshold_name", "threshold"], as_index=False)
    .agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        train_seconds_mean=("train_seconds", "mean"),
    )
)
wide = summary.pivot(index=["model", "threshold_name", "threshold"], columns="scenario",
                     values=["roc_auc_mean", "f1_mean", "precision_mean", "recall_mean"])
comparison_rows = []
for (name, threshold_name, threshold), row in wide.iterrows():
    result = {"model": name, "threshold_name": threshold_name, "threshold": threshold}
    for metric in ("roc_auc_mean", "f1_mean", "precision_mean", "recall_mean"):
        full = row[(metric, "all_features")]
        reduced = row[(metric, "without_early_return")]
        result[f"{metric}_all"] = full
        result[f"{metric}_without"] = reduced
        result[f"{metric}_delta_without_minus_all"] = reduced - full
    comparison_rows.append(result)
comparison = pd.DataFrame(comparison_rows)

OUT.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
paths = {
    "folds": OUT / f"early_return_sensitivity_folds_{stamp}.csv",
    "summary": OUT / f"early_return_sensitivity_summary_{stamp}.csv",
    "comparison": OUT / f"early_return_sensitivity_comparison_{stamp}.csv",
    "predictions": OUT / f"early_return_sensitivity_oof_{stamp}.csv",
    "metadata": OUT / f"early_return_sensitivity_metadata_{stamp}.json",
}
for key, frame in [
    ("folds", fold_table), ("summary", summary),
    ("comparison", comparison), ("predictions", predictions),
]:
    with paths[key].open("x", newline="", encoding="utf-8-sig") as stream:
        frame.to_csv(stream, index=False)
metadata = {
    "source": str(DATA), "source_sha256": digest, "split_file": str(SPLITS),
    "seed": SEED, "development_rows": len(train_ids),
    "holdout_rows_unseen": len(holdout_ids),
    "excluded_in_reduced_scenario": EARLY_RETURN,
    "numeric_features_all": NUMERIC, "categorical_features": CATEGORICAL,
    "models": {name: model.get_params() for name, model in models.items()},
    "fixed_thresholds_from_prior_experiment": THRESHOLD_BY_MODEL,
    "versions": {"pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
    "interpretation": "Sensitivity to early-return features only; a metric drop does not prove temporal leakage. Feature availability at prediction time must be checked separately.",
    "caution": "The previous_cv_choice thresholds were selected on these same development folds. Holdout and test_public were not evaluated.",
}
with paths["metadata"].open("x", encoding="utf-8") as stream:
    json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str)
print("\nPaired five-fold mean comparison (without minus all):")
print(comparison[["model", "threshold_name", "roc_auc_mean_all",
                  "roc_auc_mean_without", "roc_auc_mean_delta_without_minus_all",
                  "f1_mean_all", "f1_mean_without",
                  "f1_mean_delta_without_minus_all"]].to_string(index=False))
print("\nSaved files:")
for path in paths.values():
    print(path)
