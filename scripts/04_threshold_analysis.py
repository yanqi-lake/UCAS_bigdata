"""Five-fold out-of-fold threshold analysis for the three course models.

Run: python scripts/04_threshold_analysis.py
Uses only the saved development folds. The holdout and test set are untouched.
Preprocessing is refitted within each training fold.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = next(
    (p for p in (ROOT / "data/raw/train_public.csv", ROOT / "data/train_public.csv") if p.is_file()),
    None,
)
SPLITS = ROOT / "data/splits/split_indices.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED = 2026
THRESHOLDS = np.round(np.arange(0.05, 0.501, 0.025), 3)

NUMERIC = [
    "total_loan", "year_of_loan", "interest", "monthly_payment",
    "debt_loan_ratio", "del_in_18month", "scoring_low", "scoring_high",
    "known_outstanding_loan", "known_dero", "pub_dero_bankrup",
    "recircle_b", "recircle_u", "f0", "f1", "f2", "f3", "f4",
    "early_return", "early_return_amount", "early_return_amount_3mon",
]
CATEGORICAL = [
    "class", "employer_type", "industry", "work_year", "house_exist",
    "censor_status", "use", "region", "initial_list_status", "app_type",
]


def make_pipeline(model):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])


def scores(actual, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    return {
        "f1": f1_score(actual, predicted, zero_division=0),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "predicted_positive": int(predicted.sum()),
    }


if DATA is None:
    raise FileNotFoundError("找不到 train_public.csv")
with DATA.open("rb") as stream:
    data_sha = hashlib.file_digest(stream, "sha256").hexdigest()
split = json.loads(SPLITS.read_text(encoding="utf-8"))
if data_sha != split["source_sha256"]:
    raise ValueError("数据 SHA256 与保存切分时不同，已停止运行")
df = pd.read_csv(DATA, encoding="utf-8-sig")
if len(df) != len(split["train_indices"]) + len(split["holdout_indices"]):
    raise ValueError("数据行数与切分文件不一致")
if not set(NUMERIC + CATEGORICAL + ["isDefault"]) <= set(df.columns):
    raise ValueError("训练数据缺少脚本要求的字段")
X = df[NUMERIC + CATEGORICAL].copy()
for col in NUMERIC:
    X[col] = pd.to_numeric(X[col], errors="coerce")
for col in CATEGORICAL:
    X[col] = X[col].astype("string").fillna("MISSING").astype(str)
y = df["isDefault"].astype(int)
train_ids = set(split["train_indices"])
holdout_ids = set(split["holdout_indices"])
folds = split["cv_validation_indices"]
assert train_ids.isdisjoint(holdout_ids)
assert set().union(*(set(fold) for fold in folds)) == train_ids
assert sum(map(len, folds)) == len(train_ids)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "decision_tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20, random_state=SEED
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        n_jobs=4, random_state=SEED,
    ),
}

prediction_rows = []
fold_rows = []
for name, estimator in models.items():
    for fold_number, valid in enumerate(folds, start=1):
        valid_set = set(valid)
        assert valid_set <= train_ids and valid_set.isdisjoint(holdout_ids)
        fit_ids = sorted(train_ids - valid_set)
        pipe = make_pipeline(estimator)
        pipe.fit(X.iloc[fit_ids], y.iloc[fit_ids])
        probability = pipe.predict_proba(X.iloc[valid])[:, 1]
        actual = y.iloc[valid].to_numpy()
        auc = roc_auc_score(actual, probability)
        for row_index, label, prob in zip(valid, actual, probability):
            prediction_rows.append({
                "model": name, "fold": fold_number, "row_index": row_index,
                "isDefault": int(label), "probability": float(prob),
            })
        for threshold in THRESHOLDS:
            fold_rows.append({
                "model": name, "fold": fold_number, "threshold": float(threshold),
                "n_validation": len(valid), "roc_auc": auc,
                **scores(actual, probability, threshold),
            })
        print(f"{name:20s} fold={fold_number} AUC={auc:.4f}", flush=True)

predictions = pd.DataFrame(prediction_rows)
assert len(predictions) == len(train_ids) * len(models)
for name, part in predictions.groupby("model"):
    assert set(part["row_index"]) == train_ids
    assert not part["row_index"].duplicated().any()

fold_scores = pd.DataFrame(fold_rows)
summary = (
    fold_scores.groupby(["model", "threshold"], as_index=False)
    .agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
    )
)
pooled_rows = []
for name, part in predictions.groupby("model", sort=False):
    actual = part["isDefault"].to_numpy()
    probability = part["probability"].to_numpy()
    for threshold in THRESHOLDS:
        pooled_rows.append({
            "model": name, "threshold": float(threshold),
            "n_development": len(part), "roc_auc": roc_auc_score(actual, probability),
            **scores(actual, probability, threshold),
        })
pooled = pd.DataFrame(pooled_rows)

# This selection reuses the development predictions and is therefore exploratory.
# The reserved holdout is needed for a final independent estimate.
best = (
    summary.sort_values(["model", "f1_mean", "threshold"],
                        ascending=[True, False, True])
    .groupby("model", as_index=False).head(1)
    .sort_values("model")
)

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
paths = {
    "predictions": RESULTS / f"threshold_oof_predictions_{stamp}.csv",
    "folds": RESULTS / f"threshold_cv_folds_{stamp}.csv",
    "summary": RESULTS / f"threshold_cv_summary_{stamp}.csv",
    "pooled": RESULTS / f"threshold_oof_pooled_{stamp}.csv",
    "best": RESULTS / f"threshold_cv_best_{stamp}.csv",
    "metadata": RESULTS / f"threshold_cv_metadata_{stamp}.json",
    "f1_figure": FIGURES / f"threshold_f1_{stamp}.png",
    "pr_figure": FIGURES / f"threshold_precision_recall_{stamp}.png",
}
for key, frame in [
    ("predictions", predictions), ("folds", fold_scores),
    ("summary", summary), ("pooled", pooled), ("best", best),
]:
    with paths[key].open("x", newline="", encoding="utf-8-sig") as stream:
        frame.to_csv(stream, index=False)

metadata = {
    "source_data": str(DATA), "source_sha256": data_sha,
    "split_file": str(SPLITS), "seed": SEED,
    "development_rows": len(train_ids), "holdout_rows_unseen": len(holdout_ids),
    "threshold_grid": THRESHOLDS.tolist(),
    "model_parameters": {name: model.get_params() for name, model in models.items()},
    "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
    "package_versions": {
        "pandas": pd.__version__, "numpy": np.__version__,
        "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
    },
    "selection_rule": "highest mean five-fold F1; ties choose lower threshold",
    "caution": "Chosen thresholds and scores use the same development folds. Do not report the selected score as an independent test estimate. Holdout and test_public were not evaluated.",
}
with paths["metadata"].open("x", encoding="utf-8") as stream:
    json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str)

fig, ax = plt.subplots(figsize=(8, 5))
for name, part in summary.groupby("model"):
    ax.plot(part["threshold"], part["f1_mean"], marker="o", label=name)
ax.set(xlabel="Decision threshold", ylabel="Mean five-fold F1",
       title="F1 versus decision threshold (development CV)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(paths["f1_figure"], dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for name, part in predictions.groupby("model"):
    precision, recall, _ = precision_recall_curve(
        part["isDefault"], part["probability"]
    )
    ax.plot(recall, precision, label=name)
ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
       title="Precision–recall curves (development OOF)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(paths["pr_figure"], dpi=180)
plt.close(fig)

print("\nBest exploratory thresholds by mean fold F1:")
print(best[["model", "threshold", "f1_mean", "precision_mean",
            "recall_mean", "roc_auc_mean"]].to_string(index=False))
print("\nSaved files:")
for path in paths.values():
    print(path)
