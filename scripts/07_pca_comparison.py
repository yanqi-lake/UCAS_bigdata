"""Five-fold PCA ablation on the development set, excluding early-return fields.

Run: python scripts/07_pca_comparison.py
PCA, scaling and imputation are fitted only on each training fold.
PCA is applied to numeric features; categorical one-hot features are unchanged.
The reserved holdout and test_public are not evaluated.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from PIL import Image, ImageDraw, ImageFont
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = next((p for p in (ROOT / "data/raw/train_public.csv", ROOT / "data/train_public.csv") if p.is_file()), None)
SPLITS = ROOT / "data/splits/split_indices.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED = 2026
NUMERIC = [
    "total_loan", "year_of_loan", "interest", "monthly_payment",
    "debt_loan_ratio", "del_in_18month", "scoring_low", "scoring_high",
    "known_outstanding_loan", "known_dero", "pub_dero_bankrup",
    "recircle_b", "recircle_u", "f0", "f1", "f2", "f3", "f4",
]
CATEGORICAL = [
    "class", "employer_type", "industry", "work_year", "house_exist",
    "censor_status", "use", "region", "initial_list_status", "app_type",
]
EXCLUDED = ["early_return", "early_return_amount", "early_return_amount_3mon"]


def make_pipeline(estimator, use_pca):
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]
    if use_pca:
        numeric_steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), NUMERIC),
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
rows = []
prediction_rows = []
variance_rows = []
for scenario, use_pca in [("no_pca", False), ("numeric_pca_95", True)]:
    for name, estimator in models.items():
        for fold_number, valid in enumerate(folds, start=1):
            valid_set = set(valid)
            assert valid_set <= train_ids and valid_set.isdisjoint(holdout_ids)
            fit_ids = sorted(train_ids - valid_set)
            pipeline = make_pipeline(estimator, use_pca)
            pipeline.fit(X.iloc[fit_ids], y.iloc[fit_ids])
            probability = pipeline.predict_proba(X.iloc[valid])[:, 1]
            actual = y.iloc[valid].to_numpy()
            numeric_pipe = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
            n_before = numeric_pipe.named_steps["impute"].transform(X.iloc[fit_ids][NUMERIC][:1]).shape[1]
            if use_pca:
                pca = numeric_pipe.named_steps["pca"]
                n_after = int(pca.n_components_)
                variance_explained = float(pca.explained_variance_ratio_.sum())
                if name == "logistic_regression":
                    for component, (ratio, cumulative) in enumerate(
                        zip(pca.explained_variance_ratio_,
                            np.cumsum(pca.explained_variance_ratio_)), start=1
                    ):
                        variance_rows.append({
                            "fold": fold_number, "component": component,
                            "explained_variance_ratio": float(ratio),
                            "cumulative_explained_variance": float(cumulative),
                        })
            else:
                n_after = n_before
                variance_explained = 1.0
            predicted = (probability >= 0.5).astype(int)
            rows.append({
                "scenario": scenario, "model": name, "fold": fold_number,
                "n_train": len(fit_ids), "n_valid": len(valid),
                "numeric_dim_before": n_before, "numeric_dim_after": n_after,
                "numeric_variance_explained": variance_explained,
                "roc_auc": roc_auc_score(actual, probability),
                "average_precision": average_precision_score(actual, probability),
                "f1_at_0.5": f1_score(actual, predicted, zero_division=0),
                "precision_at_0.5": precision_score(actual, predicted, zero_division=0),
                "recall_at_0.5": recall_score(actual, predicted, zero_division=0),
            })
            for row_index, label, prob in zip(valid, actual, probability):
                prediction_rows.append({
                    "scenario": scenario, "model": name, "fold": fold_number,
                    "row_index": row_index, "isDefault": int(label),
                    "probability": float(prob),
                })
            print(f"{scenario:15s} {name:20s} fold={fold_number} "
                  f"AUC={rows[-1]['roc_auc']:.4f} numeric={n_before}->{n_after}", flush=True)

fold_table = pd.DataFrame(rows)
predictions = pd.DataFrame(prediction_rows)
variance_table = pd.DataFrame(variance_rows)
assert len(predictions) == len(train_ids) * len(models) * 2
for _, part in predictions.groupby(["scenario", "model"]):
    assert set(part["row_index"]) == train_ids
    assert not part["row_index"].duplicated().any()

summary = (
    fold_table.groupby(["scenario", "model"], as_index=False)
    .agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        f1_at_05_mean=("f1_at_0.5", "mean"),
        numeric_dim_before_mean=("numeric_dim_before", "mean"),
        numeric_dim_after_mean=("numeric_dim_after", "mean"),
        numeric_variance_explained_mean=("numeric_variance_explained", "mean"),
    )
)
comparison = summary.pivot(index="model", columns="scenario",
                           values=["roc_auc_mean", "average_precision_mean",
                                   "f1_at_05_mean"]).copy()
compare_rows = []
for name, record in comparison.iterrows():
    result = {"model": name}
    for metric in ("roc_auc_mean", "average_precision_mean", "f1_at_05_mean"):
        before = record[(metric, "no_pca")]
        after = record[(metric, "numeric_pca_95")]
        result[f"{metric}_no_pca"] = before
        result[f"{metric}_pca"] = after
        result[f"{metric}_delta_pca_minus_no_pca"] = after - before
    compare_rows.append(result)
compare_table = pd.DataFrame(compare_rows)

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
paths = {
    "folds": RESULTS / f"pca_cv_folds_{stamp}.csv",
    "summary": RESULTS / f"pca_cv_summary_{stamp}.csv",
    "comparison": RESULTS / f"pca_cv_comparison_{stamp}.csv",
    "predictions": RESULTS / f"pca_cv_oof_{stamp}.csv",
    "variance": RESULTS / f"pca_explained_variance_{stamp}.csv",
    "metadata": RESULTS / f"pca_cv_metadata_{stamp}.json",
    "figure": FIGURES / f"pca_auc_comparison_{stamp}.png",
    "variance_figure": FIGURES / f"pca_cumulative_variance_{stamp}.png",
}
for key, frame in [
    ("folds", fold_table), ("summary", summary),
    ("comparison", compare_table), ("predictions", predictions),
    ("variance", variance_table),
]:
    with paths[key].open("x", newline="", encoding="utf-8-sig") as stream:
        frame.to_csv(stream, index=False)
metadata = {
    "source": str(DATA), "source_sha256": digest,
    "split_file": str(SPLITS), "seed": SEED,
    "development_rows": len(train_ids), "holdout_rows_unseen": len(holdout_ids),
    "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
    "excluded_features": EXCLUDED,
    "pca": "Numeric imputation, standardization, then PCA retaining >=95% training-fold variance; categorical one-hot features bypass PCA.",
    "selection": "PCA is fitted only on training folds; no holdout tuning.",
    "models": {name: model.get_params() for name, model in models.items()},
    "versions": {"pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
    "metric_note": "ROC-AUC and average precision are threshold-free. F1 is shown only at the fixed 0.5 threshold; this threshold was not tuned for the reduced feature set.",
}
with paths["metadata"].open("x", encoding="utf-8") as stream:
    json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str)

image = Image.new("RGB", (1050, 520), "white")
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
draw.text((40, 25), "Five-fold ROC-AUC: numeric PCA (95% variance) versus no PCA", fill="black", font=font)
colors = {"no_pca": "#4C78A8", "numeric_pca_95": "#F58518"}
for i, name in enumerate(models):
    y0 = 90 + i * 130
    draw.text((25, y0 + 35), name, fill="black", font=font)
    for j, scenario in enumerate(("no_pca", "numeric_pca_95")):
        value = float(summary.loc[(summary["model"] == name) &
                                  (summary["scenario"] == scenario), "roc_auc_mean"].iloc[0])
        bar_y = y0 + j * 42
        draw.rectangle((220, bar_y, 220 + round(700 * value), bar_y + 27), fill=colors[scenario])
        draw.text((230 + round(700 * value), bar_y + 6), f"{value:.3f}", fill="black", font=font)
draw.rectangle((30, 480, 48, 494), fill=colors["no_pca"])
draw.text((54, 481), "No PCA", fill="black", font=font)
draw.rectangle((170, 480, 188, 494), fill=colors["numeric_pca_95"])
draw.text((194, 481), "Numeric PCA", fill="black", font=font)
image.save(paths["figure"])

image = Image.new("RGB", (960, 560), "white")
draw = ImageDraw.Draw(image)
draw.text((50, 25), "Cumulative explained variance of numeric PCA (mean over five folds)",
          fill="black", font=font)
left, top, right, bottom = 100, 70, 900, 480
draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
means = variance_table.groupby("component")["cumulative_explained_variance"].mean()
n_components = int(means.index.max())
points = []
for component, cumulative in means.items():
    x = left + (right - left) * int(component) / n_components
    y_axis = bottom - (bottom - top) * float(cumulative)
    points.append((round(x), round(y_axis)))
for point_a, point_b in zip(points, points[1:]):
    draw.line((*point_a, *point_b), fill="#4C78A8", width=4)
for x, y_axis in points:
    draw.ellipse((x - 4, y_axis - 4, x + 4, y_axis + 4), fill="#4C78A8")
y95 = bottom - (bottom - top) * .95
draw.line((left, round(y95), right, round(y95)), fill="#E45756", width=2)
draw.text((15, round(y95) - 8), "95%", fill="#E45756", font=font)
draw.text((left - 10, bottom + 12), "0", fill="black", font=font)
draw.text((right - 10, bottom + 12), str(n_components), fill="black", font=font)
draw.text((350, bottom + 45), "Number of retained numeric principal components",
          fill="black", font=font)
image.save(paths["variance_figure"])

print("\nComparison (PCA minus no PCA):")
print(compare_table.to_string(index=False))
print("\nPCA numeric dimensions and variance:")
print(summary.loc[summary["scenario"] == "numeric_pca_95",
                  ["model", "numeric_dim_before_mean", "numeric_dim_after_mean",
                   "numeric_variance_explained_mean"]].to_string(index=False))
print("\nSaved files:")
for path in paths.values():
    print(path)
