"""One-time holdout evaluation of the frozen development-set decision.

Run: python scripts/09_final_holdout.py
Trains on the 8,000 development rows and evaluates the 2,000-row holdout.
Stops if a final holdout metrics file already exists; do not tune on holdout.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from PIL import Image, ImageDraw, ImageFont
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = next((p for p in (ROOT / "data/raw/train_public.csv", ROOT / "data/train_public.csv") if p.is_file()), None)
SPLITS = ROOT / "data/splits/split_indices.json"
DECISION = ROOT / "results/reduced_model_decision_20261126T185314Z.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
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

if DATA is None:
    raise FileNotFoundError("找不到 train_public.csv")
existing_metrics = sorted(RESULTS.glob("final_holdout_metrics_*.json"))
verify_existing = sys.argv[1:] == ["--verify-existing"]
if sys.argv[1:] and not verify_existing:
    raise SystemExit("仅支持可选参数 --verify-existing")
if existing_metrics and not verify_existing:
    print("最终留出集结果已存在；未重复评价。要用固定方案复核，请加 --verify-existing。")
    raise SystemExit(0)
if verify_existing and len(existing_metrics) != 1:
    raise ValueError("复核模式要求恰好一份已有的最终留出集指标文件")
decision = json.loads(DECISION.read_text(encoding="utf-8"))
split = json.loads(SPLITS.read_text(encoding="utf-8"))
with DATA.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if digest != split["source_sha256"] or digest != decision["source_data_sha256"]:
    raise ValueError("原始数据、切分和固定决策的哈希不一致")
if decision["selected_model"] != "random_forest" or decision["selected_scenario"] != "no_pca":
    raise ValueError("此脚本仅适用于已固定的不使用 PCA 的随机森林方案")
if decision["feature_policy"] != "exclude early_return, early_return_amount, early_return_amount_3mon":
    raise ValueError("固定决策与此脚本的特征策略不一致")
threshold = float(decision["selected_threshold"])
if not 0 < threshold < 1:
    raise ValueError("分类阈值不在 (0, 1) 内")
train_ids = split["train_indices"]
holdout_ids = split["holdout_indices"]
if set(train_ids).intersection(holdout_ids):
    raise ValueError("开发集与留出集重叠")
df = pd.read_csv(DATA, encoding="utf-8-sig")
if len(df) != len(train_ids) + len(holdout_ids):
    raise ValueError("数据行数与切分文件不一致")
if not set(NUMERIC + CATEGORICAL + EXCLUDED + ["isDefault", "loan_id"]) <= set(df.columns):
    raise ValueError("训练数据缺少预期字段")
X = df[NUMERIC + CATEGORICAL].copy()
for col in NUMERIC:
    X[col] = pd.to_numeric(X[col], errors="coerce")
for col in CATEGORICAL:
    X[col] = X[col].astype("string").fillna("MISSING").astype(str)
y = df["isDefault"].astype(int)

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
pipeline = Pipeline([
    ("preprocess", ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])),
    ("model", RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        n_jobs=4, random_state=2026,
    )),
])
pipeline.fit(X.iloc[train_ids], y.iloc[train_ids])
probability = pipeline.predict_proba(X.iloc[holdout_ids])[:, 1]
actual = y.iloc[holdout_ids].to_numpy()
predicted = (probability >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
metrics = {
    "source_data": str(DATA), "source_sha256": digest,
    "split_file": str(SPLITS), "decision_file": str(DECISION),
    "train_rows": len(train_ids), "holdout_rows": len(holdout_ids),
    "train_positive": int(y.iloc[train_ids].sum()),
    "holdout_positive": int(actual.sum()),
    "scenario": decision["selected_scenario"],
    "model": decision["selected_model"], "threshold": threshold,
    "excluded_features": EXCLUDED, "pca_used": False,
    "roc_auc": float(roc_auc_score(actual, probability)),
    "average_precision": float(average_precision_score(actual, probability)),
    "f1": float(f1_score(actual, predicted, zero_division=0)),
    "precision": float(precision_score(actual, predicted, zero_division=0)),
    "recall": float(recall_score(actual, predicted, zero_division=0)),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "versions": {"pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
    "interpretation_note": "One-time independent holdout evaluation of the frozen development decision. Do not tune features, model or threshold using these results.",
}
if verify_existing:
    saved = json.loads(existing_metrics[0].read_text(encoding="utf-8"))
    if saved["source_sha256"] != digest or saved["threshold"] != threshold:
        raise ValueError("已有结果的数据哈希或阈值与固定方案不一致")
    for name in ("roc_auc", "average_precision", "f1", "precision", "recall"):
        if abs(metrics[name] - saved[name]) > 1e-8:
            raise ValueError(f"复核失败：{name} 与已保存结果不同")
    if metrics["confusion_matrix"] != saved["confusion_matrix"]:
        raise ValueError("复核失败：混淆矩阵不同")
    print("复核通过：数据哈希、固定阈值、五项指标和混淆矩阵均与已保存结果一致。")
    raise SystemExit(0)

predictions = pd.DataFrame({
    "row_index": holdout_ids,
    "loan_id": df.iloc[holdout_ids]["loan_id"].to_numpy(),
    "isDefault": actual,
    "probability": probability,
    "predicted_at_frozen_threshold": predicted,
})
preprocessor = pipeline.named_steps["preprocess"]
names = preprocessor.get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_
if len(names) != len(importances):
    raise ValueError("特征名和重要性长度不一致")
importance_table = pd.DataFrame({
    "transformed_feature": names,
    "impurity_importance": importances,
}).sort_values("impurity_importance", ascending=False)

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
paths = {
    "metrics": RESULTS / f"final_holdout_metrics_{stamp}.json",
    "predictions": RESULTS / f"final_holdout_predictions_{stamp}.csv",
    "importance": RESULTS / f"final_holdout_feature_importance_{stamp}.csv",
    "roc": FIGURES / f"final_holdout_roc_{stamp}.png",
    "pr": FIGURES / f"final_holdout_precision_recall_{stamp}.png",
}
with paths["predictions"].open("x", newline="", encoding="utf-8-sig") as stream:
    predictions.to_csv(stream, index=False)
with paths["importance"].open("x", newline="", encoding="utf-8-sig") as stream:
    importance_table.to_csv(stream, index=False)


def save_curve(x_values, y_values, title, x_label, y_label, path, baseline=None):
    image = Image.new("RGB", (850, 620), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 105, 80, 780, 535
    draw.text((45, 25), title, fill="black", font=font)
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    if baseline is not None:
        y_base = bottom - (bottom - top) * baseline
        draw.line((left, round(y_base), right, round(y_base)), fill="#999999", width=2)
    points = [
        (round(left + (right - left) * float(x)),
         round(bottom - (bottom - top) * float(y)))
        for x, y in zip(x_values, y_values)
    ]
    for a, b in zip(points, points[1:]):
        draw.line((*a, *b), fill="#4C78A8", width=3)
    draw.text((left - 10, bottom + 10), "0", fill="black", font=font)
    draw.text((right - 5, bottom + 10), "1", fill="black", font=font)
    draw.text((55, bottom - 8), "0", fill="black", font=font)
    draw.text((55, top - 5), "1", fill="black", font=font)
    draw.text((390, bottom + 38), x_label, fill="black", font=font)
    draw.text((20, 60), y_label, fill="black", font=font)
    image.save(path)


fpr, tpr, _ = roc_curve(actual, probability)
save_curve(fpr, tpr, f"Final holdout ROC (AUC={metrics['roc_auc']:.3f})",
           "False positive rate", "True positive rate", paths["roc"])
precision_curve, recall_curve, _ = precision_recall_curve(actual, probability)
save_curve(recall_curve[::-1], precision_curve[::-1],
           f"Final holdout precision-recall (AP={metrics['average_precision']:.3f})",
           "Recall", "Precision", paths["pr"], baseline=float(actual.mean()))
with paths["metrics"].open("x", encoding="utf-8") as stream:
    json.dump(metrics, stream, ensure_ascii=False, indent=2)

print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("\nTop 12 transformed-feature impurity importances:")
print(importance_table.head(12).to_string(index=False))
print("\nSaved files:")
for path in paths.values():
    print(path)
