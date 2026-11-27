"""Choose a reduced-feature model and threshold using development CV only.

Run: python scripts/08_select_reduced_model_threshold.py
Reads out-of-fold probabilities from 07_pca_comparison.py. The saved holdout
is never read. Model rule: highest mean five-fold ROC-AUC. Threshold rule:
highest mean five-fold F1 on a fixed grid; ties choose lower threshold.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SPLITS = ROOT / "data/splits/split_indices.json"
THRESHOLDS = np.round(np.arange(0.05, 0.5001, 0.005), 3)

summaries = sorted(RESULTS.glob("pca_cv_summary_*.csv"))
if not summaries:
    raise FileNotFoundError("找不到 PCA 五折汇总结果")
source_summary = summaries[-1]
source_stamp = source_summary.stem.removeprefix("pca_cv_summary_")
source_oof = RESULTS / f"pca_cv_oof_{source_stamp}.csv"
source_meta = RESULTS / f"pca_cv_metadata_{source_stamp}.json"
if not source_oof.is_file() or not source_meta.is_file():
    raise FileNotFoundError("对应的 OOF 预测或元数据不存在")
summary = pd.read_csv(source_summary, encoding="utf-8-sig")
oof = pd.read_csv(source_oof, encoding="utf-8-sig")
meta = json.loads(source_meta.read_text(encoding="utf-8"))
split = json.loads(SPLITS.read_text(encoding="utf-8"))
if meta["source_sha256"] != split["source_sha256"]:
    raise ValueError("PCA 结果与当前保存切分的数据哈希不一致")
train_ids = set(split["train_indices"])
holdout_ids = set(split["holdout_indices"])
if not train_ids.isdisjoint(holdout_ids):
    raise ValueError("开发集与留出集重叠")
if not oof["probability"].between(0, 1).all():
    raise ValueError("预测概率超出 [0, 1]")
for _, part in oof.groupby(["scenario", "model"]):
    if set(part["row_index"]) != train_ids or part["row_index"].duplicated().any():
        raise ValueError("OOF 预测未恰好覆盖每条开发集样本一次")
    if not set(part["fold"]) == {1, 2, 3, 4, 5}:
        raise ValueError("OOF 预测折数不完整")


def metrics(y, prob, threshold):
    prediction = prob >= threshold
    positive = y == 1
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn}


rows = []
for (scenario, model, fold), part in oof.groupby(["scenario", "model", "fold"]):
    actual = part["isDefault"].to_numpy()
    probability = part["probability"].to_numpy()
    for threshold in THRESHOLDS:
        rows.append({
            "scenario": scenario, "model": model, "fold": int(fold),
            "threshold": float(threshold), "n_valid": len(part),
            **metrics(actual, probability, threshold),
        })
fold_table = pd.DataFrame(rows)
threshold_summary = (
    fold_table.groupby(["scenario", "model", "threshold"], as_index=False)
    .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
         precision_mean=("precision", "mean"),
         recall_mean=("recall", "mean"))
)
best_by_model = (
    threshold_summary.sort_values(
        ["scenario", "model", "f1_mean", "threshold"],
        ascending=[True, True, False, True]
    ).groupby(["scenario", "model"], as_index=False).head(1)
    .sort_values(["scenario", "model"])
)
ranked = summary.sort_values(
    ["roc_auc_mean", "average_precision_mean", "scenario", "model"],
    ascending=[False, False, True, True]
)
chosen_model = ranked.iloc[0]
chosen_threshold = best_by_model.loc[
    (best_by_model["scenario"] == chosen_model["scenario"]) &
    (best_by_model["model"] == chosen_model["model"])
].iloc[0]

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
paths = {
    "folds": RESULTS / f"reduced_threshold_folds_{stamp}.csv",
    "summary": RESULTS / f"reduced_threshold_summary_{stamp}.csv",
    "best": RESULTS / f"reduced_threshold_best_by_model_{stamp}.csv",
    "decision": RESULTS / f"reduced_model_decision_{stamp}.json",
    "figure": FIGURES / f"reduced_threshold_f1_{stamp}.png",
}
for key, frame in [("folds", fold_table), ("summary", threshold_summary),
                   ("best", best_by_model)]:
    with paths[key].open("x", newline="", encoding="utf-8-sig") as stream:
        frame.to_csv(stream, index=False)
decision = {
    "source_pca_summary": str(source_summary),
    "source_pca_oof": str(source_oof),
    "source_data_sha256": meta["source_sha256"],
    "split_file": str(SPLITS),
    "development_rows": len(train_ids), "holdout_rows_not_read": len(holdout_ids),
    "feature_policy": "exclude early_return, early_return_amount, early_return_amount_3mon",
    "model_selection_rule": "Highest mean five-fold ROC-AUC; tie: higher mean average precision, then name",
    "threshold_selection_rule": "Highest mean five-fold F1 over 0.05..0.50 step 0.005; tie: lower threshold",
    "selected_scenario": str(chosen_model["scenario"]),
    "selected_model": str(chosen_model["model"]),
    "selected_threshold": float(chosen_threshold["threshold"]),
    "development_cv_roc_auc_mean": float(chosen_model["roc_auc_mean"]),
    "development_cv_f1_mean_at_selected_threshold": float(chosen_threshold["f1_mean"]),
    "note": "Model and threshold were chosen on the same development folds. Their CV scores are selection estimates, not an independent test. Evaluate the reserved holdout once after freezing this decision.",
}
with paths["decision"].open("x", encoding="utf-8") as stream:
    json.dump(decision, stream, ensure_ascii=False, indent=2)

# Draw the threshold curve for the selected model and its PCA counterpart.
image = Image.new("RGB", (960, 550), "white")
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
draw.text((45, 25), f"Mean five-fold F1 versus threshold: {chosen_model['model']}",
          fill="black", font=font)
left, top, right, bottom = 100, 75, 890, 470
draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
for scenario, color in [("no_pca", "#4C78A8"), ("numeric_pca_95", "#F58518")]:
    part = threshold_summary.loc[
        (threshold_summary["model"] == chosen_model["model"]) &
        (threshold_summary["scenario"] == scenario)
    ].sort_values("threshold")
    points = []
    for row in part.itertuples():
        x = left + (right - left) * (row.threshold - .05) / .45
        y = bottom - (bottom - top) * row.f1_mean / .7
        points.append((round(x), round(y)))
    for a, b in zip(points, points[1:]):
        draw.line((*a, *b), fill=color, width=3)
    legend_y = 500 if scenario == "no_pca" else 520
    draw.rectangle((45, legend_y, 65, legend_y + 12), fill=color)
    draw.text((72, legend_y), scenario, fill="black", font=font)
draw.text((left - 10, bottom + 8), "0.05", fill="black", font=font)
draw.text((right - 25, bottom + 8), "0.50", fill="black", font=font)
draw.text((410, bottom + 30), "Decision threshold", fill="black", font=font)
image.save(paths["figure"])

print("Best development thresholds by model:\n", best_by_model.to_string(index=False), sep="")
print("\nFrozen development decision:\n", json.dumps(decision, ensure_ascii=False, indent=2), sep="")
print("\nSaved files:")
for path in paths.values():
    print(path)
