"""Development-set EDA and descriptive correlations; no model is trained.

Run: python scripts/05_eda_correlations.py
Only the saved development indices are analyzed. Original CSV is read-only.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = next(
    (p for p in (ROOT / "data/raw/train_public.csv", ROOT / "data/train_public.csv") if p.is_file()),
    None,
)
SPLITS = ROOT / "data/splits/split_indices.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TARGET = "isDefault"
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

if DATA is None:
    raise FileNotFoundError("找不到 train_public.csv")
with DATA.open("rb") as stream:
    data_sha = hashlib.file_digest(stream, "sha256").hexdigest()
split = json.loads(SPLITS.read_text(encoding="utf-8"))
if data_sha != split["source_sha256"]:
    raise ValueError("数据 SHA256 与保存切分时不同，已停止运行")
data = pd.read_csv(DATA, encoding="utf-8-sig")
train_ids = split["train_indices"]
holdout_ids = split["holdout_indices"]
if len(data) != len(train_ids) + len(holdout_ids):
    raise ValueError("数据行数与切分文件不一致")
if not set(NUMERIC + CATEGORICAL + [TARGET]) <= set(data.columns):
    raise ValueError("缺少预期字段")
assert set(train_ids).isdisjoint(holdout_ids)
df = data.iloc[train_ids].copy()
assert len(df) == len(train_ids)
assert set(df[TARGET].dropna().unique()) == {0, 1}
for col in NUMERIC:
    df[col] = pd.to_numeric(df[col], errors="coerce")

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
paths = {
    "labels": RESULTS / f"eda_labels_{stamp}.csv",
    "missing": RESULTS / f"eda_missing_{stamp}.csv",
    "numeric": RESULTS / f"eda_numeric_summary_{stamp}.csv",
    "outliers": RESULTS / f"eda_iqr_outliers_{stamp}.csv",
    "pearson": RESULTS / f"eda_pearson_matrix_{stamp}.csv",
    "spearman": RESULTS / f"eda_spearman_matrix_{stamp}.csv",
    "target_corr": RESULTS / f"eda_target_correlations_{stamp}.csv",
    "categories": RESULTS / f"eda_category_default_rates_{stamp}.csv",
    "metadata": RESULTS / f"eda_metadata_{stamp}.json",
    "label_figure": FIGURES / f"eda_label_distribution_{stamp}.png",
    "missing_figure": FIGURES / f"eda_missingness_{stamp}.png",
    "correlation_figure": FIGURES / f"eda_numeric_correlations_{stamp}.png",
}

def save(frame, key):
    with paths[key].open("x", newline="", encoding="utf-8-sig") as stream:
        frame.to_csv(stream, index=False)

labels = df[TARGET].value_counts().sort_index()
label_table = pd.DataFrame({
    TARGET: labels.index.astype(int), "count": labels.to_numpy(),
    "share": labels.to_numpy() / len(df),
})
save(label_table, "labels")

missing_table = pd.DataFrame({
    "feature": df.columns,
    "dtype": [str(df[col].dtype) for col in df.columns],
    "missing_count": df.isna().sum().to_numpy(),
    "missing_rate": df.isna().mean().to_numpy(),
    "unique_nonmissing": df.nunique(dropna=True).to_numpy(),
}).sort_values(["missing_count", "feature"], ascending=[False, True])
save(missing_table, "missing")

numeric_summary = df[NUMERIC].describe(percentiles=[.01, .25, .5, .75, .99]).T
numeric_summary.insert(0, "feature", numeric_summary.index)
save(numeric_summary.reset_index(drop=True), "numeric")

outlier_rows = []
for col in NUMERIC:
    values = df[col].dropna()
    q1, q3 = values.quantile([.25, .75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_rows.append({
        "feature": col, "nonmissing": len(values), "q1": q1, "q3": q3,
        "iqr": iqr, "lower_fence": lower, "upper_fence": upper,
        "outside_count": int(((values < lower) | (values > upper)).sum()),
        "outside_rate": float(((values < lower) | (values > upper)).mean()),
    })
save(pd.DataFrame(outlier_rows), "outliers")

corr_cols = [col for col in NUMERIC if df[col].nunique(dropna=True) > 1]
pearson = df[corr_cols + [TARGET]].corr(method="pearson")
spearman = df[corr_cols + [TARGET]].corr(method="spearman")
save(pearson.rename_axis("feature").reset_index(), "pearson")
save(spearman.rename_axis("feature").reset_index(), "spearman")
target_corr = pd.DataFrame({
    "feature": corr_cols,
    "pearson_with_target": pearson.loc[corr_cols, TARGET].to_numpy(),
    "spearman_with_target": spearman.loc[corr_cols, TARGET].to_numpy(),
})
target_corr["abs_pearson"] = target_corr["pearson_with_target"].abs()
target_corr = target_corr.sort_values("abs_pearson", ascending=False)
save(target_corr, "target_corr")

category_rows = []
for col in CATEGORICAL:
    group = (
        df.assign(category=df[col].astype("string").fillna("<MISSING>"))
        .groupby("category", dropna=False)[TARGET]
        .agg(["size", "mean"])
        .sort_values("size", ascending=False)
    )
    for category, row in group.iterrows():
        category_rows.append({
            "feature": col, "category": str(category),
            "count": int(row["size"]), "default_rate": float(row["mean"]),
        })
save(pd.DataFrame(category_rows), "categories")

metadata = {
    "source": str(DATA), "source_sha256": data_sha,
    "split_file": str(SPLITS), "development_rows": len(df),
    "holdout_rows_not_inspected": len(holdout_ids),
    "numeric_columns": NUMERIC, "categorical_columns": CATEGORICAL,
    "excluded_from_numeric_correlation": [
        "loan_id", "user_id", "post_code", "title", "policy_code",
        "issue_date", "earlies_credit_mon",
    ],
    "notes": [
        "Pearson and Spearman correlations are descriptive, not causal.",
        "IQR counts flag unusual values; they are not automatically data errors.",
        "Early-return fields require a prediction-time availability audit before final modeling.",
        "No holdout scores or values were used.",
    ],
}
with paths["metadata"].open("x", encoding="utf-8") as stream:
    json.dump(metadata, stream, ensure_ascii=False, indent=2)

font = ImageFont.load_default()
image = Image.new("RGB", (900, 500), "white")
draw = ImageDraw.Draw(image)
draw.text((55, 30), "Development-set target distribution", fill="black", font=font)
max_count = int(labels.max())
for i, (label, count) in enumerate(labels.items()):
    x0 = 190 + i * 320
    height = round(330 * count / max_count)
    draw.rectangle((x0, 410 - height, x0 + 160, 410), fill=["#4C78A8", "#E45756"][i])
    draw.text((x0 + 55, 390 - height), str(count), fill="black", font=font)
    draw.text((x0 + 20, 430), ["Non-default (0)", "Default (1)"][i], fill="black", font=font)
image.save(paths["label_figure"])

nonzero = missing_table.loc[missing_table["missing_count"] > 0].sort_values("missing_rate")
image = Image.new("RGB", (1050, max(300, 90 + 43 * len(nonzero))), "white")
draw = ImageDraw.Draw(image)
draw.text((30, 25), "Missingness in development set", fill="black", font=font)
for i, row in enumerate(nonzero.itertuples()):
    y0 = 70 + 43 * i
    width = round(float(row.missing_rate) * 1000)
    draw.text((25, y0 + 5), row.feature, fill="black", font=font)
    draw.rectangle((265, y0, 265 + width, y0 + 25), fill="#F58518")
    draw.text((280 + width, y0 + 5), f"{row.missing_rate:.1%}", fill="black", font=font)
image.save(paths["missing_figure"])

cell = 33
image = Image.new("RGB", (510 + cell * len(corr_cols), 130 + cell * len(corr_cols)), "white")
draw = ImageDraw.Draw(image)
draw.text((25, 20), "Pearson correlations of numerical features (-1 blue, +1 red)", fill="black", font=font)
matrix = pearson.loc[corr_cols, corr_cols]
for i, row_name in enumerate(corr_cols):
    draw.text((10, 93 + i * cell), row_name, fill="black", font=font)
    draw.text((285 + i * cell, 72), str(i + 1), fill="black", font=font)
    for j, col_name in enumerate(corr_cols):
        value = float(matrix.loc[row_name, col_name])
        if value >= 0:
            color = (255, round(255 * (1 - value)), round(255 * (1 - value)))
        else:
            color = (round(255 * (1 + value)), round(255 * (1 + value)), 255)
        x0, y0 = 280 + j * cell, 90 + i * cell
        draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color)
image.save(paths["correlation_figure"])

print("Development rows:", len(df), "| Holdout rows not inspected:", len(holdout_ids))
print("\nTarget distribution:\n", label_table.to_string(index=False), sep="")
print("\nNonzero missingness:\n", nonzero[["feature", "missing_count", "missing_rate"]].to_string(index=False), sep="")
print("\nTop absolute Pearson correlations with target:\n", target_corr.head(8).to_string(index=False), sep="")
print("\nSaved files:")
for path in paths.values():
    print(path)
