"""Create report figures from saved final results, without retraining or evaluating."""

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
metric_files = sorted(RESULTS.glob("final_holdout_metrics_*.json"))
if len(metric_files) != 1:
    raise ValueError("预期恰好一份最终留出集指标文件")
stamp = metric_files[0].stem.removeprefix("final_holdout_metrics_")
metrics = json.loads(metric_files[0].read_text(encoding="utf-8"))
importance = pd.read_csv(RESULTS / f"final_holdout_feature_importance_{stamp}.csv", encoding="utf-8-sig")
if importance.empty:
    raise ValueError("特征重要性表为空")
importance_path = FIGURES / f"final_holdout_feature_importance_{stamp}.png"
confusion_path = FIGURES / f"final_holdout_confusion_matrix_{stamp}.png"
if importance_path.exists() and confusion_path.exists():
    print("最终图表已存在；未覆盖：")
    print(importance_path)
    print(confusion_path)
    raise SystemExit(0)
if importance_path.exists() or confusion_path.exists():
    raise FileExistsError("仅部分目标图表存在；为避免覆盖，已停止")

font = ImageFont.load_default()
top = importance.head(15).iloc[::-1]
image = Image.new("RGB", (1050, 700), "white")
draw = ImageDraw.Draw(image)
draw.text((35, 25), "Final random-forest impurity importance (top 15 transformed features)",
          fill="black", font=font)
maximum = float(top["impurity_importance"].max())
for i, row in enumerate(top.itertuples()):
    y = 75 + i * 39
    label = row.transformed_feature.removeprefix("numeric__").removeprefix("categorical__")
    draw.text((20, y + 6), label, fill="black", font=font)
    width = round(520 * float(row.impurity_importance) / maximum)
    draw.rectangle((420, y, 420 + width, y + 25), fill="#4C78A8")
    draw.text((430 + width, y + 6), f"{row.impurity_importance:.4f}", fill="black", font=font)
draw.text((25, 672), "Model-internal descriptive measure; not a causal effect.",
          fill="#555555", font=font)
image.save(importance_path)

cm = metrics["confusion_matrix"]
image = Image.new("RGB", (680, 500), "white")
draw = ImageDraw.Draw(image)
draw.text((30, 25), f"Final holdout confusion matrix (threshold={metrics['threshold']})",
          fill="black", font=font)
draw.text((220, 80), "Predicted 0", fill="black", font=font)
draw.text((430, 80), "Predicted 1", fill="black", font=font)
draw.text((25, 175), "Actual 0", fill="black", font=font)
draw.text((25, 355), "Actual 1", fill="black", font=font)
cells = [
    ("tn", 180, 120, "#B4D8EA"), ("fp", 390, 120, "#FAD4B5"),
    ("fn", 180, 300, "#FAD4B5"), ("tp", 390, 300, "#B4D8EA"),
]
for name, x, y, color in cells:
    draw.rectangle((x, y, x + 170, y + 140), fill=color)
    draw.text((x + 65, y + 50), str(cm[name]), fill="black", font=font)
    draw.text((x + 65, y + 80), name.upper(), fill="#444444", font=font)
image.save(confusion_path)
print(importance_path)
print(confusion_path)
