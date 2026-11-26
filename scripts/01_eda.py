#!/usr/bin/env python
"""P1 数据治理与探索性分析入口。

用法（项目根目录）：
    .venv/bin/python scripts/01_eda.py --section all
    .venv/bin/python scripts/01_eda.py --section quality|dist|assoc|psi

产物：
    reports/tables/*.csv   表格（缺失、异常、基数、id 关联、违约率、关联强度、PSI）
    reports/figs/*.png     图（缺失、标签分布、分层违约率、相关性、PSI）
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config, data_quality, dates, drift, eda, plotting, schema  # noqa: E402
from src.io import load_competition_data  # noqa: E402


def _dump(df: pd.DataFrame, name: str) -> Path:
    path = config.TABLES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def section_quality(tp, ti, te) -> None:
    print("\n[quality] 缺失 / 异常 / 基数 / 重复 / id 关联")
    dfs = {"train_public": tp, "train_internet": ti, "test_public": te}
    missing = data_quality.missing_summary(dfs)
    print("  缺失 Top5:", missing[missing["missing"] > 0].head(5)[["table", "column", "missing_rate"]].to_dict("records"))
    _dump(missing, "quality_missing.csv")

    numeric = schema.fields(type_="numeric")
    outliers = pd.concat(
        [
            data_quality.outlier_ranges(tp, [c for c in numeric if c in tp.columns]).assign(table="train_public"),
            data_quality.outlier_ranges(ti, [c for c in numeric if c in ti.columns]).assign(table="train_internet"),
        ],
        ignore_index=True,
    )
    print(f"  异常值区间: {len(outliers)} 个数值字段 -> reports/tables/quality_outliers.csv")
    _dump(outliers, "quality_outliers.csv")

    cat_cols = [c for c in schema.fields(type_="categorical") + schema.fields(type_="ordinal") if c in tp.columns]
    card = data_quality.category_cardinality(tp, cat_cols)
    print("  高基数字段:", card.head(4)[["column", "n_unique"]].to_dict("records"))
    _dump(card, "quality_cardinality.csv")

    dup = pd.DataFrame(
        [
            data_quality.duplicate_summary(tp, "train_public"),
            data_quality.duplicate_summary(ti, "train_internet"),
            data_quality.duplicate_summary(te, "test_public"),
        ]
    )
    _dump(dup, "quality_duplicates.csv")

    spaces, consistency, verdict = data_quality.verify_id_linkage(tp, ti, te)
    _dump(spaces, "id_space.csv")
    _dump(consistency, "id_consistency.csv")
    print("  id 空间:")
    print(spaces.to_string(index=False))
    print(f"  结论: {verdict}")


def section_dist(tp, ti, te) -> None:
    print("\n[dist] 标签分布 / 分层违约率 / 缺失图")
    overall = float(tp[config.TARGET_COL].mean())
    eda.plot_target_balance(tp, config.TARGET_COL, config.FIGS_DIR / "target_balance.png")
    eda.plot_missing_bars({"train_public": tp, "train_internet": ti}, config.FIGS_DIR / "missing_bars.png")
    print(f"  目标域违约率 {overall:.2%}，源域违约率 {float(ti[config.SOURCE_TARGET_COL].mean()):.2%}")

    tp = tp.copy()
    tp["issue_year"] = dates.issue_year(tp[config.DATE_COL])
    plans = [
        ("class", None),
        ("work_year", None),
        ("industry", None),
        ("use", None),
        ("house_exist", None),
        ("region", None),
        ("debt_loan_ratio", 10),
        ("issue_year", None),
    ]
    for col, bins in plans:
        if col not in tp.columns:
            continue
        kind = "numeric" if col == "debt_loan_ratio" else "categorical"
        table = eda.default_rate_by(tp, col, config.TARGET_COL, bins=bins, kind=kind)
        _dump(table, f"default_rate_by_{col}.csv")
        eda.plot_default_rate(table, col, config.FIGS_DIR / f"default_rate_{col}.png", overall)
        print(f"  {col}: 违约率区间 {table['rate'].min():.1%} ~ {table['rate'].max():.1%}")

    for col in ("interest", "total_loan", "monthly_payment"):
        eda.distribution(tp, col, config.FIGS_DIR / f"dist_{col}.png", kind="numeric")


def section_assoc(tp) -> None:
    print("\n[assoc] 关联强度 / 共线性")
    exclude = {config.TARGET_COL, config.ID_COL, config.USER_COL}
    numeric = [
        c for c in tp.columns if pd.api.types.is_numeric_dtype(tp[c]) and c not in exclude
    ]
    categorical = [c for c in tp.columns if c not in numeric and c not in exclude]
    assoc = eda.association_table(tp, config.TARGET_COL, numeric, categorical)
    _dump(assoc, "association.csv")
    print("  互信息 Top5:", assoc.head(5)[["feature", "kind", "mutual_info"]].to_dict("records"))

    corr = tp[numeric].corr(method="pearson")
    eda.corr_heatmap(corr, config.FIGS_DIR / "corr_heatmap.png")
    col = eda.collinearity(corr, threshold=0.9)
    _dump(col, "collinearity.csv")
    print(f"  |r|>=0.9 的字段对: {len(col)} 对")


def section_psi(tp, ti, te) -> None:
    print("\n[psi] 域间分布漂移（目标域 vs 源域）")
    tp2, ti2, te2 = tp.copy(), ti.copy(), te.copy()
    for df in (tp2, ti2, te2):
        df["issue_year"] = dates.issue_year(df[config.DATE_COL])
        df["earlies_credit_year"] = dates.parse_earlies_credit_mon(df["earlies_credit_mon"]).dt.year

    # 按 schema 决定 PSI 口径：类别字段用频次分布，数值/序数字段用分位分箱；
    # 日期字段用派生的年份代替原始字符串。
    kinds: dict[str, str] = {}
    for name in schema.common_fields():
        if name in (config.ID_COL, config.USER_COL):
            continue
        meta = schema.field_map()["fields"][name]
        if meta["type"] == "datetime":
            continue
        kinds[name] = "categorical" if meta["type"] == "categorical" else "numeric"
    kinds.update({"issue_year": "numeric", "earlies_credit_year": "numeric"})
    features = list(kinds)
    table = drift.psi_table(tp2, ti2, features, kinds=kinds)
    table = table.rename(columns={"psi": "psi_internet_vs_public"})
    table_test = drift.psi_table(tp2, te2, features, kinds=kinds)[["feature", "psi"]]
    table = table.merge(table_test.rename(columns={"psi": "psi_test_vs_public"}), on="feature", how="left")
    table["level"] = table["psi_internet_vs_public"].map(drift.psi_level)
    _dump(table, "psi.csv")

    plotting.setup_font(verbose=False)
    plot_data = table.sort_values("psi_internet_vs_public", ascending=True).tail(20)
    fig, ax = plotting.new_fig(width=8, height=max(3.5, 0.3 * len(plot_data) + 1.2))
    colors = {"stable": "#54A24B", "moderate": "#EECA3B", "shifted": "#E45756"}
    ax.barh(plot_data["feature"], plot_data["psi_internet_vs_public"],
            color=[colors[l] for l in plot_data["level"]])
    ax.axvline(0.1, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0.25, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("PSI（源域 vs 目标域）")
    ax.set_title("共有字段的域间漂移（Top20）")
    plotting.save(fig, config.FIGS_DIR / "psi_bar.png")

    print(table.head(6)[["feature", "psi_internet_vs_public", "level"]].to_string(index=False))
    shifted = table[table["level"] == "shifted"]["feature"].tolist()
    print(f"  显著漂移(>0.25): {len(shifted)} 个 -> {shifted}")


def main() -> int:
    warnings.filterwarnings("ignore")  # 常数列相关、分箱边界等告警不影响结论
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="all", choices=["all", "quality", "dist", "assoc", "psi"])
    args = parser.parse_args()

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGS_DIR.mkdir(parents=True, exist_ok=True)
    plotting.setup_font(verbose=False)

    tp, ti, te = load_competition_data()
    print(f"数据: public={tp.shape} internet={ti.shape} test={te.shape}")

    if args.section in ("all", "quality"):
        section_quality(tp, ti, te)
    if args.section in ("all", "dist"):
        section_dist(tp, ti, te)
    if args.section in ("all", "assoc"):
        section_assoc(tp)
    if args.section in ("all", "psi"):
        section_psi(tp, ti, te)

    print("\n完成。表格 -> reports/tables，图 -> reports/figs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())