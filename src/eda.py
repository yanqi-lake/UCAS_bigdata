"""探索性分析与关联强度（P1 环节 1 的后半段）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src import config, plotting, schema


def default_rate_by(
    df: pd.DataFrame,
    col: str,
    target: str = config.TARGET_COL,
    bins: int | None = None,
    top: int = 10,
    kind: str | None = None,
    min_count: int = 30,
) -> pd.DataFrame:
    """分层违约率：类别字段直接分组（Top N + OTHER，样本量 < min_count 的稀有类别并入 OTHER），
    数值字段按分位分箱。"""
    kind = kind or ("numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical")
    if kind == "numeric" or bins:
        s = pd.to_numeric(df[col], errors="coerce")
        grouped = pd.qcut(s, q=bins or 10, duplicates="drop")
        grouped = grouped.astype(str)
    else:
        s = df[col].astype("object").where(df[col].notna(), "缺失")
        counts = s.value_counts()
        keep = counts[counts >= min_count].head(top).index
        grouped = s.where(s.isin(keep), "OTHER").astype(str)

    frame = pd.DataFrame({"group": grouped, "target": df[target].astype(float)})
    out = (
        frame.groupby("group", dropna=False)
        .agg(n=("target", "size"), bad=("target", "sum"), rate=("target", "mean"))
        .reset_index()
        .sort_values("rate")
        .reset_index(drop=True)
    )
    out["share"] = (out["n"] / len(df)).round(4)
    out["rate"] = out["rate"].round(4)
    # 样本量过小的分组不进报告（避免 0%/100% 的假象）
    return out[out["n"] >= min_count].reset_index(drop=True)


def plot_default_rate(table: pd.DataFrame, col: str, path, overall: float, top_n: int = 15) -> None:
    data = table.tail(top_n)
    fig, ax = plotting.new_fig(width=8, height=max(3.0, 0.32 * len(data) + 1.5))
    bars = ax.barh(data["group"], data["rate"], color="#4C78A8")
    ax.axvline(overall, color="#E45756", linestyle="--", linewidth=1.2, label=f"整体 {overall:.1%}")
    ax.bar_label(bars, labels=[f"{r:.1%} (n={n})" for r, n in zip(data["rate"], data["n"])], padding=3, fontsize=8)
    ax.set_xlabel("违约率")
    ax.set_ylabel(col)
    ax.set_title(f"{col} 分层违约率（目标域）")
    ax.set_xlim(0, max(float(data["rate"].max()) * 1.35, overall * 1.5))
    ax.legend(loc="lower right")
    plotting.save(fig, path)


def plot_missing_bars(dfs: dict[str, pd.DataFrame], path, top_n: int = 12) -> None:
    fig, axes = plotting.new_fig(width=9, height=4.2)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (name, df) in zip(axes, dfs.items()):
        rate = (df.isna().mean() * 100).sort_values(ascending=False)
        rate = rate[rate > 0].head(top_n)
        ax.barh(rate.index[::-1], rate.values[::-1], color="#72B7B2")
        ax.set_title(f"{name} 缺失率 Top{len(rate)}")
        ax.set_xlabel("缺失率 (%)")
    plotting.save(fig, path)


def plot_target_balance(df: pd.DataFrame, target: str, path) -> None:
    vc = df[target].value_counts().sort_index()
    fig, ax = plotting.new_fig(width=4.5, height=3.2)
    bars = ax.bar([str(i) for i in vc.index], vc.values, color=["#54A24B", "#E45756"])
    ax.bar_label(bars, labels=[f"{v} ({v / len(df):.1%})" for v in vc.values], padding=3)
    ax.set_title(f"目标域标签分布（{target}）")
    ax.set_xlabel("标签")
    ax.set_ylabel("样本数")
    plotting.save(fig, path)


def distribution(df: pd.DataFrame, col: str, path, kind: str = "numeric", top: int = 12) -> None:
    fig, ax = plotting.new_fig(width=6, height=3.5)
    if kind == "numeric":
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s = s[s.between(s.quantile(0.001), s.quantile(0.999))]
        ax.hist(s, bins=40, color="#4C78A8")
        ax.set_xlabel(col)
        ax.set_ylabel("频数")
    else:
        vc = df[col].value_counts().head(top)
        ax.bar(vc.index.astype(str), vc.values, color="#4C78A8")
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel(col)
        ax.set_ylabel("频数")
    ax.set_title(f"{col} 分布")
    plotting.save(fig, path)


def association_table(
    df: pd.DataFrame,
    target: str = config.TARGET_COL,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Pearson / Spearman / 互信息三口径的关联强度表（数值列给相关系数，类别列只给互信息）。"""
    numeric_cols = numeric_cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = categorical_cols or [
        c for c in df.columns if c not in numeric_cols and c != target
    ]
    numeric_cols = [c for c in numeric_cols if c != target]
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    y = y[mask].astype(int)

    rows = []
    for col in numeric_cols:
        x = pd.to_numeric(df.loc[mask, col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "kind": "numeric",
                "pearson": round(float(x.corr(y, method="pearson")), 4),
                "spearman": round(float(x.corr(y, method="spearman")), 4),
                "mutual_info": np.nan,
            }
        )

    encoded = {}
    discrete = []
    for col in categorical_cols:
        codes, _ = pd.factorize(df.loc[mask, col].astype("object"), use_na_sentinel=True)
        encoded[col] = codes.astype(float)
        discrete.append(True)
    for col in numeric_cols:
        s = pd.to_numeric(df.loc[mask, col], errors="coerce")
        encoded[col] = s.fillna(s.median()).to_numpy()
        discrete.append(False)

    X = pd.DataFrame(encoded, index=df.index[mask])
    mi = mutual_info_classif(X.to_numpy(), y.to_numpy(), discrete_features=discrete, random_state=seed)
    mi_map = dict(zip(X.columns, mi))

    out = pd.DataFrame(rows)
    cat_rows = [
        {
            "feature": col,
            "kind": "categorical",
            "pearson": np.nan,
            "spearman": np.nan,
            "mutual_info": round(float(mi_map.get(col, np.nan)), 4),
        }
        for col in categorical_cols
    ]
    out = pd.concat([out, pd.DataFrame(cat_rows)], ignore_index=True)
    out["mutual_info"] = out.apply(
        lambda r: r["mutual_info"] if pd.notna(r["mutual_info"]) else round(float(mi_map.get(r["feature"], np.nan)), 4),
        axis=1,
    )
    return out.sort_values("mutual_info", ascending=False).reset_index(drop=True)


def collinearity(corr: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """高度共线的字段对（|r| ≥ threshold）。"""
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "pearson": round(float(r), 4)})
    return pd.DataFrame(rows).sort_values("pearson", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def corr_heatmap(corr: pd.DataFrame, path, title: str = "数值特征相关性（Pearson）") -> None:
    n = len(corr.columns)
    fig, ax = plotting.new_fig(width=max(6, 0.32 * n + 2), height=max(5, 0.32 * n + 1.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), corr.columns, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.grid(False)
    plotting.save(fig, path)