"""数据质量体检：缺失、异常值、类别基数、重复、id 关联性证伪（P1 环节 1）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, schema


def missing_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """三表缺失情况长表：table / column / dtype / missing / missing_rate / n_unique。"""
    rows = []
    for table, df in dfs.items():
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            rows.append(
                {
                    "table": table,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "missing": n_miss,
                    "missing_rate": round(n_miss / len(df), 4),
                    "n_unique": int(df[col].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows).sort_values(["table", "missing_rate"], ascending=[True, False]).reset_index(drop=True)


def outlier_ranges(df: pd.DataFrame, cols: list[str], q: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """数值字段的分位截断建议：p01/p99 之外的样本数。"""
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        lo, hi = s.quantile(q[0]), s.quantile(q[1])
        rows.append(
            {
                "column": col,
                "min": float(s.min()),
                "p01": float(lo),
                "p50": float(s.median()),
                "p99": float(hi),
                "max": float(s.max()),
                "below_p01": int((s < lo).sum()),
                "above_p99": int((s > hi).sum()),
            }
        )
    return pd.DataFrame(rows)


def category_cardinality(df: pd.DataFrame, cols: list[str], top: int = 3) -> pd.DataFrame:
    """类别字段基数与头部集中度，判断是否需要用目标编码/频次编码。"""
    rows = []
    for col in cols:
        vc = df[col].value_counts(dropna=True)
        total = max(int(vc.sum()), 1)
        head = "; ".join(f"{idx}({cnt / total:.1%})" for idx, cnt in vc.head(top).items())
        rows.append(
            {
                "column": col,
                "n_unique": int(vc.size),
                "top_share": round(float(vc.iloc[0] / total), 4) if len(vc) else np.nan,
                "top_values": head,
            }
        )
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False).reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame, table: str) -> dict:
    return {
        "table": table,
        "n_rows": int(len(df)),
        "dup_loan_id": int(df[config.ID_COL].duplicated().sum()),
        "dup_user_id": int(df[config.USER_COL].duplicated().sum()),
    }


def id_space(df: pd.DataFrame, table: str, col: str = config.USER_COL) -> dict:
    s = df[col].dropna().astype("int64")
    n = len(s)
    is_dense = bool(n > 0 and s.max() == n - 1 and s.nunique() == n)
    return {
        "table": table,
        "column": col,
        "n_rows": int(len(df)),
        "n_unique": int(s.nunique()),
        "min": int(s.min()),
        "max": int(s.max()),
        "dense_0_to_n-1": is_dense,
    }


def _equality_rate(a: pd.Series, b: pd.Series) -> float:
    both_na = a.isna() & b.isna()
    equal = (a == b) | both_na
    return float(equal.mean())


def id_consistency(train_public: pd.DataFrame, train_internet: pd.DataFrame) -> pd.DataFrame:
    """在"同 id"的样本上比较两域字段取值一致率 —— 用来判断这些 id 是否真是同一人。"""
    left = train_public.copy()
    right = train_internet.copy()
    merged = left.merge(right, on=config.USER_COL, how="inner", suffixes=("_pub", "_net"))
    rows = []
    for col in schema.common_fields():
        if col in (config.ID_COL, config.USER_COL):
            continue
        a, b = merged.get(f"{col}_pub"), merged.get(f"{col}_net")
        if a is None or b is None:
            continue
        rows.append({"field": col, "n_pairs": int(len(merged)), "equal_rate": round(_equality_rate(a, b), 4)})
    return pd.DataFrame(rows).sort_values("equal_rate", ascending=False).reset_index(drop=True)


def verify_id_linkage(
    train_public: pd.DataFrame, train_internet: pd.DataFrame, test_public: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """id 关联性证伪：源域 user_id 是稠密编号，与目标域的"重合"只是编号空间嵌套。

    返回 (id 空间表, 字段一致率表, 结论文本)。
    """
    spaces = pd.DataFrame(
        [
            id_space(train_public, "train_public"),
            id_space(train_internet, "train_internet"),
            id_space(test_public, "test_public"),
        ]
    )
    pub = set(train_public[config.USER_COL])
    net = set(train_internet[config.USER_COL])
    tst = set(test_public[config.USER_COL])
    spaces = pd.concat(
        [
            spaces,
            pd.DataFrame(
                [
                    {
                        "table": "overlap",
                        "column": config.USER_COL,
                        "n_rows": np.nan,
                        "n_unique": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                        "dense_0_to_n-1": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    for col_name in ("n_rows", "n_unique", "min", "max"):
        spaces[col_name] = spaces[col_name].astype("Int64")
    spaces["dense_0_to_n-1"] = spaces["dense_0_to_n-1"].astype("boolean")
    overlap = {
        "public∩internet": len(pub & net),
        "test∩internet": len(tst & net),
        "public∩test": len(pub & tst),
        "loan_id public∩internet": int(len(set(train_public[config.ID_COL]) & set(train_internet[config.ID_COL]))),
    }

    consistency = id_consistency(train_public, train_internet)
    low = consistency[consistency["equal_rate"] < 0.5]["field"].tolist()
    verdict = (
        f"源域 user_id 为 0..{int(spaces.iloc[1]['max'])} 的稠密编号（dense={bool(spaces.iloc[1]['dense_0_to_n-1'])}），"
        f"目标域落在 {int(spaces.iloc[0]['min'])}~{int(spaces.iloc[0]['max'])}；"
        f"重合计数 {overlap} 属编号空间嵌套；"
        f"{len(low)}/{len(consistency)} 个字段在同 id 上一致率 <50%（如 {', '.join(low[:3])}），"
        "判定为伪关联，禁止按 user_id join。"
    )
    return spaces, consistency, verdict