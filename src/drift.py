"""分布漂移：PSI（Population Stability Index）。

PSI = Σ (p - q) * ln(p / q)，p 为目标域（训练折）占比，q 为源域占比。
经验阈值：<0.1 稳定，0.1~0.25 中等，>0.25 显著（不可直接迁移）。
注意：分箱边界必须只用"期望分布"（目标域）计算，验证/测试集只做映射，避免信息泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6
_NA = "<NA>"
_OTHER = "<OTHER>"


def psi_level(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "shifted"


def _numeric_distribution(expected: pd.Series, actual: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(expected, errors="coerce").dropna()
    if values.nunique() <= 1:
        c = float(values.iloc[0]) if len(values) else 0.0
        edges = np.array([-np.inf, c, np.inf])
    else:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(values, quantiles))
        edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    p = pd.cut(pd.to_numeric(expected, errors="coerce"), edges, include_lowest=True)
    q = pd.cut(pd.to_numeric(actual, errors="coerce"), edges, include_lowest=True)
    p_cnt = p.value_counts(dropna=False)
    q_cnt = q.value_counts(dropna=False)
    # 统一成字符串标签后再对齐，避免 CategoricalIndex 因类别不同而无法 reindex
    p_cnt.index = p_cnt.index.astype(str)
    q_cnt.index = q_cnt.index.astype(str)
    q_cnt = q_cnt.reindex(p_cnt.index).fillna(0)
    return p_cnt.to_numpy(dtype=float), q_cnt.to_numpy(dtype=float)


def _categorical_distribution(expected: pd.Series, actual: pd.Series, max_levels: int = 20) -> tuple[np.ndarray, np.ndarray]:
    top = expected.astype("object").where(expected.notna(), _NA).value_counts().head(max_levels).index
    p = expected.astype("object").where(expected.notna(), _NA).where(lambda s: s.isin(top), _OTHER)
    q = actual.astype("object").where(actual.notna(), _NA).where(lambda s: s.isin(top), _OTHER)
    levels = list(pd.Index(top).tolist()) + [_OTHER, _NA]
    levels = [lv for lv in dict.fromkeys(levels)]
    p_cnt = p.value_counts().reindex(levels).fillna(0)
    q_cnt = q.value_counts().reindex(levels).fillna(0)
    return p_cnt.to_numpy(dtype=float), q_cnt.to_numpy(dtype=float)


def psi(expected: pd.Series, actual: pd.Series, kind: str = "numeric", bins: int = 10) -> float:
    """计算单字段 PSI。kind: numeric / categorical。缺失值单独成一箱。"""
    if kind == "numeric":
        p_cnt, q_cnt = _numeric_distribution(expected, actual, bins)
    else:
        p_cnt, q_cnt = _categorical_distribution(expected, actual)

    p = np.clip(p_cnt / max(p_cnt.sum(), 1), _EPS, None)
    q = np.clip(q_cnt / max(q_cnt.sum(), 1), _EPS, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((p - q) * np.log(p / q)))


def psi_table(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    columns: list[str],
    kinds: dict[str, str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """批量 PSI：expected=目标域，actual=源域。kinds 缺省时按 dtype 猜（数值→numeric，其余→categorical）。"""
    rows = []
    for col in columns:
        if col not in expected.columns or col not in actual.columns:
            continue
        kind = (kinds or {}).get(col)
        if kind is None:
            kind = "numeric" if pd.api.types.is_numeric_dtype(expected[col]) else "categorical"
        value = psi(expected[col], actual[col], kind=kind, bins=bins)
        rows.append(
            {
                "feature": col,
                "kind": kind,
                "psi": round(value, 4),
                "level": psi_level(value),
                "miss_rate_expected": round(float(expected[col].isna().mean()), 4),
                "miss_rate_actual": round(float(actual[col].isna().mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)