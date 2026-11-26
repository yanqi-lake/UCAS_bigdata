"""PSI 与日期解析的边界测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dates import parse_earlies_credit_mon
from src.drift import psi, psi_level, psi_table


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=2000))
    assert psi(s, s) < 0.01


def test_psi_disjoint_distributions_is_large():
    a = pd.Series(np.zeros(1000))
    b = pd.Series(np.ones(1000))
    assert psi(a, b) > 0.25


def test_psi_categorical_shift():
    a = pd.Series(["A"] * 800 + ["B"] * 200)
    b = pd.Series(["A"] * 200 + ["B"] * 800)
    assert psi(a, b, kind="categorical") > 0.5


def test_psi_table_sorted_and_levels():
    a = pd.DataFrame({"x": np.arange(100.0), "c": ["a"] * 50 + ["b"] * 50})
    b = pd.DataFrame({"x": np.arange(100.0) + 100, "c": ["b"] * 100})
    t = psi_table(a, b, ["x", "c"], kinds={"x": "numeric", "c": "categorical"})
    assert list(t["feature"]) == ["x", "c"] or list(t["feature"]) == ["c", "x"]
    assert t["psi"].is_monotonic_decreasing
    assert psi_level(0.05) == "stable" and psi_level(0.2) == "moderate" and psi_level(0.5) == "shifted"


def test_parse_earlies_credit_mon_formats():
    s = pd.Series(["Mar-1984", "Apr-90", "Dec-99", "1-Dec", "10-Aug"])
    out = parse_earlies_credit_mon(s)
    assert out.iloc[0] == pd.Timestamp("1984-03-01")
    assert out.iloc[1] == pd.Timestamp("1990-04-01")
    assert out.iloc[2] == pd.Timestamp("1999-12-01")
    assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])