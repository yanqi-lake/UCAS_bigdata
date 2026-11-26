"""数据质量体检的最小用例。"""
from __future__ import annotations

import pandas as pd

from src.data_quality import id_space, missing_summary, outlier_ranges, verify_id_linkage


def _tiny_frames():
    tp = pd.DataFrame(
        {
            "loan_id": [1, 2, 3],
            "user_id": [10, 11, 12],
            "region": [1, 2, 3],
            "total_loan": [1000.0, 2000.0, None],
            "isDefault": [0, 1, 0],
        }
    )
    ti = pd.DataFrame(
        {
            "loan_id": [7, 8, 9, 10],
            "user_id": [0, 1, 2, 3],
            "region": [9, 8, 7, 6],
            "total_loan": [500.0, 600.0, 700.0, 800.0],
            "is_default": [0, 1, 0, 1],
        }
    )
    te = pd.DataFrame({"loan_id": [4, 5], "user_id": [12, 13], "region": [3, 4]})
    return tp, ti, te


def test_missing_summary_counts():
    tp, ti, te = _tiny_frames()
    s = missing_summary({"train_public": tp})
    row = s[(s["table"] == "train_public") & (s["column"] == "total_loan")].iloc[0]
    assert row["missing"] == 1
    assert abs(row["missing_rate"] - 1 / 3) < 1e-4


def test_outlier_ranges_basic():
    df = pd.DataFrame({"x": list(range(100)) + [10000]})
    out = outlier_ranges(df, ["x"]).iloc[0]
    assert out["max"] == 10000
    assert out["above_p99"] >= 1


def test_id_space_dense_detection():
    df = pd.DataFrame({"user_id": [0, 1, 2, 3]})
    assert id_space(df, "t")["dense_0_to_n-1"] is True
    df2 = pd.DataFrame({"user_id": [200000, 200001]})
    assert id_space(df2, "t")["dense_0_to_n-1"] is False


def test_verify_id_linkage_flags_fake_overlap():
    tp, ti, te = _tiny_frames()
    spaces, consistency, verdict = verify_id_linkage(tp, ti, te)
    assert "禁止" in verdict
    assert len(spaces) == 4  # 三表 + overlap 行
    assert set(consistency["field"]) >= {"region", "total_loan"}