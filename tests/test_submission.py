"""提交文件校验：正常与四类异常。"""
from __future__ import annotations

import pandas as pd

from src import config
from src.validation import check_submission


def _fixture():
    test = pd.DataFrame({config.ID_COL: [11, 22, 33]})
    good = pd.DataFrame({"id": [11, 22, 33], "isDefault": [0.2, 0.7, 0.4]})
    return test, good


def test_valid_submission_passes():
    test, good = _fixture()
    r = check_submission(good, test)
    assert r.ok, r.failures


def test_wrong_columns_fail():
    test, good = _fixture()
    bad = good.rename(columns={"isDefault": "pred"})
    assert not check_submission(bad, test).ok


def test_wrong_row_count_fails():
    test, good = _fixture()
    assert not check_submission(good.iloc[:2], test).ok


def test_id_order_mismatch_fails():
    test, good = _fixture()
    bad = good.copy()
    bad["id"] = [22, 11, 33]
    assert not check_submission(bad, test).ok


def test_out_of_range_probability_fails():
    test, good = _fixture()
    bad = good.copy()
    bad.loc[0, "isDefault"] = 1.7
    assert not check_submission(bad, test).ok