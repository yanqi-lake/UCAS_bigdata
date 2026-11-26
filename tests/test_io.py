"""IO 层单测：BOM、表名分派、提交文件读写。"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.io import file_md5, make_submission, read_csv, read_table


def test_read_csv_strips_bom(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_text("loan_id,isDefault\n1,0\n", encoding="utf-8-sig")
    df = read_csv(p)
    assert list(df.columns) == ["loan_id", "isDefault"]
    assert df.loc[0, "loan_id"] == 1


def test_read_table_rejects_unknown_name():
    with pytest.raises(KeyError):
        read_table("not_a_table")


def test_make_submission_roundtrip(tmp_path):
    test = pd.DataFrame({"loan_id": [1000575, 1028125, 1010694]})
    sub = make_submission(test, [0.1, 0.9, 0.5], tmp_path / "sub.csv")
    assert list(sub.columns) == ["id", "isDefault"]
    back = pd.read_csv(tmp_path / "sub.csv")
    assert back["id"].tolist() == test["loan_id"].tolist()
    assert back["isDefault"].tolist() == pytest.approx([0.1, 0.9, 0.5])


def test_make_submission_length_mismatch():
    test = pd.DataFrame({"loan_id": [1, 2]})
    with pytest.raises(ValueError):
        make_submission(test, [0.1], "/tmp/never_written.csv")


def test_file_md5_matches_known_value(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    assert file_md5(p) == "900150983CD24FB0D6963F7D28E17F72"