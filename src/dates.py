"""日期字段解析（两域格式不同，且目标域存在年份缺失的脏值）。

实测格式：
* train_internet   : ``Mar-1984``            → Mon-YYYY
* train_public/test: ``Apr-90``（53%，Mon-YY）、``1-Dec`` / ``10-Aug``（43%，DD-Mon，年份丢失）
DD-Mon 无法还原年份，统一记为 NaT，并在 EDA 中作为数据质量问题记录。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_MON_YY = re.compile(r"^(?P<mon>[A-Za-z]{3})-(?P<yy>\d{2}|\d{4})$")


def parse_earlies_credit_mon(series: pd.Series) -> pd.Series:
    """解析 earlies_credit_mon → datetime64[ns]；DD-Mon（年份缺失）返回 NaT。"""
    s = series.astype("string").str.strip()
    ext = s.str.extract(_MON_YY)
    mon = ext["mon"].map(_MON)
    yy = pd.to_numeric(ext["yy"], errors="coerce")

    year = pd.Series(np.nan, index=s.index, dtype="float64")
    year = year.mask(yy >= 100, yy)                     # 已是四位年份
    year = year.mask((yy < 100) & (yy >= 30), yy + 1900)  # 90 -> 1990
    year = year.mask((yy < 100) & (yy < 30), yy + 2000)   # 05 -> 2005

    valid = mon.notna() & year.notna()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if valid.any():
        strs = (
            year[valid].astype(int).astype(str)
            + "-"
            + mon[valid].astype(int).astype(str).str.zfill(2)
            + "-01"
        )
        out.loc[valid] = pd.to_datetime(strs, format="%Y-%m-%d", errors="coerce").to_numpy()
    return out


def parse_issue_date(series: pd.Series) -> pd.Series:
    """issue_date：目标域 2016/10/1，源域 2015-06-01。"""
    return pd.to_datetime(series.astype("string").str.strip(), errors="coerce", format="mixed")


def issue_year(series: pd.Series) -> pd.Series:
    return parse_issue_date(series).dt.year