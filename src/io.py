"""统一的读写层。

约定：
* 竞赛 csv 全部带 UTF-8 BOM，一律用 ``encoding="utf-8-sig"`` 读取；
* 中间产物用 parquet（float32 + 列裁剪，750k 规模下比 csv 快一个量级）；
* 提交文件只保留 ``id, isDefault`` 两列，id 直接取测试集 ``loan_id``。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config

_ENCODING = "utf-8-sig"


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """读取带 BOM 的竞赛 csv。"""
    params = dict(encoding=_ENCODING, low_memory=False)
    params.update(kwargs)
    return pd.read_csv(Path(path), **params)


def read_table(name: str, **kwargs) -> pd.DataFrame:
    """按名字读取三表之一：train_public / train_internet / test_public。"""
    if name not in ("train_public", "train_internet", "test_public"):
        raise KeyError(f"未知的数据表: {name}")
    return read_csv(getattr(config.TABLES, name), **kwargs)


def load_competition_data(
    drop_duplicate_columns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """一次性读取目标域训练集、源域训练集与测试集。"""
    train_public = read_table("train_public")
    train_internet = read_table("train_internet")
    test_public = read_table("test_public")
    if drop_duplicate_columns:
        # 防御：csv 若被重复追加表头/重复列名，pandas 会生成 .1 后缀，这里直接报错而不是静默继续
        for name, df in (
            ("train_public", train_public),
            ("train_internet", train_internet),
            ("test_public", test_public),
        ):
            dup = [c for c in df.columns if c.endswith(".1")]
            if dup:
                raise ValueError(f"{name} 存在重复列名: {dup}")
    return train_public, train_internet, test_public


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """分块计算 MD5，避免把 160MB 文件整体读进内存。"""
    h = hashlib.md5()
    with open(Path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)
    return p


def read_parquet(path: str | Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(path), columns=list(columns) if columns else None)


def make_submission(test: pd.DataFrame, proba, path: str | Path) -> pd.DataFrame:
    """按赛题格式写出提交文件：id = 测试集 loan_id，isDefault = 违约概率。"""
    if len(test) != len(proba):
        raise ValueError(f"测试集 {len(test)} 行与预测 {len(proba)} 行不一致")
    sub = pd.DataFrame({"id": test[config.ID_COL].to_numpy(), "isDefault": list(proba)})
    p = Path(path)
    ensure_dir(p.parent)
    sub.to_csv(p, index=False, encoding="utf-8")
    return sub