"""字段语义层：从 configs/feature_map.yaml 读取字段类型 / 作用域 / 编码约定。

全项目（EDA、特征、建模、PSI）都以这里为唯一事实来源，避免各脚本各写一份字段清单。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from src import config

FEATURE_MAP_PATH = config.CONFIGS_DIR / "feature_map.yaml"


@lru_cache(maxsize=1)
def field_map(path: str | None = None) -> dict:
    p = Path(path) if path else FEATURE_MAP_PATH
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def target_columns() -> dict:
    return field_map()["target"]


def fields(scope: str | None = None, type_: str | None = None, transferable: bool | None = None) -> list[str]:
    """按作用域/类型/可迁移性筛选字段名。"""
    out = []
    for name, meta in field_map()["fields"].items():
        if scope and meta.get("scope") != scope:
            continue
        if type_ and meta.get("type") != type_:
            continue
        if transferable is not None and bool(meta.get("transferable")) != transferable:
            continue
        out.append(name)
    return out


def fields_by_type(kind: str) -> list[str]:
    return fields(type_=kind)


def common_fields() -> list[str]:
    return fields(scope="common")


def feature_fields() -> list[str]:
    """所有可作为模型输入的字段（排除 id 与目标）。"""
    return [f for f, m in field_map()["fields"].items() if m.get("type") != "id"]


def missing_strategy(name: str) -> str:
    return field_map()["fields"][name].get("missing", "none")