"""项目级配置：目录、随机种子、数据表与官方校验值。

所有路径都相对项目根目录解析，避免依赖调用者的当前工作目录。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OFFICIAL_DIR = DATA_DIR / "official"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGS_DIR = REPORTS_DIR / "figs"
TABLES_DIR = REPORTS_DIR / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 2026
N_FOLDS = 5
TARGET_COL = "isDefault"          # 目标域标签
SOURCE_TARGET_COL = "is_default"  # 源域标签
ID_COL = "loan_id"
USER_COL = "user_id"
DATE_COL = "issue_date"

TABLES = SimpleNamespace(
    train_public=RAW_DIR / "train_public.csv",
    train_internet=RAW_DIR / "train_internet.csv",
    test_public=RAW_DIR / "test_public.csv",
    submit_example=DATA_DIR / "submit_example.csv",
)

# 官网公布的文件 MD5（scripts/00_verify_data.py 用来自检）
OFFICIAL_MD5 = {
    "official/train_dataset.zip": "192D3D30A9252E849C600C0931C8C1FB",
    "official/test_public.csv": "D4A2AB57C4B6188CC606949501750420",
    "official/submit_example.csv": "5F272DFEE05DF1C950009A6E5700E509",
    "official/个贷违约预测baseline.zip": "0D4D985CB15DB705E89A3DD8CFCDEEC5",
    "official/【更新1117】个贷违约预测Baseline.zip": "2DAA6FEB80626560FC09D448AFF3FBE3",
}

# 三表规模（用于冒烟检查，行数不含表头）
EXPECTED_SHAPES = {
    "train_public": (10000, 39),
    "train_internet": (750000, 42),
    "test_public": (5000, 38),
}


def load_config(path: str | Path | None = None) -> dict:
    """读取 yaml 配置（默认 configs/base.yaml），并与内置常量合并返回普通 dict。"""
    cfg_path = Path(path) if path is not None else CONFIGS_DIR / "base.yaml"
    cfg: dict = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    cfg.setdefault("seed", SEED)
    cfg.setdefault("n_folds", N_FOLDS)
    cfg.setdefault("target_col", TARGET_COL)
    cfg.setdefault("source_target_col", SOURCE_TARGET_COL)
    cfg["project_root"] = str(PROJECT_ROOT)
    return cfg