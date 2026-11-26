"""数据与提交文件的校验逻辑（脚本与单测共用）。

设计原则：返回结构化结果而不是到处 print，方便被脚本、测试和报告复用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import config


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        flag = "PASS" if self.ok else "FAIL"
        return f"[{flag}] {self.name}: {self.detail}"


def check_shapes(
    train_public: pd.DataFrame,
    train_internet: pd.DataFrame,
    test_public: pd.DataFrame,
    expected: dict | None = None,
) -> list[CheckResult]:
    """三表行列数是否与官网一致（默认读 config.EXPECTED_SHAPES）。"""
    expected = expected or config.EXPECTED_SHAPES
    frames = {
        "train_public": train_public,
        "train_internet": train_internet,
        "test_public": test_public,
    }
    out = []
    for name, df in frames.items():
        exp_rows, exp_cols = expected[name]
        ok = df.shape == (exp_rows, exp_cols)
        out.append(
            CheckResult(
                name=f"shape:{name}",
                ok=ok,
                detail=f"实际 {df.shape}, 期望 ({exp_rows}, {exp_cols})",
            )
        )
    return out


def check_test_columns(train_public: pd.DataFrame, test_public: pd.DataFrame) -> CheckResult:
    """测试集列 = 训练集列去掉标签列，且顺序一致。"""
    expected = [c for c in train_public.columns if c != config.TARGET_COL]
    actual = list(test_public.columns)
    ok = actual == expected
    return CheckResult(
        name="columns:test_vs_train",
        ok=ok,
        detail="列集合与顺序一致" if ok else f"缺失={sorted(set(expected) - set(actual))} 多余={sorted(set(actual) - set(expected))}",
    )


def check_submit_example(submit: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """提交样例的 id 必须与测试集 loan_id 完全同序。"""
    if "id" not in submit.columns or config.ID_COL not in test.columns:
        return CheckResult("submit_example:id_order", False, "缺少 id 或 loan_id 列")
    same_order = submit["id"].tolist() == test[config.ID_COL].tolist()
    return CheckResult(
        name="submit_example:id_order",
        ok=same_order,
        detail=f"样例 {len(submit)} 行 / 测试 {len(test)} 行，同序={same_order}",
        failures=[] if same_order else ["样例 id 与测试集 loan_id 顺序不一致"],
    )


def check_submission(sub: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """提交文件三项硬校验：列名、行数与 id 对齐、取值域。"""
    failures: list[str] = []
    if list(sub.columns) != ["id", "isDefault"]:
        failures.append(f"列名应为 ['id','isDefault']，实际 {list(sub.columns)}")
    if len(sub) != len(test):
        failures.append(f"行数应为 {len(test)}，实际 {len(sub)}")
    if len(sub) == len(test) and "id" in sub.columns:
        if sub["id"].tolist() != test[config.ID_COL].tolist():
            failures.append("id 与测试集 loan_id 未同序对齐")
    if "isDefault" in sub.columns:
        s = pd.to_numeric(sub["isDefault"], errors="coerce")
        if s.isna().any():
            failures.append(f"isDefault 存在 {int(s.isna().sum())} 个非数值")
        elif ((s < 0) | (s > 1)).any():
            failures.append("isDefault 存在超出 [0,1] 的取值")
    return CheckResult(
        name="submission",
        ok=not failures,
        detail=f"{len(sub)} 行，列={list(sub.columns)}",
        failures=failures,
    )