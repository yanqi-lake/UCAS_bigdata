#!/usr/bin/env python
"""P0 数据自检：MD5、行列数、测试集列、提交样例 id 对齐。

用法（项目根目录）：
    .venv/bin/python scripts/00_verify_data.py

退出码 0 表示全部通过，1 表示至少一项失败（便于接进 CI / Makefile）。
结果同时写到 reports/tables/data_verification.csv。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.io import file_md5, load_competition_data, read_csv  # noqa: E402
from src.validation import (  # noqa: E402
    check_shapes,
    check_submit_example,
    check_test_columns,
)


def verify_official_md5() -> list[dict]:
    rows = []
    for rel, expected in config.OFFICIAL_MD5.items():
        path = config.DATA_DIR / rel
        if not path.exists():
            rows.append({"file": rel, "expected": expected, "actual": "<missing>", "ok": False})
            continue
        actual = file_md5(path)
        rows.append({"file": rel, "expected": expected, "actual": actual, "ok": actual == expected})
    return rows


def verify_raw_md5() -> list[dict]:
    """记录解包后 csv 的 MD5（官网未公布单文件值，仅作留档）。"""
    out = []
    for name in ("train_public.csv", "train_internet.csv", "test_public.csv"):
        p = config.RAW_DIR / name
        out.append({"file": f"raw/{name}", "md5": file_md5(p) if p.exists() else "<missing>"})
    return out


def main() -> int:
    print("=" * 78)
    print("P0 数据自检")
    print("=" * 78)

    print("\n[1/4] 官网原件 MD5")
    official = verify_official_md5()
    for r in official:
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"  [{flag}] {r['file']}")
        if not r["ok"]:
            print(f"         期望 {r['expected']}\n         实际 {r['actual']}")

    print("\n[2/4] 解包 csv 的 MD5（留档，官网未公布单文件值）")
    raw_rows = verify_raw_md5()
    for r in raw_rows:
        print(f"  {r['file']}: {r['md5']}")

    print("\n[3/4] 三表规模与列结构")
    train_public, train_internet, test_public = load_competition_data()
    print(f"  train_public   {train_public.shape}")
    print(f"  train_internet {train_internet.shape}")
    print(f"  test_public    {test_public.shape}")

    checks = check_shapes(train_public, train_internet, test_public)
    checks.append(check_test_columns(train_public, test_public))

    print("\n[4/4] 提交样例与测试集对账")
    submit = read_csv(config.TABLES.submit_example)
    checks.append(check_submit_example(submit, test_public))

    for c in checks:
        print(f"  {c}")
        for f in c.failures:
            print(f"         - {f}")
    print(f"  样例 isDefault 取值: {sorted(submit['isDefault'].unique().tolist())[:5]}")

    ok_md5 = all(r["ok"] for r in official)
    ok_checks = all(c.ok for c in checks)
    total_ok = ok_md5 and ok_checks

    # 落盘留痕
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"item": r["file"], "type": "official_md5", "expected": r["expected"],
             "actual": r["actual"], "ok": r["ok"]} for r in official]
    rows += [{"item": r["file"], "type": "raw_md5", "expected": "-", "actual": r["md5"], "ok": True}
             for r in raw_rows]
    rows += [{"item": c.name, "type": "check", "expected": "-", "actual": c.detail,
              "ok": c.ok} for c in checks]
    out_path = config.TABLES_DIR / "data_verification.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n结果已写入 {out_path.relative_to(ROOT)}")

    print("\n" + ("=" * 78))
    print("结论: " + ("全部通过 ✅" if total_ok else "存在失败项 ❌"))
    print("=" * 78)
    return 0 if total_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())