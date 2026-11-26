# P1 交接说明（数据治理与 EDA）

> 交付日期：2026-11-26　执行：组员 B　下一棒：组员 B（P2 特征工程）
> 对应计划：`docs/PLAN.md` P1　状态：**✅ 已完成，P2 可以开工**
> 成果报告：`reports/R1_data.md`

## 1. 一句话结论

`make eda` 一条命令生成 17 张表 + 15 张图，`reports/R1_data.md` 已写完全部结论；19 个单测通过。
三条最关键的事实：**`user_id` 不可 join**、**PSI 只对 6 个字段告警**、**`earlies_credit_mon` 目标域 46.7% 缺年份**。

## 2. 本次交付的文件

| 文件 | 作用 |
|---|---|
| `src/data_quality.py` | 缺失/异常/基数/重复汇总、`verify_id_linkage()` 伪关联证伪 |
| `src/eda.py` | 分层违约率（含稀有类别合并）、关联强度（Pearson/Spearman/互信息）、共线性、绘图 |
| `src/drift.py` | PSI（数值按分位分箱、类别按频次、缺失单独成箱） |
| `src/dates.py` | `issue_date` / `earlies_credit_mon` 解析（含 DD-Mon 记 NaT 的规则） |
| `src/schema.py` | 读 `configs/feature_map.yaml`，全项目字段类型的唯一来源 |
| `configs/feature_map.yaml` | 46 个字段的类型/作用域/编码/缺失策略/可迁移性（已按 PSI 更新） |
| `scripts/01_eda.py` | `--section quality\|dist\|assoc\|psi\|all` 入口 |
| `tests/test_drift.py`、`tests/test_data_quality.py` | PSI 边界、日期解析、质量函数单测（共 9 个） |
| `reports/R1_data.md` | 环节小结（含全部数字、图、给 P2 的 7 条输入） |
| `reports/tables/*`、`reports/figs/*` | 17 张表、15 张图 |

`Makefile` 新增 `eda` 目标。

## 3. 怎么复现

```bash
cd /home/lll/workspace/big_data_analytics
make eda      # 约 2 分钟（750k 源域参与 PSI 计算）
make test     # 19 passed
```

单独跑某一节：`.venv/bin/python scripts/01_eda.py --section psi`。

## 4. P2 待办（按文件拆）

| 任务 | 文件 | 关键函数 | 验收 |
|---|---|---|---|
| 字段统一 | `src/feature_map.py` | `unify_domain(df, domain)`：按 yaml 统一 dtype、统一 `house_exist↔house_ownership`、`isDefault↔is_default` | 两域同名字段 dtype 一致 |
| 编码器 | `src/features.py` | `TargetEncoderCV`（K 折内拟合）、`FrequencyEncoder(max_levels)`、`OneHot`、`Ordinal` | 单测：验证折不参与统计量拟合 |
| 派生特征 | `src/features.py` | `add_ratios()`、`add_history_len()`、`add_interactions()`、`add_missing_flags()` | 每个新特征写入 `reports/tables/feat_audit.csv` |
| 去重 | `src/dedup.py` | MinHash+LSH（Jaccard≥0.9） | 输出重复簇 `tables/dup_clusters.csv` |
| 降维 | `src/reduce.py` | SVD/PCA 保留 90% 方差 | 两版数据集（原始 / 降维）供 E4 |
| 数据集落盘 | `src/prepare.py` | `build()` → `data/processed/{target_train,source_train,test}.parquet` | 行数与原表一致、无 ID 泄漏 |
| 单测 | `tests/test_features.py` | 目标编码泄漏、缺失指示、编码可逆 | 全绿 |

## 5. 已知坑

1. **字体**：matplotlib 需要中文字体。`src/plotting.py` 会自动把 `C:\Windows\Fonts\msyh.ttc` 复制到 `~/.local/share/fonts/` 再注册；新机器上第一次 `make eda` 会自动完成，无需手动。
2. **`earlies_credit_mon`**：目标域 46.7% 是 `DD-Mon`（无年份），解析后是 `NaT`；不要用 `pd.to_datetime` 直接解析（会把 `1-Dec` 解成 2021 年）。
3. **PSI 分箱**：当前用全量目标域分箱，P3 必须改成训练折内拟合，否则是隐性泄漏。
4. **稀有类别**：分组统计统一用 `min_count=30` 合并/剔除，否则会出现 0%/100% 的假象。
5. **类别字段编码**：`region`/`post_code`/`title` 两域编码体系不同，禁止按取值等值匹配；只能分别编码。
6. **数值列 id**：`loan_id`/`user_id` 是整数，做相关性/PSI 时必须显式排除（脚本已排除，新代码注意）。

## 6. 目录现状

```
reports/
├── R1_data.md          ✅
├── tables/             ✅ 17 个 csv
└── figs/               ✅ 15 个 png
src/
├── data_quality.py     ✅
├── eda.py              ✅
├── drift.py            ✅
├── dates.py            ✅
├── schema.py           ✅
└── （feature_map.py / features.py / dedup.py / reduce.py / prepare.py）  ⬜ P2
data/processed/         ⬜ 空（P2 产出）
```

> 变更记录：2026-11-26 初版（P1 完成）。