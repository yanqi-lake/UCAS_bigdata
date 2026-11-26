# P0 交接说明（环境与脚手架）

> 交付日期：2026-11-26　执行：组长 A　下一棒：组员 B（P1 数据治理与 EDA）
> 对应计划：`docs/PLAN.md` 第 3 节 P0　状态：**✅ 已完成，P1 可以开工**

---

## 1. 一句话结论

WSL 里已建好可复现的 `.venv`（25 个包，版本锁定），项目骨架、统一读写层、提交校验、数据自检脚本与 10 个单测全部跑通：
`make verify` 输出「全部通过」，`make test` 输出 `10 passed`。

---

## 2. 本次交付的文件

| 文件 | 作用 | 状态 |
|---|---|---|
| `requirements.txt` | 依赖锁定（pandas 2.2.3 / numpy 1.26.4 / sklearn 1.5.2 / lightgbm 4.5.0 …） | ✅ |
| `Makefile` | `setup / verify / test / clean` 四个入口 | ✅ 已实测 |
| `pytest.ini` | `pythonpath = .`，测试可直接 import `src` | ✅ |
| `configs/base.yaml` | 全局配置：种子 2026、5 折、列名、目录 | ✅ |
| `src/config.py` | 路径常量、`TABLES`、官网 MD5、期望形状、`load_config()` | ✅ |
| `src/io.py` | `read_csv`（强制 utf-8-sig）、`read_table`、`load_competition_data`、`file_md5`、`make_submission`、parquet 读写 | ✅ |
| `src/validation.py` | `check_shapes` / `check_test_columns` / `check_submit_example` / `check_submission` | ✅ |
| `scripts/00_verify_data.py` | 数据自检入口，退出码可用于 CI；结果写 `reports/tables/data_verification.csv` | ✅ 已跑通 |
| `tests/test_io.py`（5 个） | BOM 处理、表名分派、提交文件往返、行数不一致报错、MD5 | ✅ |
| `tests/test_submission.py`（5 个） | 正常提交 + 四类异常（列名/行数/id 顺序/取值越界） | ✅ |
| `README.md`（根） | 项目总览 + 快速开始 + 数据约定 | ✅ |
| `.gitignore` | 忽略 `.venv`、原始 csv、官方 zip、parquet/pkl | ✅ |
| `docs/PLAN.md` | 完整任务计划（P0—P5） | ✅ 上一轮已交付 |

**已生成的数据产物**：`reports/tables/data_verification.csv`（自检留痕）。
**空目录已就位**（等后续阶段填）：`configs/experiments/`、`data/processed/`、`data/splits/`、`results/`、`reports/figs/`。

---

## 3. 环境怎么用

```bash
# 所有命令都在 WSL 里、项目根目录执行
cd /home/lll/workspace/big_data_analytics

make verify    # 数据自检
make test      # 单测
source .venv/bin/activate   # 需要交互式跑 notebook / python 时
```

- 解释器：`.venv/bin/python`（Python 3.12.3）。**不要用系统 python**（pandas 3.0.2，会让官方 baseline 的 `inplace` 写法报错）。
- Windows 侧 `python` 是应用商店占位程序（退出码 9009），不可用；一切在 WSL 里跑。
- 需要装新包时用清华镜像（已实测 2.8MB/s）：
  ```bash
  .venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
  ```
  装完请把版本补进 `requirements.txt`，保持可复现。

已装版本（`pip freeze` 摘要）：pandas 2.2.3、numpy 1.26.4、scipy 1.13.1、scikit-learn 1.5.2、lightgbm 4.5.0、matplotlib 3.9.2、seaborn 0.13.2、joblib 1.4.2、PyYAML 6.0.2、datasketch 1.6.5、pytest 8.3.3。

---

## 4. P0 的验收证据（可直接贴进报告附录）

`make verify` 实测输出要点：

- 官网 5 个原件 MD5 全部 PASS（train_dataset.zip `192D3D30…`、test_public `D4A2AB57…`、submit_example `5F272DFE…`、两个 baseline zip）
- 三表规模 (10000, 39) / (750000, 42) / (5000, 38) 全对
- `test_public` 列 = `train_public` 去标签列，且顺序一致
- `submit_example` 的 5000 个 `id` 与测试集 `loan_id` **同序**
- 解包 csv 的 MD5：`train_public 3F7A7170…`、`train_internet 0BE09D53…`、`test_public D4A2AB57…`

`make test`：`10 passed in 0.83s`。

---

## 5. P1 待办（下一阶段，按文件拆到函数）

> 目标：3 人日内产出 `reports/R1_data.md` + 图表，并定下"可迁移字段清单"。
> 原则：脚本能重跑，结论落文字，图有标题与解读。

### 5.1 数据质量体检 —— `src/data_quality.py`（0.5 天）

| 函数 | 输入 → 输出 | 说明 |
|---|---|---|
| `missing_summary(df)` | → DataFrame（列名/缺失数/缺失率） | 三表各跑一次，合并成 `reports/tables/quality_overview.csv` |
| `outlier_ranges(df, cols)` | → 分位 1%/99% 截断表 | 重点看 `interest`、`total_loan`、`monthly_payment`、`debt_loan_ratio` |
| `category_cardinality(df, cols)` | → 基数表 | `region`、`post_code`、`title`、`industry` 高基数需在 P2 用目标/频次编码 |
| `duplicate_users(df)` | → 重复 `user_id` 计数 | 已知目标域 0 重复，写进表即可 |
| `verify_id_linkage(train_public, train_internet, test)` | → id 空间与一致率表 | **固化"user_id 不可 join"的证伪证据**（源域 id 是 0..749999 稠密编号；字段一致率 region 5.7%、post_code 0.3%） |

### 5.2 分布与分层违约率 —— `src/eda.py`（0.75 天）

- `default_rate_by(df, col, bins=None)` → 表 + 条形图；至少覆盖 `class`、`work_year`、`region`（Top10）、`use`、`debt_loan_ratio`（分箱）、`issue_date`（按年）
- `plot_missing_heatmap(df)` → `reports/figs/missing_heatmap.png`
- 产出：`reports/tables/default_rate_by_*.csv` + `reports/figs/dist_*.png`（≥5 张）

### 5.3 关联强度 —— `src/eda.py`（0.5 天）

- `association_table(df, target)`：数值列做 Pearson / Spearman，类别列做互信息（`sklearn.feature_selection.mutual_info_classif`）
- 标出 |r| > 0.9 的共线对（如 `scoring_low/high`、`f0~f4`）
- 产出：`reports/tables/association.csv`、`reports/figs/corr_heatmap.png`

### 5.4 域间漂移 PSI —— `src/drift.py`（0.5 天）

- `psi(expected, actual, bins=10)`：分箱用**训练折**的边界，防止信息泄漏
- `psi_table(train_public, train_internet, cols)`：对 35 个共有字段逐列计算
- 产出：`reports/tables/psi.csv`（按 <0.1 / 0.1–0.25 / >0.25 分档）、`reports/figs/psi_bar.png`
- 结论要回答：**哪些字段可迁移**（PSI 低）、哪些要丢弃或重编码

### 5.5 字段映射草案 —— `configs/feature_map.yaml`（0.25 天）

- 35 个共有字段：统一类型（数值/类别/序数/日期）、缺失填充策略、编码方式（目标编码/频次编码/one-hot）
- 注意两域编码体系不同：`region`、`post_code`、`title` 的取值不是同一套字典，**不能按等值匹配**，只能分别编码后建模
- 同义字段映射：`house_exist ↔ house_ownership`、`class`、`work_year`、`earlies_credit_mon`（含 2021 占位日期问题）、`employer_type`、`industry`

### 5.6 环节小结 —— `reports/R1_data.md`（0.5 天）

用 4 段写：数据规模与质量 → 分布与违约率 → 关联强度 → PSI 与可迁移字段；附 3 张表 + 5 张图；末尾给 3 条给 P2 的明确输入。

### 5.7 建议新增的入口脚本

- `scripts/01_eda.py --section quality|dist|assoc|psi`，并在 `Makefile` 加 `eda:` 目标

### 5.8 P1 完成定义（DoD）

- [ ] `make eda` 一条命令生成全部表与图
- [ ] `reports/R1_data.md` 有结论、有图、有对 P2 的输入
- [ ] `configs/feature_map.yaml` 覆盖全部 35 个共有字段
- [ ] 单测补 `tests/test_drift.py`（PSI 边界：分布相同→0、完全错位→1）

---

## 6. 已知坑与约定（踩过的不再踩）

1. **BOM**：三份 csv 带 UTF-8 BOM，必须 `encoding="utf-8-sig"`；已封装在 `src.io.read_csv`，不要绕过它。
2. **pandas 版本**：系统解释器是 3.0.2，官方 baseline 会挂；项目固定 2.2.3，新代码别用 3.0 独有 API（`pd.options.future.infer_string` 等）。
3. **写文件到 `\\wsl.localhost` 共享路径**：某些原子写（临时文件 + rename）会报 `ENOTSUP`。用编辑器直接保存、或在 WSL 内用 Python 写即可；生成产物尽量走脚本。
4. **`user_id` 伪关联**：源域 750k 用户的 id 是 0..749999 稠密编号，与目标域 20 万段"重合"纯属编号嵌套；**禁止 join**，否则等于给测试集塞噪声。
5. **内存**：750k×42 约 0.5–1GB，脚本里避免 `df.copy()` 满天飞；中间产物存 parquet + float32。
6. **时间切分**：三表时间范围重叠（2007–2018），时间切分只能"防高估"，不能代表真实外推；主结论仍用分层 5 折。
7. **提交 id**：`id` = 测试集 `loan_id`（**不是行号**），已封装进 `io.make_submission`。
8. **切分文件**：P3 起所有实验共用 `data/splits/*.json`，禁止各自 `random_state`；P1 暂时不用切分，但 PSI 分箱边界要按训练折算。

---

## 7. 目录现状（交接时的真实状态）

```
big_data_analytics/
├── applicationform.pdf       申请书
├── README.md                 项目总览（已写）
├── requirements.txt          ✅
├── Makefile                  ✅ 实测
├── pytest.ini                ✅
├── .venv/                    ✅ 25 个包
├── src/                      ✅ config / io / validation（+__init__）
├── scripts/                  ✅ 00_verify_data.py
├── tests/                    ✅ 10 个用例
├── configs/
│   ├── base.yaml             ✅
│   └── experiments/          ⬜ 空（P4 用）
├── data/
│   ├── raw/                  ✅ 三表
│   ├── official/             ✅ 5 个官网原件
│   ├── baseline/             ✅ 0923 / 1117 两版官方代码
│   ├── processed/            ⬜ 空（P2 用）
│   └── splits/               ⬜ 空（P3 用）
├── results/                  ⬜ 空（P3/P4 用）
├── reports/
│   ├── figs/                 ⬜ 空（P1 开始填）
│   ├── tables/               ✅ data_verification.csv，其余待生成
│   └── R1_data.md            ⬜ P1 交付
└── docs/
    ├── PLAN.md               ✅
    └── HANDOFF_P0.md         ✅ 本文件
```

---

## 8. 下一步（按顺序）

1. 读 `docs/PLAN.md` 第 0 节与第 3 节 P1（10 分钟）
2. 跑 `make verify`，打开 `reports/tables/data_verification.csv` 看一眼字段（5 分钟）
3. 按第 5.1 节写 `src/data_quality.py`，先交 `verify_id_linkage()`（这块是报告里最容易被追问的点）
4. 依次做 5.2–5.6，最后补 `Makefile` 的 `eda:` 目标

> 变更记录：2026-11-26 初版（P0 完成）。