# 交接文档（当前状态：P0 + P1 已完成）

> 仓库：`git@github.com:yanqi-lake/UCAS_bigdata.git`（原 `UCAS--`，已改名，remote 已同步）
> 最近提交：`48f92af` P1 数据治理与 EDA　存档标签：`checkpoint-p1`
> 更新日期：2026-11-26　下一阶段：P2 特征工程

---

## 1. 项目是什么

复现并改进 CCF BDCI 2021 中原银行「个贷违约预测」赛题（DataFountain #530）：用源域「网络信用贷」750k 条记录，
辅助目标域「个人贷款」10k 条记录的违约概率预测，**指标 AUC**。课程要求四个环节 + E1—E4 对照实验，见 `docs/PLAN.md`。

组内以角色代称：组长 A（迁移与实验设计）、组员 B（数据与特征、工程框架）、组员 C（建模与评估）。

## 2. 现在到哪了

| 阶段 | 内容 | 状态 | 依据 |
|---|---|---|---|
| P0 | 环境、项目骨架、数据自检、单测 | ✅ | `docs/HANDOFF_P0.md`、`make verify` |
| P1 | 数据治理、EDA、关联强度、PSI、字段映射 | ✅ | `reports/R1_data.md`、`make eda` |
| P2 | 特征工程、去重、降维、数据集落盘 | ⬜ 下一步 | 本文第 7 节 |
| P3 | 建模与评估（E1） | ⬜ | `docs/PLAN.md` |
| P4 | 跨域迁移（E2—E4） | ⬜ | `docs/PLAN.md` |
| P5 | 报告与答辩 | ⬜ | `docs/PLAN.md` |

```text
big_data_analytics/
├── docs/            PLAN.md、HANDOFF.md（本文件）、HANDOFF_P0.md、HANDOFF_P1.md
├── configs/         base.yaml、feature_map.yaml（46 字段的类型/编码/可迁移性）
├── src/             config、io、validation、schema、data_quality、eda、drift、dates、plotting
├── scripts/         00_verify_data.py、01_eda.py
├── tests/           19 个用例（pytest）
├── data/            raw（三表）、official（官网原件）、baseline（官方代码）、processed⬜、splits⬜
├── reports/         R1_data.md、tables/（17 个 csv）、figs/（15 个 png）
├── results/         ⬜ 空（P3/P4 实验记录）
└── Makefile         setup / verify / eda / test / clean
```

## 3. 怎么跑（WSL，项目根目录）

```bash
make setup     # 建 .venv 并装依赖（首次约 2 分钟，清华镜像）
make verify    # 数据自检：5 个官网 MD5 + 三表行列数 + 提交样例对齐
make eda       # 生成全部表格与图（约 15 秒）
make test      # 单测
```

- 解释器固定 `.venv/bin/python`（Python 3.12 + pandas 2.2.3）。**不要用系统 python**（pandas 3.0，官方 baseline 会报错）。
- Windows 侧 `python` 是应用商店占位程序，不可用。
- 装新包：`.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>`，并回填 `requirements.txt`。

## 4. 数据（不进 git，需组内共享）

| 表 | 规模 | 标签 | 标签率 |
|---|---|---|---|
| `data/raw/train_public.csv` | 10,000 × 39 | `isDefault` | 16.83% |
| `data/raw/train_internet.csv` | 750,000 × 42 | `is_default` | 19.96% |
| `data/raw/test_public.csv` | 5,000 × 38 | — | — |

- 官网 5 个原件在 `data/official/`，MD5 与官网一致（`make verify` 会核）。
- **三表带 UTF-8 BOM**，统一走 `src.io.read_csv`（`encoding="utf-8-sig"`）。
- **提交文件**：`id` 取测试集 `loan_id`（不是行号），`isDefault` 填概率；样例见 `data/submit_example.csv`。

## 5. P1 已交付的关键结论（细节见 `reports/R1_data.md`）

1. `user_id` **不可关联**：源域是 0..749999 稠密编号，20 万段的"重合"是编号嵌套；同 id 上 27/33 字段一致率 <50%。迁移只能走特征空间与样本加权。
2. 域间漂移集中在 6 个字段：`earlies_credit_year` 7.10、`scoring_low` 5.31、`scoring_high` 4.90、`early_return_amount` 2.85、`early_return` 1.71、`early_return_amount_3mon` 0.78；其余 27 个共有字段 PSI < 0.1。
3. 目标域 `earlies_credit_mon` 有 46.7% 是 `1-Dec` 这类无年份格式，解析后记 NaT；源域是 `Mar-1984`。
4. 信号偏弱且分散：互信息最高 0.123（`early_return_amount`，负相关）；`class` 分层违约率 5.9%→37.7% 单调。
5. `total_loan` 与 `monthly_payment` 共线 r=0.925，建议改比值。

## 6. 三条纪律

1. **切分文件共享**：P3 起所有实验复用 `data/splits/*.json`，禁止各自 `random_state`。
2. **防泄漏**：目标编码、分箱边界、标准化参数只在训练折拟合（PSI 当前用全量目标域，P3 必须改）。
3. **产物归位**：脚本进 `scripts/`，模块进 `src/`，表格进 `reports/tables/`，图进 `reports/figs/`，实验记录进 `results/`。

## 7. 下一步：P2 特征工程（按文件拆）

| 顺序 | 文件 | 关键函数 | 验收 |
|---|---|---|---|
| 1 | `src/feature_map.py` | `unify_domain(df, domain)`：按 `configs/feature_map.yaml` 统一 dtype，`house_exist↔house_ownership`、`isDefault↔is_default` | 两域同名字段 dtype 一致，单测覆盖 |
| 2 | `src/features.py` | `TargetEncoderCV(k=5)`（折内拟合）、`FrequencyEncoder(max_levels)`、`OrdinalEncoderMap`、`OneHotMap` | 单测：验证折不参与统计量拟合 |
| 3 | `src/features.py` | `add_ratios()`：`monthly_payment/total_loan`、`early_return_amount/monthly_payment`；`add_history_len()`：`issue_date-earlies_credit_mon`（53.3% 可算 + 缺失指示）；`add_interactions()`：`debt_loan_ratio×del_in_18month`；`add_missing_flags()` | 每个新特征进 `reports/tables/feat_audit.csv`（定义/缺失率/单变量 AUC） |
| 4 | `src/dedup.py` | MinHash+LSH（Jaccard≥0.9，`datasketch` 已装） | 输出 `tables/dup_clusters.csv` 与去重前后样本量 |
| 5 | `src/reduce.py` | SVD/PCA 保留 90% 方差 | 两版数据集供 E4 消融 |
| 6 | `src/prepare.py` | `build()` → `data/processed/{target_train,source_train,test}.parquet` | 行数与原表一致；不做全表目标编码 |
| 7 | `tests/test_features.py` | 泄漏、缺失指示、编码可逆 | `make test` 全绿 |
| 8 | `reports/R2_features.md` | 特征清单、编码对照、去重/降维结果 | 有数字、有表 |

P2 的字段处理规则（P1 已定，直接执行）：

- 直接迁移：27 个 PSI<0.1 的共有字段；
- 域内变换：`early_return*` 三项 → `log1p` + 域内分位秩 + `is_source` 指示；
- 不直接迁移：`scoring_low/high`、`earlies_credit_mon`（除非域内标准化 + 缺失指示）；
- 编码：`title`/`post_code` 频次（Top-100+OTHER）；`class`/`work_year`/`offsprings` 序数；`industry`/`use`/`employer_type`/`region` 目标编码；`house_exist`/`censor_status`/`initial_list_status`/`policy_code`/`app_type` one-hot。

## 8. 已知坑（踩过的别再踩）

1. **中文字体**：`src/plotting.py` 会自动把 `C:\Windows\Fonts\msyh.ttc` 复制到 `~/.local/share/fonts/`；新机器首次 `make eda` 自动完成。
2. **BOM**：直接用 `pd.read_csv` 会让第一列变成 `\ufeffloan_id`。
3. **pandas 3.0**：官方 baseline 用 1.x 写法，必须跑在 `.venv`（2.2.3）。
4. **`\\wsl.localhost` 写文件**：编辑器/工具的原子写（临时文件+rename）可能报 `ENOTSUP`，用脚本或 WSL 内写入。
5. **PSI 分箱**：当前用全量目标域，P3 必须训练折内拟合。
6. **稀有类别**：分组统计统一 `min_count=30`，否则出现 0%/100% 假象。
7. **数值 id**：`loan_id`/`user_id` 是整数，做相关/PSI 必须显式排除。
8. **仓库已改名**：`UCAS--` → `UCAS_bigdata`，老链接会重定向，remote 已改好。

## 9. 协作约定

- 主分支 `main`，功能分支 `p2-features` 之类；提交信息用「阶段: 做了什么」。
- 不提交：原始 csv、`.venv`、`data/processed`、`data/splits`、`*.parquet`（`.gitignore` 已排除）。
- 文档里用角色代称（组长 A / 组员 B / 组员 C），不写真实姓名。

## 10. 存档

- 分支 `main` 最新提交 `48f92af`（P1 完成）；
- 标签 `checkpoint-p1` 指向同一提交，可 `git checkout checkpoint-p1` 取回本状态；
- 阶段交接细节：`docs/HANDOFF_P0.md`（环境/骨架）、`docs/HANDOFF_P1.md`（数据治理）。