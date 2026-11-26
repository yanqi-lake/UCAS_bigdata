# 个贷违约预测 —— 课程项目任务计划（v1）

> 依据：《大数据分析》课程小组作业申请书（选题：面向新客群冷启动的信贷违约风险建模 / DataFountain #530）
> 覆盖申请书的四个环节（数据治理 → 特征工程 → 建模与评估 → 跨域迁移验证）与 E1—E4 对照实验
---

## 0. 现状盘点（已核实的事实，写在报告第一章用）

| 项目 | 现状 |
|---|---|
| 数据 | 已就位 `data/raw/`：train_public（10,000×39）、train_internet（750,000×42）、test_public（5,000×38）；`data/official/` 存官网原件，5 个文件 MD5 全部与官网一致 |
| 提交样例 | `data/submit_example.csv`，`id` 为测试集 `loan_id`，5000 行，与 test 行序一致 |
| 基线代码 | `data/baseline/0923`（初版）、`data/baseline/1117`（CatBoost 更新版）已解包 |
| 标签率 | 目标域 16.83%，源域 19.96% |
| 共有字段 | 35 个（含 `loan_id`、`user_id`）；仅目标域有 `known_outstanding_loan/known_dero/app_type`；仅源域有 `sub_class/work_type/house_loan_status/marriage/offsprings/f5` |
| 时间跨度 | 2007-06 ~ 2018-12，test 与训练同期（不是未来时间）→ 时间切分只能作"防高估"的辅助验证 |
| 环境 | WSL Ubuntu-24.04 / Python 3.12.3 / pandas 3.0.2；**缺** scikit-learn、lightgbm、xgboost、catboost、seaborn、joblib；Windows 侧 python 不可用 |
| 编码 | 三份 csv 均带 UTF-8 BOM → 统一用 `encoding="utf-8-sig"` |

**必须先纠正的一条（影响后续所有设计）**：`user_id` 不能用来关联两域。
源域 `user_id` 是 `0..749999` 的稠密编号（750,000 个用户各一条记录），而目标域 `user_id` 落在 `200,006 ~ 249,998`；两者"交集 100%"只是编号空间嵌套的假象，字段一致性检查（region 一致率 5.7%、post_code 0.3%）也说明不是同一人。`loan_id` 两域无交集。
→ 结论：**迁移只能走特征空间与样本加权，不能做 id 级 join**。这一条要在 P1 用脚本量化并写进报告，避免"伪关联"污染特征。

---

## 1. 目标与验收物（对齐申请书"预期成果"）

| 申请书成果 | 落地文件 | 验收标准 |
|---|---|---|
| ① 可复现的数据治理、特征与建模代码 | `src/` + `scripts/` + `tests/` + `REPRODUCE.md` | 一条命令从原始 csv 复现到 `submission.csv`；测试全绿 |
| ② 数据分布与相关性分析图表 | `reports/figs/*.png`（≥8 张）+ `reports/R1_data.md` | 缺失/违约率/相关性/互信息/PSI 五类齐备，图有标题与解读 |
| ③ E1—E4 指标对照表与迁移结论 | `results/runs.csv` + `reports/tables/E1_E4.csv` + `reports/R2_transfer.md` | 同一切分下 Group×Model×Fold 全记录，含均值±波动与显著性判断 |
| ④ 符合赛题格式的提交文件 | `submission.csv` | 5000 行、`id` 与 test 的 `loan_id` 完全一致、`isDefault` ∈ [0,1] |
| ⑤ 终期报告与答辩 PPT | `reports/final_report.docx`、`reports/slides.pptx` | 章节匹配申请书四个环节 + E1—E4 结论 |

**主指标**：目标域 AUC（分层 5 折均值 ± 标准差）。**辅指标**：KS、Recall@10%、AUC@时间切分。

---

## 2. 项目骨架（先搭空壳，再逐个填）

```
big_data_analytics/
├── applicationform.pdf
├── README.md                      # 项目总览：怎么跑、目录说明
├── requirements.txt               # 固定版本，见 P0
├── Makefile                       # make setup / make data / make exp / make report
├── configs/
│   ├── base.yaml                  # 路径、随机种子、折数、目标列
│   ├── feature_map.yaml           # 两域字段语义对齐表（P1 产出）
│   └── experiments/
│       ├── e1_target_only.yaml
│       ├── e2_direct_merge.yaml
│       ├── e3_weighted.yaml
│       └── e4_ablation.yaml
├── src/
│   ├── __init__.py
│   ├── config.py                  # 读 yaml → Config 对象，统一路径
│   ├── io.py                      # read_csv(utf-8-sig)、读写 parquet、提交文件写出
│   ├── data_quality.py            # 缺失/异常/重复/类别基数 + id 不可关联性验证（P1）
│   ├── eda.py                     # 分布、分层违约率、相关性、互信息（P1）
│   ├── drift.py                   # PSI / 域分类器 AUC（P1）
│   ├── feature_map.py             # 字段统一与取值编码映射（P2）
│   ├── features.py                # 目标编码/频次编码/派生变量/聚合（P2）
│   ├── dedup.py                   # MinHash+LSH 近重复检测（P2）
│   ├── reduce.py                  # PCA/SVD 降维对照（P2）
│   ├── prepare.py                 # 串起预处理 → data/processed/*.parquet（P2）
│   ├── splits.py                  # 分层 5 折 + 时间切分，落盘复用（P3）
│   ├── models.py                  # LR/DT/RF/LightGBM 统一接口（P3）
│   ├── metrics.py                 # AUC/KS/Recall@k/校准（P3）
│   ├── transfer.py                # 直接合并/相似度加权/预训练微调（P4）
│   └── experiment.py              # 实验注册、跑批、写 results/runs.csv（P4）
├── scripts/
│   ├── 00_verify_data.py          # MD5+行数+列数+提交格式校验（可立即写）
│   ├── 01_eda.py                  # 生成 reports/figs + tables
│   ├── 02_prepare.py              # 特征与数据集落盘
│   ├── 03_baseline.py             # E1
│   ├── 04_experiments.py          # E2/E3
│   ├── 05_ablation.py             # E4
│   ├── 06_submission.py           # 生成 submission.csv
│   └── 99_check_submission.py     # 提交前校验
├── tests/
│   ├── test_io.py
│   ├── test_features.py           # 目标编码无泄漏
│   └── test_submission.py
├── data/                          # 已有：raw/official/baseline
│   ├── processed/                 # parquet 中间件
│   └── splits/                    # folds.json / time_split.json
├── results/                       # runs.csv、各实验目录
└── reports/
    ├── figs/  tables/             # 图表
    ├── R1_data.md  R2_transfer.md # 环节小结
    ├── final_report.docx
    └── slides.pptx
```

---

## 3. 分阶段任务（含文件级验收）

### P0 环境与脚手架 —— 0.5 人日（组员 B）
| 任务 | 文件 | 验收 |
|---|---|---|
| 建 WSL venv 并锁版本 | `requirements.txt`（pandas 2.2.*、numpy<2.1、scikit-learn 1.5.*、lightgbm 4.3.*、scipy、matplotlib、seaborn、joblib、pyyaml、datasketch、pytest） | `pip install -r` 在 WSL 成功；`pytest` 可跑 |
| 配置与 IO 层 | `src/config.py`、`src/io.py` | 读三表统一走 `io.read_csv`；单测覆盖 BOM 列名 |
| 数据校验脚本 | `scripts/00_verify_data.py`、`tests/test_io.py` | 输出 5 个文件 MD5 + 行数/列数，与官网一致 |

> 注：官方 baseline 是 pandas 1.x 写法，本机 pandas 3.0 会报 `inplace`/`fillna` 相关错误；**venv 里固定 pandas 2.2，不复用系统解释器**，这样 baseline 可原样跑通作对照。

### P1 数据治理与探索性分析 —— 3 人日（组员 B 主，组员 C 配图）
| 任务 | 文件 | 验收标准 |
|---|---|---|
| 数据质量体检 | `src/data_quality.py` → `reports/tables/quality_overview.csv` | 缺失率、异常值（如 `interest`/`total_loan` 分位截断）、重复 `user_id`、类别基数全部成表 |
| id 关联性证伪 | 同上（`verify_id_linkage()`） | 输出 id 空间范围、字段一致率表，写进 R1：**不可 join** |
| 分布与分层违约率 | `src/eda.py` → `figs/dist_*.png`、`tables/default_rate_by_*.csv` | 至少 5 个关键字段（class、work_year、region、use、debt_loan_ratio 分箱）的分层违约率 |
| 关联强度筛选 | 同上 → `tables/mi_pearson_spearman.csv` | 相关系数（Pearson/Spearman）+ 互信息排名，标出共线对（|r|>0.9） |
| 域间漂移 PSI | `src/drift.py` → `tables/psi.csv`、`figs/psi.png` | 35 个共有字段 PSI 全表，按 <0.1/0.1~0.25/>0.25 分档 |
| 字段映射草案 | `configs/feature_map.yaml` | 每个同义字段给出统一后的类型与编码规则（`house_exist↔house_ownership`、`class`、`work_year`、`earlies_credit_mon` 等） |
| 环节小结 | `reports/R1_data.md` | 结论明确回答：哪些字段可迁移（PSI 低）、哪些是伪关联 |

### P2 特征工程 —— 4 人日（组员 B）
| 任务 | 文件 | 验收标准 |
|---|---|---|
| 两域字段统一 | `src/feature_map.py` + `configs/feature_map.yaml` | 同一语义字段合并后 dtype 一致；给出映射覆盖率 |
| 编码 | `src/features.py`：`TargetEncoderCV`（K 折内拟合，训练折外预测）、`FrequencyEncoder` | 单测断言：验证折统计量不进入训练折；无 NaN 泄漏 |
| 派生变量 | 同上：`月供/贷款额`、`债务收入比×近18月逾期`、`信用历史长度(issue_date - earlies_credit_mon)`、`early_return` 三项比率、借款人级聚合（源域按 user_id 聚合后按分布分箱） | 每个新特征在 `tables/feat_audit.csv` 里记录定义、缺失率、与标签的 AUC 单变量值 |
| 近重复去重 | `src/dedup.py`（MinHash+LSH，Jaccard≥0.9） | 输出重复簇清单 `tables/dup_clusters.csv` 与去重前后样本量差异 |
| 降维对照 | `src/reduce.py`（SVD/PCA 保留 90% 方差） | 产出 `processed/` 两版数据集（原始 vs 降维）供 E4 用 |
| 数据集落盘 | `src/prepare.py` → `data/processed/{target_train,source_train,test}.parquet` | 行数与原表一致；`processed/` 可被 P3/P4 直接读取 |
| 单测 | `tests/test_features.py` | 目标编码泄漏测试通过 |

### P3 建模与评估 —— 4 人日（组员 C）
| 任务 | 文件 | 验收标准 |
|---|---|---|
| 统一切分 | `src/splits.py` → `data/splits/folds.json`、`time_split.json` | StratifiedKFold(5, seed=2026)；时间切分按 `issue_date` 排序取后 20%，两文件落盘后全组复用 |
| 指标 | `src/metrics.py` | AUC、KS、Recall@10%、Brier/校准曲线四件套，单测覆盖 KS 计算 |
| 四模型 | `src/models.py` | LR（标准化+one-hot/目标编码）、DT（max_depth 网格）、RF（n_estimators=500）、LightGBM（早停）统一 `fit/predict_proba` |
| E1 基线跑批 | `scripts/03_baseline.py` → `results/e1/*`、`reports/tables/E1_baseline.csv` | 4 模型 × 5 折 × 2 种切分齐全，输出 均值±std |
| 图与解释 | `reports/figs/roc_e1.png`、`feat_importance_lgbm.png`、`calibration.png` | ROC 叠加图、Top20 重要性、校准曲线 |
| 提交链路 | `scripts/06_submission.py`、`scripts/99_check_submission.py`、`tests/test_submission.py` | 生成 `submission.csv` 并通过 5000 行/`id` 对齐/取值域三项校验 |

### P4 跨域迁移与 E1—E4 —— 5 人日（组长 A 主，组员 B 供数据处理）
| 任务 | 文件 | 验收标准 |
|---|---|---|
| 统一实验框架 | `src/experiment.py` | 实验名→配置→统一切分→写 `results/runs.csv`（一行为一次 Group×Model×Fold） |
| E2 直接融合 | `src/transfer.py: merge_all()` + `configs/experiments/e2_direct_merge.yaml` | 35 共有字段对齐后合并 760k 样本；记录训练耗时 |
| E3 相似度加权 | `transfer.py: weight_by_domain()`（域分类器输出概率→权重，或特征空间 KNN 密度比） | 权重分布图 + 有效样本量；对比加权前后 AUC |
| 预训练微调 | `transfer.py: pretrain_finetune()`（源域预训练 LightGBM → 目标域 `init_score`/低学习率续训） | 三策略同一折上可比 |
| E4 消融 | `scripts/05_ablation.py` | 依次去掉 派生特征 / 去重 / 降维 / 样本加权，四行结果可归因 |
| 结果汇总 | `reports/tables/E1_E4.csv`、`reports/R2_transfer.md` | 给出"迁移是否有效"的明确结论（含负迁移证据与阈值） |

### P5 汇总与答辩 —— 3 人日（全员）
| 任务 | 文件 | 验收 |
|---|---|---|
| 一键复现 | `Makefile`、`REPRODUCE.md` | 空目录起，`make setup && make all` 产出 submission 与全部表图 |
| 终期报告 | `reports/final_report.docx`（可用 office 技能生成） | 章节 = 申请书四环节 + 对照实验 + 结论 + 局限 |
| 答辩 PPT | `reports/slides.pptx` | 12~15 页，含 E1—E4 主表与一张迁移示意图 |

---

## 4. E1—E4 实验矩阵（统一在 P3 的切分文件上跑）

| 组 | 训练数据 | 模型 | 目的 | 输出 |
|---|---|---|---|---|
| E1 | 仅目标域 10k | LR / DT / LGBM | 目标域性能基线（RF 训练慢，作为补充） | `results/e1/` |
| E2 | 目标域 + 源域全部（35 共有字段对齐，约 760k） | LGBM（+LR 对照） | 检验样本扩充是否负迁移 | `results/e2/` |
| E3 | 目标域 + 源域按相似度加权（权重分 3 档做敏感性） | LGBM | 分布对齐的增益 | `results/e3/` |
| E4 | 在最优组上依次移除：派生特征→去重→降维→加权 | LGBM | 各环节独立贡献 | `results/e4/`，`reports/tables/E1_E4.csv` |

补充（申请书写明"其余条件保持一致"）：同折同种子（2026）、同早停轮数、同评估口径；源域 750k 训练耗时若超预算，先在 E3 上用分层抽样 200k 做预实验，再决定是否全量。

---

## 5. 防泄漏与可复现清单（每次提交前过一遍）

- [ ] 目标编码/缺失填充/标准化只在**训练折**拟合，验证折只做 transform
- [ ] 所有实验复用 `data/splits/*.json`，不各自 random_state
- [ ] 不读取 `test_public.csv` 的任何标签（本来也没有），不使用外部数据
- [ ] 特征筛选、PSI 分箱阈值在训练折内确定
- [ ] `results/runs.csv` 每次运行追加，含 git commit、参数、耗时
- [ ] 提交前跑 `scripts/99_check_submission.py`

---

## 6. 排期（3 周，按 15 个工作日）

| 周 | 主要交付 | 里程碑 |
|---|---|---|
| W1 | P0 + P1 全部 + P2 字段统一与编码 | 周三：`make data` 跑通；周五：R1 完成，PSI 表定下可迁移字段 |
| W2 | P2 剩余 + P3 全部 | 周三：E1 四模型出分；周五：`submission.csv` 首次生成并校验通过 |
| W3 | P4 + P5 | 周二：E1—E4 主表定稿；周四：报告 + PPT 初稿；周五：复现验收 |

若课程截止时间更早，可压缩顺序为：P0 → P1（只做 PSI 与缺失）→ P3（E1）→ P4（E2/E3）→ 补 P2 + P5。

---

## 7. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| pandas 3.0 与官方 baseline 不兼容 | 基线跑不动 | venv 固定 pandas 2.2；或按 pandas 3 重写那两处 `inplace` 调用 |
| 源域 750k 训练慢 | 实验轮次受限 | parquet 缓存 + LightGBM `max_bin=127`；先 200k 子采样预实验 |
| 目标编码高基数泄漏 | 虚高 AUC | K 折内拟合 + 单测 + 与频次编码对照 |
| 负迁移 | E2 低于 E1 | 以 E3/E4 结论为准，报告如实呈现，不强行合并 |
| 时间切分样本少（约 2k） | 波动大 | 报告 std 与重复 5 次不同时间窗；主结论仍以分层 5 折为准 |
| 数据规模/内存 | 750k×42 尚可 | 全程 float32 + 只用共有字段；必要时按 user 分层抽样 |

---

## 8. 现在就能动手的第一步（P0，约 2 小时）

1. 建 venv 与依赖：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
2. 写 `src/io.py` + `src/config.py`（路径常量、`read_csv` 统一 `utf-8-sig`）
3. 写 `scripts/00_verify_data.py`：5 个 MD5 + 三表行列数 + 提交样例与 test 的 `loan_id` 对齐检查（现有 md 里的结论代码化）
4. 写 `tests/test_io.py`、`tests/test_submission.py` 空跑通过
5. 用 `data/raw` 跑一次 `python scripts/00_verify_data.py`，把输出贴进 `reports/R1_data.md` 开头

完成后即可进入 P1，把"id 不可关联"的证伪脚本固化下来。