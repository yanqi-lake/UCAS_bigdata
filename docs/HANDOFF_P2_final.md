# P2 交接说明（特征工程）

> 阶段记录。项目总览见 [HANDOFF.md](HANDOFF.md)，前序见 [HANDOFF_P1.md](HANDOFF_P1.md)。
> 交付日期：2026-11-27　执行：组员 B　下一棒：组员 C（P3）/ 组长 A（P4）
> 对应计划：`docs/PLAN.md` P2　状态：**✅ P2 CSV 核心流程与五折/时间切分验收完成；原仓库完整回归与 Parquet 未验收**
> 成果报告：`reports/R2_features.md`；验收：`reports/tables/p2_acceptance.csv`。

## 1. 一句话结论

11 个 P2 单测通过；本轮 14 组真实数据矩阵回读通过。五折 AUC（LR 接口测试）0.8742±0.0097，时间切分 AUC 0.8493；全量融合上次计算通过，训练 757,999×69、PCA 28维（90.42%）。本次全量源域已完成 CSV 特征矩阵、元数据及降维矩阵写出和回读比对。 目标域各折 85–86 维、PCA 31–32 维，时间切分 85→30 维。可交接 P3/P4 数据接口，不需人工填写报告。

## 2. 本次交付的文件

| 文件 | 作用 |
|---|---|
| `src/feature_map.py` | 字段映射、dtype、标签统一 |
| `src/features.py` | 折内目标/频次/序数/One-hot 编码、派生特征、聚合、缺失/异常处理 |
| `src/dedup.py` | MinHash + LSH 同域训练折内近重复检测 |
| `src/reduce.py` | PCA/SVD 90% 方差降维 |
| `src/prepare.py` | 对齐、共享切分、特征流水线、审计与数据落盘 |
| `src/experiment.py` | AUC/KS/Recall@10%、样本权重与实验记录接口 |
| `scripts/02_prepare.py` | 特征工程 CLI |
| `scripts/p2_validate.py` | 真实数据验收、回读、消融、模型接口验证 |
| `tests/test_features.py` | 11 项 P2 单测 |
| `reports/R2_features.md` | 完整实测结论 |
| `reports/tables/` | P2 汇总、特征审计、PSI、去重及映射覆盖率 |
| `reports/p2_validation/` | 原始验收日志、各实验匿名统计及文件校验哈希 |
| `docs/P2_TEST_LOG.txt` | 完整测试记录（实际输出） |
| `requirements-p2.txt` | 可选 Parquet 依赖；CSV 不需安装 |

## 3. 怎么复现

项目根目录（WSL，使用原 `.venv`，三表放 `data/raw/`）：

```bash
make setup  # 原环境已有可跳过
.venv/bin/python -m pytest tests/test_features.py -q
.venv/bin/python scripts/02_prepare.py --fold all --format csv --run-name p2_target_all
.venv/bin/python scripts/p2_validate.py --format csv --existing-target data/processed/p2_target_all --full
```

单跑全量融合：

```bash
.venv/bin/python scripts/02_prepare.py --scope common --with-source --include-domain --fold 0 --format csv --run-name e2_common
```

`--no-derived` / `--no-dedup` / `--reduction none|pca|svd` / `--exclude-early` 为消融开关。不同配置使用不同 run-name。Parquet 是原计划格式；安装 `requirements-p2.txt` 后改为 `--format parquet`。当前环境不能安装该引擎，故本次实际使用 CSV。

| 验收 | 结果 |
|---|---|
| 数据映射覆盖率 | 三表 100%，共有字段 35/35 dtype 一致（100行抽样检查） |
| P2 单测 | 11 passed |
| CLI 五折 + 时间 | 6/6，写出和回读成功 |
| 派生/去重/降维/提前还款开关 | 通过回读 |
| 全量融合 | 本次 CSV 回读通过 |
| 原仓库完整回归、Parquet | 环境受限，未验收；不冒充通过 |

## 4. P3/P4 待办（按角色拆）

| 任务 | 负责人 | 接口与验收 |
|---|---|---|
| P3 四模型 | C | 共用 `data/splits/`，读取 `X_*` 与 `meta_*`，输出 AUC/KS/Recall 五折及时间切分结果 |
| P4 E2/E3 | A | `--scope common --with-source`，使用统一切分、相同模型与参数，样本加权仅用训练数据 |
| P4 E4 | A/C | `--no-derived`、`--no-dedup`、`--reduction none`，单因素比较 |
| 原仓库合并 | B/A | 合入 P2 后运行 `make test`；如需 Parquet 按上述命令复现，记录真实结果 |

标签在 `meta_train/val.isDefault`，测试 ID 从 `meta_test.loan_id` 读取；ID 和标签不进 X。全组共享 seed=2026 五折索引；时间截止 2016-10-01（7953/2047），时间融合仅使用截止日前源域。

## 5. 已知坑

1. 三表各自用户 ID 唯一，不可跨域 join；借款人聚合在当前数据为常数，不能声称有效增益。
2. 目标域最早信用日期 4,672 条训练、2,285 条测试无年份，保留 NaT 与缺失标识，不猜测年份。
3. 目标编码 OOF，外层训练折拟合、验证/测试只转换；时间验证内部 OOF 为随机分层折，并非严格逐时编码。
4. 提前还款可能涉及贷后信息，排除后 LR AUC 0.7159；正式预测先明确预测时点。
5. MinHash/LSH 存在截断桶，0 删除不等于无重复。跨域类别映射、特征空间与正式迁移对照口径须一致。
6. 原仓库 Git clone 失败、Parquet 引擎未安装；实际运行使用 P1 兼容依赖重建，原仓库完整回归不在本次通过范围。不要覆盖已有 P1 文件。
7. 原始 CSV、带 ID 矩阵及重复对不进 Git；结果表为匿名统计。

## 6. 目录现状

```text
src/                   ✅ 六个 P2 模块
scripts/02_prepare.py  ✅
scripts/p2_validate.py ✅
tests/test_features.py ✅ 11 passed
docs/HANDOFF_P2.md     ✅ 本文件
docs/P2_TEST_LOG.txt   ✅ 实际日志
reports/R2_features.md ✅ 全部实验结果已填
reports/tables/        ✅ P2 匿名统计表
reports/p2_validation/ ✅ 本轮及前轮验收证据
```

> 变更记录：2026-11-27 最终交付（五折、时间切分、消融及 CSV 回读实测；未宣称原仓库集成、Parquet 验收）。
