# 课程模型实验交接（待合并的独立补充）

本目录记录 `train_public.csv` 上的目标域课程模型实验，**不替代**仓库原有的 `docs/HANDOFF.md`、P2 特征工程和 E1–E4 跨域迁移计划。原有仓库的 P0/P1 状态不因此改变。

## 研究口径

- 原始数据：DataFountain 2021 CCF BDCI 个贷违约预测 `train_public.csv`，10,000 行、39 列。SHA256：`797ee36cf3e610413c015c763a37582121a2fb5cb2c4d22ff8fa48524693e0e7`。
- 切分：`scripts/02_make_splits.py` 用种子 2026 生成分层开发集 8,000 条、留出集 2,000 条，并在开发集内划分五折。`data/splits/` 不进 Git，需从相同原始 CSV 重新生成。
- 每折的缺失填补、标准化、独热编码和 PCA 均只在该折训练样本上拟合。
- 为研究贷款发放前的风险预测，最终方案排除三个预测时点可用性未确认的字段：`early_return`、`early_return_amount`、`early_return_amount_3mon`。排除是保守建模选择，不代表已证明这些字段泄漏。
- 本实验只用目标域；`train_internet.csv` 的迁移学习工作仍按仓库原计划推进。

## 实验链与结果

| 入口 | 用途 | 关键结果 |
| --- | --- | --- |
| `03_compare_course_models.py` | 全部特征的逻辑回归、决策树、随机森林五折基线 | AUC 0.8588 / 0.8423 / 0.8757 |
| `04_threshold_analysis.py` | 全部特征阈值探索 | 只作开发集探索，不作为最终成绩 |
| `05_eda_correlations.py` | 缺失、标签、相关性、异常值统计 | 开发集违约率 16.825% |
| `06_early_return_sensitivity.py` | 排除三个还款字段的敏感性 | AUC 0.6829 / 0.6135 / 0.6848 |
| `07_pca_comparison.py` | 数值 PCA 95% 与不降维对照 | 24→15 维、平均保留 95.11% 方差；PCA 后 AUC 0.6815 / 0.6317 / 0.6793 |
| `08_select_reduced_model_threshold.py` | 开发集固定模型与阈值 | 不使用 PCA 的随机森林，阈值 0.19 |
| `09_final_holdout.py` | 一次性留出集评价 | AUC 0.6794、AP 0.3018、F1 0.3733、Precision 0.2716、Recall 0.5964 |
| `10_plot_final_assets.py` | 用已保存结果绘图 | 不重新训练、不重新评价 |

最终混淆矩阵 TN 1124、FP 539、FN 136、TP 201。开发集五折最终方案 AUC 0.6848、固定阈值 0.19 的 F1 0.3692。详细表格在 `results/`，论文素材在 `reports/experiment_notes_20261126T185621Z.md`，本批次独立脚本生成的图在 `figures/`（原仓库 P1 图仍在 `reports/figs/`）。图表和结果文件名中的时间戳为 UTC。

## 复现顺序

1. 将原始 `train_public.csv` 放到 `data/raw/`（脚本也兼容 `data/`）。先运行 `02_make_splits.py`；其余脚本会校验数据 SHA256。
2. 按编号运行 `03` 至 `08`。`08` 从最新的 PCA 结果中选择模型和阈值；如仓库中已有其他 PCA 试验，请明确指定本批次 `20261126T184807Z` 的文件，避免误取。
3. `09` 引用本批次固定决策文件 `reduced_model_decision_20261126T185314Z.json`。已有最终结果时普通运行会直接说明并退出，不重新评价；若要复核同一固定方案，可显式运行 `python scripts/09_final_holdout.py --verify-existing`，它只比较已有指标和混淆矩阵，不写新结果。**已有留出集成绩，不应以新一轮调参覆盖它。**
4. `10` 仅依据保存的最终结果生成图表；图已存在时退出而不覆盖。

脚本使用 pandas、numpy、scikit-learn、scipy、Pillow；`04` 另用 matplotlib。实际运行记录：Python 3.12.14、pandas 3.0.1、numpy 2.3.5、scikit-learn 1.9.1、scipy 1.16.3。原仓库 WSL 环境可能使用不同版本，合并前应在该环境运行语法检查和必要的复核；不要把上述版本直接覆盖仓库统一依赖。

## 上传边界与待办

- `notebooks/shaky_reproduction/lgb_v1.py` 是本地官方 Baseline 的逐字节副本；`config.py` 是项目相对路径适配版。环境包记录见 `results/reproduction_packages.txt`，复现过程的已知事实与未核实项见 `docs/REPRODUCTION_LOG.md`。没有可核验的 Baseline 指标日志，不能将课程模型的成绩写成该 Baseline 的复现成绩。
- 本批次仅上传脚本、汇总/逐折指标、无逐条客户记录的元数据、图表和交接文档。**不上传**原始 CSV、压缩包、虚拟环境、切分文件中包含的贷款 ID、OOF/留出集逐条预测。
- 原有 `docs/HANDOFF.md` 应增加指向本文件的链接；避免覆盖既有 P0/P1 交接。本批次图表使用 `figures/` 以保持独立脚本可直接运行，仓库原有的 `reports/figs/`、`configs/`、`src/` 结构及 E1–E4 工作保持原计划。
- 网上 LightGBM Baseline 的复现与本课程模型实验分开记录。`docs/REPRODUCTION_LOG.md` 仍有来源版本、改动和运行指标待核实，不能把空白项当作已完成。
- 后续任务：确认预测时点与三个还款字段的业务定义；撰写课程论文；继续原仓库的跨域迁移实验。留出集已评价，不再据此调参。
