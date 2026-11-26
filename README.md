# 个贷违约预测 —— 跨域迁移与冷启动建模

《大数据分析》课程小组作业。复现并改进 CCF BDCI 2021 中原银行「个贷违约预测」赛题（DataFountain
[#530](https://www.datafountain.cn/competitions/530)）：用源域「网络信用贷」数据辅助目标域
「个人贷款」的违约概率预测，评价指标 AUC。

- 小组：3 人（组长 A、组员 B、组员 C）
- 计划：[docs/PLAN.md](docs/PLAN.md)
- P0 交接：[docs/HANDOFF_P0.md](docs/HANDOFF_P0.md)
- 数据说明：[data/README.md](data/README.md)

## 快速开始

```bash
# WSL 里执行，项目根目录
make setup     # 建 .venv 并装依赖（首次约 2 分钟）
make verify    # 数据自检：MD5 / 行列数 / 提交样例对齐
make test      # 单测
```

Windows 侧 `python` 是应用商店占位程序不可用，**所有命令都在 WSL（Ubuntu-24.04）里跑**。

## 协作与数据获取

仓库**不含**原始数据（约 165MB）和虚拟环境，克隆后需要：

1. 把三份 csv 放到 `data/raw/`（组内共享，说明见 [data/README.md](data/README.md)），官网原件放 `data/official/`
2. `make setup` 建环境，`make verify` 自检（5 个 MD5 与三表行列数应全绿）

约定：`scripts/` 一个入口只做一件事；实验参数写 `configs/experiments/*.yaml`；跑批结果进 `results/`；图表与表格进 `reports/`。
## 目录

```
src/         可复用模块（config / io / validation，后续 data_quality、features、models、transfer）
scripts/     可执行入口（00_verify_data 已完成；01_eda 起为后续阶段）
configs/     base.yaml + experiments/*.yaml
tests/       单测（pytest）
data/        raw（官网 csv）/ official（官网原件）/ baseline（官方基线代码）/ processed / splits
reports/     figs、tables、环节小结 R1/R2
results/     实验跑批结果 runs.csv
docs/        PLAN.md、HANDOFF_P0.md
```

## 数据约定

| 表 | 行 | 列 | 说明 |
|---|---|---|---|
| `data/raw/train_public.csv` | 10,000 | 39 | 目标域，标签 `isDefault` |
| `data/raw/train_internet.csv` | 750,000 | 42 | 源域，标签 `is_default` |
| `data/raw/test_public.csv` | 5,000 | 38 | 测试集，无标签 |

- 三份 csv 带 **UTF-8 BOM**，统一由 `src.io.read_csv` 处理（`encoding="utf-8-sig"`）。
- 提交文件：`id` 取测试集 `loan_id`（不是 0 起始行号），`isDefault` 为违约概率。
- `user_id` 两域**不可关联**（源域是 0..749999 稠密编号），详见 [docs/PLAN.md](docs/PLAN.md) 第 0 节。