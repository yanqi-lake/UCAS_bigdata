# 个贷违约预测（CCF BDCI 2021 / DataFountain #530）数据集

赛题页：<https://www.datafountain.cn/competitions/530>
任务：用「个人贷款」与「网络信用贷」两批有差异的信贷数据做迁移学习，预测目标业务的用户是否违约。
评测指标：**AUC**（越高越好）。

## 目录结构

```
data/
├── official/                 官网原始文件，逐字节未改动（MD5 已与官网公布值全部核对一致）
│   ├── train_dataset.zip
│   ├── test_public.csv
│   ├── submit_example.csv
│   ├── 个贷违约预测baseline.zip
│   └── 【更新1117】个贷违约预测Baseline.zip
├── raw/                      解包 / 直接使用的 csv
│   ├── train_public.csv
│   ├── train_internet.csv
│   └── test_public.csv
├── baseline/                 官方 baseline 代码（从上面两个 zip 解出）
│   ├── 0923/                 credit_baseline.py / .ipynb
│   └── 1117/                 credit_preprocessing.py + credit_predict.py
├── submit_example.csv        官网提交样例（id = 测试集 loan_id）
└── README.md
```

## 官网 MD5 校验（全部一致 ✅）

| 文件 | 官网公布 MD5 | 本地实测 | 结果 |
|---|---|---|---|
| `official/train_dataset.zip` | `192D3D30A9252E849C600C0931C8C1FB` | 同 | ✅ |
| `official/test_public.csv` | `D4A2AB57C4B6188CC606949501750420` | 同 | ✅ |
| `official/submit_example.csv` | `5F272DFEE05DF1C950009A6E5700E509` | 同 | ✅ |
| `official/个贷违约预测baseline.zip` | `0D4D985CB15DB705E89A3DD8CFCDEEC5` | 同 | ✅ |
| `official/【更新1117】个贷违约预测Baseline.zip` | `2DAA6FEB80626560FC09D448AFF3FBE3` | 同 | ✅ |

## 数据文件

| 文件 | 说明 | 行数（不含表头） | 列数 | 大小 | MD5 |
|---|---|---|---|---|---|
| `raw/train_public.csv` | 个人贷款违约记录，标签 `isDefault`（目标域） | 10,000 | 39 | 2,046,257 B | `3F7A7170FC437D3CB8BCC0E2F4A0DAF7` |
| `raw/train_internet.csv` | 网络信用贷违约记录，标签 `is_default`（辅助域） | 750,000 | 42 | 162,877,235 B | `0BE09D538ED87794A86388B73896DCA0` |
| `raw/test_public.csv` | 测试集，无标签 | 5,000 | 38 | 1,025,658 B | `D4A2AB57C4B6188CC606949501750420` |

`raw/` 里的两个训练 csv 就是从 `official/train_dataset.zip` 解出来的，MD5 与 zip 内文件一致；`raw/test_public.csv` 与 `official/test_public.csv` 为同一文件。
（此前用社区镜像试下的一版与官网原件逐字节一致，`official/` 目录以官网原件为准。）
训练集正样本率 16.83%（8317 / 1683）。

## 提交格式（以官网样例为准）

```
id,isDefault
1000575,0
1028125,0
...
```

- `id` 是**测试集里的 `loan_id`**（不是 0 起始的行号），按 `test_public.csv` 的行顺序给出；
- 共 5000 行 + 表头；
- `isDefault` 填 0/1 或违约概率，评测按 AUC 计算；
- 样例文件带 UTF-8 BOM，提交文件建议同样保持 `id,isDefault` 两列。

## 官方 baseline 说明

- `baseline/0923/credit_baseline.py`（另有 `.ipynb`）：取两域**共有字段**拼成迁移训练集，做基础特征处理后训练模型并输出提交文件，是最初版官方基线。
- `baseline/1117/credit_preprocessing.py` + `credit_predict.py`：更新版。预处理脚本要求先把 `train_public.csv` / `test_public.csv` **另存为** `train_public2.csv` / `test_public2.csv`（脚本借此修正 `earlies_credit_mon` 这个 2021 年占位日期），之后生成 `train1.pkl` / `train2.pkl` / `test.pkl`；预测脚本用 **CatBoost**（3000 轮、depth=7、lr=0.001、AUC 评估）在 `train2` 上训练，并构造 `issue_date_diff` / `earlies_credit_mon_diff` / `issue_earlies_diff` 三个日期差特征。跑之前把这些文件放在同一目录，并把路径改成指向 `data/raw/`。

## 使用注意

1. **csv 带 UTF-8 BOM**：pandas 读取请用 `encoding='utf-8-sig'`，否则第一列列名会带 BOM 前缀（`loan_id` 变 `\ufeffloan_id`）。
2. 两域字段命名/编码不同：`train_public.csv` 用 `class` / `employer_type` / `issue_date` / `house_exist`，`train_internet.csv` 用 `class` / `sub_class` / `work_type` / `house_ownership`；目标列分别是 `isDefault` 与 `is_default`。
3. 测试集字段 = 训练集字段去掉标签列（38 列）；两域与测试集间有极少量重叠 `user_id`，属赛题刻意设计的迁移场景。
4. 缺失值：`f1`、`work_year`、`f0`~`f4` 等列有空值，需自行处理。
5. 数据仅可用于本比赛，不得用于商业用途（见赛题页「参赛规则」）。