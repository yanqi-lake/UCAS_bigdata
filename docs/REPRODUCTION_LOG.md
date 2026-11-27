# 网上代码复现记录

每次运行复制下面的记录块，填写实际观察到的结果。尚未验证的项目写“待核实”，不要填估计值。

## 运行记录 01

| 项目 | 实际记录 |
|---|---|
| 代码来源链接 | DataFountain 2021 CCF BDCI 个贷违约预测官方 Baseline 压缩包；确切下载页和版本待核实 |
| 来源版本或提交号、取得日期 | 待填写 |
| 本项目代码位置 | `notebooks/shaky_reproduction/lgb_v1.py`（运行前核对） |
| 对原代码的改动及原因 | `lgb_v1.py` 与本地 `baseline/CCF2021-master/lgb_v1.py` 的 SHA256 一致；`config.py` 改为项目相对路径。候选上传版进一步优先读取仓库约定的 `data/raw/`，兼容本地旧的 `data/`。 |
| 操作系统、Python 版本 | Windows PowerShell；Python 3.13.2（用户终端截图确认） |
| 依赖版本 | 见 `results/reproduction_packages.txt`；用户终端确认 pandas 3.1.1、scikit-learn 1.9.1、LightGBM 4.7.0、toad 0.1.7 |
| 实际执行命令和工作目录 | 用户截图显示在 `C:\Users\12208\Desktop\LoanDefaultPrediction` 执行 `python .\notebooks\shaky_reproduction\lgb_v1.py`；是否完整成功及输出日志待补 |
| 读取的数据文件及 SHA256 | 待填写 |
| 数据行数、标签字段 | 待填写 |
| 切分方法、随机种子、各部分样本量 | 待填写 |
| 预处理、特征工程 | 待填写 |
| 模型与超参数 | 待填写 |
| AUC、F1、Precision、Recall | 未留存可核验的复现输出，暂不填写；不得用课程实验的结果代替 |
| 分类阈值 | 待填写；AUC 使用概率而非阈值后的类别 |
| 训练耗时与硬件 | 待填写 |
| 生成文件及其位置 | 待填写 |
| 是否可由同一命令重复得到相近结果 | 待核实 |

## 统一实验口径

用户终端已确认可通过 `python` 使用 Python 3.13.2 及模型依赖；此前 Codex 工具环境无法访问该解释器，不能据此判断用户环境失效。网上代码曾由用户运行成功，但本记录尚缺该次运行的完整日志与指标。统一切分文件由标准库生成，三模型五折结果另存于 `results/course_models_cv_*`。

1. 主研究数据是带 `isDefault` 标签的 `train_public.csv`。`test_public.csv` 没有标签，不用于本地指标计算。
2. 运行 `scripts/02_make_splits.py` 后，所有模型读取 `data/splits/split_indices.json`。先在 `train_indices` 内做 5 折开发与调参，方法固定后再评估一次 `holdout_indices`。
3. 缺失填补、编码、标准化、PCA、特征筛选均仅用相应训练折拟合，再转换验证折。禁止使用整个数据集计算这些参数。
4. 记录 AUC 时使用违约概率。F1、Precision、Recall 同时记录阈值。保存完整参数、随机种子、运行时间和代码版本。
5. 网上复现代码如使用不同数据域、切分或指标，原结果单列记录，不与课程主线模型的成绩直接排名。

