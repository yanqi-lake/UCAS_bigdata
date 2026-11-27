from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 仓库使用 data/raw；保留本地复现时使用的 data/ 作为兼容路径。
DATA_DIR = PROJECT_ROOT / "data" / "raw"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PUBLIC_PATH = DATA_DIR / "train_public.csv"
TRAIN_INTERNET_PATH = DATA_DIR / "train_internet.csv"
TEST_PUBLIC_PATH = DATA_DIR / "test_public.csv"
