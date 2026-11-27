"""Create reproducible stratified holdout and five CV folds, without third-party packages.

Run from anywhere: python scripts/02_make_splits.py
Only reads train_public.csv. Never changes source data or existing split files.
"""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = next((p for p in (ROOT / 'data/raw/train_public.csv', ROOT / 'data/train_public.csv') if p.is_file()), None)
if SOURCE is None:
    raise FileNotFoundError('找不到 train_public.csv（检查 data/raw/ 或 data/）')
OUT = ROOT / 'data' / 'splits'
SEED = 2026
HOLDOUT_FRACTION = 0.2
FOLDS = 5

with SOURCE.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
with SOURCE.open('r', encoding='utf-8-sig', newline='') as stream:
    reader = csv.DictReader(stream)
    if not reader.fieldnames or not {'loan_id', 'isDefault'} <= set(reader.fieldnames):
        raise ValueError('缺少 loan_id 或 isDefault 字段')
    records = [(i, row['loan_id'], row['isDefault']) for i, row in enumerate(reader)]
if not records or len({r[1] for r in records}) != len(records):
    raise ValueError('训练集为空或 loan_id 不唯一')
if set(r[2] for r in records) != {'0', '1'}:
    raise ValueError('目标标签应恰好包含 0 和 1')

groups = defaultdict(list)
for i, loan_id, label in records:
    groups[label].append(i)
rng = random.Random(SEED)
train, holdout = [], []
for label in sorted(groups):
    ids = groups[label][:]
    rng.shuffle(ids)
    cut = round(len(ids) * HOLDOUT_FRACTION)
    holdout.extend(ids[:cut])
    train.extend(ids[cut:])
train.sort()
holdout.sort()

folds = [[] for _ in range(FOLDS)]
train_set = set(train)
for label in sorted(groups):
    ids = [i for i in groups[label] if i in train_set]
    rng.shuffle(ids)
    for j, i in enumerate(ids):
        folds[j % FOLDS].append(i)
for fold in folds:
    fold.sort()

assert len(train_set | set(holdout)) == len(records) and not train_set.intersection(holdout)
assert set().union(*(set(f) for f in folds)) == train_set
assert sum(map(len, folds)) == len(train)

labels = {i: label for i, _, label in records}
loan_ids = {i: loan_id for i, loan_id, _ in records}
def describe(indices):
    return {'rows': len(indices), 'label_counts': dict(Counter(labels[i] for i in indices))}

manifest = {
    'source': str(SOURCE), 'source_sha256': digest,
    'method': 'Python random.Random(seed), per-label shuffle; round(20%) holdout; per-label reshuffle and round-robin assignment to 5 CV folds',
    'seed': SEED, 'holdout_fraction': HOLDOUT_FRACTION, 'folds': FOLDS,
    'index_basis': 'zero-based CSV data row, excluding header',
    'all': describe(range(len(records))), 'train': describe(train),
    'holdout': describe(holdout),
    'cv_validation_folds': [describe(f) for f in folds],
    'usage': 'Use train for model development and CV. Do not inspect holdout performance until the modeling choices are fixed. Fit preprocessing within each training fold.'
}
split = {
    'source_sha256': digest, 'seed': SEED,
    'train_indices': train, 'holdout_indices': holdout,
    'train_loan_ids': [loan_ids[i] for i in train],
    'holdout_loan_ids': [loan_ids[i] for i in holdout],
    'cv_validation_indices': folds,
}
OUT.mkdir(parents=True, exist_ok=True)
for name, obj in [('split_manifest.json', manifest), ('split_indices.json', split)]:
    path = OUT / name
    if path.exists():
        with path.open('r', encoding='utf-8') as stream:
            if json.load(stream) != obj:
                raise FileExistsError(f'{path} 已存在且内容不同；为保护已有切分，未覆盖')
    else:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2)
    print(path)
print(json.dumps({k: manifest[k] for k in ('all', 'train', 'holdout', 'cv_validation_folds')}, ensure_ascii=False, indent=2))
