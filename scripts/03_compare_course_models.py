"""Compare LR, decision tree and random forest on one saved five-fold split.

Run: python scripts/03_compare_course_models.py
Uses only the 8,000 development rows. The 2,000-row holdout is untouched.
All imputers, encoders and scalers are fitted inside each training fold.
"""
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = next((p for p in (ROOT / 'data/raw/train_public.csv', ROOT / 'data/train_public.csv') if p.exists()), None)
SPLITS = ROOT / 'data/splits/split_indices.json'
OUT = ROOT / 'results'
SEED = 2026
THRESHOLD = 0.5

NUMERIC = [
    'total_loan', 'year_of_loan', 'interest', 'monthly_payment',
    'debt_loan_ratio', 'del_in_18month', 'scoring_low', 'scoring_high',
    'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
    'recircle_b', 'recircle_u', 'f0', 'f1', 'f2', 'f3', 'f4',
    'early_return', 'early_return_amount', 'early_return_amount_3mon',
]
CATEGORICAL = [
    'class', 'employer_type', 'industry', 'work_year', 'house_exist',
    'censor_status', 'use', 'region', 'initial_list_status', 'app_type',
]


def make_pipeline(model):
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    pre = ColumnTransformer([
        ('numeric', numeric, NUMERIC),
        ('categorical', categorical, CATEGORICAL),
    ])
    return Pipeline([('preprocess', pre), ('model', model)])


if DATA is None:
    raise FileNotFoundError('找不到 train_public.csv')
with DATA.open('rb') as stream:
    data_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
split = json.loads(SPLITS.read_text(encoding='utf-8'))
if data_sha != split['source_sha256']:
    raise ValueError('数据 SHA256 与保存切分时不同，已停止运行')
df = pd.read_csv(DATA, encoding='utf-8-sig')
if len(df) != len(split['train_indices']) + len(split['holdout_indices']):
    raise ValueError('数据行数与切分文件不一致')
if not set(NUMERIC + CATEGORICAL + ['isDefault']) <= set(df.columns):
    raise ValueError('训练数据缺少脚本要求的字段')
X = df[NUMERIC + CATEGORICAL].copy()
for col in NUMERIC:
    X[col] = pd.to_numeric(X[col], errors='coerce')
for col in CATEGORICAL:
    X[col] = X[col].astype('string').fillna('MISSING').astype(str)
y = df['isDefault'].astype(int)
train_ids = set(split['train_indices'])
holdout_ids = set(split['holdout_indices'])
folds = split['cv_validation_indices']
assert train_ids.isdisjoint(holdout_ids)
assert set().union(*(set(f) for f in folds)) == train_ids

models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'decision_tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=SEED),
    'random_forest': RandomForestClassifier(n_estimators=200, max_depth=12,
                                            min_samples_leaf=10, n_jobs=4,
                                            random_state=SEED),
}
rows = []
for name, estimator in models.items():
    for fold_no, valid in enumerate(folds, start=1):
        valid_set = set(valid)
        fit_ids = sorted(train_ids - valid_set)
        assert valid_set <= train_ids and not valid_set.intersection(holdout_ids)
        pipe = make_pipeline(estimator)
        start = time.perf_counter()
        pipe.fit(X.iloc[fit_ids], y.iloc[fit_ids])
        seconds = time.perf_counter() - start
        prob = pipe.predict_proba(X.iloc[valid])[:, 1]
        predicted = (prob >= THRESHOLD).astype(int)
        actual = y.iloc[valid]
        row = {
            'model': name, 'fold': fold_no, 'n_train': len(fit_ids),
            'n_valid': len(valid), 'roc_auc': roc_auc_score(actual, prob),
            'f1': f1_score(actual, predicted, zero_division=0),
            'precision': precision_score(actual, predicted, zero_division=0),
            'recall': recall_score(actual, predicted, zero_division=0),
            'threshold': THRESHOLD, 'train_seconds': round(seconds, 3),
            'seed': SEED,
        }
        rows.append(row)
        print(f"{name:20s} fold={fold_no} AUC={row['roc_auc']:.4f} F1={row['f1']:.4f}", flush=True)

OUT.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
fold_path = OUT / f'course_models_cv_folds_{stamp}.csv'
summary_path = OUT / f'course_models_cv_summary_{stamp}.csv'
meta_path = OUT / f'course_models_cv_metadata_{stamp}.json'
with fold_path.open('x', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = pd.DataFrame(rows).groupby('model')[['roc_auc', 'f1', 'precision', 'recall', 'train_seconds']].agg(['mean', 'std'])
summary.to_csv(summary_path, encoding='utf-8-sig')
meta = {
    'data': str(DATA), 'data_sha256': data_sha, 'split_file': str(SPLITS),
    'python_packages': {'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
    'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
    'excluded_columns': ['loan_id', 'user_id', 'issue_date', 'earlies_credit_mon', 'post_code', 'title', 'policy_code'],
    'models': {name: model.get_params() for name, model in models.items()},
    'threshold': THRESHOLD,
    'note': 'Five-fold development-set CV only. Holdout and test_public were not evaluated.',
}
with meta_path.open('x', encoding='utf-8') as stream:
    json.dump(meta, stream, ensure_ascii=False, indent=2, default=str)
print('\nSaved:', fold_path, summary_path, meta_path, sep='\n')
print(summary.to_string())
