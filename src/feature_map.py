"""P2: deterministic schema alignment, without learning from validation/test."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'isDefault'
EARLY = ('early_return', 'early_return_amount', 'early_return_amount_3mon')

def field_map(path=None):
    with open(path or ROOT / 'configs/feature_map.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)['fields']

def unify_domain(df, domain, mapping=None):
    from src.dates import parse_issue_date, parse_earlies_credit_mon
    if domain not in ('public', 'internet', 'test'):
        raise ValueError(domain)
    mapping = field_map() if mapping is None else mapping
    df = df.copy().reset_index(drop=True)
    for old, new in [('house_ownership','house_exist'), ('is_default',LABEL)]:
        if old in df:
            if new in df:
                raise ValueError(f'Both {old} and {new} exist; resolve explicitly')
            df = df.rename(columns={old:new})
    if df.columns.duplicated().any():
        raise ValueError('Duplicate columns')
    required = [c for c,m in mapping.items() if m['scope'] == 'common' or
                m['scope'] == ('internet_only' if domain=='internet' else 'public_only')]
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f'{domain}: missing schema columns {sorted(missing)}')
    out = pd.DataFrame(index=df.index)
    for c in required:
        m = mapping[c]; s = df[c]
        if m['type'] == 'id':
            out[c] = pd.to_numeric(s, errors='raise').astype('Int64')
            if out[c].isna().any(): raise ValueError(f'Missing {c}')
        elif m['type'] == 'datetime':
            out[c] = (parse_issue_date if c=='issue_date' else parse_earlies_credit_mon)(s)
        elif m['type']=='numeric' or c=='year_of_loan':
            out[c] = pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf], np.nan).astype('float64')
        else:
            out[c] = s.astype('string').str.strip().str.replace(r'^(-?\d+)\.0+$',r'\1',regex=True).replace('',pd.NA)
    if domain != 'test':
        if LABEL not in df: raise ValueError(f'{domain}: missing label')
        y = pd.to_numeric(df[LABEL], errors='raise')
        if not y.isin([0,1]).all(): raise ValueError('Non-binary/missing label')
        out[LABEL] = y.astype('int8')
    elif LABEL in df:
        raise ValueError('Test contains label')
    out['is_source'] = np.int8(domain=='internet')
    if 'loan_id' in out and not out.loan_id.is_unique: raise ValueError('Duplicate loan_id')
    return out
