"""P2 preprocessing + split-local feature matrices. No training model here."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.feature_map import LABEL, field_map, unify_domain
from src.features import FeaturePipeline, DEFINITIONS, num
from src.dedup import near_duplicates
from src.reduce import FeatureReducer

ROOT=Path(__file__).resolve().parents[1]

def save_frame(df,path,fmt='parquet'):
    p=Path(path).with_suffix('.'+fmt); p.parent.mkdir(parents=True,exist_ok=True)
    if fmt=='parquet': df.to_parquet(p,index=False)
    elif fmt=='csv': df.to_csv(p,index=False)
    else: raise ValueError(fmt)
    return p

def build(raw_dir=None,processed_dir=None,fmt='parquet',mapping=None):
    """Row-preserving schema alignment; no fitted transforms, labels standardized to isDefault."""
    from src.io import read_csv
    raw=Path(raw_dir or ROOT/'data/raw'); processed=Path(processed_dir or ROOT/'data/processed')
    names=[('train_public','public','target_train'),('train_internet','internet','source_train'),('test_public','test','test')]
    mapping=field_map() if mapping is None else mapping
    result=[]
    for fname,dom,out in names:
        f=raw/(fname+'.csv')
        if not f.exists() and dom=='internet': f=raw/'train_internet_public.csv'
        d=read_csv(f); aligned=unify_domain(d,dom,mapping)
        assert len(d)==len(aligned)
        save_frame(aligned,processed/out,fmt);result.append(aligned)
    return tuple(result)

def make_splits(target, directory, seed=2026):
    """Stable row-position indices, with loan-id order fingerprint. Refuse conflicting existing files."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    y=target[LABEL].to_numpy(); fp=hashlib.sha256(pd.util.hash_pandas_object(target[['loan_id',LABEL,'issue_date']],index=False).to_numpy().tobytes()).hexdigest()
    if pd.Series(y).value_counts().min()<5 or len(np.unique(y))!=2: raise ValueError('Need at least five rows of each class')
    folds=[{'fold':i,'train_idx':a.tolist(),'val_idx':b.tolist()} for i,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(target,y))]
    dates=target.issue_date
    if dates.isna().any(): raise ValueError('Missing issue_date: resolve explicitly before time split')
    ordered=np.sort(dates.to_numpy()); cutoff=ordered[int(len(ordered)*.8)]
    tr=np.flatnonzero(dates.to_numpy()<cutoff); va=np.flatnonzero(dates.to_numpy()>=cutoff)
    if not len(tr) or not len(va) or len(np.unique(y[tr]))<2 or len(np.unique(y[va]))<2: raise ValueError('Time split does not contain both classes in each part')
    time={'fold':'time','train_idx':tr.tolist(),'val_idx':va.tolist(),'cutoff':str(pd.Timestamp(cutoff))}
    for name,payload in [('folds.json',{'seed':seed,'fingerprint':fp,'folds':folds}),('time_split.json',{'seed':seed,'fingerprint':fp,**time})]:
        p=directory/name
        if p.exists() and json.loads(p.read_text())!=payload: raise ValueError(f'Existing split conflicts: {p}; do not overwrite shared split')
        p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return folds,time

def audit_features(train,pipe):
    d=pipe._base(train); d['user_prior_count']=pipe._history(d); rows=[]
    for c,definition in DEFINITIONS.items():
        if c not in d: continue
        s=num(d[c]); ok=s.notna(); auc=np.nan
        if s[ok].nunique()>1 and d.loc[ok,LABEL].nunique()==2: auc=roc_auc_score(d.loc[ok,LABEL],s[ok])
        rows.append({'feature':c,'definition':definition,'missing_rate':float(1-ok.mean()),'univariate_auc_raw_direction':auc,'used_in_model':c in pipe.output_columns})
    return pd.DataFrame(rows,columns=['feature','definition','missing_rate','univariate_auc_raw_direction','used_in_model'])

def fold_psi(target,source,mapping):
    """Diagnostic only. Bins/categories defined by outer target training rows, not validation/test."""
    rows=[]
    for c,m in mapping.items():
        if m['scope']!='common' or m['type']=='id' or c not in target or c not in source: continue
        a,b=target[c],source[c]
        if m['type']=='datetime': a,b=a.dt.year,b.dt.year
        if m['type'] in ('numeric','datetime'):
            a,b=num(a),num(b); edges=np.unique(a.dropna().quantile(np.linspace(0,1,11)).to_numpy())
            edges=np.r_[-np.inf,edges[1:-1],np.inf] if len(edges)>1 else np.array([-np.inf,np.inf]);
            def bin_(s): return pd.cut(s,edges,include_lowest=True).astype('string').fillna('__MISSING__')
            a,b=bin_(a),bin_(b)
        else:
            a,b=a.astype('string').fillna('__MISSING__'),b.astype('string').fillna('__MISSING__'); levels=set(a.value_counts().head(100).index); a=a.where(a.isin(levels),'__OTHER__'); b=b.where(b.isin(levels),'__OTHER__')
        keys=a.value_counts().index.union(b.value_counts().index); p=a.value_counts(normalize=True).reindex(keys,fill_value=0).to_numpy();q=b.value_counts(normalize=True).reindex(keys,fill_value=0).to_numpy();p=np.maximum(p,1e-6);q=np.maximum(q,1e-6);p/=p.sum();q/=q.sum()
        rows.append({'feature':c,'psi':float(np.sum((q-p)*np.log(q/p)))})
    return pd.DataFrame(rows,columns=['feature','psi'])

def prepare_fold(target,source,test,split,mapping=None,scope='target',use_source=False,dedup=True,derived=True,reduction='pca',early_policy='rank',include_domain=False,exclude_early=False,source_limit=None,seed=2026):
    mapping=field_map() if mapping is None else mapping
    if use_source and scope!='common': raise ValueError('Source requires common scope')
    if source_limit is not None and source_limit<1: raise ValueError('source_limit must be positive')
    tr=target.iloc[split['train_idx']].reset_index(drop=True);va=target.iloc[split['val_idx']].reset_index(drop=True)
    eligible=source
    if split.get('cutoff'):
        eligible=source.loc[source.issue_date < pd.Timestamp(split['cutoff'])]
    if eligible.empty: raise ValueError('No eligible source rows')
    src=eligible.reset_index(drop=True) if source_limit is None else eligible.sample(n=min(source_limit,len(eligible)),random_state=seed).sort_index().reset_index(drop=True)
    psi=fold_psi(tr,src,mapping); logs=[]; counts=[]
    def clean(d,dom):
        cols=[c for c,m in mapping.items() if c in d and m['type'] not in ('id','datetime') and (not exclude_early or not c.startswith('early_return')) and (scope!='common' or (m['scope']=='common' and m.get('transferable')))]
        if not dedup:
            counts.append({'domain':dom,'input_rows':len(d),'removed':0,'conflicts':0,'truncated_buckets':0,'candidate_checks':0});return d
        keep,rep,stats=near_duplicates(d,cols,seed=seed)
        if len(rep):
            rep['domain']=dom;rep['loan_id']=d.loan_id.iloc[rep.row.to_numpy(dtype=int)].to_numpy();rep['representative_loan_id']=d.loan_id.iloc[rep.representative.to_numpy(dtype=int)].to_numpy();logs.append(rep)
        counts.append({'domain':dom,**stats}); return d.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    tr=clean(tr,'target'); src=clean(src,'source') if use_source else src
    train=pd.concat([tr,src],ignore_index=True) if use_source else tr
    pipe=FeaturePipeline(mapping,scope,derived,seed,early_policy,include_domain,exclude_early)
    X=pipe.fit_transform(train);V=pipe.transform(va);T=pipe.transform(test)
    audit=audit_features(tr,pipe)
    matrices={'train':X,'val':V,'test':T}; reduced=None; info={}
    if reduction!='none':
        red=FeatureReducer(reduction,seed=seed).fit(X);reduced={k:red.transform(v) for k,v in matrices.items()};info={'method':reduction,'input_features':X.shape[1],'output_features':int(red.n),'explained_variance_ratio':red.explained}
    meta={}
    for k,d in [('train',train),('val',va),('test',test)]:
        meta[k]=d[[c for c in ['loan_id','user_id','is_source',LABEL] if c in d]].reset_index(drop=True)
        assert len(meta[k])==len(matrices[k])
    return {'matrices':matrices,'reduced':reduced,'meta':meta,'pipeline':pipe,'reduction':info,'audit':audit,'psi':psi,'dup_clusters':pd.concat(logs,ignore_index=True) if logs else pd.DataFrame(columns=['row','representative','jaccard','label_conflict','domain','loan_id','representative_loan_id']),'dup_counts':pd.DataFrame(counts)}

def write_fold(result,directory,fmt='parquet'):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    for k,v in result['matrices'].items():
        save_frame(v,out/f'X_{k}',fmt);save_frame(result['meta'][k],out/f'meta_{k}',fmt)
    if result['reduced'] is not None:
        for k,v in result['reduced'].items(): save_frame(v,out/f'X_{k}_reduced',fmt)
    for k,name in [('audit','feat_audit'),('psi','psi_train_fold'),('dup_clusters','dup_clusters'),('dup_counts','dup_counts')]:
        result[k].to_csv(out/(name+'.csv'),index=False)
    (out/'reduction.json').write_text(json.dumps(result['reduction'],indent=2),encoding='utf-8')
