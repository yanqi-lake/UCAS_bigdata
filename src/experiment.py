"""Shared experiment adapter: use any sklearn-style model supplied by P3/P4.
No model selection, domain weighting or test labels are implemented here.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from src.feature_map import LABEL


def evaluate_fold(result, estimator, *, group, model_name, fold, reduced=False, sample_weight=None, recall_fraction=.10):
    matrices=result['reduced'] if reduced else result['matrices']
    if matrices is None: raise ValueError('Reduced matrices were not generated')
    X,V=matrices['train'],matrices['val'];meta=result['meta'];y=meta['train'][LABEL].to_numpy();v=meta['val'][LABEL].to_numpy()
    if not 0<recall_fraction<=1: raise ValueError('recall_fraction must be in (0,1]')
    if sample_weight is not None:
        w=np.asarray(sample_weight,dtype=float)
        if w.shape!=(len(X),) or not np.isfinite(w).all() or (w<0).any() or not w.sum()>0: raise ValueError('Invalid sample weights')
    if len(np.unique(v))!=2: raise ValueError('Validation requires both labels')
    start=time.perf_counter()
    estimator.fit(X,y,**({'sample_weight':w} if sample_weight is not None else {}))
    positive=np.flatnonzero(np.asarray(estimator.classes_)==1)
    if len(positive)!=1: raise ValueError('Model has no unique class 1')
    p=estimator.predict_proba(V)[:,positive[0]]
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Invalid probabilities')
    fpr,tpr,_=roc_curve(v,p);k=max(1,int(np.ceil(len(v)*recall_fraction)));top=np.argsort(-p,kind='stable')[:k]
    return {'group':str(group),'model':str(model_name),'fold':str(fold),'reduced':bool(reduced),'auc':float(roc_auc_score(v,p)),'ks':float(np.max(tpr-fpr)),'recall_at_fraction':float(v[top].sum()/v.sum()),'recall_fraction':recall_fraction,'n_train':len(X),'n_val':len(V),'n_features':X.shape[1],'elapsed_seconds':time.perf_counter()-start}


def append_run(record,path):
    """Serial execution only. Refuse duplicate group/model/fold/reduced records."""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);new=pd.DataFrame([record]);keys=['group','model','fold','reduced']
    if p.exists():
        old=pd.read_csv(p,dtype={'group':str,'model':str,'fold':str});combined=pd.concat([old,new],ignore_index=True)
        if combined.duplicated(keys).any(): raise ValueError('Duplicate experiment key; use separate run file or group name')
    else: combined=new
    combined.to_csv(p,index=False)
    return combined
