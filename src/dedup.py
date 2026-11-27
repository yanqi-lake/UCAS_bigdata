"""Approximate MinHash/LSH candidate search + exact Jaccard on quantized feature tokens.
Train only; same-domain only; conflicting labels are retained. Bounded candidate checks.
"""
import numpy as np
import pandas as pd
from src.feature_map import LABEL


def near_duplicates(df, columns, threshold=.9, seed=2026, bands=8, rows=4, bucket_cap=12):
    if not 0<threshold<=1 or bands<1 or rows<1 or bucket_cap<1: raise ValueError('Invalid LSH parameters')
    df=df.reset_index(drop=True); n=len(df); keep=np.ones(n,bool)
    reports=[]; stats={'input_rows':n,'removed':0,'conflicts':0,'truncated_buckets':0,'candidate_checks':0}
    if n<2 or not columns: return keep,pd.DataFrame(columns=['row','representative','jaccard','label_conflict']),stats
    tokens=[]
    for j,c in enumerate(columns):
        v=df[c]
        if pd.api.types.is_numeric_dtype(v) and v.nunique()>20:
            edges=np.unique(v.dropna().quantile(np.linspace(0,1,21)).to_numpy(dtype=float))
            a=np.searchsorted(edges,v.to_numpy(dtype=float),side='right').astype('int64'); a[v.isna()]=-1
            s=pd.Series(a).astype(str)
        else: s=v.astype('string').fillna('__MISSING__').astype(str)
        h=pd.util.hash_pandas_object(s,index=False).to_numpy(dtype=np.uint64)
        # Distinguish different columns even when values are equal.
        tokens.append(h ^ np.uint64((j+1)*2654435761))
    t=np.column_stack(tokens); rng=np.random.default_rng(seed); p=len(columns)
    # Every field has exactly one token. Set intersection is number of matching fields.
    # 32 MinHash values per row, hashed into eight four-value LSH bands.
    sig=np.empty((n,bands*rows),np.uint32)
    for k in range(bands*rows):
        a=np.uint64(rng.integers(1,2**32,dtype=np.uint64)) | np.uint64(1); b=np.uint64(rng.integers(0,2**32,dtype=np.uint64))
        # mixed high bits of affine hash (uint64 overflow intentionally wraps)
        sig[:,k]=((t*a+b)>>np.uint64(32)).min(axis=1).astype(np.uint32)
    labels=df[LABEL].to_numpy() if LABEL in df else np.zeros(n)
    parent=np.arange(n); conflict_seen=set()
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for k in range(bands):
        block=np.ascontiguousarray(sig[:,k*rows:(k+1)*rows]); key=block.view(np.dtype((np.void,block.dtype.itemsize*rows))).ravel()
        order=np.argsort(key,kind='stable'); sorted_keys=key[order]
        cuts=np.r_[0,np.flatnonzero(sorted_keys[1:]!=sorted_keys[:-1])+1,n]
        for l,r in zip(cuts[:-1],cuts[1:]):
            if r-l<2: continue
            if r-l>bucket_cap+1: stats['truncated_buckets']+=1
            reps=[]
            for i in order[l:r]:
                if not keep[i]: continue
                merged=False
                for j in reps[:bucket_cap]:
                    j=root(j)
                    if i==j: continue
                    stats['candidate_checks']+=1
                    same=int(np.count_nonzero(t[i]==t[j])); score=same/(2*p-same)
                    if score<threshold: continue
                    if labels[i]!=labels[j]:
                        pair=tuple(sorted((int(i),int(j))))
                        if pair not in conflict_seen:
                            conflict_seen.add(pair);reports.append((int(i),int(j),score,True)); stats['conflicts']+=1
                        continue
                    parent[i]=j;keep[i]=False;reports.append((int(i),int(j),score,False));merged=True;break
                if not merged and len(reps)<bucket_cap: reps.append(int(i))
    stats['removed']=int((~keep).sum())
    return keep,pd.DataFrame(reports,columns=['row','representative','jaccard','label_conflict']),stats
