"""Fold-local features. fit_transform uses out-of-fold target encoding."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from src.feature_map import EARLY, LABEL, field_map

MISSING='__MISSING__'; OTHER='__OTHER__'
def cat(s): return s.astype('string').fillna(MISSING).astype(str)
def num(s): return pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(float)
def ratio(a,b): return num(a)/num(b).where(num(b)>0)

DEFINITIONS = {'payment_loan_ratio':'monthly_payment / total_loan',
 'early_payment_ratio':'early_return_amount / monthly_payment',
 'early_3mon_share':'early_return_amount_3mon / early_return_amount',
 'debt_delinquency':'debt_loan_ratio * del_in_18month',
 'credit_history_months':'issue_date minus earlies_credit_mon, in months',
 'has_credit_history':'valid credit history indicator', 'user_prior_count':'strictly earlier records in same domain and training reference'}

def add_ratios(df):
    d=df.copy()
    for name,a,b in [('payment_loan_ratio','monthly_payment','total_loan'),('early_payment_ratio','early_return_amount','monthly_payment'),('early_3mon_share','early_return_amount_3mon','early_return_amount')]:
        if a in d and b in d: d[name]=ratio(d[a],d[b])
    return d

def add_history_len(df):
    d=df.copy()
    if 'issue_date' in d and 'earlies_credit_mon' in d:
        a,b=d.issue_date,d.earlies_credit_mon
        m=(a.dt.year-b.dt.year)*12+a.dt.month-b.dt.month
        d['credit_history_months']=m.where(m>=0)
        d['has_credit_history']=d.credit_history_months.notna().astype(float)
    return d

def add_interactions(df):
    d=df.copy()
    if {'debt_loan_ratio','del_in_18month'} <= set(d):
        d['debt_delinquency']=num(d.debt_loan_ratio)*num(d.del_in_18month)
    return d

def add_missing_flags(df, columns):
    return pd.DataFrame({c+'__missing':df[c].isna().astype('float32') for c in columns},index=df.index)

class TargetEncoderCV:
    def __init__(self,k=5,smoothing=20,seed=2026): self.k=k; self.smoothing=smoothing; self.seed=seed
    def fit(self,s,y):
        s=cat(s).reset_index(drop=True); y=np.asarray(y,dtype=float)
        self.prior=float(y.mean()) if len(y) else .5
        g=pd.DataFrame({'c':s,'y':y}).groupby('c').y.agg(['sum','count'])
        self.mapping=(g['sum']+self.smoothing*self.prior)/(g['count']+self.smoothing)
        return self
    def transform(self,s): return cat(s).map(self.mapping).fillna(self.prior).to_numpy(dtype=float)
    def fit_transform(self,s,y):
        s=s.reset_index(drop=True); y=np.asarray(y); n=len(y); z=np.full(n,.5)
        if n>=2:
            counts=pd.Series(y).value_counts()
            k=min(self.k,n)
            if len(counts)==2 and counts.min()>=2:
                split=StratifiedKFold(min(k,int(counts.min())),shuffle=True,random_state=self.seed).split(s,y)
            else: split=KFold(k,shuffle=True,random_state=self.seed).split(s)
            for tr,va in split:
                enc=TargetEncoderCV(self.k,self.smoothing,self.seed).fit(s.iloc[tr],y[tr]); z[va]=enc.transform(s.iloc[va])
        self.fit(s,y); return z

class FrequencyEncoder:
    def __init__(self,max_levels=100): self.max_levels=max_levels
    def fit(self,s):
        s=cat(s); self.top=set(s.value_counts().head(self.max_levels).index)
        self.freq=s.where(s.isin(self.top),OTHER).value_counts(normalize=True); return self
    def transform(self,s):
        s=cat(s); return s.where(s.isin(self.top),OTHER).map(self.freq).fillna(0).to_numpy(float)

class OrdinalEncoderMap:
    @staticmethod
    def transform(s,name):
        if name=='class': return cat(s).str.upper().map(dict(zip('ABCDEFG',range(7)))) .astype(float)
        if name=='work_year':
            t=cat(s); v=pd.to_numeric(t.str.extract(r'(\d+)')[0],errors='coerce'); return v.mask(t.str.startswith('<'),0).clip(0,10)
        return num(s)

class OneHotMap:
    def fit(self,s): self.levels=sorted(set(cat(s))-{OTHER}); return self
    def transform(self,s):
        s=cat(s); s=s.where(s.isin(self.levels),OTHER)
        return {f'{i}':(s==v).to_numpy(float) for i,v in enumerate(self.levels+[OTHER])}

class FeaturePipeline:
    def __init__(self, mapping=None, scope='target', derived=True, seed=2026, early_policy='rank', include_domain=False, exclude_early=False):
        self.mapping=field_map() if mapping is None else mapping
        if scope not in ('target','common'): raise ValueError(scope)
        if early_policy not in ('rank','raw'): raise ValueError(early_policy)
        self.scope=scope; self.derived=derived; self.seed=seed; self.early_policy=early_policy; self.include_domain=include_domain; self.exclude_early=exclude_early
    def _base(self,d):
        d=d.reset_index(drop=True).copy()
        if self.derived:
            d=add_interactions(add_history_len(add_ratios(d)))
        return d
    def _history(self,d):
        result=np.zeros(len(d),float)
        if self.user_history is None or self.user_history.empty: return result
        q=d[['is_source','user_id','issue_date']].copy(); q['_row']=np.arange(len(d))
        q=q.dropna().sort_values('issue_date')
        if len(q):
            m=pd.merge_asof(q,self.user_history,on='issue_date',by=['is_source','user_id'],direction='backward',allow_exact_matches=False)
            result[m['_row'].to_numpy()]=m['_count'].fillna(0).to_numpy()
        return result
    def _columns(self,d):
        for c,m in self.mapping.items():
            if c not in d or m['type']=='id': continue
            if self.exclude_early and c in EARLY: continue
            if self.scope=='common' and (m['scope']!='common' or not m.get('transferable')): continue
            if self.scope=='target' and m['scope']=='internet_only': continue
            yield c,m['encoding']
        if self.derived:
            for c in DEFINITIONS:
                if c not in d or c=='user_prior_count': continue
                if self.scope=='common' and c in ('credit_history_months','has_credit_history','early_payment_ratio','early_3mon_share'): continue
                if self.exclude_early and c.startswith('early_'): continue
                yield c,'none'
    def _numeric(self,s,c,fit):
        s=num(s); key=c
        if fit:
            finite=s.dropna(); lo,hi=(finite.quantile(.01),finite.quantile(.99)) if c in ('debt_loan_ratio','recircle_b') and len(finite) else (-np.inf,np.inf)
            m=self.mapping.get(c,{}).get('missing')
            fill=(float(finite.mode().iloc[0]) if len(finite) else 0.) if m=='mode' or c=='work_year' else (0. if m=='constant' else (float(finite.median()) if len(finite) else 0.))
            self.numeric[key]=(lo,hi,fill)
        lo,hi,fill=self.numeric[key]
        return s.clip(lo,hi).fillna(fill).to_numpy(float)
    def _encode(self,d,fit=False,y=None):
        d=self._base(d); out={}; domain=d.is_source.to_numpy()
        if self.derived: out['user_prior_count']=self._history(d)
        if self.include_domain: out['is_source']=domain.astype(float)
        cols=list(self._columns(d)) if fit else self.columns
        if fit: self.columns=cols
        for c,kind in cols:
            s=d[c] if c in d else pd.Series(np.nan,index=d.index)
            if kind=='datetime_parts':
                if c=='earlies_credit_mon': continue # use derived history only
                for part,v in [('year',s.dt.year),('month',s.dt.month)]:
                    out[f'{c}_{part}']=self._numeric(v,f'{c}_{part}',fit)
                out[c+'__missing']=s.isna().to_numpy(float); continue
            if kind in ('target','frequency'):
                values=np.zeros(len(d),float)
                for dom in (0,1):
                    ix=np.flatnonzero(domain==dom); key=(c,dom)
                    if fit and len(ix):
                        enc=TargetEncoderCV(seed=self.seed) if kind=='target' else FrequencyEncoder()
                        values[ix]=(enc.fit_transform(s.iloc[ix],y[ix]) if kind=='target' else enc.fit(s.iloc[ix]).transform(s.iloc[ix])); self.encoders[key]=enc
                    elif len(ix):
                        enc=self.encoders.get(key); values[ix]=enc.transform(s.iloc[ix]) if enc is not None else (.5 if kind=='target' else 0.)
                out[c+'__'+kind]=values
            elif kind=='onehot':
                if fit: self.encoders[c]=OneHotMap().fit(s)
                for k,v in self.encoders[c].transform(s).items(): out[c+'__oh_'+k]=v
            else:
                v=OrdinalEncoderMap.transform(s,c) if kind=='ordinal' else num(s)
                if c in EARLY and self.scope=='common' and self.early_policy=='rank':
                    vals=np.zeros(len(d),float)
                    for dom in (0,1):
                        ix=np.flatnonzero(domain==dom); a=np.log1p(v.iloc[ix].where(v.iloc[ix]>=0)).to_numpy(float); key=(c,dom)
                        if fit: self.ranks[key]=np.sort(a[np.isfinite(a)])
                        ref=self.ranks.get(key,np.array([])); vals[ix]=np.where(np.isfinite(a)&(len(ref)>0),np.searchsorted(ref,a,side='right')/max(1,len(ref)),0.)
                    out[c+'__rank']=vals
                else: out[c]=self._numeric(v,c,fit)
                out[c+'__missing']=v.isna().to_numpy(float)
        result=pd.DataFrame(out,dtype='float32')
        if not np.isfinite(result.to_numpy()).all(): raise ValueError('Non-finite encoded feature')
        return result
    def fit_transform(self,d,y=None):
        y=np.asarray(d[LABEL] if y is None else y); self.numeric={};self.encoders={};self.ranks={}; self.user_history=None
        if self.derived and {'is_source','user_id','issue_date'}<=set(d):
            keys=['is_source','user_id','issue_date']
            h=d[keys].dropna().groupby(keys,sort=True).size().rename('_count').reset_index()
            h['_count']=h.groupby(keys[:2])['_count'].cumsum()
            self.user_history=h.sort_values('issue_date')
        X=self._encode(d,True,y); self.output_columns=list(X); return X
    def transform(self,d): return self._encode(d).reindex(columns=self.output_columns,fill_value=0)
