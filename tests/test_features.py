import numpy as np
import pandas as pd
from src.feature_map import unify_domain, LABEL
from src.features import TargetEncoderCV, FrequencyEncoder, FeaturePipeline
from src.dedup import near_duplicates
from src.reduce import FeatureReducer
from src.prepare import make_splits,prepare_fold,write_fold

def mapping():
    def m(t='numeric',e='none',scope='common',transferable=True):return dict(type=t,encoding=e,scope=scope,transferable=transferable,missing='median')
    return {'loan_id':m('id',transferable=False),'user_id':m('id',transferable=False),'total_loan':m(),'monthly_payment':m(),'industry':m('categorical','target'),'house_exist':m('categorical','onehot'),'title':m('categorical','frequency'),'class':m('ordinal','ordinal'),'debt_loan_ratio':m(),'del_in_18month':m(),'early_return_amount':m(),'issue_date':m('datetime','datetime_parts'),'earlies_credit_mon':m('datetime','datetime_parts',transferable=False),'f0':m()}

def raw(n=100,source=False,test=False):
    rng=np.random.default_rng(7+int(source)+2*int(test)); d=pd.DataFrame({'loan_id':np.arange(n)+1000*int(source)+2000*int(test),'user_id':np.arange(n)//2,'total_loan':rng.uniform(100,10000,n),'monthly_payment':rng.uniform(20,500,n),'industry':rng.choice(['A','B','C'],n),'house_exist':rng.choice([0,1],n),'title':rng.choice(['a','b','c'],n),'class':rng.choice(list('ABC'),n),'debt_loan_ratio':rng.uniform(0,30,n),'del_in_18month':rng.integers(0,4,n),'early_return_amount':rng.uniform(0,500,n),'issue_date':pd.date_range('2015-01-01',periods=n,freq='D').strftime('%Y/%m/%d'),'earlies_credit_mon':np.where(np.arange(n)%2==0,'Apr-90','1-Dec'),'f0':rng.uniform(0,1,n)})
    d.loc[0,'total_loan']=0;d.loc[1,'f0']=np.nan
    if not test:d[LABEL]=np.arange(n)%2
    if source:d=d.rename(columns={LABEL:'is_default','house_exist':'house_ownership'})
    return d

def data():return tuple(unify_domain(raw(n,source=dom=='internet',test=dom=='test'),dom,mapping()) for n,dom in [(100,'public'),(120,'internet'),(20,'test')])

def test_alignment_and_date():
    t,s,x=data();assert LABEL in s and 'is_default' not in s and LABEL not in x;assert t.house_exist.dtype==s.house_exist.dtype;assert t.earlies_credit_mon.isna().sum()==50

def test_target_encoding_oof_and_unseen():
    s=pd.Series(['a','b','c','d']);y=np.array([0,1,0,1]);enc=TargetEncoderCV(k=2);z=enc.fit_transform(s,y);assert np.allclose(z,.5);assert enc.transform(pd.Series(['unknown']))[0]==.5
    singleton=TargetEncoderCV().fit_transform(pd.Series(['x']),[1]);assert singleton[0]==.5

def test_frequency_unknown():
    enc=FrequencyEncoder(1).fit(pd.Series(['a','a','b']));assert np.isclose(enc.transform(pd.Series(['new']))[0],1/3)

def test_pipeline_validation_isolation():
    t,s,x=data();p=FeaturePipeline(mapping());a=p.fit_transform(t.iloc[:80]);v=t.iloc[80:].copy();b=p.transform(v);v[LABEL]=1-v[LABEL];assert p.transform(v).equals(b);assert list(a)==list(b);assert np.isfinite(a.to_numpy()).all();assert LABEL not in a and 'loan_id' not in a
    assert a.loc[0,'payment_loan_ratio__missing']==1
    assert a.loc[1,'user_prior_count']==1

def test_dedup_conflicts_retained():
    d=pd.DataFrame({'a':[1,1,1,9],'b':['x','x','x','z'],LABEL:[0,0,1,0]});keep,log,stats=near_duplicates(d,['a','b']);assert keep.tolist()==[True,False,True,True];assert stats['removed']==1 and stats['conflicts']>=1

def test_reduction():
    X=pd.DataFrame(np.random.default_rng(4).normal(size=(100,6)));p=FeatureReducer('pca').fit(X.iloc[:80]);v=p.transform(X.iloc[80:]);assert len(v)==20 and p.explained>=.9-1e-8
    s=FeatureReducer('svd').fit(X.iloc[:80]);assert len(s.transform(X.iloc[80:]))==20

def test_fold_end_to_end(tmp_path):
    t,s,x=data();f,ts=make_splits(t,tmp_path/'splits');assert make_splits(t,tmp_path/'splits')==(f,ts);assert not set(f[0]['train_idx'])&set(f[0]['val_idx']);assert t.iloc[ts['train_idx']].issue_date.max()<t.iloc[ts['val_idx']].issue_date.min()
    r=prepare_fold(t,s,x,f[0],mapping(),scope='common',use_source=True,source_limit=60,dedup=False);assert len(r['matrices']['train'])==140;assert len(r['matrices']['val'])==20;assert len(r['matrices']['test'])==20;assert LABEL not in r['meta']['test'];assert 'credit_history_months' not in r['matrices']['train'];write_fold(r,tmp_path/'out','csv');assert (tmp_path/'out/X_train_reduced.csv').exists()
    r2=prepare_fold(t,s,x,ts,mapping(),scope='common',use_source=True,dedup=False,derived=False,reduction='none');assert len(r2['matrices']['train'])==160;assert 'payment_loan_ratio' not in r2['matrices']['train']

def test_experiment_adapter(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from src.experiment import evaluate_fold,append_run
    t,s,x=data();f,_=make_splits(t,tmp_path/'splits');r=prepare_fold(t,s,x,f[0],mapping(),dedup=False,reduction='none');record=evaluate_fold(r,LogisticRegression(max_iter=1500,solver='liblinear'),group='smoke',model_name='LR',fold=0);assert 0<=record['auc']<=1;assert 0<=record['ks']<=1;append_run(record,tmp_path/'runs.csv');assert len(pd.read_csv(tmp_path/'runs.csv'))==1

def test_build_csv_io(tmp_path):
    from src.prepare import build
    rawdir=tmp_path/'raw';rawdir.mkdir()
    for name,n,source,test in [('train_public',100,False,False),('train_internet',120,True,False),('test_public',20,False,True)]:
        raw(n,source,test).to_csv(rawdir/(name+'.csv'),encoding='utf-8-sig',index=False)
    t,s,x=build(rawdir,tmp_path/'processed','csv',mapping());assert (len(t),len(s),len(x))==(100,120,20);assert (tmp_path/'processed/source_train.csv').exists()
