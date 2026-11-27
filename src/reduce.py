"""Train-fold fit only. PCA or uncentered SVD, with train-derived scaling."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

class FeatureReducer:
    def __init__(self,method='pca',variance=.90,seed=2026):
        if method not in ('pca','svd'): raise ValueError(method)
        if not 0<variance<1: raise ValueError(variance)
        self.method=method;self.variance=variance;self.seed=seed
    def fit(self,X):
        self.scaler=StandardScaler(with_mean=self.method=='pca').fit(X)
        A=self.scaler.transform(X)
        if min(A.shape)<2: raise ValueError('Need at least two rows and features')
        if self.method=='pca':
            self.model=PCA(n_components=self.variance,svd_solver='full').fit(A); self.n=self.model.n_components_
        else:
            k=min(A.shape)
            self.model=TruncatedSVD(n_components=k,algorithm='randomized',random_state=self.seed).fit(A)
            cumulative=np.cumsum(self.model.explained_variance_ratio_)
            self.n=min(k,int(np.searchsorted(cumulative,self.variance))+1)
        self.explained=float(np.sum(self.model.explained_variance_ratio_[:self.n]));return self
    def transform(self,X):
        A=self.model.transform(self.scaler.transform(X))[:,:self.n]
        return pd.DataFrame(A,dtype='float32',columns=[f'{self.method}_{i+1:03}' for i in range(A.shape[1])])
    def fit_transform(self,X): return self.fit(X).transform(X)
