import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LevelMulti(BaseEstimator, TransformerMixin):
    ''' Encodes target value as a multi-classification value (low/normal/high) '''
    def __init__(self, targetmin=0.2, targetmax=0.8):
        ''' Called when initializing the transformer.  '''      
        self.targetmin = targetmin
        self.targetmax = targetmax   
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # set value to 0|1|2
        Xdata = np.zeros(X.shape[0], dtype=int)
        norm = np.logical_and(self.targetmin <= X, X <= self.targetmax)
        Xdata[norm] = 1
        Xdata[X > self.targetmax] = 2           
        
        return Xdata
