
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LevelBinary(BaseEstimator, TransformerMixin):
    ''' Encodes target value as a binary classification value (normal/abnormal) '''
    def __init__(self, targetmin=0.2, targetmax=0.8):
        ''' Called when initializing the transformer.  '''  
        self.targetmin = targetmin
        self.targetmax = targetmax   
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # set value to one if level is out of range
        Xdata = np.zeros(X.shape[0], dtype=int)
        outliers = np.logical_or(X < self.targetmin, X > self.targetmax)     
        Xdata[outliers] = 1 

        return Xdata
