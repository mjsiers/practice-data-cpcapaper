import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data

class BaselineRubberband(BaseEstimator, TransformerMixin):
    ''' Baseline correction using rubberband method. '''
    def __init__(self, xmin=200, xmax=400):
        ''' Called when initializing the transformer.  '''
        self.xmin = xmin
        self.xmax = xmax   
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
