import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def fitpoly(y, xvals=None, polyorder=2):     
    coeffs = np.polyfit(xvals, y, polyorder)
    ypoly = np.polyval(coeffs, xvals)
    return ypoly

class Baseline(BaseEstimator, TransformerMixin):
    ''' Baseline correction using polynomial. '''
    def __init__(self, polyorder=2, weight=0.95, allowneg=False, negthreshold=0.0001, outbaseline=False):
        ''' Called when initializing the transformer.  '''
        self.polyorder = polyorder
        self.weight = weight
        self.allowneg = allowneg
        self.negthreshold = negthreshold
        self.outbaseline = outbaseline
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # determine the x-values by just using the X matrix shape value
        # (same logic used to generate the simulated data)
        x = np.arange(0, X.shape[1], 1.0)       

        # compute baseline for each row and return baseline corrected dataset
        Xbase = (np.apply_along_axis(func1d=fitpoly, axis=1, arr=X, xvals=x, polyorder=self.polyorder) * self.weight)
        Xdata = X - Xbase
        if not self.allowneg:
            # set all negative values to zero
            Xdata[Xdata < self.negthreshold] = 0.0
        
        # check to see if we need to output the baseline and not the corrected signal
        if self.outbaseline:
            return Xbase

        return Xdata
