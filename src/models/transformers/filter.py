import scipy.signal as sps
from sklearn.base import BaseEstimator, TransformerMixin

class Filter(BaseEstimator, TransformerMixin):
    ''' Filters/smoothes data using Savitsky-Golay algorithm. '''
    def __init__(self, windowsize=7, polyorder=2):
        ''' Called when initializing the transformer.  '''
        self.windowsize = windowsize
        self.polyorder = polyorder   
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # return only the requested slice of the matrix
        Xdata = sps.savgol_filter(X, self.windowsize, self.polyorder, axis=1)
        return Xdata
