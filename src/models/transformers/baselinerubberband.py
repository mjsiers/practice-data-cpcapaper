import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from sklearn.base import BaseEstimator, TransformerMixin

# https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data

def fitband(y, usespline=True, usepoly=False, polyorder=5):
    x = np.arange(0, y.shape[0], 1.0) 

    # find the convex hull
    pts = np.array(list(zip(x, y)))
    v = ConvexHull(pts).vertices 

    # rotate convex hull vertices until they start from the lowest one
    v = np.roll(v, -v.argmin())
    # leave only the ascending ones
    v = v[:v.argmax()]

    if usespline:
        # fit spline to the convex hull
        coeffs = UnivariateSpline(x[v], y[v], s=None)
        baseline = coeffs(x)
    elif usepoly:
        # fit polynomial to the convex hull data only
        polycoeffs = np.polyfit(x[v], y[v], polyorder)
        baseline = np.polyval(polycoeffs, x)
    else:
        # create baseline using linear interpolation between vertices
        baseline = np.interp(x, x[v], y[v])
    
    return baseline

class BaselineRubberband(BaseEstimator, TransformerMixin):
    ''' Baseline correction using rubberband method. '''
    def __init__(self, weight=0.95):
        ''' Called when initializing the transformer.  '''
        self.weight = weight        
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # compute baseline for each row and return baseline corrected dataset
        Xbase = np.apply_along_axis(func1d=fitband, axis=1, arr=X)
        Xdata = X - Xbase
        return Xbase      
