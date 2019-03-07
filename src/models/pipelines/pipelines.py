#import numpy as np
#from scipy.spatial.distance import euclidean
#from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from src.models.transformers.filter import Filter
from src.models.transformers.baseline import Baseline
from src.models.transformers.truncate import Truncate

def baseline_pipeline(skipbaseline=False, outbaseline=False):
    pipeline = Pipeline([
        ('filter', Filter(windowsize=15, polyorder=2))    
    ])
    if not skipbaseline:
        pipeline.steps.append(('baseline', Baseline(polyorder=3, weight=0.95, outbaseline=outbaseline)))    
    return pipeline

def preprocess_pipeline(skipbaseline=False, xmin=200, xmax=450):
    # get baseline pipeline and append on the truncate step
    pipeline = baseline_pipeline(skipbaseline=skipbaseline, outbaseline=False)
    pipeline.steps.append(('truncate', Truncate(xmin=xmin, xmax=xmax)))
    return pipeline

def pca_pipeline(ncomponents=0.95, scalestd=True):
    pipeline = Pipeline([
        ('scale', StandardScaler(with_mean=True, with_std=scalestd)),
        ('project', PCA(n_components=ncomponents, whiten=False))
    ])
    return pipeline

def knn_estimator(pipeline, nneighbors=5):
    if pipeline is None:
        pipeline = Pipeline([])    
    pipeline.steps.append(('predict', KNeighborsClassifier(n_neighbors=nneighbors)))
    return pipeline

def rf_estimator(pipeline, nestimators=10):
    if pipeline is None:
        pipeline = Pipeline([])
    pipeline.steps.append(('predict', RandomForestClassifier(n_estimators=nestimators)))
    return pipeline
