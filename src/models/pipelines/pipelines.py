import numpy as np
#from scipy.spatial.distance import euclidean
#from scipy.spatial.distance import mahalanobis
#from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from src.models.encoders import LevelBinary
from src.models.encoders import LevelMulti
from src.models.transformers import Truncate
from src.models.transformers import BaselinePoly
from src.models.transformers import BaselineRubberband

def analysis():
    pipeline = Pipeline([
        ('baseline', BaselineRubberband()),          
        ('truncate', Truncate())
    ])
    return pipeline
