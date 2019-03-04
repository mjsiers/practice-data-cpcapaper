
from sklearn.base import BaseEstimator, TransformerMixin

class LevelMulti(BaseEstimator, TransformerMixin):
    ''' Encodes target value as a multi-classification value (low/normal/high) '''
    def __init__(self, target="level", targetmin=0.2, targetmax=0.8):
        ''' Called when initializing the transformer.  '''
        self.target = target        
        self.targetmin = targetmin
        self.targetmax = targetmax   
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # create array to hold values
        Xlabels = []

        # loop over all the data rows
        for idx, row in X.iterrows():
            itemValue = row[self.target]
            if itemValue < self.targetmin:
                labelValue = 0                  
            elif itemValue <= self.targetmax:
                labelValue = 1                                                                              
            else:
                labelValue = 2
    
            # add label to the array
            Xlabels.append(labelValue)

        return Xlabels