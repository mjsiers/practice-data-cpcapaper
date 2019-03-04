
from sklearn.base import BaseEstimator, TransformerMixin

class LevelBinary(BaseEstimator, TransformerMixin):
    ''' Encodes target value as a binary classification value (normal/abnormal) '''
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
            if (itemValue < self.targetmin) or (itemValue > self.targetmax):
                labelValue = 1                                                                                            
            else:
                labelValue = 0
    
            # add label to the array
            Xlabels.append(labelValue)

        return Xlabels