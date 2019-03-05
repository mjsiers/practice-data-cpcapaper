import os
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.models.transformers.filter import Filter
from src.models.transformers.baselinepoly import BaselinePoly
from src.models.transformers.truncate import Truncate

def main(fname='', xmin=200, xmax=450):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read in the CSV file and ensure we drop the concentration level column
    dfFile = pd.read_csv(fname, index_col=0)
    Xlevel = dfFile['level'].values.copy()    
    dfX = dfFile.drop(['level'], axis=1).copy()

    # setup pipeline and transform the data
    datapipeline = Pipeline([
        ('filter', Filter(windowsize=17, polyorder=3)),
        ('baseline', BaselinePoly(polyorder=3, weight=0.95)),        
        ('truncate', Truncate(xmin=xmin, xmax=xmax))  
    ])
    Xdata = datapipeline.transform(dfX.values)

    # covert data to dataframe in order to save the output to a file
    xcols = np.arange(xmin, xmax+1, dtype=int)      
    dfProcessed = pd.DataFrame(Xdata, columns=xcols)
    dfProcessed.insert(0, 'level', Xlevel) 
    print(dfProcessed.head())
 

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(fname='./data/generated/ds0001-baseline-train.csv')
