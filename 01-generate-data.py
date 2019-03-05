import os
import logging
import numpy as np
import pandas as pd
from src.data import generators

def generate_dataset(nitems, outpath, filename, gcurve):
    # generate the training datasets using specified baseline curve value     
    xvalues, yvalues, blexps, ydata = generators.data_generator(nitems, gcurve=gcurve) 

    # create dataframes from the generated dataset
    fname = os.path.join(outpath, filename)
    dftrain = pd.DataFrame(ydata, columns=xvalues.astype(int))
    dftrain.insert(0, 'blexp', blexps)
    dftrain.insert(0, 'level', yvalues)    
    dftrain.to_csv(fname, index_label='index', float_format='%.6f') 

def main(version, outpath, ntrain=150, ntest=50):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # define baseline curve arrays to loop over
    blnames = ["baseline", "baseline-neg", "baseline-pos"]
    blcurves = [0, -1, 1]

    # loop over all the baseline curves
    for i, c in enumerate(blcurves):
        # generate the training dataset using specified baseline curve value  
        fname = 'ds{0:04d}-{1}-train.csv'.format(version, blnames[i]) 
        generate_dataset(ntrain, outpath, fname, c)

        # generate the testing dataset using specified baseline curve value  
        fname = 'ds{0:04d}-{1}-test.csv'.format(version, blnames[i]) 
        generate_dataset(ntest, outpath, fname, c)        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # for dataset version one fixing the random to make generate reproducable
    np.random.seed(42)
    main(version=1, outpath='./data/generated')
