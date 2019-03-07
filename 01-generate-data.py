import os
import logging
import numpy as np
import pandas as pd
from src.data import generators

def generate_dataset(nitems, outpath, filename, baselineonly=False):
    # generate the training datasets using specified baseline curve value     
    xvalues, yvalues, blexps, ydata = generators.data_generator(nitems, baselineonly=baselineonly) 

    # create dataframes from the generated dataset
    fname = os.path.join(outpath, filename)
    dftrain = pd.DataFrame(ydata, columns=xvalues.astype(int))
    dftrain.insert(0, 'blexp', blexps)
    dftrain.insert(0, 'level', yvalues)
    dftrain.to_csv(fname, index_label='index', float_format='%.6f') 

def main(version, outpath, ntrain=150, nbackground=150, ntest=50):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('generating simulated raw spectral data')

    # generate the training dataset using specified baseline curve value  
    fname = 'ds{0:04d}-raw-train.csv'.format(version) 
    generate_dataset(ntrain, outpath, fname)

    # generate the testing dataset using specified baseline curve value  
    fname = 'ds{0:04d}-raw-test.csv'.format(version) 
    generate_dataset(ntest, outpath, fname) 

    # generate the background training dataset using specified baseline curve value  
    fname = 'ds{0:04d}-raw-background.csv'.format(version) 
    generate_dataset(nbackground, outpath, fname, baselineonly=False)  
    fname = 'ds{0:04d}-raw-background-baseline.csv'.format(version) 
    generate_dataset(nbackground, outpath, fname, baselineonly=True)             

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # for dataset version one fixing the random to make generate reproducable
    np.random.seed(42)
    main(version=1, outpath='./data/generated')    
    np.random.seed(43)
    main(version=2, outpath='./data/generated')
