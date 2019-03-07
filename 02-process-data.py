import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.pipelines.pipelines import preprocess_pipeline

def save_dataset(df, filename, outpath):
    fname = os.path.join(outpath, filename)
    df.to_csv(fname, index_label='index', float_format='%.6f') 

def transform_dataset(fname, skipbaseline=False, xmin=200, xmax=450):
    # read in the CSV file and ensure we drop the concentration level column
    dfFile = pd.read_csv(fname, index_col=0)
    ylevel = dfFile['level'].values.copy()  
    blexps = dfFile['blexp'].values.copy()       
    dfX = dfFile.drop(['level', 'blexp'], axis=1).copy()

    # get the pipeline and transform the data
    pipeline = preprocess_pipeline(skipbaseline=skipbaseline, xmin=xmin, xmax=xmax)
    Xdata = pipeline.transform(dfX.values)

    # covert data to dataframe in order to save the output to a file
    xcols = np.arange(xmin, xmax+1, dtype=int)      
    dfProcessed = pd.DataFrame(Xdata, columns=xcols)
    dfProcessed.insert(0, 'blexp', blexps)     
    dfProcessed.insert(0, 'level', ylevel)   
    return dfProcessed

def main(version, inpath, outpath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('processing simulated raw spectral data files')

    # loop over all the file names/types
    files = ['train', 'background', 'background-baseline', 'test']
    for f in files:
        # determine the current file and ensure it exists
        fname = 'ds{0:04d}-raw-{1}.csv'.format(version, f)
        fpath = os.path.join(inpath, fname)
        fobj = Path(fpath)
        if fobj.exists():
            # determine output file name and transform/save the data file
            ofname = 'ds{0:04d}-filtered-{1}.csv'.format(version, f)
            dfData = transform_dataset(fpath, skipbaseline=True)
            save_dataset(dfData, ofname, outpath)

            # determine output file name and transform/save the data file
            ofname = 'ds{0:04d}-baseline-{1}.csv'.format(version, f)            
            dfData = transform_dataset(fpath, skipbaseline=False) 
            save_dataset(dfData, ofname, outpath)           

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # for dataset version one fixing the random to make generate reproducable
    np.random.seed(42)
    main(version=1, inpath='./data/generated', outpath='./data/processed') 
    np.random.seed(43)
    main(version=2, inpath='./data/generated', outpath='./data/processed') 
    