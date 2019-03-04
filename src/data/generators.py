import os
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd 
from scipy.stats import norm 

logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data")

def baseline_generator(num, x, gcurve=0, noise=0.00100):
    # initialize the output array for the specified number of curves
    baselines = np.zeros((num, x.shape[0]))
    logger.info('BCurves shape: [%s]', baselines.shape)    
    for i in range(num):
        # generate random value for the exponent and compute baseline curve
        bexp = np.random.uniform(2.1, 2.2)
        logger.debug('Baseline Exponent: [%.4f]', bexp)
        blc = (-1e-7*x**bexp)
        bl = blc + np.min(blc)*-1.0

        # determine if we need to add/subtract in the gaussian curve
        if gcurve != 0:
            # define the optional gaussian curve
            S_3 = norm.pdf(x, loc=360.0, scale=120.0)
            if gcurve < 0:
                logger.debug('Baseline Curve: [NEG]')                     
                bl = bl - S_3        
            else:
                logger.debug('Baseline Curve: [POS]')                     
                bl = bl + S_3

        # determine if we need to add in some random noise
        if (noise > 0.0001): 
            xnum = x.shape[0]           
            bnoise = noise * np.random.normal(size=xnum)
            bl = bl + bnoise

        # save off the generated baseline curve into the output array
        baselines[i] = bl

    return baselines

def signal_generator(x, cpeaks, noise=0.00075):
    # create the required signal curves
    S_1 = norm.pdf(x, loc=310.0, scale=40.0)
    S_2 = norm.pdf(x, loc=390.0, scale=20.0)
    S_true = np.vstack((S_1, S_2))

    # initialize the output array for the specified number of curves
    cnum = cpeaks.shape[0]
    signals = np.zeros((cnum, x.shape[0]))  
    logger.info('Signals shape: [%s]', signals.shape)         
    for i in range(cnum):     
        # generate the signals from the input concentration levels
        s = np.dot(cpeaks[i], S_true)
        if (noise > 0.0001):
            # generate the random noise  
            snoise = noise * np.random.normal(size=x.shape[0])
            s = s + snoise

        # save out the generated signal
        signals[i] = s

    return signals

def data_generator(cnum, xnum=600, gcurve=0):
    # setup the x-axis values
    x = np.arange(0, xnum, 1.0)

    # generate some random concentration levels and compute weight value for each signal peak
    c = np.random.random(cnum)
    cpeaks = np.vstack((c, (1.0-c))).T
    logger.info('CLevels shape: [%s]', cpeaks.shape)             

    # generate the requested baselines and signals
    baselines = baseline_generator(cnum, x, gcurve)    
    signals = signal_generator(x, cpeaks)
    results = baselines+signals
    logger.info('Results shape: [%s]', results.shape)         

    return x, c, results

if __name__ == "__main__":
    x, targets, signals = data_generator(5, gcurve=0)
    x, targets, signals = data_generator(10, gcurve=1)
    x, targets, signals = data_generator(15, gcurve=-1)
