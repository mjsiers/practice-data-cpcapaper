#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import norm    
from sklearn.pipeline import Pipeline
from src.models.transformers.filter import Filter
from src.models.transformers.baseline import Baseline
from src.models.transformers.truncate import Truncate

mpl.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (12, 5)

#%%
xnum = 600
x = np.arange(0, xnum, 1.0)

# setup the signal generation distributions
S_1 = norm.pdf(x, loc=310.0, scale=40.0)
S_2 = norm.pdf(x, loc=390.0, scale=20.0)
S_true = np.vstack((S_1, S_2))

#%%
dfFile = pd.read_csv('./data/generated/ds0001-base0-train.csv', index_col=0)
ylevel = dfFile['level'].values.copy()
blexps = dfFile['blexp'].values.copy()
dfX = dfFile.drop(['level', 'blexp'], axis=1).copy()
xvalues = dfX.columns.values.astype(float)

xmin = 200
xmax = 450
datapipeline = Pipeline([
    ('filter', Filter(windowsize=15, polyorder=2)),
    ('baseline', Baseline(polyorder=3, weight=0.95)),        
    ('truncate', Truncate(xmin=xmin, xmax=xmax))   
])

ydata0 = dfX.values.copy()
ydata1 = datapipeline.named_steps['filter'].transform(ydata0.copy())
ydata2 = datapipeline.named_steps['baseline'].transform(ydata1.copy())
ydata3 = datapipeline.named_steps['truncate'].transform(ydata2.copy())

for i in range(10):
    target = ylevel[i]
    C_true = np.array([[target, 1.-target]])
    signal = np.dot(C_true, S_true)

    fig, axs = plt.subplots()
    axs.plot(xvalues[xmin:xmax+1], signal[0, xmin:xmax+1], label='signal')    
    #axs.plot(xvalues[xmin:xmax+1], ydata0[i, xmin:xmax+1], label='raw')
    #axs.plot(xvalues[xmin:xmax+1], ydata1[i, xmin:xmax+1], label='filter')
    axs.plot(xvalues[xmin:xmax+1], ydata2[i, xmin:xmax+1], label='baseline')
    #axs.plot(xvalues[xmin:xmax+1], ydata3[i], label='trunc')
    fig.suptitle('Sample:[{0}] Baseline:[{1:.4f}] Target:[{2:.4f}]'.format(i, blexps[i], target))         
    plt.legend()
