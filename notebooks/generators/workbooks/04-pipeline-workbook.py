#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.pipeline import Pipeline
from src.data.generators import data_generator_levels, data_generator_signals
from src.models.transformers.filter import Filter
from src.models.transformers.baseline import Baseline
from src.models.transformers.truncate import Truncate

mpl.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (12, 5)

#%%
cvals = np.array([0.15, 0.50, 0.85])
xvalues, ylevel, blexps, ydata0 = data_generator_levels(cvals)
signals = data_generator_signals(cvals, noise=0.0)

xmin = 200
xmax = 450
datapipeline = Pipeline([
    ('filter', Filter(windowsize=15, polyorder=2)),
    ('baseline', Baseline(polyorder=3, weight=0.95, outbaseline=True)),  
    ('correct', Baseline(polyorder=3, weight=0.95)),             
    ('truncate', Truncate(xmin=xmin, xmax=xmax))   
])

ydata_fl = datapipeline.named_steps['filter'].transform(ydata0.copy())
ydata_bl = datapipeline.named_steps['baseline'].transform(ydata_fl.copy())
ydata_cs = datapipeline.named_steps['correct'].transform(ydata_fl.copy())
ydata_tr = datapipeline.named_steps['truncate'].transform(ydata_cs.copy())

#%%
for i in range(3):
    fig, axs = plt.subplots()
    axs.plot(xvalues, signals[i], label='signal') 
    axs.plot(xvalues, ydata0[i], label='raw')
    axs.plot(xvalues, ydata_bl[i], label='baseline')    
    fig.suptitle('Sample:[{0}] Baseline:[{1:.4f}] Target:[{2:.4f}]'.format(i, blexps[i], ylevel[i]))         
    plt.legend()

#%%
for i in range(3):
    fig, axs = plt.subplots()
    axs.plot(xvalues[xmin:xmax+1], signals[i, xmin:xmax+1], label='signal')    
    axs.plot(xvalues[xmin:xmax+1], ydata0[i, xmin:xmax+1], label='raw')
    axs.plot(xvalues[xmin:xmax+1], ydata_fl[i, xmin:xmax+1], label='filter')    
    axs.plot(xvalues[xmin:xmax+1], ydata_bl[i, xmin:xmax+1], label='baseline')    
    fig.suptitle('Sample:[{0}] Baseline:[{1:.4f}] Target:[{2:.4f}]'.format(i, blexps[i], ylevel[i]))         
    plt.legend()

#%%
for i in range(3):
    fig, axs = plt.subplots()
    axs.plot(xvalues[xmin:xmax+1], signals[i, xmin:xmax+1], label='signal')    
    #axs.plot(xvalues[xmin:xmax+1], ydata_cs[i, xmin:xmax+1], label='corrected')
    axs.plot(xvalues[xmin:xmax+1], ydata_tr[i], label='corrected')
    fig.suptitle('Sample:[{0}] Baseline:[{1:.4f}] Target:[{2:.4f}]'.format(i, blexps[i], ylevel[i]))         
    plt.legend()
