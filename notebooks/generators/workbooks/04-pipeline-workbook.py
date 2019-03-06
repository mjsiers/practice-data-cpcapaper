#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.pipeline import Pipeline
from src.models.transformers.filter import Filter
from src.models.transformers.baseline import Baseline
from src.models.transformers.truncate import Truncate

mpl.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (12, 5)

#%%
dfFile = pd.read_csv('./data/generated/ds0001-base0-train.csv', index_col=0)
ylevel = dfFile['level'].values.copy()
blexps = dfFile['blexp'].values.copy()
dfX = dfFile.drop(['level', 'blexp'], axis=1).copy()
xvalues = dfX.columns.values.astype(float)

xmin = 200
xmax = 450
datapipeline = Pipeline([
    ('filter', Filter(windowsize=17, polyorder=3)),
    ('baseline', Baseline(polyorder=3, weight=0.95)),        
    ('truncate', Truncate(xmin=xmin, xmax=xmax))   
])

ydata0 = dfX.values.copy()
ydata1 = datapipeline.named_steps['filter'].transform(ydata0.copy())
ydata2 = datapipeline.named_steps['baseline'].transform(ydata1.copy())
ydata3 = datapipeline.named_steps['truncate'].transform(ydata2.copy())

for i in range(10):
    fig, axs = plt.subplots()
    axs.plot(xvalues[xmin:xmax+1], ydata0[i, xmin:xmax+1], label='raw')
    axs.plot(xvalues[xmin:xmax+1], ydata1[i, xmin:xmax+1], label='filter')
    axs.plot(xvalues[xmin:xmax+1], ydata2[i, xmin:xmax+1], label='baseline')
    axs.plot(xvalues[xmin:xmax+1], ydata3[i], label='trunc')
    fig.suptitle('Sample:[{0}] Baseline:[{1:.4f}] Target:[{2:.4f}]'.format(i, blexps[i], ylevel[i]))         
    plt.legend()
