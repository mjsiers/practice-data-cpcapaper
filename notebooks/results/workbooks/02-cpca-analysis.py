#%% [markdown]
# ## Contrastive PCA Analysis Results
# The goal of this notebook is to document the Contrastive PCA results on the simulated datasets.

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn import metrics
from contrastive import CPCA
from src.models.encoders.levelmulti import LevelMulti

mpl.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (12, 9)

#%%
def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    ylevel = df['level'].values.copy()  
    #blexps = dfFile['blexp'].values.copy()       
    dfData = df.drop(['level', 'blexp'], axis=1).copy()
    return ylevel, dfData.values.copy()

def cpca_plot(dsver, dsname):
    ydata, Xdata = load_data('./data/processed/ds{0:04d}-{1}-train.csv'.format(dsver, dsname))
    ylabels = LevelMulti(targetmin=0.2, targetmax=0.8).transform(ydata.copy())

    _, Xback = load_data('./data/processed/ds{0:04d}-{1}-background-signal.csv'.format(dsver, dsname))
    CPCA().fit_transform(Xdata, Xback, plot=True, active_labels=ylabels)
    #CPCA().fit_transform(Xdata, Xback, plot=True, active_labels=ylabels, n_alphas=10, max_log_alpha=2, n_alphas_to_return=4)

    _, Xback = load_data('./data/processed/ds{0:04d}-{1}-background-nosignal.csv'.format(dsver, dsname))
    CPCA().fit_transform(Xdata, Xback, plot=True, active_labels=ylabels)  
    #CPCA().fit_transform(Xdata, Xback, plot=True, active_labels=ylabels, n_alphas=10, max_log_alpha=2, n_alphas_to_return=4)      

def cpca_score(dsver, dsname, bgname, alpha):
    _, Xback = load_data('./data/processed/ds{0:04d}-{1}-background-{2}.csv'.format(dsver, dsname, bgname))    
    ydata, Xdata = load_data('./data/processed/ds{0:04d}-{1}-train.csv'.format(dsver, dsname))
    ylabels = LevelMulti(targetmin=0.2, targetmax=0.8).transform(ydata.copy())   

    Xpca = CPCA().fit_transform(Xdata, Xback, alpha_selection='manual', alpha_value=alpha)  
    sscore = metrics.silhouette_score(Xpca, ylabels)  
    print('CPCA {0}-{1} Silhouette Score: {2:.4f} alpha={3:.2f}'.format(dsname.capitalize(), bgname.capitalize(), sscore, alpha))     

#%%
cpca_plot(2, 'filtered')

#%%
cpca_score(2, 'filtered', 'nosignal', 0.01)
cpca_score(2, 'filtered', 'nosignal', 46.42)
cpca_score(2, 'filtered', 'nosignal', 151.18)
cpca_score(2, 'filtered', 'nosignal', 492.39)

#%%
cpca_plot(2, 'baseline')

#%%
cpca_score(2, 'baseline', 'nosignal', 0.01)
cpca_score(2, 'baseline', 'nosignal', 58.78)
cpca_score(2, 'baseline', 'nosignal', 242.45)
cpca_score(2, 'baseline', 'nosignal', 789.65)
