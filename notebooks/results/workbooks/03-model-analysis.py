#%% [markdown]
# ## Classification Model Analysis
# The goal of this notebook is to build some different classification models using both the
# datasets from both PCA algorithms (normal and contrastive). The model results will be used
# as another comparison point between the datasets. 

#%%
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from contrastive import CPCA
from src.models.encoders.levelmulti import LevelMulti
from src.models.pipelines.pipelines import pca_pipeline

mpl.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (12, 9)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

#%%
def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    ylevel = df['level'].values.copy()      
    dfData = df.drop(['level', 'blexp'], axis=1).copy()
    return ylevel, dfData.values.copy()

def pca_data(dsver, dsname, dstype='train'):
    ydata, Xdata = load_data('./data/processed/ds{0:04d}-{1}-{2}.csv'.format(dsver, dsname, dstype))
    ylabels = LevelMulti(targetmin=0.2, targetmax=0.8).transform(ydata.copy())   

    ppca = pca_pipeline(ncomponents=2)
    Xpca = ppca.fit_transform(Xdata)
    return ylabels, Xpca     

def cpca_data(dsver, dsname, alpha, dstype='train', bgname='nosignal'):
    _, Xback = load_data('./data/processed/ds{0:04d}-{1}-background-{2}.csv'.format(dsver, dsname, bgname))    
    ydata, Xdata = load_data('./data/processed/ds{0:04d}-{1}-{2}.csv'.format(dsver, dsname, dstype))
    ylabels = LevelMulti(targetmin=0.2, targetmax=0.8).transform(ydata.copy())   

    Xpca = CPCA().fit_transform(Xdata, Xback, alpha_selection='manual', alpha_value=alpha)  
    return ylabels, Xpca

#%%
def fit_model(name, estimator, ytrain, Xtrain, ytest, Xtest):
    estimator.fit(Xtrain.copy(), ytrain)
    ypred = estimator.predict(Xtest.copy())

    cmatrix = confusion_matrix(ytest, ypred)
    print('{0} CF Matrix:'.format(name))
    print(cmatrix)

    labels = ['low', 'norm', 'high'] 
    print(classification_report(ytest, ypred, target_names=labels))

    f1score = f1_score(ytest, ypred, average='macro')
    print('{0} F1 Score: {1:.4f}\n'.format(name, f1score))       

def run_models(ytrain, Xtrain, ytest, Xtest):
    estimator = KNeighborsClassifier(n_neighbors=5)
    fit_model('KNN', estimator, ytrain, Xtrain, ytest, Xtest)

    #estimator = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3,)
    #fit_model('DTREE', estimator, ytrain, Xtrain, ytest, Xtest)

    parms = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}
    estimator = GridSearchCV(SVC(kernel='rbf'), param_grid=parms, cv=10, scoring='accuracy') 
    fit_model('SVC', estimator, ytrain, Xtrain, ytest, Xtest)   

#%% [markdown]
# ## PCA Model Analysis
# TBD. 

#%%
ytrain, Xtrain = pca_data(2, 'baseline', dstype='train')
ytest, Xtest = pca_data(2, 'baseline', dstype='test')
run_models(ytrain, Xtrain, ytest, Xtest)

#%%
ytrain, Xtrain = pca_data(2, 'filtered', dstype='train')
ytest, Xtest = pca_data(2, 'filtered', dstype='test')
run_models(ytrain, Xtrain, ytest, Xtest)

#%% [markdown]
# ## Contrastive PCA Model Analysis
# TBD. 

#%%
alpha = 151.18
ytrain, Xtrain = cpca_data(2, 'filtered', dstype='train', alpha=alpha)
ytest, Xtest = cpca_data(2, 'filtered', dstype='test', alpha=alpha)
run_models(ytrain, Xtrain, ytest, Xtest)

#%%
alpha = 46.42
ytrain, Xtrain = cpca_data(2, 'filtered', dstype='train', alpha=alpha)
ytest, Xtest = cpca_data(2, 'filtered', dstype='test', alpha=alpha)
run_models(ytrain, Xtrain, ytest, Xtest)

#%%
alpha = 58.78
ytrain, Xtrain = cpca_data(2, 'baseline', dstype='train', alpha=alpha)
ytest, Xtest = cpca_data(2, 'baseline', dstype='test', alpha=alpha)
run_models(ytrain, Xtrain, ytest, Xtest)

#%% [markdown]
# ## Classification Model Results
# TBD. 
