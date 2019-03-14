#%% [markdown]
# ## Classification Model Analysis
# The goal of this notebook is to build some different classification models using the
# datasets from both PCA algorithms (normal and contrastive). The model results will be used
# as another comparison point between the datasets. 

#%%
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from contrastive import CPCA
from src.models.encoders.levelmulti import LevelMulti
from src.models.pipelines.pipelines import pca_pipeline

mpl.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (12, 9)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

#%% [markdown]
# ### Data Processing
# The following functions are used to load and transform the data into the format needed for
# training the models.  The model input data will only consist of the first two components
# from both PCA algorithms.

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

    Xpca = CPCA(n_components=2).fit_transform(Xdata, Xback, alpha_selection='manual', alpha_value=alpha) 
    return ylabels, Xpca

#%% [markdown]
# ### Model Training and Testing
# The following functions are used to evaluate a set of different machine learning models.  After 
# a model has been trained, the test dataset will be passed into the model to determine how well the
# model performed.  A confusion matrix for the test dataset will be displayed along with some other 
# model scoring metrics.
#
# Currently only two types of models will be evaluated. The K-Nearest Neighbors algorithm and
# the Support Vector Machine Classifier algorithm.  A small parameter grid search is performed
# with the Support Vector Machine algorithm to provide the best model results for comparison
# between the datasets.

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

    parms = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}
    estimator = GridSearchCV(SVC(kernel='rbf'), param_grid=parms, cv=10, scoring='accuracy') 
    fit_model('SVC', estimator, ytrain, Xtrain, ytest, Xtest)   

#%% [markdown]
# ### PCA Model Analysis
# The two different model types will be evaluated on both the filtered and baselined PCA datasets.  The
# following cells will build and evaluate the models on these datasets.  The results will be displayed
# in the cell outputs.

#%%
ytrain, Xtrain = pca_data(2, 'filtered', dstype='train')
ytest, Xtest = pca_data(2, 'filtered', dstype='test')
run_models(ytrain, Xtrain, ytest, Xtest)

#%%
ytrain, Xtrain = pca_data(2, 'baseline', dstype='train')
ytest, Xtest = pca_data(2, 'baseline', dstype='test')
run_models(ytrain, Xtrain, ytest, Xtest)

#%% [markdown]
# ### PCA Model Results
# The cells above show that for the simulated data, the baselined corrected dataset performs
# best with the PCA algorithm.  On this dataset, the K-Nearest Neighbors model gives the highest
# F1 score of **0.9637** and only misclassifies 3 test samples.  The table below shows the results
# from the best model on each dataset.
#
# | Dataset       | Model     | F1 Score   | Misclassified Samples |
# |:--------------|:----------|:-----------|:----------------------|
# | filtered      | SVC       | 0.8605     | 11                    |
# | **baseline**  | **KNN**   | **0.9637** | 3                     |

#%% [markdown]
# ### Contrastive PCA Model Analysis
# The following cells perform the model evaluations on the Contrastive PCA datasets.  The filtered 
# dataset will be evaluated at two different alpha values.  The results will be displayed
# in the cell outputs.

#%%
alpha = 151.18
ytrain, Xtrain = cpca_data(2, 'filtered', dstype='train', alpha=alpha)
ytest, Xtest = cpca_data(2, 'filtered', dstype='test', alpha=alpha)
run_models(ytrain, Xtrain, ytest, Xtest)

#%%
alpha = 58.78
ytrain, Xtrain = cpca_data(2, 'baseline', dstype='train', alpha=alpha)
ytest, Xtest = cpca_data(2, 'baseline', dstype='test', alpha=alpha)
run_models(ytrain, Xtrain, ytest, Xtest)

#%% [markdown]
# ### Contrastive PCA Model Results
# The cells above show that for the simulated data, the baseline dataset also performs better 
# for the Contrastive PCA algorithm.  On both datasets, the Support Vector Machine Classifier gives 
# the higher F1 score.   The table below shows the results from the best model on each dataset.
#
# | Dataset       | Model     | F1 Score   | Misclassified Samples |
# |:--------------|:----------|:-----------|:----------------------|
# | filtered      | SVC       | 0.8569     | 11                    |
# | **baseline**  | **SVC**   | **0.9507** | 4                     |

#%% [markdown]
# ### Classification Model Results
# On this simulated dataset, the Contrastive PCA algorithm was not able to out perform the standard
# PCA algorithm on a baseline corrected dataset.  This experiment should probably be executed again
# using a more complex baseline to see if it would perform better under these conditions.  
