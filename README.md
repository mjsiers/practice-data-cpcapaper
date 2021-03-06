Examine Use of Contrastive PCA on Simulated Spectral Data
=========================================================
The purpose of this learning project is to explore the use of the [**Contrastive PCA**](https://arxiv.org/abs/1709.06716) 
algorithm on simulated spectroscopy data.  The [**references**](./references) folder contains links too and copies of a 
few different papers that describe this variation on the standard PCA algorithm.

An important step in the processing of spectroscopy data is determining the best method for removing the 
baseline signal.  The details of the Raman signal can be obscured by the baseline signal which is always present and 
caused by the fluorescence of the sample.  This process is known as baseline correction.

The **Contrastive PCA** algorithm was developed to compute the PCA components by maximizing the target variance 
while trying to minimize the background variance.  This algorithm requires the use of two datasets.  The background
dataset provides the algorithm with samples to describe the patterns of the background signal.  The target dataset
will provide the full signal (target and background).

The primary goal of this project is to determine how well the Contrastive PCA algorithm can be used as a replacement 
for the baseline correction process.

## Project Tasks
- [X] Implement Python Generator to generate sample spectroscopy signal for analysis.
- [X] Implement Python Generator to generate multiple variations of sample baseline signals for analysis.
- [X] Implement Python Generator to generate target dataset (target signal and baseline signal).
- [X] Implement Python methods to generate baseline dataset (baseline extracted from target signal samples).
- [X] Generate analysis plots for comparing results between Standard PCA and Contrastive PCA.
- [X] Build models and compare results between Standard PCA and Contrastive PCA.

## Project Environment
This project was developed using Python.  The **requirements.txt** can be used to set up a new development
environment.  All development was done on Windows 10 using Anaconda and the Visual Studio Code editor.

## Project Results
The project results are available in a Jupyter Notebook format and are located in the [**notebooks**](./notebooks) 
folder.  Each stage of the project development is contained in a seperate sub-folder.  The Jupyter Notebook reports
are actually generated by exporting the output from Visual Studio Code.  The Python source file for each notebook 
is located in the **workbooks** folder.

The majority of the source code used to implement this project can be found in the [**src**](./src) folder.  The 
Python scripts in the root folder are used to generate and process the datasets used in this analysis.  The 
project final report is contained in the [**Model Analysis**](./notebooks/results/03-model-analysis.ipynb) notebook.

## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── generated      <- Simulated spectroscopy data generated from the included source code.
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │   └── workbooks      <- Python source files that are used to create the Jupyter notebook files (VSCode).
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data
    │   │              
    │   └── models         <- Scripts to process raw data files     
    │       ├── encoders
    │       ├── pipelines
    │       └── transformers    
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
