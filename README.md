Examine Use of Contrastive PCA Algorihtm on Simulated Spectroscopy Data
=======================================================================
The purpose of this learning project is to explore the use of the [**Contrastive PCA**](https://arxiv.org/abs/1709.06716) 
algorithm on simulated spectroscopy data.  The references folder contains links too and copies of a few different papers 
that describe this variation on the standard PCA algorithm.

An important step in the processing of Raman spectroscopy data is determining the best method for removing the 
background signal.  The Raman signal can be obscured by the background signal which is always present and caused by the
fluorescence of the sample.

The **Contrastive PCA** algorithm was developed to compute the PCA components by maximizing the target variance 
while trying to minimize the background variance.  This algorithm requires the use of two datasets.  The background
dataset provides the algorithm with samples to describe the patterns of the background signal.  The target dataset
will provide the full signal (target and background).

## Project Goals
------------
- [ ] Implement Python Generator to generate sample spectroscopy signal for analysis.
- [ ] Implement Python Generator to generate multiple variations of sample background signals found with spectroscopy data.
- [ ] Generate analysis plots for comparing results between Standard PCA and Contrastive PCA.
- [ ] Generate anomaly detection models and compare results between Standard PCA and Contrastive PCA.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── generated      <- Simulated spectroscopy data generated from the included source code.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
