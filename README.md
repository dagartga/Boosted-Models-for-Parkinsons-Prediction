parkinsons_prediction
==============================

## Overview

Using the first 12 months of doctor's visits where protein mass spectometry data has been recorded, the model is meant to assist doctors in determining whether a patient is likely to develop moderate-to-severe parkinsons for the UPDRS 1, 2, and 3. A categorical prediction of 1 means the patient is predicted to have moderate-to-severe UPDRS rating at some point in the future. A categorical prediction of 0 means the patient is predicted to have none-to-mild UPDRS ratings in the future. If a protein or peptide column is not present in the data, then it is given a value of 0, meaning it is not present in the sample. The visit month is defined as the months since the first recorded visit. It is necessary for predicting the UPDRS score with these models. The column upd23b_clinical_state_on_medication is based on whether the patient was taking medication during the clinical evaluation and can be values "On", "Off", or NaN.

- updrs 1 categorical ratings: 10 and below is mild, 11 to 21 is moderate, 22 and above is severe
- updrs 2 categorical ratings: 12 and below is mild, 13 to 29 is moderate, 30 and above is severe
- updrs 3 categorical ratings: 32 and below is mild, 33 to 58 is moderate, 59 and above is severe
- updrs 4 was dropped due to too few samples for training

### Data Source
The raw data can be found at [Kaggle Parkinsons Dataset](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data)

## To Use this Project

### Make Predictions
#### Option 1:
Take the Kaggle dataset and get predictions for each of the patients
- Take the file train_peptides.csv, train_proteins.csv, train_clinical_data.csv from Kaggle link in the Data Source section of this README. Place those files csv files in the ./data/raw/ directory
- Create a python virtual environment
- Use the Makefile to install requirements: `$ make install`
- CD into the src/ directory and run the prediciton pipeline: `$ python pred_pipeline.py`
- This will process all of the raw data and run predictions with the trained models, which can be found in ./models/prod_models/, and a new file called full_updrs_preds.csv will be created in the ./data/predictions/ directory
   - The predictions will have the column names: 
    - "updrs_1_cat_preds" 
    - "updrs_2_cat_preds"
    - "updrs_3_cat_preds"

#### Option 2:
Take user input of protein and peptide data and perform a prediction, or use the example .json files from ./data/api_examples/ to return a prediction.
- Build the docker image
- Run the docker container in port 5000
- Confirm it is running by visiting http://localhost:5000 in the web browser
- Make an API request: `$ python api_request.py`
- A json file will be stored in ./data/predictions/ with the name visit_id_prediction.json


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>