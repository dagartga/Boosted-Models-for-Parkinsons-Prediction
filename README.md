parkinsons_prediction
==============================

## Overview

Using the first 12 months of doctor's visits where protein mass spectometry data has been recorded, the model is meant to assist doctors in determining whether a patient is likely to develop moderate-to-severe parkinsons for the UPDRS 1, 2, and 3. A categorical prediction of 1 means the patient is predicted to have moderate-to-severe UPDRS rating at some point in the future. A categorical prediction of 0 means the patient is predicted to have none-to-mild UPDRS ratings in the future. If a protein or peptide column is not present in the data, then it is given a value of 0, meaning it is not present in the sample. The visit month is defined as the months since the first recorded visit. It is necessary for predicting the UPDRS score with these models. The column upd23b_clinical_state_on_medication is based on whether the patient was taking medication during the clinical evaluation and can be values "On", "Off", or NaN.

- updrs 1 categorical ratings: 10 and below is mild, 11 to 21 is moderate, 22 and above is severe
- updrs 2 categorical ratings: 12 and below is mild, 13 to 29 is moderate, 30 and above is severe
- updrs 3 categorical ratings: 32 and below is mild, 33 to 58 is moderate, 59 and above is severe
- updrs 4 was dropped due to too few samples for training

### Project Write-Up
[Comparison of Three Boosting Models on Parkinsons Prediction](https://dagartga.github.io/parkinsons_project/)

### Data Source
The raw data can be found at [Kaggle Parkinsons Dataset](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data)

## To Use this Project

### Make Predictions
#### Option 1:
Take the Kaggle dataset and get predictions for each of the patients
- Take the file train_peptides.csv, train_proteins.csv, train_clinical_data.csv from Kaggle link in the Data Source section of this README. Place those files csv files in the ./data/raw/ directory
- Create a python virtual environment
- Use the Makefile to install requirements:<br> 
`$ make install`
- CD into the src/ directory and run the prediciton pipeline:<br>
`$ python pred_pipeline.py`
- This will process all of the raw data and run predictions with the trained models, which can be found in ./models/prod_models/, and a new file called full_updrs_preds.csv will be created in the ./data/predictions/ directory
   - The predictions will have the column names: 
    - "updrs_1_cat_preds" 
    - "updrs_2_cat_preds"
    - "updrs_3_cat_preds"


#### Option 2:
Use your own input of protein and peptide data that is a .json file with "visit_month", "patient_id", and the protein and peptide names:values. Or use the examples in ./data/api_examples/ to return a prediction.

- Create a virtual environment
- Install the dependencies:<br> 
`$ make install`
- Change to the src directory:<br> 
`$ cd src`
- Run the prediction pipeline file with your data filepath:<br> 
`$ python pred_pipeline_user_input.py file/path/to/data.json`
- The raw data and predictions are stored in ./data/predictions/ with the name {visit_id}_predictions.json
    - If "visit_id" is in the input data file keys then that will be used, otherwise "visit_id" is the {patient_id}_{visit_month}

#### Option 3:
Take user input of protein and peptide data and perform a prediction, or use the example .json files from ./data/api_examples/ to return a prediction.
- Build the docker image:<br> 
`docker build -t parkinsons-predict .`
- Confirm the docker images is listed:<br> 
`docker images parkinsons-predict`
- Run the docker container in port 5000:<br>
`docker run -p 5000:5000 -d --name parkinsons-predict parkinsons-predict`
- Confirm it is running by visiting http://localhost:5000 in the web browser
    - It should read "Welcome to the Parkinsons Prediction API"
- Run a automatic test prediction by visiting http://localhost:5000/test_predict
    - It should return a json string with the predictions and the visit_id
- Make an API request with the example data:<br>
`$ python api_request.py ./data/api_examples/16566_24_data.json`
- Make an API request with your own json data: <br>
`$ python api_request.py file/path/to/data.json`
- A json file will be stored in ./data/predictions/ with the name visit_id_prediction.json


### Notebook Descriptions

- Compare_medication_SMOTE_1yr_Models
    - This notebook compares the performance of models that were trained using SMOTE data, including the medication data, for visits that fall within 12 months or less. These models were tuned using the hyperopt package. Two of the final models are in this notebook.

- Compare_medication_SMOTE_1yr_Baseline_Models
    - This notebook compares the performance of models using the default parameters. The data for evaluation includes the SMOTE data and medication data for visits that fall within 12 months or less.

- Compare_SMOTE_1yr_Models:
    - This notebook compares the performance of models that were trained using SMOTE data, but without medication data, for visits that fall within 12 months or less. These models were tuned using the hyperopt package. One of the final models is in this notebook.

- Compare_finetune_1yr_Cat_Results:
    - This notebook compares the performance of models that were trained using data that did not have class imbalanced processing performed and without medication data for patient visits that fall within 12 months or less. These models were tuned manually.

- Compare_1yr_Cat_Results:
    - This notebook compares the performance of models using their default settings on data that was not preprocessed and does not include any medication data for patient visits that fall within 12 months or less.

- Compare_Categorical_Results:
    - This notebook compares the performance of models using their default settings on data that was not preprocessed and does not include any medication data and includes data from all patient visits.

- Raw_Data_EDA:

- Patient_Data_EDA:

- First_12_Months_EDA:

- EDA_RandomForest:

- 
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
