import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import json
from webapp.pred_pipeline_user_input_app import get_all_updrs_preds


def load_lottiefile(filepath:str):
    with open(filepath,'r') as f:
        lottie_json = json.load(f)
    return lottie_json

filename = load_lottiefile("doctor_animation.json")
st_lottie(filename, speed=1, height=200)

st.title('Parkinsons Severity Prediction')

tab1, tab2, tab3 = st.tabs(['Overview', 'Prediction', 'Data'])


with tab1:
    st.header('Project Overview')
    """
    Using the first 12 months of doctor's visits where protein mass spectometry data has been recorded, 
    the model is meant to assist doctors in determining whether a patient is likely to develop 
    moderate-to-severe parkinsons for the UPDRS 1, 2, and 3. 
    
    A categorical prediction of 1 means the patient is predicted to have moderate-to-severe UPDRS rating 
    at some point in the future. A categorical prediction of 0 means the patient is predicted to have 
    none-to-mild UPDRS ratings in the future. If a protein or peptide column is not present in the data, 
    then it is given a value of 0, meaning it is not present in the sample. The visit month is defined as 
    the months since the first recorded visit. It is necessary for predicting the UPDRS score with these 
    models. 
    
    The column upd23b_clinical_state_on_medication is based on whether the patient was taking medication 
    during the clinical evaluation and can be values "On", "Off", or NaN.

    - **UPDRS 1 categorical ratings**: 10 and below is mild, 11 to 21 is moderate, 22 and above is severe
    - **UPDRS 2 categorical ratings**: 12 and below is mild, 13 to 29 is moderate, 30 and above is severe
    - **UPDRS 3 categorical ratings**: 32 and below is mild, 33 to 58 is moderate, 59 and above is severe
    - **UPDRS 4 was dropped due to too few samples for training**
    """
    



with tab2:
    
    # read in the protein and updrs data
    updrs1_df = pd.read_csv('./data/processed/train_updrs_1_cat.csv')
    updrs2_df = pd.read_csv('./data/processed/train_updrs_2_cat.csv')
    updrs3_df = pd.read_csv('./data/processed/train_updrs_3_cat.csv')
    
    # get only the 12 months of patient data
    updrs1_df = updrs1_df[updrs1_df['visit_month'] <= 12]
    updrs2_df = updrs2_df[updrs2_df['visit_month'] <= 12]
    updrs3_df = updrs3_df[updrs3_df['visit_month'] <= 12]
    
    st.header('Parkinsons Severity Prediction')
    patient_id = st.selectbox('Patient ID', updrs1_df.sort_values(by='patient_id')['patient_id'].unique())
    patient_updrs1_df = updrs1_df[updrs1_df['patient_id'] == patient_id]
    patient_updrs2_df = updrs2_df[updrs2_df['patient_id'] == patient_id]
    patient_updrs3_df = updrs3_df[updrs3_df['patient_id'] == patient_id]

    
    # get the predictions
    updrs_preds = get_all_updrs_preds(patient_updrs1_df, col_dir='./dataframe_cols/')

    # convert preds to json
    #predictions = updrs_preds.to_json(orient="records")