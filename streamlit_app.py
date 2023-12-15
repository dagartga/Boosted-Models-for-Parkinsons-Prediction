import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import json
import joblib
import pickle
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
    updrs1_df = pd.read_csv('./streamlit_data/full_pred_updrs_1.csv')
    updrs2_df = pd.read_csv('./streamlit_data/full_pred_updrs_2.csv')
    updrs3_df = pd.read_csv('./streamlit_data/full_pred_updrs_3.csv')
    
    patient_id = updrs1_df['patient_id'].unique()[0]
    st.header('Parkinsons Severity Prediction')
    patient_id = st.selectbox('Patient ID', updrs1_df.sort_values(by='patient_id')['patient_id'].unique())
    patient_updrs1_df = updrs1_df[updrs1_df['patient_id'] == patient_id]
    patient_updrs2_df = updrs2_df[updrs2_df['patient_id'] == patient_id]
    patient_updrs3_df = updrs3_df[updrs3_df['patient_id'] == patient_id]

    
    # updrs values by visit month
    visit_updrs1_df = patient_updrs1_df[['updrs_1', 'visit_month']].rename(columns={'updrs_1': 'value'})
    visit_updrs2_df = patient_updrs2_df[['updrs_2', 'visit_month']].rename(columns={'updrs_2': 'value'})
    visit_updrs3_df = patient_updrs3_df[['updrs_3', 'visit_month']].rename(columns={'updrs_3': 'value'})
    (visit_updrs1_df['updrs'], visit_updrs2_df['updrs'], visit_updrs3_df['updrs']) = ('UPDRS 1', 'UPDRS 2', 'UPDRS 3')
    
    updrs_vals = pd.concat([visit_updrs1_df[['updrs', 'value', 'visit_month']], 
                            visit_updrs2_df[['updrs', 'value', 'visit_month']], 
                            visit_updrs3_df[['updrs', 'value', 'visit_month']]], axis=0)

    # display dataframe of predicted updrs and the visit month
    st.write('**UPDRS Max Predictions**')
    pred_df = pd.merge(patient_updrs1_df[['visit_month', 'updrs_1_max_cat_preds']], 
                        patient_updrs2_df[['visit_month', 'updrs_2_max_cat_preds']], on='visit_month')
    pred_df = pd.merge(pred_df, patient_updrs3_df[['visit_month', 'updrs_3_max_cat_preds']], on='visit_month')
    pred_df = pred_df.sort_values(by=['visit_month']).set_index('visit_month')
    st.dataframe(pred_df.rename(columns={'updrs_1_max_cat_preds': 'Max Predicted UPDRS 1',
                                         'updrs_2_max_cat_preds': 'Max Predicted UPDRS 2',
                                         'updrs_3_max_cat_preds': 'Max Predicted UPDRS 3'}))
    
    # write the max updrs value predicted
    max_updrs1 = patient_updrs1_df['updrs_1_max_cat_preds'].values.max()
    max_updrs2 = patient_updrs2_df['updrs_2_max_cat_preds'].values.max()
    max_updrs3 = patient_updrs3_df['updrs_3_max_cat_preds'].values.max()
    
    max_updrs1 = 'Moderate-to-Severe' if max_updrs1 == 1 else 'None-to-Mild'
    max_updrs2 = 'Moderate-to-Severe' if max_updrs2 == 1 else 'None-to-Mild'
    max_updrs3 = 'Moderate-to-Severe' if max_updrs3 == 1 else 'None-to-Mild'
        
    st.write(f'**UPDRS 1 Max Prediction**: {max_updrs1}')
    st.write(f'**UPDRS 2 Max Prediction**: {max_updrs2}')
    st.write(f'**UPDRS 3 Max Prediction**: {max_updrs3}')
    
    if patient_updrs1_df['visit_month'].nunique() > 1:
        # plot the updrs values by visit month
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=updrs_vals, x='visit_month', y='value', hue='updrs', ax=ax)
        ax.set_title(f'UPDRS Values for Patient {patient_id}')
        ax.set_xlabel('Visit Month')
        ax.set_ylabel('UPDRS Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        st.pyplot(fig)
    else:
        st.markdown('*Only One Visit for this Patient*')
        # plot as a bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=updrs_vals, x='updrs', y='value', hue='visit_month', ax=ax)
        ax.set_title(f'UPDRS Values for Patient {patient_id}')
        ax.set_xlabel('UPDRS')
        ax.set_ylabel('UPDRS Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Visit Month')
        st.pyplot(fig)
        
        
        
    # import the input data used for modeling
    input_updrs1_df = pd.read_csv('./streamlit_data/updrs_1_model_input.csv')
    input_updrs2_df = pd.read_csv('./streamlit_data/updrs_2_model_input.csv')
    input_updrs3_df = pd.read_csv('./streamlit_data/updrs_3_model_input.csv')
    
    
    # UPDRS 1
    # Load the saved model
    model = joblib.load("./webapp/catboost_updrs_1_model_hyperopt_smote.sav")

    # Make predictions on the data
    preds = model.predict(input_updrs1_df)
    
    
    # UPDRS 2
    # Load the saved model
    model = joblib.load("./webapp/catboost_updrs_2_model_hyperopt_smote_meds.sav")
    # make predictions on the data
    preds = model.predict(input_updrs2_df)
    
    
    # UPDRS 3
    # Load the saved model
    filename = "./webapp/lgboost_updrs_3_model_hyperopt_smote_meds.sav"
    model = pickle.load(open(filename, "rb"))

    # Make predictions on the data
    preds = model.predict(input_updrs3_df)