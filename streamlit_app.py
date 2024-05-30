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
import shap
from webapp.pred_pipeline_user_input_app import get_all_updrs_preds

st.set_page_config(layout="centered")


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        lottie_json = json.load(f)
    return lottie_json


filename = load_lottiefile("./streamlit_data/doctor_animation.json")
st_lottie(filename, speed=1, height=200)

st.title("Parkinsons Severity Prediction")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Prediction",
        "Overview",
        "UPDRS 1 Proteins",
        "UPDRS 2 Proteins",
        "UPDRS 3 Proteins",
    ]
)


with tab2:
    st.header("Project Overview")
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


with tab1:
    # read in the protein and updrs data
    updrs1_df = pd.read_csv("./streamlit_data/full_pred_updrs_1.csv")
    updrs2_df = pd.read_csv("./streamlit_data/full_pred_updrs_2.csv")
    updrs3_df = pd.read_csv("./streamlit_data/full_pred_updrs_3.csv")

    # import patient updrs values
    patient_updrs_df = pd.read_csv("./streamlit_data/patient_updrs_values.csv")

    # import the input data used for modeling
    input_updrs1_df = pd.read_csv("./streamlit_data/updrs_1_model_input.csv")
    input_updrs2_df = pd.read_csv("./streamlit_data/updrs_2_model_input.csv")
    input_updrs3_df = pd.read_csv("./streamlit_data/updrs_3_model_input.csv")

    st.header("Parkinsons Severity Prediction")
    # have the user select the patient id
    patient_id = st.selectbox(
        "Patient ID", updrs1_df.sort_values(by="patient_id")["patient_id"].unique()
    )
    patient_updrs1_df = updrs1_df[updrs1_df["patient_id"] == patient_id]
    patient_updrs2_df = updrs2_df[updrs2_df["patient_id"] == patient_id]
    patient_updrs3_df = updrs3_df[updrs3_df["patient_id"] == patient_id]

    # updrs values by visit month
    visit_updrs1_df = patient_updrs1_df[["updrs_1", "visit_month"]].rename(
        columns={"updrs_1": "value"}
    )
    visit_updrs2_df = patient_updrs2_df[["updrs_2", "visit_month"]].rename(
        columns={"updrs_2": "value"}
    )
    visit_updrs3_df = patient_updrs3_df[["updrs_3", "visit_month"]].rename(
        columns={"updrs_3": "value"}
    )
    (visit_updrs1_df["updrs"], visit_updrs2_df["updrs"], visit_updrs3_df["updrs"]) = (
        "UPDRS 1",
        "UPDRS 2",
        "UPDRS 3",
    )

    updrs_vals = pd.concat(
        [
            visit_updrs1_df[["updrs", "value", "visit_month"]],
            visit_updrs2_df[["updrs", "value", "visit_month"]],
            visit_updrs3_df[["updrs", "value", "visit_month"]],
        ],
        axis=0,
    )

    # display dataframe of predicted updrs and the visit month
    """ ### UPDRS Max Predictions
    **The model uses only the protein and peptide data from visit months 0 - 12 to predict whether the patient will have moderate-to-severe max UPDRS rating**
    
    Below you can see the **"Max Predicted UPDRS Score"** for each UPDRS
    """

    pred_df = pd.merge(
        patient_updrs1_df[["visit_month", "updrs_1_max_cat_preds"]],
        patient_updrs2_df[["visit_month", "updrs_2_max_cat_preds"]],
        on="visit_month",
    )
    pred_df = pd.merge(
        pred_df,
        patient_updrs3_df[["visit_month", "updrs_3_max_cat_preds"]],
        on="visit_month",
    )
    pred_df = pred_df.sort_values(by=["visit_month"]).set_index("visit_month")
    for i in range(1, 4):
        if i == 1:
            pred_df[f"updrs_{i}_max_cat_preds"] = pred_df[
                f"updrs_{i}_max_cat_preds"
            ].apply(
                lambda x: "> 10 (Moderate-to-Severe)"
                if x == 1
                else "< 11 (None-to-Mild)"
            )
        elif i == 2:
            pred_df[f"updrs_{i}_max_cat_preds"] = pred_df[
                f"updrs_{i}_max_cat_preds"
            ].apply(
                lambda x: "> 12 (Moderate-to-Severe)"
                if x == 1
                else "< 13 (None-to-Mild)"
            )
        elif i == 3:
            pred_df[f"updrs_{i}_max_cat_preds"] = pred_df[
                f"updrs_{i}_max_cat_preds"
            ].apply(
                lambda x: "> 32 (Moderate-to-Severe)"
                if x == 1
                else "< 33 (None-to-Mild)"
            )
    st.dataframe(
        pred_df.rename(
            columns={
                "updrs_1_max_cat_preds": "Max Predicted UPDRS 1",
                "updrs_2_max_cat_preds": "Max Predicted UPDRS 2",
                "updrs_3_max_cat_preds": "Max Predicted UPDRS 3",
            }
        )
    )

    """
    - **UPDRS 1 categorical ratings**: 10 and below is mild, 11 to 21 is moderate, 22 and above is severe
    - **UPDRS 2 categorical ratings**: 12 and below is mild, 13 to 29 is moderate, 30 and above is severe
    - **UPDRS 3 categorical ratings**: 32 and below is mild, 33 to 58 is moderate, 59 and above is severe
    """

    # filter out the input data for the patient
    patient_values = patient_updrs_df[patient_updrs_df["patient_id"] == patient_id]

    """### View all of actual UPDRS values for the patient below:"""
    if patient_values["visit_month"].nunique() > 1:
        # plot the updrs values by visit month
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=patient_values,
            x="visit_month",
            y="value",
            hue="updrs",
            ax=ax,
        )
        ax.set_title(f"UPDRS Values for Patient {patient_id}")
        ax.set_xlabel("Visit Month")
        ax.set_ylabel("UPDRS Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        st.pyplot(fig)
    else:
        st.markdown("*Only One Visit for this Patient*")
        # plot as a bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=patient_values,
            x="updrs",
            y="value",
            hue="visit_month",
            ax=ax,
        )
        ax.set_title(f"UPDRS Values for Patient {patient_id}")
        ax.set_xlabel("UPDRS")
        ax.set_ylabel("UPDRS Value")
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title="Visit Month"
        )
        st.pyplot(fig)

    st.header("Explanation of Model Predictions")
    st.write(
        "The following plots show the **top ten features (proteins)** that contributed to the model prediction for the **inputed patient and visit month**. The features are ranked by their SHAP values."
    )
    st.write(
        "**Choose a visit month to see the explanation of the model prediction for the input patient**"
    )
    # user selects the visit month to make predictions on
    visit_month = st.selectbox("Visit Month", patient_updrs1_df["visit_month"].unique())

    st.subheader("UPDRS 1")

    # UPDRS 1
    # Load the saved model
    model = joblib.load("./webapp/catboost_updrs_1_model_hyperopt_smote.sav")
    # filter out the input data for the patient
    drop_col = [
        "patient_id",
        "upd23b_clinical_state_on_medication_On",
        "upd23b_clinical_state_on_medication_Unknown",
    ]
    input_updrs1_df = input_updrs1_df[input_updrs1_df["patient_id"] == patient_id].drop(
        columns=drop_col
    )
    # filter for the visit month
    input_updrs1_df = input_updrs1_df[input_updrs1_df["visit_month"] == visit_month]

    # make predictions on the data
    # preds = model.predict(input_updrs1_df)

    # plot the shap values
    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    input_shap_values = explainer.shap_values(input_updrs1_df)
    # create a dataframe of the shap values with the column names
    input_shap_df = pd.DataFrame(
        input_shap_values, columns=input_updrs1_df.columns
    ).T.reset_index()
    input_shap_df.columns = ["feature", "shap_value"]

    # SHAP force plot for inputed instance predicted class
    fig, ax = plt.subplots()
    # plot a vertical bar for the top ten features
    sns.barplot(
        data=input_shap_df.sort_values(by="shap_value", ascending=False).head(10),
        x="shap_value",
        y="feature",
        ax=ax,
    )
    plt.title(
        "Features (Proteins) Towards Severe UPDRS 1 Model Prediction", fontsize=14
    )
    plt.ylabel("")
    plt.xlabel("")
    st.pyplot(fig)

    st.subheader("UPDRS 2")
    # UPDRS 2
    # Load the saved model
    model = joblib.load("./webapp/catboost_updrs_2_model_hyperopt_smote_meds.sav")
    # filter out the input data for the patient
    input_updrs2_df = input_updrs2_df[input_updrs2_df["patient_id"] == patient_id].drop(
        columns=["patient_id"]
    )
    # filter for the visit month
    input_updrs2_df = input_updrs2_df[input_updrs2_df["visit_month"] == visit_month]
    # make predictions on the data
    # preds = model.predict(input_updrs2_df)

    # plot the shap values
    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    input_shap_values = explainer.shap_values(input_updrs2_df)
    # create a dataframe of the shap values with the column names
    input_shap_df = pd.DataFrame(
        input_shap_values, columns=input_updrs2_df.columns
    ).T.reset_index()
    input_shap_df.columns = ["feature", "shap_value"]

    # SHAP force plot for inputed instance predicted class
    fig, ax = plt.subplots()
    # plot a vertical bar for the top ten features
    sns.barplot(
        data=input_shap_df.sort_values(by="shap_value", ascending=False).head(10),
        x="shap_value",
        y="feature",
        ax=ax,
    )
    plt.title("Feature (Proteins) Towards Severe UPDRS 2 Model Prediction", fontsize=14)
    plt.ylabel("")
    plt.xlabel("")
    st.pyplot(fig)

    st.subheader("UPDRS 3")
    # UPDRS 3
    # Load the saved model
    filename = "./webapp/lgboost_updrs_3_model_hyperopt_smote_meds.sav"
    model = pickle.load(open(filename, "rb"))
    # filter out the input data for the patient
    input_updrs3_df = input_updrs3_df[input_updrs3_df["patient_id"] == patient_id].drop(
        columns=["patient_id"]
    )
    # filter for the visit month
    input_updrs3_df = input_updrs3_df[input_updrs3_df["visit_month"] == visit_month]
    # make predictions on the data
    # preds = model.predict(input_updrs3_df)

    # plot the shap values
    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    input_shap_values = explainer.shap_values(input_updrs3_df)
    # create a dataframe of the shap values with the column names
    input_shap_df = pd.DataFrame(
        input_shap_values[0], columns=input_updrs3_df.columns
    ).T.reset_index()
    input_shap_df.columns = ["feature", "shap_value"]

    # SHAP force plot for inputed instance predicted class
    fig, ax = plt.subplots()
    # plot a vertical bar for the top ten features
    sns.barplot(
        data=input_shap_df.sort_values(by="shap_value", ascending=False).head(10),
        x="shap_value",
        y="feature",
        ax=ax,
    )
    plt.title(
        "Features (Proteins) Towards Severe UPDRS 3 Model Prediction", fontsize=14
    )
    plt.ylabel("")
    plt.xlabel("")
    st.pyplot(fig)

with tab3:
    # show the feature importances from the saved csv files
    st.header("Feature Importances")
    st.subheader("UPDRS 1")
    updrs1_feat_imp = pd.read_csv("./webapp/updrs_1_feat_imp.csv")
    updrs1_feat_imp = updrs1_feat_imp.sort_values(by="importance", ascending=False)
    top_ten_updrs1_feats = updrs1_feat_imp.head(10)
    fig, ax = plt.subplots()
    sns.barplot(data=top_ten_updrs1_feats, x="importance", y="feature", ax=ax)
    plt.title("Top Ten Features for UPDRS 1 Model", fontsize=14)
    plt.ylabel("")
    plt.xlabel("")
    st.pyplot(fig)

    # import the Uniprot data
    uniprot_df = pd.read_csv("./webapp/UniprotProteinLookup.csv")
    # combine the protein and the uniprot data
    top_ten_updrs1_feats["protein"] = top_ten_updrs1_feats["feature"].apply(
        lambda x: x.split("_")[1] if "_" in x else x
    )
    top_ten_updrs1_feats = pd.merge(
        top_ten_updrs1_feats, uniprot_df, left_on="protein", right_on="UniProt"
    )
    top_ten_updrs1_feats = top_ten_updrs1_feats.fillna("Unknown")
    # display the protein information
    st.subheader("Top Proteins for UPDRS 1 Information")
    st.write(
        "**If a protein is missing it is because it is not in the Uniprot database**"
    )
    st.write("-------------------")
    for i, row in top_ten_updrs1_feats.iterrows():
        st.markdown(f"**Protein Peptide**: {row['feature']}")
        st.markdown(f"**Protein Name**: {row['Protein names']}")
        st.markdown(f"**Gene Name**: {row['Gene Names']}")
        st.markdown(f"**Length**: {row['Length']}")
        st.write("-------------------")

with tab4:
    # show the feature importances from the saved csv files
    st.header("Feature Importances")
    st.subheader("UPDRS 2")
    updrs2_feat_imp = pd.read_csv("./webapp/updrs_2_feat_imp.csv")
    updrs2_feat_imp = updrs2_feat_imp.sort_values(by="importance", ascending=False)
    top_ten_updrs2_feats = updrs2_feat_imp.head(10)
    fig, ax = plt.subplots()
    sns.barplot(data=top_ten_updrs2_feats, x="importance", y="feature", ax=ax)
    plt.title("Top Ten Features for UPDRS 2 Model", fontsize=14)
    plt.ylabel("")
    plt.xlabel("")
    st.pyplot(fig)

    # combine the protein and the uniprot data
    top_ten_updrs2_feats["protein"] = top_ten_updrs2_feats["feature"].apply(
        lambda x: x.split("_")[1] if "_" in x else x
    )
    top_ten_updrs2_feats = pd.merge(
        top_ten_updrs2_feats, uniprot_df, left_on="protein", right_on="UniProt"
    )
    top_ten_updrs2_feats = top_ten_updrs2_feats.fillna("Unknown")
    # display the protein information
    # display the protein information
    st.subheader("Top Proteins for UPDRS 2 Information")
    st.write(
        "**If a protein is missing it is because it is not in the Uniprot database**"
    )
    st.write("-------------------")
    for i, row in top_ten_updrs2_feats.iterrows():
        st.markdown(f"**Protein Peptide**: {row['feature']}")
        st.markdown(f"**Protein Name**: {row['Protein names']}")
        st.markdown(f"**Gene Name**: {row['Gene Names']}")
        st.markdown(f"**Length**: {row['Length']}")
        st.write("-------------------")

with tab5:
    # show the feature importances from the saved csv files
    st.header("Feature Importances")
    st.subheader("UPDRS 3")
    updrs3_feat_imp = pd.read_csv("./webapp/updrs_3_feat_imp.csv")
    updrs3_feat_imp = updrs3_feat_imp.sort_values(by="importance", ascending=False)
    top_ten_updrs3_feats = updrs3_feat_imp.head(10)
    fig, ax = plt.subplots()
    sns.barplot(data=top_ten_updrs3_feats, x="importance", y="feature", ax=ax)
    plt.title("Top Ten Features for UPDRS 3 Model", fontsize=14)
    plt.ylabel("")
    plt.xlabel("")
    st.pyplot(fig)

    # combine the protein and the uniprot data
    top_ten_updrs3_feats["protein"] = top_ten_updrs3_feats["feature"].apply(
        lambda x: x.split("_")[1] if "_" in x else x
    )
    top_ten_updrs3_feats = pd.merge(
        top_ten_updrs3_feats, uniprot_df, left_on="protein", right_on="UniProt"
    )
    top_ten_updrs3_feats = top_ten_updrs3_feats.fillna("Unknown")
    # display the protein information
    # display the protein information
    st.subheader("Top Proteins for UPDRS 3 Information")
    st.write(
        "**If a protein is missing it is because it is not in the Uniprot database**"
    )
    st.write("-------------------")
    for i, row in top_ten_updrs3_feats.iterrows():
        st.markdown(f"**Protein Peptide**: {row['feature']}")
        st.markdown(f"**Protein Name**: {row['Protein names']}")
        st.markdown(f"**Gene Name**: {row['Gene Names']}")
        st.markdown(f"**Length**: {row['Length']}")
        st.write("-------------------")
