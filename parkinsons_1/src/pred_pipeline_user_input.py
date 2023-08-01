import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import re
import json
import pickle
import warnings
import argparse
from catboost_future_cat_12m_train import convert_df_to_1yr
from data.make_dataset import preprocess_train_df

warnings.filterwarnings("ignore")


def preprocess_input_data(input_data, cols):
    input_df = pd.DataFrame(input_data, index=[0])

    # list of columns for information
    info_cols = [
        "visit_id",
        "patient_id",
        "visit_month",
    ]

    if "upd23b_clinical_state_on_medication" in input_df.columns:
        info_cols.append("upd23b_clinical_state_on_medication")

    info_df = input_df[info_cols]
    prot_pep_df = input_df.drop(columns=info_cols)

    # use regex to get the peptide columns which have a pattern "_"
    regex = re.compile(".*_.*")
    # list of peptide columns
    peptide_list = list(filter(regex.match, prot_pep_df.columns))
    # get the list of protein columns
    protein_list = list(set(prot_pep_df.columns) - set(peptide_list))

    prot_pep_cols = protein_list + peptide_list

    # add a column for the number of proteins and peptides present
    prot_pep_df["num_prot_pep"] = prot_pep_df[prot_pep_cols].sum(axis=1)

    # number of proteins
    prot_pep_df["num_prot"] = prot_pep_df[protein_list].sum(axis=1)

    # number of peptides
    prot_pep_df["num_pept"] = prot_pep_df[peptide_list].sum(axis=1)

    return info_df, prot_pep_df


def add_med_data(info_df, prot_pep_df):
    if "upd23b_clinical_state_on_medication" not in info_df.columns:
        info_df["upd23b_clinical_state_on_medication"] = "Unknown"
    elif (
        ~info_df["upd23b_clinical_state_on_medication"]
        .isin(["On", "Off", "Unknown", None])
        .any()
    ):
        raise ValueError(
            "upd23b_clinical_state_on_medication column must contain only On, Off, Unknown, or None"
        )
    else:
        info_df["upd23b_clinical_state_on_medication"] = info_df[
            "upd23b_clinical_state_on_medication"
        ].fillna("Unknown")

    info_df["upd23b_clinical_state_on_medication_On"] = np.where(
        info_df["upd23b_clinical_state_on_medication"] == "On", 1, 0
    )
    info_df["upd23b_clinical_state_on_medication_Unknown"] = np.where(
        info_df["upd23b_clinical_state_on_medication"] == "Unknown", 1, 0
    )

    # drop the original column
    info_df.drop("upd23b_clinical_state_on_medication", axis=1, inplace=True)
    # list of dummy columns
    dummy_cols = [
        "upd23b_clinical_state_on_medication_On",
        "upd23b_clinical_state_on_medication_Unknown",
    ]
    # add the dummy columns to the prot_pep_df
    final_df = pd.concat([info_df[dummy_cols], prot_pep_df], axis=1)

    return final_df


def predict_updrs1(df):
    # Load the saved model
    model = joblib.load("../models/catboost_updrs_1_model_hyperopt_smote.sav")

    # Make predictions on the test data
    X = df

    preds = model.predict_proba(X)[:, 1]

    # use threshold of 0.48 to get the predicted updrs_1_cat
    updrs_1_cat_preds = np.where(preds >= 0.46, 1, 0)

    # add the column to the dataframe
    df["updrs_1_max_cat_preds"] = updrs_1_cat_preds

    return df


def predict_updrs2(df):
    model = joblib.load("../models/catboost_updrs_2_model_hyperopt_smote_meds.sav")

    # Make predictions on the data
    X = df

    preds = model.predict_proba(X)[:, 1]

    # use threshold of 0.22 to get the predicted updrs_2_cat
    updrs_2_cat_preds = np.where(preds >= 0.22, 1, 0)

    # add the column to the dataframe
    df["updrs_2_max_cat_preds"] = updrs_2_cat_preds

    return df


def predict_updrs3(df):
    # Load the saved model
    filename = "../models/lgboost_updrs_3_model_hyperopt_smote_meds.sav"
    model = pickle.load(open(filename, "rb"))

    # Make predictions on the data
    X = df

    preds = model.predict_proba(X, verbose=-100)[:, 1]

    # use threshold of 0.28 to get the predicted updrs_3_cat
    updrs_3_cat_preds = np.where(preds >= 0.28, 1, 0)

    # add the column to the dataframe
    df["updrs_3_max_cat_preds"] = updrs_3_cat_preds

    return df


if __name__ == "__main__":
    # get the model_columns.txt data from json string
    with open("model_columns.txt", "r") as f:
        cols = json.load(f)

    cols = cols["columns"]

    input_data1 = pd.read_csv("../data/processed/train_updrs_1.csv")
    input_data1 = input_data1.drop(columns=["updrs_1", "kfold"])

    input_data1 = input_data1.iloc[0:1, :]

    df1, df2 = preprocess_input_data(input_data1, cols)

    final_df = add_med_data(df1, df2)

    # add the visit_month column
    final_df["visit_month"] = input_data1["visit_month"]

    # get the columns for updrs_1 from updrs1_cols.txt
    with open("updrs1_cols.txt", "r") as f:
        updrs1_cols = json.load(f)

    updrs1_df = final_df[updrs1_cols]

    updrs1_preds = predict_updrs1(updrs1_df)

    print(updrs1_preds["updrs_1_max_cat_preds"])
