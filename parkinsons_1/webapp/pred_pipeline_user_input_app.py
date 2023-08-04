import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
import joblib
import re
import json
import pickle
import warnings
import argparse

warnings.filterwarnings("ignore")


def fill_columns(df, updrs):
    # get the columns for updrs from updrs_cols.txt
    with open(f"{updrs}_cols.txt", "r") as f:
        updrs_cols = json.load(f)

    info_cols = [
        "visit_month",
        "num_prot_pep",
        "num_prot",
        "num_pept",
        "upd23b_clinical_state_on_medication",
        "visit_id",
        "patient_id",
        "kfold",
        updrs,
    ]

    prot_pep_cols = [col for col in updrs_cols if col not in info_cols]

    missing_cols = [col for col in prot_pep_cols if col not in df.columns]

    # fill the missing columns with 0
    df[missing_cols] = 0

    return df


def preprocess_input_data(input_data):
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
    model = joblib.load("./catboost_updrs_1_model_hyperopt_smote.sav")

    # Make predictions on the test data
    X = df

    preds = model.predict_proba(X)[:, 1]

    # use threshold of 0.48 to get the predicted updrs_1_cat
    updrs_1_cat_preds = np.where(preds >= 0.46, 1, 0)

    # add the column to the dataframe
    df["updrs_1_max_cat_preds"] = updrs_1_cat_preds

    return df


def predict_updrs2(df):
    model = joblib.load("./catboost_updrs_2_model_hyperopt_smote_meds.sav")

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
    filename = "./lgboost_updrs_3_model_hyperopt_smote_meds.sav"
    model = pickle.load(open(filename, "rb"))

    # Make predictions on the data
    X = df

    preds = model.predict_proba(X, verbose=-100)[:, 1]

    # use threshold of 0.28 to get the predicted updrs_3_cat
    updrs_3_cat_preds = np.where(preds >= 0.28, 1, 0)

    # add the column to the dataframe
    df["updrs_3_max_cat_preds"] = updrs_3_cat_preds

    return df


def get_all_updrs_preds(input_data):
    # convert the json input to a dataframe
    input_df = pd.DataFrame(input_data, index=[0])

    for updrs in ["updrs_1", "updrs_2", "updrs_3"]:
        with open(f"{updrs}_cols.txt", "r") as f:
            cols = json.load(f)

        if updrs in input_df.columns:
            input_df = input_df.drop(columns=[f"{updrs}"])
        if "kfold" in input_df.columns:
            input_df = input_df.drop(columns=["kfold"])

        full_input = fill_columns(input_df, updrs)

        info_df, prot_pep_df = preprocess_input_data(full_input)

        final_df = add_med_data(info_df, prot_pep_df)

        # add the visit_month column
        final_df["visit_month"] = input_df["visit_month"]

        # make sure there are no duplicates
        final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()

        updrs_df = final_df[cols]

        if updrs == "updrs_1":
            updrs1_preds = predict_updrs1(updrs_df)

            print(
                f"UPDRS 1 Max Category Prediction: {updrs1_preds[f'{updrs}_max_cat_preds'].values[0]}"
            )

        if updrs == "updrs_2":
            updrs2_preds = predict_updrs2(updrs_df)

            print(
                f"UPDRS 2 Max Category Prediction: {updrs2_preds[f'{updrs}_max_cat_preds'].values[0]}"
            )

        if updrs == "updrs_3":
            updrs3_preds = predict_updrs3(updrs_df)

            print(
                f"UPDRS 3 Max Category Prediction: {updrs3_preds[f'{updrs}_max_cat_preds'].values[0]}"
            )


if __name__ == "__main__":
    input_data = {
        "visit_month": 60,
        "patient_id": 24343,
        "visit_id": 1,
    }
    get_all_updrs_preds(input_data)
