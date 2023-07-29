import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import pickle
import warnings
import argparse
from catboost_future_cat_12m_train import convert_df_to_1yr
from data.make_dataset import preprocess_train_df

warnings.filterwarnings("ignore")


def make_categorical_dataset(processed_dfs, proteins_df):
    """
    Turns the train_updrs.csv into a categorical dataset
    based on the ratings:
    updrs 1 categorical ratings: 10 and below is mild, 11 to 21 is moderate, 22 and above is severe
    updrs 2 categorical ratings: 12 and below is mild, 13 to 29 is moderate, 30 and above is severe
    updrs 3 categorical ratings: 32 and below is mild, 33 to 58 is moderate, 59 and above is severe
    updrs 4 categorical ratings: 4 and below is mild, 5 to 12 is moderate, 13 and above is severe
    """
    # read the data
    updrs1_df = processed_dfs["updrs_1"]
    updrs2_df = processed_dfs["updrs_2"]
    updrs3_df = processed_dfs["updrs_3"]
    updrs4_df = processed_dfs["updrs_4"]

    protein_list = list(proteins_df["UniProt"].unique())

    # list of columns for information
    info_cols = [
        "visit_id",
        "patient_id",
        "visit_month",
        "updrs_1",
        "updrs_2",
        "updrs_3",
        "updrs_4",
        "kfold",
    ]

    # protein and peptide columns
    peptide_list = [
        col
        for col in updrs1_df.columns
        if col not in protein_list and col not in info_cols
    ]
    prot_pep_cols = protein_list + peptide_list

    # add a column for the number of proteins and peptides present
    updrs1_df["num_prot_pep"] = updrs1_df[prot_pep_cols].sum(axis=1)
    updrs2_df["num_prot_pep"] = updrs2_df[prot_pep_cols].sum(axis=1)
    updrs3_df["num_prot_pep"] = updrs3_df[prot_pep_cols].sum(axis=1)
    updrs4_df["num_prot_pep"] = updrs4_df[prot_pep_cols].sum(axis=1)
    # number of proteins
    updrs1_df["num_prot"] = updrs1_df[protein_list].sum(axis=1)
    updrs2_df["num_prot"] = updrs2_df[protein_list].sum(axis=1)
    updrs3_df["num_prot"] = updrs3_df[protein_list].sum(axis=1)
    updrs4_df["num_prot"] = updrs4_df[protein_list].sum(axis=1)
    # number of peptides
    updrs1_df["num_pept"] = updrs1_df[peptide_list].sum(axis=1)
    updrs2_df["num_pept"] = updrs2_df[peptide_list].sum(axis=1)
    updrs3_df["num_pept"] = updrs3_df[peptide_list].sum(axis=1)
    updrs4_df["num_pept"] = updrs4_df[peptide_list].sum(axis=1)

    # apply the categorical ratings
    updrs1_df["updrs_1_cat"] = np.where(
        updrs1_df["updrs_1"] <= 10,
        "mild",
        np.where(updrs1_df["updrs_1"] <= 21, "moderate", "severe"),
    )
    updrs2_df["updrs_2_cat"] = np.where(
        updrs2_df["updrs_2"] <= 12,
        "mild",
        np.where(updrs2_df["updrs_2"] <= 29, "moderate", "severe"),
    )
    updrs3_df["updrs_3_cat"] = np.where(
        updrs3_df["updrs_3"] <= 32,
        "mild",
        np.where(updrs3_df["updrs_3"] <= 58, "moderate", "severe"),
    )
    updrs4_df["updrs_4_cat"] = np.where(
        updrs4_df["updrs_4"] <= 4,
        "mild",
        np.where(updrs4_df["updrs_4"] <= 12, "moderate", "severe"),
    )

    categorical_dfs = {
        "updrs_1": updrs1_df,
        "updrs_2": updrs2_df,
        "updrs_3": updrs3_df,
        "updrs_4": updrs4_df,
    }

    return categorical_dfs


def add_med_data(clin_df, updrs_df):
    clin_df["upd23b_clinical_state_on_medication"] = clin_df[
        "upd23b_clinical_state_on_medication"
    ].fillna("Unknown")

    # get dummies for on_medication column
    clin_df_dummies = pd.get_dummies(
        clin_df, columns=["upd23b_clinical_state_on_medication"], drop_first=True
    )

    clin_df_dummies = clin_df_dummies[
        [
            "visit_id",
            "upd23b_clinical_state_on_medication_On",
            "upd23b_clinical_state_on_medication_Unknown",
        ]
    ]

    # merge the updrs data with the clinical data for dummy columns
    updrs_df = pd.merge(updrs_df, clin_df_dummies, on="visit_id")

    return updrs_df


def predict_updrs1(df):
    # Load the saved model
    model = joblib.load("../models/catboost_updrs_1_model_hyperopt_smote.sav")

    # Make predictions on the test data
    X = df.drop(columns=["updrs_1_cat", "kfold", "visit_id", "patient_id", "updrs_1"])

    preds = model.predict_proba(X)[:, 1]

    # use threshold of 0.48 to get the predicted updrs_1_cat
    updrs_1_cat_preds = np.where(preds >= 0.46, 1, 0)

    # add the column to the dataframe
    df["updrs_1_cat_preds"] = updrs_1_cat_preds

    return df


def predict_updrs2(df):
    model = joblib.load("../models/catboost_updrs_2_model_hyperopt_smote_meds.sav")

    # Make predictions on the test data
    X = df.drop(columns=["updrs_2_cat", "kfold", "visit_id", "patient_id", "updrs_2"])

    preds = model.predict_proba(X)[:, 1]

    # use threshold of 0.22 to get the predicted updrs_2_cat
    updrs_2_cat_preds = np.where(preds >= 0.22, 1, 0)

    # add the column to the dataframe
    df["updrs_2_cat_preds"] = updrs_2_cat_preds

    return df


def predict_updrs3(df):
    # Load the saved model
    filename = "../models/lgboost_updrs_3_model_hyperopt_smote_meds.sav"
    model = pickle.load(open(filename, "rb"))

    # Make predictions on the test data
    X = df.drop(columns=["updrs_3_cat", "kfold", "visit_id", "patient_id", "updrs_3"])

    preds = model.predict_proba(X, verbose=-100)[:, 1]

    # use threshold of 0.28 to get the predicted updrs_3_cat
    updrs_3_cat_preds = np.where(preds >= 0.28, 1, 0)

    # add the column to the dataframe
    df["updrs_3_cat_preds"] = updrs_3_cat_preds

    return df


if __name__ == "__main__":
    # use argparse to get the data paths
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_data", help="input data path for prediction", type=str, nargs=1
    )
    args = parser.parse_args()
    file_path = args.input_data[0]

    input_df = pd.read_csv(file_path)

    # TODO:create a preprocess function that takes in data in a tabular format
    processed_dfs = preprocess_train_df(
        train_clin_df, train_prot_df, train_pep_df, save_data=False
    )

    categorical_dfs = make_categorical_dataset(processed_dfs, train_prot_df)

    categorical_dfs["updrs_2"] = add_med_data(train_clin_df, categorical_dfs["updrs_2"])
    categorical_dfs["updrs_3"] = add_med_data(train_clin_df, categorical_dfs["updrs_3"])

    updrs1_df = predict_updrs1(categorical_dfs["updrs_1"])
    updrs2_df = predict_updrs2(categorical_dfs["updrs_2"])
    updrs3_df = predict_updrs3(categorical_dfs["updrs_3"])

    # combine prediction columns into one dataframe
    updrs_preds = pd.merge(
        updrs1_df,
        updrs2_df[["visit_id", "updrs_2_cat", "updrs_2_cat_preds"]],
        on="visit_id",
    )

    updrs_preds = pd.merge(
        updrs_preds,
        updrs3_df[["visit_id", "updrs_3_cat", "updrs_3_cat_preds"]],
        on="visit_id",
        how="left",
    )

    # save the dataframe as a csv
    updrs_preds.to_csv("../data/predictions/full_updrs_preds.csv", index=False)
