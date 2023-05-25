# updrs 1 categorical ratings: 10 and below is mild, 11 to 21 is moderate, 22 and above is severe
# updrs 2 categorical ratings: 12 and below is mild, 13 to 29 is moderate, 30 and above is severe
# updrs 3 categorical ratings: 32 and below is mild, 33 to 58 is moderate, 59 and above is severe
# updrs 4 categorical ratings: 4 and below is mild, 5 to 12 is moderate, 13 and above is severe

import pandas as pd
import numpy as np


def make_categorical_dataset():
    """
    Turns the train_updrs.csv into a categorical dataset
    based on the ratings:
    updrs 1 categorical ratings: 10 and below is mild, 11 to 21 is moderate, 22 and above is severe
    updrs 2 categorical ratings: 12 and below is mild, 13 to 29 is moderate, 30 and above is severe
    updrs 3 categorical ratings: 32 and below is mild, 33 to 58 is moderate, 59 and above is severe
    updrs 4 categorical ratings: 4 and below is mild, 5 to 12 is moderate, 13 and above is severe
    """
    # read the data
    updrs1_df = pd.read_csv("../../data/processed/train_updrs_1.csv")
    updrs2_df = pd.read_csv("../../data/processed/train_updrs_2.csv")
    updrs3_df = pd.read_csv("../../data/processed/train_updrs_3.csv")
    updrs4_df = pd.read_csv("../../data/processed/train_updrs_4.csv")

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
    protein_cols = [col for col in updrs1_df.columns if col not in info_cols]

    # add a column for the number of proteins and peptides present
    updrs1_df["num_proteins"] = updrs1_df[protein_cols].sum(axis=1)
    updrs2_df["num_proteins"] = updrs2_df[protein_cols].sum(axis=1)
    updrs3_df["num_proteins"] = updrs3_df[protein_cols].sum(axis=1)
    updrs4_df["num_proteins"] = updrs4_df[protein_cols].sum(axis=1)

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

    # save the data
    updrs1_df.to_csv("../../data/processed/train_updrs_1_cat.csv", index=False)
    updrs2_df.to_csv("../../data/processed/train_updrs_2_cat.csv", index=False)
    updrs3_df.to_csv("../../data/processed/train_updrs_3_cat.csv", index=False)
    updrs4_df.to_csv("../../data/processed/train_updrs_4_cat.csv", index=False)


if __name__ == "__main__":
    make_categorical_dataset()
