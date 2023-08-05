import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


def preprocess_train_df(train_clin_df, train_prot_df, train_pep_df, save_data=False):
    """Takes in the raw data from the Kaggle datasets and combines the data into a single dataframe for each UPDRS.
       The dataframe returned has one row for each visit_id with all the protein and peptide abundances for that visit.
       The dataframe also has the updrs_1, updrs_2, updrs_3, and updrs_4 columns for that visit.
       5 fold stratified cross validation is also performed on the data and a kfold column is added to the dataframe.
       Each dataframe is stored in a dictionary with the key being the UPDRS name.

    Args:
        train_clin_df (dataframe): this dataframe has the UPDRS scores for each visit_id
        train_prot_df (dataframe): this dataframe has the protein abundances for each visit_id
        train_pep_df (dataframe): this dataframe has the peptide abundances for each visit_id
        save_data (bool, optional): whether to store each UPDRS dataframe as a csv file named 'train_{updrs}.csv'. Defaults to False.

    Returns:
        dictionary of dataframes: dictionary of dataframes with the key being the UPDRS name
    """

    # drop the medication column
    train_clin_df = train_clin_df.drop(columns=["upd23b_clinical_state_on_medication"])

    # create a column with the UniProt and Peptide name combined
    train_pep_df["peptide_uniprot"] = (
        train_pep_df["Peptide"] + "_" + train_pep_df["UniProt"]
    )

    # create a table with the visit_id as the index and the proteins or peptides as the feature and the abundance as the values
    train_prot_pivot = train_prot_df.pivot(
        index="visit_id", values="NPX", columns="UniProt"
    )
    train_pep_pivot = train_pep_df.pivot(
        index="visit_id", values="PeptideAbundance", columns="peptide_uniprot"
    )

    # combine the two tables on the visit_id
    full_prot_train_df = train_prot_pivot.join(train_pep_pivot)

    # fill nan with 0 for this first round
    full_prot_train_df = full_prot_train_df.fillna(0)

    full_train_df = train_clin_df.merge(
        full_prot_train_df, how="inner", left_on="visit_id", right_on="visit_id"
    )
    full_train_df = full_train_df.sample(frac=1).reset_index(drop=True)

    updrs = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

    final_dfs = dict()

    for target in updrs:
        to_remove = [updr for updr in updrs if updr != target]

        temp_train_df = full_train_df.drop(to_remove, axis=1)
        temp_train_df = temp_train_df.dropna()

        # calculate the number of bins by Sturge's rule
        num_bins = int(np.floor(1 + np.log2(len(full_train_df))))
        temp_train_df.loc[:, "bins"] = pd.cut(
            temp_train_df[target], bins=num_bins, labels=False
        )

        temp_train_df = temp_train_df.dropna().reset_index(drop=True)

        # initiate the kfold class from sklearn
        kf = StratifiedKFold(n_splits=5)

        # create a kfold column
        temp_train_df["kfold"] = -1

        # fill the kfold column
        for f, (t_, v_) in enumerate(
            kf.split(X=temp_train_df, y=temp_train_df["bins"].values)
        ):
            temp_train_df.loc[v_, "kfold"] = f

        # drop the bins column
        temp_train_df = temp_train_df.drop("bins", axis=1)

        if save_data:
            temp_train_df.to_csv(
                f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1//data/processed/train_{target}.csv",
                index=False,
            )

        else:
            final_dfs[target] = temp_train_df

    return final_dfs


if __name__ == "__main__":
    train_clin_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_clinical_data.csv"
    )
    train_prot_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_proteins.csv"
    )
    train_pep_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_peptides.csv"
    )

    preprocess_train_df(train_clin_df, train_prot_df, train_pep_df, save_data=True)
