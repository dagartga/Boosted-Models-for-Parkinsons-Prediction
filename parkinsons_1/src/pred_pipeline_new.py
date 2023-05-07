import pandas as pd
import pickle

import datetime as dt
import warnings

warnings.filterwarnings("ignore")


def preprocess_test_df(test_clin_df, test_prot_df, test_pep_df, save_data=False):
    if "upd23b_clinical_state_on_medication" in test_clin_df.columns:
        # drop the medication column
        test_clin_df = test_clin_df.drop(
            columns=["upd23b_clinical_state_on_medication"]
        )

    # create a column with the UniProt and Peptide name combined
    test_pep_df["peptide_uniprot"] = (
        test_pep_df["Peptide"] + "_" + test_pep_df["UniProt"]
    )

    # create a table with the visit_id as the index and the proteins or peptides as the feature and the abundance as the values
    test_prot_pivot = test_prot_df.pivot(
        index="visit_id", values="NPX", columns="UniProt"
    )
    test_pep_pivot = test_pep_df.pivot(
        index="visit_id", values="PeptideAbundance", columns="peptide_uniprot"
    )

    # combine the two tables on the visit_id
    full_prot_test_df = test_prot_pivot.join(test_pep_pivot)

    # fill nan with 0
    full_prot_test_df = full_prot_test_df.fillna(0)

    full_test_df = test_clin_df.merge(
        full_prot_test_df, how="inner", left_on="visit_id", right_on="visit_id"
    )
    full_test_df = full_test_df.sample(frac=1).reset_index(drop=True)

    return full_test_df


def prepare_model_df(model_df, target, visit_month=0):
    train_df = pd.read_csv(
        f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv"
    )
    pred_cols = [
        col
        for col in train_df.columns
        if col not in ["visit_id", "patient_id", target, "kfold"]
    ]

    # find the columns in preds_cols that are not in the model_df.columns
    not_in_pred_cols = [col for col in pred_cols if col not in model_df.columns]

    # create an empty dataframe with the columns in not_in_pred_cols
    not_in_preds_df = pd.DataFrame(columns=not_in_pred_cols)

    # combine the model_df and the not_in_preds_df so all the needed columns are in dataframe
    new_model_df = pd.concat([model_df, not_in_preds_df], axis=1)

    # fill the nan values with 0
    new_model_df = new_model_df.fillna(0)

    # filter the new_model_df to only include the columns in pred_cols with the correct order
    return new_model_df[pred_cols]


def create_first_prediction_df(test_df, prot_test_df, pep_test_df, save_data=False):
    full_test_df = preprocess_test_df(
        test_df, prot_test_df, pep_test_df, save_data=False
    )

    final_pred_df = pd.DataFrame()

    # do the first protein values visit predictions
    for updr in ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]:
        updr_df = full_test_df[full_test_df["updrs_test"] == updr]
        info_cols = [
            "visit_id",
            "visit_month",
            "patient_id",
            "updrs_test",
            "row_id",
            "group_key",
        ]
        updr_info = updr_df[info_cols]
        model_df = updr_df.drop(columns=info_cols)

        # prepare the model_df for the correct month -----issue-----
        model_df = prepare_model_df(model_df, updr)
        model_df["visit_month"] = updr_info["visit_month"]

        # Load the saved model from file for the first
        model_path = f"..\models\model_rf_reg_{updr}_05-06-2023.pkl"

        with open(model_path, "rb") as f:
            rf_reg = pickle.load(f)

        # Use the imported model to make predictions
        y_pred = rf_reg.predict(model_df.values)

        updr_info[f"preds"] = y_pred

        final_pred_df = pd.concat([final_pred_df, updr_info])

        protein_data = full_test_df[
            ["visit_id"] + list(full_test_df.columns[6:])
        ].drop_duplicates()
        # join the final_pred_df with the protein_data
        first_pred_prot_df = final_pred_df.merge(
            protein_data, how="left", left_on="visit_id", right_on="visit_id"
        )

    return first_pred_prot_df


def forecast_updr(testing, model, updr, month_diff):
    test_df = testing[testing["updrs_test"] == updr]
    test_df = test_df.rename(columns={"preds": f"{updr}"})
    test_df = test_df.drop(columns=["updrs_test", "row_id", "group_key", "visit_id"])

    # Load the saved model from file
    model_path = f"..\models\\forecast_model_{model}_{updr}.pkl"

    with open(model_path, "rb") as f:
        rf_reg = pickle.load(f)

    test_df["visit_month_diff"] = month_diff

    df = pd.read_csv(
        f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_forecast_visitdiff_data_{updr}.csv"
    )

    train_data = df.drop(columns=["updrs_diff", "visit_id", "patient_id"])
    train_cols = train_data.columns

    # get the columns that are in train_cols but not in test_df
    not_in_test_cols = [col for col in train_cols if col not in test_df.columns]

    # create a dataframe with the columns in not_in_test_cols
    not_in_test_df = pd.DataFrame(columns=not_in_test_cols)

    # combine the test_df and the not_in_test_df so all the needed columns are in dataframe
    new_test_df = pd.concat([test_df, not_in_test_df], axis=1).fillna(0)

    # filter the new_test_df to only include the columns in train_cols with the correct order
    new_test_df = new_test_df[train_cols]

    preds = rf_reg.predict(new_test_df.values)

    new_test_df[f"{updr}_plus_{month_diff}"] = new_test_df[updr] + preds

    return new_test_df


if __name__ == "__main__":
    test_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test.csv"
    )
    prot_test_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test_proteins.csv"
    )
    pep_test_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test_peptides.csv"
    )

    testing = create_first_prediction_df(
        test_df, prot_test_df, pep_test_df, save_data=True
    )

    final_forecast_df = pd.DataFrame()

    for updr in ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]:
        for month_diff in [6, 12, 24]:
            test_df = forecast_updr(testing, model, updr, month_diff)

            final_forecast_df = pd.concat([final_forecast_df, test_df])

    testing = testing.reset_index()
    final_forecast_df = final_forecast_df.reset_index()

    final_forecast_df = final_forecast_df.merge(
        testing[["visit_id", "index"]], how="left", left_on="index", right_on="index"
    )
