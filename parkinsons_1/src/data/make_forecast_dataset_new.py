import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import itertools

updr = "updrs_2"
patient_id = 55


# iterate through each and give the visit_month difference and updrs difference
def calculate_updrs_diff(forecast_df, patient_id, updr):
    """
    Calculate the difference in UPDRS score between visits for each row in the dataset.

    Args:
        df (pd.DataFrame): The dataset for only one patient_id and only the rows that have protein data.

    Returns:
        pd.DataFrame: The dataset with the updrs_diff column added.
    """

    forecast_df = forecast_df[forecast_df["patient_id"] == patient_id]
    df = forecast_df[[updr, "visit_month", "patient_id"]]

    # generate all possible pairs of visit months
    pairs = list(itertools.product(df["visit_month"], df["visit_month"]))

    # create a dataframe with all possible pairs of visit months
    joined_df = pd.DataFrame(
        {
            "visit_month": [pair[0] for pair in pairs],
            "visit_month_2": [pair[1] for pair in pairs],
        }
    )

    # calculate the difference in months between the two visits
    joined_df["visit_month_diff"] = (
        joined_df["visit_month_2"] - joined_df["visit_month"]
    )

    # drop visit_month_dif == 0 rows
    joined_df = joined_df[joined_df["visit_month_diff"] != 0]

    # join this back to the original df
    joined_df = pd.merge(joined_df, df, on=["visit_month"], how="inner")

    # calculate the mean UPDRS score for each visit month
    updrs = joined_df.groupby(["visit_month"])[updr].mean().reset_index()
    updrs = updrs.rename(columns={updr: f"{updr}_visit2"})

    # join this back to the original df
    joined_df = joined_df.merge(
        updrs, left_on=["visit_month_2"], right_on=["visit_month"], how="left"
    )
    joined_df = joined_df.drop(columns=["visit_month_y"]).rename(
        columns={"visit_month_x": "visit_month"}
    )

    # calculate the difference in UPDRS score between the two visits
    joined_df["updrs_diff"] = joined_df[f"{updr}_visit2"] - joined_df[updr]

    # drop the unneeded columns
    joined_df = joined_df.drop(columns=["visit_month_2", f"{updr}_visit2"])

    final_df = joined_df.merge(
        forecast_df, on=["patient_id", "visit_month"], how="left"
    )

    final_df = final_df.drop(columns=[f"{updr}_y"]).rename(columns={f"{updr}_x": updr})

    return final_df


if __name__ == "__main__":
    train_clin_full_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_clinical_data.csv"
    )

    final_forecast_df = pd.DataFrame()

    updr = "updrs_2"
    patient_id = 55

    for updr in ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]:
        # get only the data for updr, visit_id, visit_month, patient_id
        train_clin_df = train_clin_full_df[
            [updr, "visit_id", "visit_month", "patient_id"]
        ]

        # get the months that have the protein data
        forecast_data = pd.read_csv(
            f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{updr}.csv"
        )

        # join the train_clin with the forecast_data
        forecast_df = pd.merge(
            train_clin_df,
            forecast_data.drop(columns=["visit_month", "patient_id", updr]),
            on=["visit_id"],
            how="inner",
        )

        for patient_id in forecast_df["patient_id"].unique():
            temp_df = calculate_updrs_diff(forecast_df, patient_id, updr)
            final_forecast_df = pd.concat([final_forecast_df, temp_df])

        # drop the kfold column
        final_forecast_df = final_forecast_df.drop(columns=["kfold"])

        # save the final results
        final_forecast_df.to_csv(
            f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_forecast_visitdiff_data_{updr}.csv",
            index=False,
        )

        final_forecast_df = pd.DataFrame()
