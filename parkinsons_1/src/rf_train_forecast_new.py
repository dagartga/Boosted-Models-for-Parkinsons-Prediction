# src/train.py

import os
import argparse
import datetime as dt
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def smape(y_true, y_pred):
    return round(
        np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
        * 100,
        2,
    )


updr = "updrs_1"
model = RandomForestRegressor(random_state=42)


def run(model, updr):
    # read the training data with folds
    df = pd.read_csv(
        f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_forecast_visitdiff_data_{updr}.csv"
    )

    X = df.drop(columns=["updrs_diff", "visit_id", "patient_id"])
    y = df["updrs_diff"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = X_train.values
    X_valid = X_valid.values

    reg = model
    reg.fit(X_train, y_train)
    preds = reg.predict(X_valid)

    r2 = metrics.r2_score(y_valid, preds)
    mape = metrics.mean_absolute_percentage_error(y_valid, preds)
    s_mape = smape(y_valid, preds)
    mae = metrics.mean_absolute_error(y_valid, preds)

    return s_mape, r2, mape, mae


if __name__ == "__main__":
    models = [
        ("rf_reg", RandomForestRegressor(random_state=42)),
        # ("xgb_reg", XGBRegressor(random_state=42)),
        # ("lin_reg", LinearRegression()),
    ]

    results = []

    date = dt.datetime.today().strftime("%Y-%m-%d")

    for model_name, model in models:
        for target in ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]:
            try:
                print("Running model: ", model_name, "for target: ", target)
                s_mape, r2, mape, mae = run(model, target)
                results.append(
                    {
                        "Model": model_name,
                        "Target": target,
                        "SMAPE": s_mape,
                        "R2": r2,
                        "MAPE": mape,
                        "MAE": mae,
                    }
                )
                print("SMAPE: ", s_mape, "R2: ", r2, "MAPE: ", mape, "MAE: ", mae, "\n")
                # store the model
                model_file = f"../models/forecast_model_{model_name}_{target}.pkl"
                pickle.dump(model, open(model_file, "wb"))
            except:
                print("---------------------------------------------------")
                print(
                    "Error running model: ",
                    model_name,
                    "for target: ",
                    target,
                    "\n",
                )

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(
        f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/rfreg_{date}_forecast_results.csv"
    )
