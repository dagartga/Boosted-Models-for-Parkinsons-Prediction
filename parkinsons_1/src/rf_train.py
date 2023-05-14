# src/train.py

import os
import argparse

import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


def smape(y_true, y_pred):
    return round(
        np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
        * 100,
        2,
    )


def run(model, target):
    # read the training data with folds
    df = pd.read_csv(
        f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv"
    )
    df = df.drop(columns=["visit_id", "patient_id"])

    x_train = df.drop([target, "kfold"], axis=1).values
    y_train = df[target].values

    reg = model
    reg.fit(x_train, y_train)
    preds = reg.predict(x_train)

    r2 = metrics.r2_score(y_train, preds)
    mape = metrics.mean_absolute_percentage_error(y_train, preds)
    s_mape = smape(y_train, preds)
    mae = metrics.mean_absolute_error(y_train, preds)

    print(f"SMAPE = {s_mape}, R2 = {r2}, MAPE = {mape}, MAE = {mae}")

    return s_mape, r2, mape, mae


if __name__ == "__main__":
    date = dt.datetime.now().strftime("%Y-%m-%d")

    models = [
        (
            "rf_reg",
            RandomForestRegressor(
                random_state=42,
            ),
        )
    ]

    results = []

    for model_name, model in models:
        for target in ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]:
            try:
                print("Running model: ", model_name, "for target: ", target)
                s, r, m, mae = run(model, target)
                results.append(
                    {
                        "Model": model_name,
                        "Target": target,
                        "SMAPE": s,
                        "R2": r,
                        "MAPE": m,
                        "MAE": mae,
                    }
                )
                print("SMAPE: ", s, "R2: ", r, "MAPE: ", m, "MAE: ", mae, "\n")
                # store the model
                # model_file = f"../models/model_{model_name}_{target}_{date}.pkl"
                # pickle.dump(model, open(model_file, "wb"))
            except:
                print("Error running model: ", model_name, "for target: ", target, "\n")

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/rfreg_visit0_default_results.csv"
    )
