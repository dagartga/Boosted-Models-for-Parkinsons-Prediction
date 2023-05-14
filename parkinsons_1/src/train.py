# src/train.py

import os
import argparse

import datetime as dt
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
)  # Number of trees in random forest


def smape(y_true, y_pred):
    return round(
        np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
        * 100,
        2,
    )


def run(fold, model, target):
    # read the training data with folds
    df = pd.read_csv(
        f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv"
    )

    # remove unnecessary columns
    df = df.drop(columns=["visit_id", "patient_id"])

    # fill na with 0
    df = df.fillna(0)

    df_train = df[df["kfold"] != fold].reset_index(drop=True)

    df_valid = df[df["kfold"] == fold].reset_index(drop=True)

    x_train = df_train.drop([target, "kfold"], axis=1).values
    y_train = df_train[target].values

    x_valid = df_valid.drop([target, "kfold"], axis=1).values
    y_valid = df_valid[target].values

    reg = model
    reg.fit(x_train, y_train)
    preds = reg.predict(x_valid)

    feature_scores = pd.Series(
        reg.feature_importances_, index=df_train.drop([target, "kfold"], axis=1).columns
    ).sort_values(ascending=False)

    r2 = metrics.r2_score(y_valid, preds)
    mape = metrics.mean_absolute_percentage_error(y_valid, preds)
    s_mape = smape(y_valid, preds)

    print(f"Fold = {fold}, SMAPE = {s_mape}, R2 = {r2}, MAPE = {mape}")

    return fold, s_mape, r2, mape, feature_scores


if __name__ == "__main__":
    # use the date for file name later
    date = dt.date.today()
    date = date.strftime("%Y-%m-%d")

    model = RandomForestRegressor()

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    results = []

    for model_name, model in ("rf_reg", model):
        for target in ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]:
            for fold in [0, 1, 2, 3, 4]:
                try:
                    print(
                        "Running model: ",
                        model_name,
                        "for target: ",
                        target,
                        "fold: ",
                        fold,
                    )
                    f, s, r, m, feat_scores = run(fold, model, target)
                    results.append(
                        {
                            "Model": model_name,
                            "Target": target,
                            "Fold": f,
                            "SMAPE": s,
                            "R2": r,
                            "MAPE": m,
                        }
                    )
                    print("SMAPE: ", s, "R2: ", r, "MAPE: ", m, "\n")
                except:
                    print(
                        "Error running model: ",
                        model_name,
                        "for target: ",
                        target,
                        "for fold: ",
                        fold,
                        "\n",
                    )

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(
        f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/baseline_{date}_fold_results.csv"
    )
    for target in ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]:
        updr_results_df = results_df[results_df["Target"] == target]
        best_model = (
            updr_results_df.groupby("Model")["SMAPE"]
            .mean()
            .sort_values()
            .reset_index()
            .iloc[0, :]
        )
        print(f"For {target} the best model is: ", best_model["Model"])
        print(f"For {target} best model SMAPE is: ", best_model["SMAPE"])

    # get overall results
    no_fold_df = (
        results_df.groupby(["Model", "Target"])["SMAPE"]
        .mean()
        .sort_values()
        .reset_index()
    )
    overall_results = (
        no_fold_df.groupby("Model")["SMAPE"].mean().sort_values().reset_index()
    )
    overall_results.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/baseline_visit0_summary_results.csv"
    )
