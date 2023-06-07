# lgboost_extrafeats_future_cat_train.py

import numpy as np
import pandas as pd
import pickle
from functools import partial
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from typing import Any, Dict, Union

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope


def hyperparameter_tuning(
    space: Dict[str, Union[float, int]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    early_stopping_rounds: int = 50,
    metric: callable = roc_auc_score,
) -> Dict[str, Any]:
    init_vals = ["max_depth", "reg_alpha"]
    space = {k: (int(val) if k in init_vals else val) for k, val in space.items()}
    space["early_stopping_rounds"] = early_stopping_rounds
    model = LGBMClassifier(**space)
    evaluation = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=evaluation, verbose=False)

    pred = model.predict(X_test)
    score = metric(y_test, pred)
    return {"loss": -score, "status": STATUS_OK, "model": model}


def preprocess_categorical_df(df, target):
    # look at visits within the first year to predict future updrs category
    df = df[df["visit_month"] <= 12]

    # convert the categorical updrs scores to numerical
    # and combine severe and moderate since severe has very few observations
    df[f"{target}_num_cat"] = df[f"{target}_cat"].map(
        {"mild": 0, "moderate": 1, "severe": 1}
    )

    # get the max category for each patient
    max_df = df.groupby(["patient_id"])[f"{target}_num_cat"].max().reset_index()

    df = df.drop(columns=[f"{target}_num_cat"])

    # merge the max category with the original dataframe
    df = df.merge(max_df, on=["patient_id"], how="left")

    return df


if __name__ == "__main__":
    # read the training data
    updrs1_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1_cat.csv"
    )

    updrs2_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_2_cat.csv"
    )

    updrs3_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_3_cat.csv"
    )

    train_data_new_feats = pd.read_csv("../data/processed/train_data_new_feats.csv")

    updrs1_cols = [
        "visit_id",
        "patient_id",
        "visit_month",
        "updrs_1",
        "num_prot_pep",
        "num_prot",
        "num_pept",
        "updrs_1_cat",
    ]
    updrs2_cols = [
        "visit_id",
        "patient_id",
        "visit_month",
        "updrs_2",
        "num_prot_pep",
        "num_prot",
        "num_pept",
        "updrs_2_cat",
    ]
    updrs3_cols = [
        "visit_id",
        "patient_id",
        "visit_month",
        "updrs_3",
        "num_prot_pep",
        "num_prot",
        "num_pept",
        "updrs_3_cat",
    ]

    new_updrs1_df = updrs1_df[updrs1_cols].merge(
        train_data_new_feats, how="left", on=["visit_id", "patient_id"]
    )
    new_updrs2_df = updrs2_df[updrs2_cols].merge(
        train_data_new_feats, how="left", on=["visit_id", "patient_id"]
    )
    new_updrs3_df = updrs3_df[updrs3_cols].merge(
        train_data_new_feats, how="left", on=["visit_id", "patient_id"]
    )

    updrs_results = dict()

    for updrs, df in zip(
        ["updrs_1", "updrs_2", "updrs_3"], [new_updrs1_df, new_updrs2_df, new_updrs3_df]
    ):
        # preprocess the df
        df = preprocess_categorical_df(df, updrs)

        train = df[df["kfold"] != 4].reset_index(drop=True)
        test = df[df["kfold"] == 4].reset_index(drop=True)

        X_train = train.drop(
            columns=[
                "visit_id",
                "patient_id",
                updrs,
                "kfold",
                f"{updrs}_cat",
                f"{updrs}_num_cat",
            ]
        )
        y_train = train[f"{updrs}_num_cat"]
        X_test = test.drop(
            columns=[
                "visit_id",
                "patient_id",
                updrs,
                "kfold",
                f"{updrs}_cat",
                f"{updrs}_num_cat",
            ]
        )
        y_test = test[f"{updrs}_num_cat"]

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        options = {
            "max_depth": hp.quniform("max_depth", 1, 8, 1),
            "min_child_weight": hp.loguniform("min_child_weight", -2, 3),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "reg_alpha": hp.uniform("reg_alpha", 0, 10),
            "reg_lambda": hp.uniform("reg_lambda", 1, 10),
            "min_split_gain": hp.loguniform("min_split_gain", -10, 10),
            "learning_rate": hp.loguniform("learning_rate", -7, 0),
            "random_state": 42,
        }

        trials = Trials()
        best = fmin(
            fn=lambda space: hyperparameter_tuning(
                space, X_train, y_train, X_test, y_test
            ),
            space=options,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
        )

        updrs_results[updrs] = best

    # save the results
    updrs_results_df = pd.DataFrame(updrs_results)

    # save as a csv file
    updrs_results_df.to_csv(
        "../data/processed/lgboost_extrafeats_future_cat_hyperparam_results.csv",
        index=True,
    )
