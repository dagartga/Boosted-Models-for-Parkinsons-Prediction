import os
import numpy as np
import pandas as pd
import pickle
from functools import partial
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
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
    init_vals = ["max_depth"]
    space = {k: (int(val) if k in init_vals else val) for k, val in space.items()}
    space["early_stopping_rounds"] = early_stopping_rounds
    model = CatBoostClassifier(**space)
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
    # create file paths
    updrs1_path = os.path.join("..", "data", "processed", "train_updrs_1_cat.csv")
    updrs2_path = os.path.join("..", "data", "processed", "train_updrs_2_cat.csv")
    updrs3_path = os.path.join("..", "data", "processed", "train_updrs_3_cat.csv")

    # read the training data
    updrs1_df = pd.read_csv(updrs1_path)
    updrs2_df = pd.read_csv(updrs2_path)
    updrs3_df = pd.read_csv(updrs3_path)

    df = updrs1_df
    updrs = "updrs_1"

    updrs_results = dict()

    for updrs, df in zip(
        ["updrs_1", "updrs_2", "updrs_3"], [updrs1_df, updrs2_df, updrs3_df]
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
            "depth": scope.int(hp.quniform("depth", 4, 10, 1)),
            "learning_rate": hp.loguniform("learning_rate", -7, 0),
            "bagging_temperature": scope.int(hp.uniform("bagging_temperature", 1, 10)),
            "min_data_in_leaf": scope.int(hp.quniform("min_data_in_leaf", 1, 10, 1)),
            "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1, 10),
            # "max_leaves": scope.int(hp.quniform("max_leaves", 1, 30, 1)),
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

    # file path for saving
    catboost_hyperparams_path = os.path.join(
        "..", "data", "processed", "catboost_future_cat_hyperparam_results.csv"
    )

    # save as a csv file
    updrs_results_df.to_csv(catboost_hyperparams_path, index=True)
