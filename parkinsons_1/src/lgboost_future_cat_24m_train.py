# xgboost_train.py

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


def optimize(params, x, y):
    """
    Optimizer function for the hyperparameter tuning
    """
    model = LGBMClassifier(**params, n_estimators=500)
    kf = model_selection.StratifiedKFold(n_splits=5)
    auc = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_auc = roc_auc_score(ytest, preds)
        auc.append(fold_auc)

    return -1.0 * np.mean(auc)


if __name__ == "__main__":
    # read the training data
    df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_24month_protein_data.csv"
    )

    # get only the updrs of interest
    # the updrs_1_max is the target, which means the highest categorical value
    # 0 is mild parkinsons, and 1 is moderate to severe parkinsons as the max
    updrs1_df = df.drop(columns=["updrs_2_max", "updrs_3_max"])
    updrs2_df = df.drop(columns=["updrs_1_max", "updrs_3_max"])
    updrs3_df = df.drop(columns=["updrs_1_max", "updrs_2_max"])

    df = updrs1_df
    updrs = "updrs_1"

    updrs_results = dict()

    for updrs, df in zip(
        ["updrs_1", "updrs_2", "updrs_3"], [updrs1_df, updrs2_df, updrs3_df]
    ):
        X = df.drop(
            columns=[
                "patient_id",
                f"{updrs}_max",
            ]
        ).values
        y = df[f"{updrs}_max"].values
        # encode the target
        y = LabelEncoder().fit_transform(y)

        param_space = {
            "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
            "min_data_in_leaf": scope.int(hp.quniform("min_data_in_leaf", 10, 30, 1)),
            "bagging_freq": scope.int(hp.uniform("bagging_freq", 1, 10)),
            "bagging_fraction": hp.uniform("bagging_fraction", 0.3, 0.9),
            "tree_learner": hp.choice(
                "tree_learner", ["serial", "feature", "data", "voting"]
            ),
            "is_unbalance": hp.choice("is_unbalance", [True, False]),
            "boosting": hp.choice("boosting", ["gbdt", "dart", "rf"]),
            "lambda_l2": hp.uniform("lambda_l2", 0, 10),
            "lambda_l1": hp.uniform("lambda_l1", 0, 10),
            "feature_fraction": hp.uniform("feature_fraction", 0.5, 0.9),
            "learning_rate": hp.loguniform("learning_rate", -7, 0),
            "max_delta_step": scope.int(hp.quniform("max_delta_step", 1, 10, 1)),
            "random_state": 42,
        }

        # partial function
        optimization_function = partial(optimize, x=X, y=y)

        # initialize trials to keep logging information
        trials = Trials()

        # run hyperopt
        best = fmin(
            fn=optimization_function,
            space=param_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
        )

        updrs_results[updrs] = best

    # save the results
    updrs_results_df = pd.DataFrame(updrs_results)

    # save as a csv file
    updrs_results_df.to_csv(
        "../data/processed/lgboost_future_cat_24m_hyperparam_results.csv", index=True
    )
