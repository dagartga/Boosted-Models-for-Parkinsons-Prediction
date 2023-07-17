# xgboost_train.py

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
    init_vals = ["max_depth", "reg_alpha"]
    space = {k: (int(val) if k in init_vals else val) for k, val in space.items()}
    space["early_stopping_rounds"] = early_stopping_rounds
    model = CatBoostClassifier(**space)
    evaluation = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=evaluation, verbose=False)

    pred = model.predict(X_test)
    score = metric(y_test, pred)
    return {"loss": -score, "status": STATUS_OK, "model": model}


def create_kfolds(df, updrs):
    # create a new column for kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins using Sturge's rule
    # I am using the max here to ensure that the number of bins is at least 5
    # and at most 12
    num_bins = int(np.floor(1 + np.log2(len(df))))

    # bin targets
    df.loc[:, "bins"] = pd.cut(df[f"{updrs}_cat"], bins=num_bins, labels=False)

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    # note that instead of targets we are using bins!
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[val_idx, "kfold"] = fold

    # drop the bins column
    df = df.drop("bins", axis=1)

    # return dataframe with folds
    return df

def convert_df_to_1yr(df, updrs):
    # get the max category for each patient
    max_df = df.groupby(['patient_id'])[f'{updrs}_cat'].max().reset_index()
    max_df = max_df.rename(columns={f'{updrs}_cat': f'{updrs}_max_cat'})
    # merge the max category with the original dataframe
    updrs_df = df.merge(max_df, on=['patient_id'], how='left')
    # take only the visit months that are 12 or less
    updrs_yr_df = updrs_df[updrs_df['visit_month'] <= 12]
    updrs_yr_df = updrs_yr_df.drop(columns=[f'{updrs}_cat'])
    updrs_yr_df.rename(columns={f'{updrs}_max_cat': f'{updrs}_cat'}, inplace=True)
    
    return updrs_yr_df



if __name__ == "__main__":
    
    # read the training data
    # read in the protein and updrs data
    updrs1_df = pd.read_csv('../data/processed/train_updrs_1_cat.csv')
    updrs2_df = pd.read_csv('../data/processed/train_updrs_2_cat.csv')
    updrs3_df = pd.read_csv('../data/processed/train_updrs_3_cat.csv')

    # replace the categorical updrs scores with numerical for mild, moderate and severe
    ## combine the moderate and severe categories since there are very few severe observations
    updrs1_df['updrs_1_cat'] = updrs1_df['updrs_1_cat'].map({'mild': 0, 'moderate': 1, 'severe': 1})
    updrs2_df['updrs_2_cat'] = updrs2_df['updrs_2_cat'].map({'mild': 0, 'moderate': 1, 'severe': 1})
    updrs3_df['updrs_3_cat'] = updrs3_df['updrs_3_cat'].map({'mild': 0, 'moderate': 1, 'severe': 1})

    updrs1_df = convert_df_to_1yr(updrs1_df, 'updrs_1')
    updrs2_df = convert_df_to_1yr(updrs2_df, 'updrs_2')
    updrs3_df = convert_df_to_1yr(updrs3_df, 'updrs_3')

    try:
        updrs1_df = updrs1_df.drop(columns=['updrs_1'])
        updrs2_df = updrs2_df.drop(columns=['updrs_2'])
        updrs3_df = updrs3_df.drop(columns=['updrs_3'])
    except:
        print('UPDRS values not in Dataframe')

    updrs_results = dict()

    for updrs, df in zip(
        ["updrs_1", "updrs_2", "updrs_3"], [updrs1_df, updrs2_df, updrs3_df]
    ):
    

        train = df[df["kfold"] != 4].reset_index(drop=True)
        test = df[df["kfold"] == 4].reset_index(drop=True)

        X_train = train.drop(
            columns=[
                "patient_id",
                "kfold",
                f"{updrs}_cat",
            ]
        )
        y_train = train[f"{updrs}_cat"]
        X_test = test.drop(
            columns=[
                "patient_id",
                "kfold",
                f"{updrs}_cat",
            ]
        )
        y_test = test[f"{updrs}_cat"]

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        options = {
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 500, 1)),
            "max_depth": hp.quniform("max_depth", 1, 8, 1),
            # "min_child_weight": hp.loguniform("min_child_weight", -2, 3),
            "subsample": hp.uniform("subsample", 0.5, 1),
            # "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            # "reg_alpha": hp.uniform("reg_alpha", 0, 10),
            # "reg_lambda": hp.uniform("reg_lambda", 1, 10),
            # "min_split_gain": hp.loguniform("min_split_gain", -10, 10),
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
        "../data/processed/catboost_future_cat_12m_hyperparam_results.csv", index=True
    )
