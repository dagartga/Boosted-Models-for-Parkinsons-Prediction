# xgboost_train.py

import numpy as np
import pandas as pd
import pickle
from functools import partial
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection


# i.	max_delta_step
# ii.	scale_pos_weight
# iii.	subsampling
# iv.	sampling_method
# v.	colsample_bytree
# vi.	max_leaves


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
    df.loc[:, "bins"] = pd.cut(df[f"{updrs}_max"], bins=num_bins, labels=False)

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


def prepare_xgboost_model(xgb_hyperparams_df, target, test_param):
    # train the model using the hyperparameters from the hyperparameter tuning
    updrs_hp = xgb_hyperparams_df[target].to_dict()
    test_param_key = test_param[0]
    test_param_value = test_param[1]
    updrs_hp[test_param_key] = test_param_value
    updrs_hp["max_depth"] = int(updrs_hp["max_depth"])
    model = XGBClassifier(**updrs_hp)
    return model


def cross_fold_validation(df, model, target):
    updrs_results = dict()

    for fold in range(0, 5):
        # get the train and test data for the current fold
        train = df[df["kfold"] != fold].reset_index(drop=True)
        test = df[df["kfold"] == fold].reset_index(drop=True)

        # get the train and test data for the current fold
        drop_cols = ["patient_id", f"{target}_max", "kfold"]
        X_train = train.drop(columns=drop_cols)
        y_train = train[f"{target}_max"]
        X_test = test.drop(columns=drop_cols)
        y_test = test[f"{target}_max"]

        # train the model
        model.fit(X_train, y_train)

        # make predictions
        preds = model.predict(X_test)

        # save the results
        updrs_results[f"{target}_fold_{fold}"] = {
            "auc_score": roc_auc_score(y_test, preds),
            "acc_score": accuracy_score(y_test, preds),
            "precision_score": precision_score(y_test, preds),
            "recall_score": recall_score(y_test, preds),
        }

    mean_auc = np.mean(
        [updrs_results[f"{target}_fold_{fold}"]["auc_score"] for fold in range(0, 5)]
    )
    mean_acc = np.mean(
        [updrs_results[f"{target}_fold_{fold}"]["acc_score"] for fold in range(0, 5)]
    )
    mean_precision = np.mean(
        [
            updrs_results[f"{target}_fold_{fold}"]["precision_score"]
            for fold in range(0, 5)
        ]
    )
    mean_recall = np.mean(
        [updrs_results[f"{target}_fold_{fold}"]["recall_score"] for fold in range(0, 5)]
    )

    return mean_auc, mean_acc, mean_precision, mean_recall


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

    # params to test
    # bagging_freq = [1, 3, 5, 7, 9] # needs bagging_fraction also
    tree_learner = ["serial", "feature", "data", "voting"]
    boosting = ["gbdt", "dart"]
    feature_fraction = [0.4, 0.6, 0.8, 1.0]
    is_unbalance = [True, False]
    min_data_in_leaf = [10, 20, 30, 40, 50]
    max_depth = [1, 5, 10, 15, 20]
    val_list = (
        tree_learner
        + boosting
        + feature_fraction
        + is_unbalance
        + min_data_in_leaf
        + max_depth
    )
    param_names = (
        ["tree_learner"] * len(tree_learner)
        + ["boosting"] * len(boosting)
        + ["feature_fraction"] * len(feature_fraction)
        + ["is_unbalance"] * len(is_unbalance)
        + ["min_data_in_leaf"] * len(min_data_in_leaf)
        + ["max_depth"] * len(max_depth)
    )

    updrs_results = dict()

    test_params_dict = dict()
    final_df = pd.DataFrame()

    for updrs, df in zip(
        ["updrs_1", "updrs_2", "updrs_3"], [updrs1_df, updrs2_df, updrs3_df]
    ):
        # preprocess the df
        df = create_kfolds(df, updrs)

        train = df[df["kfold"] != 4].reset_index(drop=True)
        test = df[df["kfold"] == 4].reset_index(drop=True)

        X_train = train.drop(
            columns=[
                "patient_id",
                "kfold",
                f"{updrs}_max",
            ]
        )
        y_train = train[f"{updrs}_max"]
        X_test = test.drop(
            columns=[
                "patient_id",
                "kfold",
                f"{updrs}_max",
            ]
        )
        y_test = test[f"{updrs}_max"]

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        xgb_hyperparams_df = pd.read_csv(
            "../data/processed/xgboost_future_cat_24m_hyperparam_results.csv",
            index_col=0,
        )

        for param, val in zip(param_names, val_list):
            # prepare the model
            test_param = [param, val]
            model = prepare_xgboost_model(xgb_hyperparams_df, updrs, test_param)
            auc, acc, prec, recall = cross_fold_validation(df, model, updrs)

            param_val = f"{param}_{val}"
            test_params_dict[param_val] = {
                "AUC": auc,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": recall,
            }

        # save the results
        final_results = pd.DataFrame(test_params_dict)
        test_params_dict = dict()
        final_results.to_csv(
            f"./models/xgboost_24m_hyperparam_finetune_results_{updrs}.csv", index=True
        )
