import os
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
        drop_cols = ["patient_id", f"{target}_cat", "kfold"]
        X_train = train.drop(columns=drop_cols)
        y_train = train[f"{target}_cat"]
        X_test = test.drop(columns=drop_cols)
        y_test = test[f"{target}_cat"]

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


def convert_df_to_1yr(df, updrs):
    # get the max category for each patient
    max_df = df.groupby(["patient_id"])[f"{updrs}_cat"].max().reset_index()
    max_df = max_df.rename(columns={f"{updrs}_cat": f"{updrs}_max_cat"})
    # merge the max category with the original dataframe
    updrs_df = df.merge(max_df, on=["patient_id"], how="left")
    # take only the visit months that are 12 or less
    updrs_yr_df = updrs_df[updrs_df["visit_month"] <= 12]
    updrs_yr_df = updrs_yr_df.drop(columns=[f"{updrs}_cat"])
    updrs_yr_df.rename(columns={f"{updrs}_max_cat": f"{updrs}_cat"}, inplace=True)

    return updrs_yr_df


if __name__ == "__main__":
    # create file paths
    updrs1_path = os.path.join("..", "data", "processed", "train_updrs_1_cat.csv")
    updrs2_path = os.path.join("..", "data", "processed", "train_updrs_2_cat.csv")
    updrs3_path = os.path.join("..", "data", "processed", "train_updrs_3_cat.csv")

    # read the training data
    updrs1_df = pd.read_csv(updrs1_path)
    updrs2_df = pd.read_csv(updrs2_path)
    updrs3_df = pd.read_csv(updrs3_path)

    # replace the categorical updrs scores with numerical for mild, moderate and severe
    ## combine the moderate and severe categories since there are very few severe observations
    updrs1_df["updrs_1_cat"] = updrs1_df["updrs_1_cat"].map(
        {"mild": 0, "moderate": 1, "severe": 1}
    )
    updrs2_df["updrs_2_cat"] = updrs2_df["updrs_2_cat"].map(
        {"mild": 0, "moderate": 1, "severe": 1}
    )
    updrs3_df["updrs_3_cat"] = updrs3_df["updrs_3_cat"].map(
        {"mild": 0, "moderate": 1, "severe": 1}
    )

    updrs1_df = convert_df_to_1yr(updrs1_df, "updrs_1")
    updrs2_df = convert_df_to_1yr(updrs2_df, "updrs_2")
    updrs3_df = convert_df_to_1yr(updrs3_df, "updrs_3")

    updrs1_df = updrs1_df.drop(columns=["kfold"])
    updrs2_df = updrs2_df.drop(columns=["kfold"])
    updrs3_df = updrs3_df.drop(columns=["kfold"])

    # params to test
    max_delta_step = [1, 3, 5, 7, 9]
    scale_pos_weight = [1.1, 1.3, 1.7, 2, 2.3]
    subsample = [0.3, 0.5, 0.7, 0.9]
    colsample_bytree = [0.3, 0.5, 0.7, 0.9]

    val_list = max_delta_step + scale_pos_weight + subsample + colsample_bytree

    param_names = (
        ["max_delta_step"] * len(max_delta_step)
        + ["scale_pos_weight"] * len(scale_pos_weight)
        + ["subsample"] * len(subsample)
        + ["colsample_bytree"] * len(colsample_bytree)
    )

    test_params_df = pd.DataFrame(
        columns=["updrs", "param", "val", "auc", "acc", "prec", "recall"]
    )
    final_df = pd.DataFrame(columns=["param", "val", "auc", "acc", "prec", "recall"])

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

        xgb_hyperparam_path = os.path.join(
            "..", "data", "processed", "xgboost_future_cat_12m_hyperparam_results.csv"
        )
        xgb_hyperparams_df = pd.read_csv(
            xgb_hyperparam_path,
            index_col=0,
        )

        i = 0
        for param, val in zip(param_names, val_list):
            # prepare the model
            test_param = [param, val]
            model = prepare_xgboost_model(xgb_hyperparams_df, updrs, test_param)
            auc, acc, prec, recall = cross_fold_validation(df, model, updrs)

            test_params_df.loc[i, "updrs"] = updrs
            test_params_df.loc[i, "param"] = param
            test_params_df.loc[i, "val"] = val
            test_params_df.loc[i, "auc"] = auc
            test_params_df.loc[i, "acc"] = acc
            test_params_df.loc[i, "prec"] = prec
            test_params_df.loc[i, "recall"] = recall
            i += 1

        # save the results
        xgb_finetune_path = os.path.join(
            ".", "models", f"xgboost_24m_hyperparam_finetune_results_{updrs}.csv"
        )
        test_params_df.to_csv(xgb_finetune_path, index=True)
