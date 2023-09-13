import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from functools import partial
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import pickle

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope


def optmize(params, param_names, x, y):
    """
    The main optimization function. This function will take all the arguments
    from the search space and training features and labels. It will then initialize
    the models by setting the chosen parameters and run cross-validation and return a
    mean squared error score as a result.

    Args:
        params (list): list of params from gp_minimize
        param_names (list): parameter names
        x (dataframe): training data
        y (series): target values

    return: mean squared error after cross validation

    """

    # convert params to dictionary
    params = dict(zip(param_names, params))

    # initialize model with current parameters
    model = LGBMRegressor(**params)

    # initialize stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=4)

    # initialize mse list
    mse_list = []

    # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, val_idx = idx[0], idx[1]

        # training and validation data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # fit model for current fold
        model.fit(X_train, y_train)

        # create predictions
        preds = model.predict(X_val)

        # calculate and append mse score
        fold_mse = mean_squared_error(y_val, preds)
        mse_list.append(fold_mse)

    # return mean of mse scores
    return np.mean(mse_list)


if __name__ == "__main__":
    # create file paths
    updrs1_path = os.path.join("..", "data", "processed", "train_updrs_1_cat.csv")
    updrs2_path = os.path.join("..", "data", "processed", "train_updrs_2_cat.csv")
    updrs3_path = os.path.join("..", "data", "processed", "train_updrs_3_cat.csv")

    # read the training data
    updrs1_df = pd.read_csv(updrs1_path)
    updrs2_df = pd.read_csv(updrs2_path)
    updrs3_df = pd.read_csv(updrs3_path)

    for updrs, df in zip(
        ["updrs_1", "updrs_2", "updrs_3"], [updrs1_df, updrs2_df, updrs3_df]
    ):
        X = df.drop(columns=["visit_id", "patient_id", updrs, "kfold"])
        y = df[updrs]

        # define the parameter space
        param_space = [
            scope.int(hp.quniform("max_depth", 3, 20, 1)),
            scope.int(hp.quniform("min_child_weight", 1, 7, 1)),
            scope.float(hp.uniform("reg_alpha", 0.1, 1)),
            scope.float(hp.uniform("reg_lambda", 0.01, 1)),
            scope.float(hp.uniform("min_split_gain", 0.05, 1)),
            scope.float(hp.uniform("colsample_bytree", 0.6, 1)),
            scope.float(hp.uniform("subsample", 0.6, 1)),
            scope.float(hp.uniform("learning_rate", 0.01, 0.1)),
        ]

        # give the param names
        param_names = [
            "max_depth",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "min_split_gain",
            "colsample_bytree",
            "subsample",
            "learning_rate",
        ]

        # partial function
        optimization_function = partial(optmize, param_names=param_names, x=X, y=y)

        # initialize trials to keep logging information
        trials = Trials()

        # run hyperopt
        hopt = fmin(
            fn=optimization_function,
            space=param_space,
            algo=tpe.suggest,
            max_evals=15,
            trials=trials,
        )

        print("updrs: ", updrs, "hopt:")
        print(hopt, "\n")

        # format max_depth and min_child_weight to int
        hopt["max_depth"] = int(hopt["max_depth"])
        hopt["min_child_weight"] = int(hopt["min_child_weight"])

        # save the best params to a csv file
        best_params = pd.DataFrame(columns=["updrs", "params"])
        best_params["updrs"] = [updrs]
        best_params["params"] = [hopt]
        results_path = os.path.join(
            "..", "models", "model_results", f"lightgbm_best_params_{updrs}.csv"
        )
        best_params.to_csv(
            results_path,
            index=False,
        )
