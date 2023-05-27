# src/test_models.py

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb
import lightgbm as lgb
import pickle


# create a function to calculate regression error metrics on the validation set after training
def test_models(df, model, target):
    # train the model on the stratified kfolds 0-3 and test on kfold 4
    train_df = df[df["kfold"] != 4].reset_index(drop=True)
    test_df = df[df["kfold"] == 4].reset_index(drop=True)

    X_train = train_df.drop(columns=[target, "visit_id", "patient_id", "kfold"])

    X_test = test_df.drop(columns=[target, "visit_id", "patient_id", "kfold"])

    y_train = train_df[target]
    y_test = test_df[target]

    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    test_df["preds"] = y_preds

    # calculate and print error metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    mae = mean_absolute_error(y_test, y_preds)
    r2 = r2_score(y_test, y_preds)

    print(f"UPDRS {target} Results:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}\n")

    return test_df, rmse, mae, r2


# create a function to take in the csv of hyperparameters and return the best model
def load_hyperparam_model(model_name, target):
    hyperparams_df = pd.read_csv(
        f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/{model_name}_best_params_{target}.csv"
    )
    params = hyperparams_df["params"].values[0]

    params = ast.literal_eval(params)

    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])

    if model_name == "lightgbm":
        model = lgb.LGBMRegressor(**params)
    elif model_name == "xgboost":
        model = xgb.XGBRegressor(**params)

    print(f"Loaded {model_name} model")

    return model


if __name__ == "__main__":
    updrs1_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1.csv"
    )

    updrs2_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_2.csv"
    )

    updrs3_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_3.csv"
    )

    for model_name in ["lightgbm", "xgboost"]:
        for updrs, df in zip(
            ["updrs_1", "updrs_2", "updrs_3"], [updrs1_df, updrs2_df, updrs3_df]
        ):
            model = load_hyperparam_model(model_name, updrs)
            test_df, rmse, mae, r2 = test_models(df, model, updrs)

            # save the model
            pickle.dump(model, open(f"models/{updrs}_model.pkl", "wb"))

            # save the test_df
            test_df.to_csv(f"models/{model_name}_{updrs}_test_df.csv", index=False)

            # save the error metrics
            error_metrics = pd.DataFrame(
                {
                    "rmse": [rmse],
                    "mae": [mae],
                    "r2": [r2],
                }
            )
            error_metrics.to_csv(
                f"models/{model_name}_{updrs}_error_metrics.csv", index=False
            )

            # save the feature importances
            feature_importances = pd.DataFrame(
                {
                    "feature": df.drop(
                        columns=[updrs, "visit_id", "patient_id", "kfold"]
                    ).columns,
                    "importance": model.feature_importances_,
                }
            )
            feature_importances.to_csv(
                f"models/{model_name}_{updrs}_feature_importances.csv", index=False
            )

            # save the residuals
            residuals_df = pd.DataFrame(
                {
                    "patient_id": test_df["patient_id"],
                    "visit_id": test_df["visit_id"],
                    "residuals": test_df[updrs] - test_df["preds"],
                }
            )
            residuals_df.to_csv(
                f"models/{model_name}_{updrs}_residuals.csv", index=False
            )
