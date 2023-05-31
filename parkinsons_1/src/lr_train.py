import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def smape(y_true, y_pred):
    return round(
        np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
        * 100,
        2,
    )


def cv_lr_model(feats):
    test_smape_vals = {"updrs_1": [], "updrs_2": [], "updrs_3": [], "updrs_4": []}

    # loop through the updrs and train linear regression model and make predictions
    for k, v in {
        "updrs_1": train_updrs1_df,
        "updrs_2": train_updrs2_df,
        "updrs_3": train_updrs3_df,
        "updrs_4": train_updrs4_df,
    }.items():
        for fold in range(5):
            train = v[v["kfold"] != fold]
            valid = v[v["kfold"] == fold]

            X_train = train["visit_month"].values.reshape(-1, 1)
            y_train = train[k]

            X_valid = valid["visit_month"].values.reshape(-1, 1)
            y_valid = valid[k]

            reg = LinearRegression()
            reg.fit(X_train, y_train)
            preds = reg.predict(X_valid)

            print(f"\nupdrs: {k}, fold: {fold}")
            test_smape = smape(y_valid, preds)
            print(f"smape: {test_smape}")

            test_smape_vals[k].append(test_smape)

    # print the average smape for each updrs
    for updrs in test_smape_vals.keys():
        print(f"\nupdrs: {updrs}, avg smape: {np.mean(test_smape_vals[updrs])}")


def train_lr_model(df, feats, target):
    X = df[feats].values.reshape(-1, 1)
    y = df[target]
    lr = LinearRegression()
    lr.fit(X, y)
    train_preds = lr.predict(X)
    print(f"\nupdrs: {target}")
    print(f"train smape: {smape(y, train_preds)}")

    return lr, train_preds


def test_lr_model(df, feats, target):
    # create stratified kfold test selection
    X_train = df[df["kfold"] != 4].reset_index(drop=True)
    X_test = df[df["kfold"] == 4].reset_index(drop=True)
    y_train = X_train[target]
    y_test = X_test[target]

    lr = LinearRegression()
    lr.fit(X_train[feats], y_train)
    test_preds = lr.predict(X_test[feats])
    print(f"\nupdrs: {target}")
    mae = mean_absolute_error(y_test, test_preds)
    print(f"test mae: {mae}")
    mse = mean_squared_error(y_test, test_preds)
    print(f"test mse: {mse}")
    r2 = r2_score(y_test, test_preds)
    print(f"test r2: {r2}")

    # predict on all folds
    all_preds = lr.predict(df[feats])

    test_metrics = {target: {"MAE": mae, "MSE": mse, "R2": r2}}

    return test_metrics, all_preds


if __name__ == "__main__":
    train_updrs1_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1.csv"
    )
    train_updrs2_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_2.csv"
    )
    train_updrs3_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_3.csv"
    )
    train_updrs4_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_4.csv"
    )

    test_metrics, test_df = test_lr_model(train_updrs1_df, ["visit_month"], "updrs_1")

    feats = [
        "visit_month",
    ]
    cv_lr_model(feats)

    # test the model
    test_metrics_udprs1, test_updrs1_df = test_lr_model(
        train_updrs1_df, feats, "updrs_1"
    )
    test_metrics_updrs2, test_updrs2_df = test_lr_model(
        train_updrs2_df, feats, "updrs_2"
    )
    test_metrics_updrs3, test_updrs3_df = test_lr_model(
        train_updrs3_df, feats, "updrs_3"
    )
    test_metrics_updrs4, test_updrs4_df = test_lr_model(
        train_updrs4_df, feats, "updrs_4"
    )

    train_updrs1_df["preds"] = test_updrs1_df
    train_updrs2_df["preds"] = test_updrs2_df
    train_updrs3_df["preds"] = test_updrs3_df
    train_updrs4_df["preds"] = test_updrs4_df

    # convert the results dictionary into a dataframe
    test_metrics1_df = pd.DataFrame.from_dict(test_metrics_udprs1, orient="index")
    test_metrics2_df = pd.DataFrame.from_dict(test_metrics_updrs2, orient="index")
    test_metrics3_df = pd.DataFrame.from_dict(test_metrics_updrs3, orient="index")
    test_metrics4_df = pd.DataFrame.from_dict(test_metrics_updrs4, orient="index")
    test_metrics_df = pd.concat(
        [test_metrics1_df, test_metrics2_df, test_metrics3_df, test_metrics4_df], axis=0
    ).reset_index()
    test_metrics_df.columns = ["updrs", "MAE", "MSE", "R2"]

    # save the results to a csv
    test_metrics_df.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/lr_test_metrics.csv",
        index=False,
    )
    train_updrs1_df.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/lr_train_updrs1_preds.csv",
        index=False,
    )
    train_updrs2_df.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/lr_train_updrs2_preds.csv",
        index=False,
    )
    train_updrs3_df.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/lr_train_updrs3_preds.csv",
        index=False,
    )
    train_updrs4_df.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/lr_train_updrs4_preds.csv",
        index=False,
    )
