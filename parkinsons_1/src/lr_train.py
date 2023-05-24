import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    feats = [
        "visit_month",
    ]
    cv_lr_model(feats)

    # train the model on the entire training data
    lr_updrs1, train_updrs1_df["preds"] = train_lr_model(
        train_updrs1_df, "visit_month", "updrs_1"
    )
    lr_updrs1, train_updrs2_df["preds"] = train_lr_model(
        train_updrs2_df, "visit_month", "updrs_2"
    )
    lr_updrs1, train_updrs3_df["preds"] = train_lr_model(
        train_updrs3_df, "visit_month", "updrs_3"
    )
    lr_updrs1, train_updrs4_df["preds"] = train_lr_model(
        train_updrs4_df, "visit_month", "updrs_4"
    )

    # save the predictions
    final_lr_df = pd.concat(
        [train_updrs1_df, train_updrs2_df, train_updrs3_df, train_updrs4_df]
    )
    final_lr_df.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/lr_preds_1.csv",
        index=False,
    )

    # plot the predictions
    plt.figure(figsize=(12, 8))
    plt.scatter(
        train_updrs1_df["visit_month"], train_updrs1_df["updrs_1"], label="updrs_1"
    )
    plt.plot(train_updrs1_df["visit_month"], train_updrs1_df["preds"], label="preds")
    plt.legend()
    plt.title("updrs_1 linear regression results", fontsize=20)
    plt.xlabel("visit_month", fontsize=16)
    plt.ylabel("updrs_1", fontsize=16)
    plt.show()

    # plot the predictions
    plt.figure(figsize=(12, 8))
    plt.scatter(
        train_updrs2_df["visit_month"], train_updrs2_df["updrs_2"], label="updrs_2"
    )
    plt.plot(train_updrs2_df["visit_month"], train_updrs2_df["preds"], label="preds")
    plt.legend()
    plt.title("updrs_2 linear regression results", fontsize=20)
    plt.xlabel("visit_month", fontsize=16)
    plt.ylabel("updrs_2", fontsize=16)
    plt.show()

    # plot the predictions
    plt.figure(figsize=(12, 8))
    plt.scatter(
        train_updrs3_df["visit_month"], train_updrs3_df["updrs_3"], label="updrs_3"
    )
    plt.plot(train_updrs3_df["visit_month"], train_updrs3_df["preds"], label="preds")
    plt.legend()
    plt.title("updrs_3 linear regression results", fontsize=20)
    plt.xlabel("visit_month", fontsize=16)
    plt.ylabel("updrs_3", fontsize=16)
    plt.show()

    # plot the predictions
    plt.figure(figsize=(12, 8))
    plt.scatter(
        train_updrs4_df["visit_month"], train_updrs4_df["updrs_4"], label="updrs_4"
    )
    plt.plot(train_updrs4_df["visit_month"], train_updrs4_df["preds"], label="preds")
    plt.legend()
    plt.title("updrs_4 linear regression results", fontsize=20)
    plt.xlabel("visit_month", fontsize=16)
    plt.ylabel("updrs_4", fontsize=16)
    plt.show()
