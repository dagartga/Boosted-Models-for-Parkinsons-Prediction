import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import json


if __name__ == "__main__":
    filename = "../webapp/lgboost_updrs_3_model_hyperopt_smote_meds.sav"
    updrs3_model = pickle.load(open(filename, "rb"))
    updrs3_feat_imp = updrs3_model.feature_importances_

    # File path where you want to save the column names
    file_path = "updrs_3_cols.txt"

    # Initialize an empty list to store the retrieved column names
    retrieved_column_names = []

    # Open the file in read mode and read each line as a column name
    with open(file_path, "r") as file:
        for line in file:
            retrieved_column_names.append(line.strip())  # Remove newline characters

    # Create a dataframe with the retrieved column names and feature importances
    updrs3_feat_imp_df = pd.DataFrame(
        {"feature": retrieved_column_names, "importance": updrs3_feat_imp}
    )

    # get the top 10 features
    top_ten_feats = updrs3_feat_imp_df.sort_values(by="importance", ascending=False)[
        :10
    ]
    top_ten_feats = top_ten_feats.reset_index()
    top_ten_feats["index"] = top_ten_feats["index"].astype(str)

    # create a barplot of the top 10 features
    sns.barplot(
        x="importance",
        y="index",
        data=top_ten_feats,
    )
    plt.title("Feature Importance for UPDRS 3")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("updrs3_feat_imp.png")
