# src/data/pca_features.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_pca(df, updrs):
    # Create a DataFrame with only the feature columns
    feature_df = df.drop(columns=updrs)

    # Standardize the feature data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    # Apply PCA
    pca = PCA()
    pca.fit(scaled_features)

    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Print the explained variance ratio
    print("Explained Variance Ratio:")
    print(explained_variance_ratio)

    # Determine the optimal number of components
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    optimal_num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print("Optimal Number of Components:", optimal_num_components)

    # Apply PCA with the optimal number of components
    pca = PCA(n_components=optimal_num_components)
    pca_features = pca.fit_transform(scaled_features)

    # Create a new DataFrame with the PCA features
    pca_df = pd.DataFrame(
        data=pca_features,
        columns=[f"PC{i}" for i in range(1, optimal_num_components + 1)],
    )

    # Concatenate the PCA features with the target column (if applicable)
    if updrs in df.columns:
        pca_df[updrs] = df[updrs]

    return pca_df


if __name__ == "__main__":
    updrs1_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1_cat.csv"
    )
    updrs2_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_2_cat.csv"
    )
    updrs3_df = pd.read_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_3_cat.csv"
    )

    updrs1_pca_df = updrs1_df.drop(
        columns=["visit_id", "patient_id", "updrs_1", "kfold"]
    )
    updrs1_pca = perform_pca(updrs1_pca_df, "updrs_1_cat")
    updrs1_final_pca = pd.concat(
        [updrs1_df[["visit_id", "patient_id", "updrs_1", "kfold"]], updrs1_pca], axis=1
    )

    updrs2_pca_df = updrs2_df.drop(
        columns=["visit_id", "patient_id", "updrs_2", "kfold"]
    )
    updrs2_pca = perform_pca(updrs2_pca_df, "updrs_2_cat")
    updrs2_final_pca = pd.concat(
        [updrs2_df[["visit_id", "patient_id", "updrs_2", "kfold"]], updrs2_pca], axis=1
    )

    updrs3_pca_df = updrs3_df.drop(
        columns=["visit_id", "patient_id", "updrs_3", "kfold"]
    )
    updrs3_pca = perform_pca(updrs3_pca_df, "updrs_3_cat")
    updrs3_final_pca = pd.concat(
        [updrs3_df[["visit_id", "patient_id", "updrs_3", "kfold"]], updrs3_pca], axis=1
    )

    # save the pca dataframes
    updrs1_final_pca.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1_pca.csv",
        index=False,
    )
    updrs2_final_pca.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_2_pca.csv",
        index=False,
    )
    updrs3_final_pca.to_csv(
        "~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_3_pca.csv",
        index=False,
    )
