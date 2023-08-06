# src/visualization/visualize.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_residuals(target):
    for model_name in ["lightgbm", "xgboost"]:
        residuals = pd.read_csv(
            f"~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/{model_name}_{target}_residuals.csv"
        )

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=residuals["visit_month"], y=residuals["residuals"], alpha=0.5)
        plt.title(f"{target} {model_name} Residuals vs. Visit Month", fontsize=20)
        plt.xlabel("Visit Month", fontsize=14)
        plt.ylabel("Residuals", fontsize=14)
        plt.savefig(f"residual_plots/{model_name}_{target}_0_residuals.png")
        plt.show()


if __name__ == "__main__":
    for target in ["updrs_1", "updrs_2", "updrs_3"]:
        plot_residuals(target)
        print(f"{target} Done!")
