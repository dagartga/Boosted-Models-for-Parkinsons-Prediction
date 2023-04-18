# src/train_1.py

import os
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor




def smape(y_true, y_pred):

    return round(np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred))/2)) * 100, 2)




def run(model, target, month_diff):
    # read the training data with folds
    df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/forecast_train_{target}.csv')
    
    forecast_cols = [col for col in df.columns if 'updrs' in col]

    drop_cols = [col for col in forecast_cols if col not in  [f'{target}_0', f'{target}_{month_diff}']]

    df = df.drop(columns=drop_cols)
    df = df.drop(columns=['visit_id', 'visit_month'])
    
    target_mo = f'{target}_{month_diff}'
    # drop nan rows for target column
    df = df.dropna(subset=[target_mo])
    
    x_train, x_valid, y_train, y_valid = train_test_split(df, df[target_mo], test_size=0.2, random_state=42)
    
    x_train = x_train.drop([target_mo], axis=1).values
    x_valid = x_valid.drop([target_mo], axis=1).values

    
    reg = model
    reg.fit(x_train, y_train)
    preds = reg.predict(x_valid)
    
    r2 = metrics.r2_score(y_valid, preds)
    mape = metrics.mean_absolute_percentage_error(y_valid, preds)
    s_mape = smape(y_valid, preds)
    
    
    return s_mape, r2, mape
            
            
    
            
    
    
    
if __name__ == '__main__':
    
    
    models = [('rf_reg', RandomForestRegressor(random_state = 42)),
              ('xgboost', XGBRegressor(random_state = 42)),
              ('lgbm', LGBMRegressor(random_state = 42))]
    
    results = []
    
    for model_name, model in models:
        for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
            for month_diff in [6, 12, 24]:
                try:
                    print('Running model: ', model_name, 'for target: ', target, 'for month_diff: ', month_diff)
                    s, r, m = run(model, target, month_diff)
                    results.append({"Model":model_name, "Target":target, "Month_Diff":month_diff, "SMAPE":s, "R2":r, "MAPE":m})
                    print('SMAPE: ', s, 'R2: ', r, 'MAPE: ', m, '\n')
                except:
                    print('Error running model: ', model_name, 'for target: ', target, 'for month_diff: ', month_diff, '\n')
    
    results_df = pd.DataFrame(results).set_index('Model')
    results_df.to_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/baseline_forecast_results.csv')
    for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
        updr_results_df = results_df[results_df['Target'] == target]
        best_model = updr_results_df.groupby('Model')['SMAPE'].mean().sort_values().reset_index().iloc[0,:]
        print(f'For {target} the best model is: ', best_model['Model'])
        print(f'For {target} best model SMAPE is: ', best_model['SMAPE'])