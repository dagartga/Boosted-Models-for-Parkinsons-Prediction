# src/train_1.py

import os
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor



def smape(y_true, y_pred):

    return round(np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred))/2)) * 100, 2)



def run(fold, model, target, month_diff):
    # read the training data with folds
    df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/forecast_train_{target}.csv')
    # change target to the updrs_diff
    target = f'{target}_diff'
    
    df = df.drop(columns=['visit_id', 'patient_id', 'visit_month'])
    df = df[df['visit_month_diff'] == month_diff]
    
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    
    x_train = df_train.drop([target, 'kfold'], axis=1).values
    y_train = df_train[target].values
    
    x_valid = df_valid.drop([target, 'kfold'], axis=1).values
    y_valid = df_valid[target].values
    
    reg = model
    reg.fit(x_train, y_train)
    preds = reg.predict(x_valid)
    
    r2 = metrics.r2_score(y_valid, preds)
    mape = metrics.mean_absolute_percentage_error(y_valid, preds)
    s_mape = smape(y_valid, preds)
    
    
    # save the model    
    #joblib.dump(clf, f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/{target}_model_{fold}.bin')
    
    return fold, s_mape, r2, mape
            
            
    
        
    
    
    
    
    
if __name__ == '__main__':
    
    
    models = [('rf_reg', RandomForestRegressor(random_state = 42)),
              ('xgboost', XGBRegressor(random_state = 42)),
              ('lgbm', LGBMRegressor(random_state = 42))]
    
    results = []
    
    for model_name, model in models:
        for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
            for month_diff in [6, 12, 24]:
                for fold in range(5):
                    print('Running model: ', model_name, 'for target: ', target, 'for month_diff: ', month_diff, 'for fold: ', fold)
                    f, s, r, m = run(fold, model, target, month_diff)
                    results.append({"Model":model_name, "Target":target, "Month_Diff":month_diff, "Fold":f, "SMAPE":s, "R2":r, "MAPE":m})
                    print('SMAPE: ', s, 'R2: ', r, 'MAPE: ', m, '\n')
    
    results_df = pd.DataFrame(results).set_index('Model')
    results_df.to_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/baseline_forecast_results.csv')
    for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
        updr_results_df = results_df[results_df['Target'] == target]
        best_model = updr_results_df.groupby('Model')['SMAPE'].mean().sort_values().reset_index().iloc[0,:]
        print(f'For {target} the best model is: ', best_model['Model'])
        print(f'For {target} best model SMAPE is: ', best_model['SMAPE'])