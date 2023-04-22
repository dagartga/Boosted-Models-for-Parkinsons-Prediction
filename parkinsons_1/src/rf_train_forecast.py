# src/train.py

import os
import argparse

import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



def smape(y_true, y_pred):

    return round(np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred))/2)) * 100, 2)

def run(model, target, month_diff):
    # read the training data with folds
    df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/forecast_train_{target}.csv')
    
    month_diff = 6
    target = 'updrs_4'
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
    
    models = [('rf_reg', RandomForestRegressor(random_state = 42))]
    
    results = []
    
    
    for model_name, model in models:
        for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
            for month_diff in [6, 12, 24]:
                try:
                    print('Running model: ', model_name, 'for target: ', target)
                    s, r, m = run(model, target, month_diff)
                    results.append({"Model":model_name, "Target":target, "SMAPE":s, "R2":r, "MAPE":m, 'Visit_Month_Diff':month_diff})
                    print('SMAPE: ', s, 'R2: ', r, 'MAPE: ', m, '\n')
                    # store the model
                    model_file = f'../models/model_{model_name}_{target}_{month_diff}.pkl'
                    pickle.dump(model, open(model_file, 'wb'))
                except:
                    print('---------------------------------------------------')
                    print('Error running model: ', model_name, 'for target: ', target, '\n')
        
    results_df = pd.DataFrame(results).set_index('Model')
    results_df.to_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/model_results/rfreg_{date}_results.csv')
