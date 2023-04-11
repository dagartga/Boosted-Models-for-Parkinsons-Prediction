# src/train.py

import os
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics

import model_dispatcher



def smape(y_true, y_pred):

    return round(np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred))/2)) * 100, 2)



def run(fold=0, model='randomforest_reg', target='updrs_1'):
    # read the training data with folds
    df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv')
    df = df.drop(columns=['visit_id', 'patient_id'])
    
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    
    x_train = df_train.drop([target, 'kfold'], axis=1).values
    y_train = df_train[target].values
    
    x_valid = df_valid.drop([target, 'kfold'], axis=1).values
    y_valid = df_valid[target].values
    
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)
    
    # save the model    
    #joblib.dump(clf, f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/{target}_model_{fold}.bin')
    
    r2 = metrics.r2_score(y_valid, preds)
    mape = metrics.mean_absolute_percentage_error(y_valid, preds)
    s_mape = smape(y_valid, preds)
    
    print(f'Fold = {fold}, SMAPE = {s_mape}, R2 = {r2}, MAPE = {mape}')
    
    return fold, s_mape, r2, mape
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    for updrs in ['updrs_1']:
        all_smapes = []
        
        f, s, r, m = run(fold=args.fold, model=args.model, target=updrs)
        all_smapes.append(s)
        
        print(f'Average SMAPE for {updrs} = {np.mean(all_smapes)}')
            
    