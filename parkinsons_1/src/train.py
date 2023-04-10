# src/train.py

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1.csv')
df.columns[-1]

def run(fold, target):
    
    # read the training data with folds
    df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv')
    
    df_train = df[df['bins'] != fold].reset_index(drop=True)
    
    df_valid = df[df['bins'] == fold].reset_index(drop=True)
    
    x_train = df_train.drop([target, 'bins'], axis=1).values
    y_train = df_train[target].values
    
    x_valid = df_valid.drop([target, 'bins'], axis=1).values
    y_valid = df_valid[target].values
    
    clf = RandomForestRegressor()
    
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)
    
    # save the model    
    joblib.dump(clf, f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/models/{target}_model_{fold}.bin')
    
    r2 = metrics.r2_score(y_valid, preds)
    mape = metrics.mean_absolute_percentage_error(y_valid, preds)
    
    print(f'Fold = {fold}, R2 = {r2}, MAPE = {mape}')
    
if __name__ == '__main__':
    
    for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
        for fold in range(5):
            run(fold, target)
            
    