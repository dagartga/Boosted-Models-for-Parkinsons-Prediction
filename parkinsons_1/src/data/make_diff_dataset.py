import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np



def preprocess_forecast_train_df(forecast_data, target):
    
    temp_df = forecast_data[['visit_id', 'patient_id', target, 'visit_month']].sort_values(by=['patient_id', 'visit_month']).reset_index(drop=True)
    temp_df['visit_month_diff'] = temp_df['visit_month'] - temp_df['visit_month'].shift(1)
    temp_df['prev_patient'] = temp_df['patient_id'].shift(1)
    
    # if next_patient does not equal patient_id, then visit_month_diff should be NaN
    temp_df['visit_month_diff'] = np.where(temp_df['prev_patient'] != temp_df['patient_id'], np.nan, temp_df['visit_month_diff'])
    temp_df['visit_month_diff'] = temp_df['visit_month_diff'].fillna(0)

    # find difference between the target values
    temp_df[f'{target}_diff'] = temp_df[target] - temp_df[target].shift(1)
    temp_df[f'{target}_diff'] = np.where(temp_df['prev_patient'] != temp_df['patient_id'], np.nan, temp_df[f'{target}_diff'])
    temp_df = temp_df.drop(columns=['prev_patient', 'patient_id', 'visit_month', target])
    
    # merge the temp_df with the forecast_data on the visit_id
    diff_df = forecast_data.merge(temp_df, on='visit_id', how='left')
    
    target_diff = f'{target}_diff'
    
    #calculate the number of bins by Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(diff_df))))
    diff_df.loc[:, "bins"] = pd.cut(diff_df[target_diff], bins=num_bins, labels=False)

    diff_df = diff_df.dropna().reset_index(drop=True)
        
    # initiate the kfold class from sklearn
    kf = StratifiedKFold(n_splits=5)
        
    # create a kfold column
    diff_df['kfold'] = -1

    # fill the kfold column
    for f, (t_, v_) in enumerate(kf.split(X=diff_df, y=diff_df['bins'].values)):
        diff_df.loc[v_, 'kfold'] = f
            
    # drop the bins column
    diff_df = diff_df.drop('bins', axis=1)

    file_path = f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/forecast_train_{target}.csv'

    print('Preprocessing complete for target: ', target)
    print('Saving file to: ', file_path, '\n')
    
    diff_df.to_csv(file_path, index=False)




if __name__ == '__main__':
    
    for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
    
        forecast_data = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv')
        preprocess_forecast_train_df(forecast_data, target)