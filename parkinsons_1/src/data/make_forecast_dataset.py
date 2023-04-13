import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np



def preprocess_forecast_train_df(forecast_data, target):
    
    temp_df = forecast_data[['visit_id', 'patient_id', target, 'visit_month']].sort_values(by=['patient_id', 'visit_month']).reset_index(drop=True)
    temp_pivot = temp_df.pivot(columns='visit_month', values=target, index='patient_id')
    temp_pivot = temp_pivot.reset_index()
    
    cols = [f'{target}_{month}' for month in temp_pivot.columns[1:]]
    temp_pivot.columns = ['patient_id'] + cols
    
    forecast_final = forecast_data[forecast_data['visit_month'] == 0]
    forecast_final.head()
    
    final_df = forecast_final.merge(temp_pivot, on=['patient_id'], how='left')
    
    final_df = final_df.drop(columns=['patient_id', target])
    if 'kfold' in final_df.columns:
        final_df = final_df.drop(columns=['kfold'])
        
    
    file_path = f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/forecast_train_{target}.csv'

    print('Preprocessing complete for target: ', target)
    print('Saving file to: ', file_path, '\n')
    
    final_df.to_csv(file_path, index=False)





if __name__ == '__main__':
    
    for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
    
        forecast_data = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv')
        preprocess_forecast_train_df(forecast_data, target)
        