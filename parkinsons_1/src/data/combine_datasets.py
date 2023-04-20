import pandas as pd

def combine_datasets(target):
     # read the training data with folds
    df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv')
    # read in the training data with new features
    df2 = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_data_new_feats.csv')
    
    # get the columns to keep from each df
    df_cols = ['visit_id', 'patient_id', 'kfold', 'visit_month', 'updrs_1']
    df2 = df2.drop(columns = ['patient_id', 'kfold'])
    # merge the data together with the necessary columns
    df = df[train_cols].merge(df2, how='left', left_on='visit_id', right_on='visit_id')
    
    df.to_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_new_feats_{target}.csv', index=False)
    
    
    
if __name__ == '__main__':
    
    for target in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
        combine_datasets(target)