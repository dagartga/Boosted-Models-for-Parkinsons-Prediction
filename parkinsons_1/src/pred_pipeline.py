import pandas as pd
import pickle

import warnings
warnings.filterwarnings('ignore')



def preprocess_test_df(test_clin_df, test_prot_df, test_pep_df, save_data=False):
    
    if 'upd23b_clinical_state_on_medication' in test_clin_df.columns:
        # drop the medication column
        test_clin_df = test_clin_df.drop(columns=['upd23b_clinical_state_on_medication'])
    
    # create a column with the UniProt and Peptide name combined
    test_pep_df['peptide_uniprot'] = test_pep_df['Peptide'] + '_'+ test_pep_df['UniProt']

    # create a table with the visit_id as the index and the proteins or peptides as the feature and the abundance as the values
    test_prot_pivot = test_prot_df.pivot(index='visit_id', values='NPX', columns='UniProt')
    test_pep_pivot = test_pep_df.pivot(index='visit_id', values='PeptideAbundance', columns='peptide_uniprot')

    # combine the two tables on the visit_id
    full_prot_test_df = test_prot_pivot.join(test_pep_pivot)

    # fill nan with 0 
    full_prot_test_df = full_prot_test_df.fillna(0)

    full_test_df = test_clin_df.merge(full_prot_test_df, how='inner', left_on='visit_id', right_on='visit_id')
    full_test_df = full_test_df.sample(frac=1).reset_index(drop=True)

    
    return full_test_df



def prepare_model_df(model_df, target, visit_month=0):
    
    train_df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv')
    pred_cols = [col for col in train_df.columns if col not in ['visit_id', 'patient_id', target, 'kfold']]

    # add visit_month if it is not in the model_df.columns
    if 'visit_month' not in model_df.columns:
        model_df['visit_month'] = visit_month
    
    # find the columns in preds_cols that are not in the model_df.columns
    not_in_pred_cols = [col for col in pred_cols if col not in model_df.columns]

    # create an empty dataframe with the columns in not_in_pred_cols
    not_in_preds_df = pd.DataFrame(columns=not_in_pred_cols)

    # combine the model_df and the not_in_preds_df so all the needed columns are in dataframe
    new_model_df = pd.concat([model_df, not_in_preds_df], axis=1)
    
    # fill the nan values with 0
    new_model_df = new_model_df.fillna(0)

    # filter the new_model_df to only include the columns in pred_cols with the correct order
    return new_model_df[pred_cols]






def create_submission_df(test_df, prot_test_df, pep_test_df, save_data=False):
    
    full_test_df = preprocess_test_df(test_df, prot_test_df, pep_test_df, save_data=False)

    final_pred_df = pd.DataFrame()

    for updr in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
        updr_df = full_test_df[full_test_df['updrs_test'] == updr]
        info_cols = ['visit_id', 'visit_month', 'patient_id', 'updrs_test', 'row_id', 'group_key']
        updr_info = updr_df[info_cols]
        model_df = updr_df.drop(columns=info_cols)
        

        for month in [0, 6, 12, 24]:
            
            # temporary fix for missing model
            if updr == 'updrs_4' and month == 6:
                print('----------RUNNING TEMPORARY FIX FOR MISSING MODEL----------')
                # prepare the model_df for the correct month
                model_df = prepare_model_df(model_df, updr, visit_month=0)
            
                # Load the saved model from file
                model_path = f'..\models\model_rf_reg_updrs_4_0.pkl'
                
                
                with open(model_path, 'rb') as f:
                    rf_reg = pickle.load(f)

                # Use the imported model to make predictions
                y_pred_1 = rf_reg.predict(model_df.values)
                
        
                # prepare the model_df for the correct month
                model_df = prepare_model_df(model_df, updr, visit_month=12)
                # Load the saved model from file
                model_path = f'..\models\model_rf_reg_updrs_4_12.pkl'
                
                
                with open(model_path, 'rb') as f:
                    rf_reg = pickle.load(f)

                # Use the imported model to make predictions
                y_pred_2 = rf_reg.predict(model_df.values)
                
                y_pred = (y_pred_1 + y_pred_2) / len(y_pred_1)
                
                
                updr_info[f'plus_{month}_months'] = y_pred
                
                
            else:
                # prepare the model_df for the correct month
                model_df = prepare_model_df(model_df, updr, visit_month=month)
            
                # Load the saved model from file
                model_path = f'..\models\model_rf_reg_{updr}_{month}.pkl'
                
                
                with open(model_path, 'rb') as f:
                    rf_reg = pickle.load(f)

                # Use the imported model to make predictions
                y_pred = rf_reg.predict(model_df.values)
                
                updr_info[f'plus_{month}_months'] = y_pred
        
        
        final_pred_df = pd.concat([final_pred_df, updr_info])
        
    submit_df = pd.DataFrame(columns=['prediction_id', 'rating', 'group_key'])

    for i, row in final_pred_df.iterrows():
        for col in ['plus_0_months', 'plus_6_months', 'plus_12_months', 'plus_24_months']:
            submit_df = submit_df.append({'prediction_id': row['row_id']+'_'+col, 'rating': row[col], 'group_key': row['group_key']}, ignore_index=True)

    if save_data:
        submit_df.to_csv(f'../data/submission.csv', index=False)


    
    

if __name__ == '__main__':
    
    test_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test.csv')
    prot_test_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test_proteins.csv')
    pep_test_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test_peptides.csv')

    
    create_submission_df(test_df, prot_test_df, pep_test_df, save_data=True)