import pandas as pd
import pickle




def preprocess_test_df(test_clin_df, test_prot_df, test_pep_df, save_data=False):
    
    if 'upd23b_clinical_state_on_medication' in test_clin_df.columns:
        # drop the medication column
        test_clin_df = test_clin_df.drop(columns=['upd23b_clinical_state_on_medication'])
    
    # create a column with the UniProt and Peptide name combined
    test_pep_df['peptide_uniprot'] = test_pep_df['Peptide'] + '_'+ test_pep_df['UniProt']

    # create a table with the visit_id as the index and the proteins or peptides as the feature and the abundance as the values
    train_prot_pivot = test_prot_df.pivot(index='visit_id', values='NPX', columns='UniProt')
    train_pep_pivot = test_pep_df.pivot(index='visit_id', values='PeptideAbundance', columns='peptide_uniprot')

    # combine the two tables on the visit_id
    full_prot_train_df = train_prot_pivot.join(train_pep_pivot)

    # fill nan with 0 
    full_prot_train_df = full_prot_train_df.fillna(0)

    full_train_df = test_clin_df.merge(full_prot_train_df, how='inner', left_on='visit_id', right_on='visit_id')
    full_train_df = full_train_df.sample(frac=1).reset_index(drop=True)

    
    return full_train_df



test_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test.csv')


prot_test_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test_proteins.csv')

pep_test_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/test_peptides.csv')



full_test_df = preprocess_test_df(test_df, prot_test_df, pep_test_df, save_data=False)

updr = 'updrs_1'
month = 0

for updr in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
    updr_df = full_test_df[full_test_df['updrs_test'] == updr]
    info_cols = ['visit_id', 'visit_month', 'patient_id', 'updrs_test', 'row_id', 'group_key']
    updr_info = updr_df[info_cols]
    model_df = updr_df.drop(columns=info_cols)
    
    for month in [0, 6, 12, 24]:
        # Load the saved model from file
        model_path = f'..\models\model_rf_reg_updrs_1_0.pkl'
        
        with open(model_path, 'rb') as f:
            rf_reg = pickle.load(f)

        # Use the imported model to make predictions
        y_pred = rf_reg.predict(model_df)
        

    

target = 'updrs_4'
train_df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_{target}.csv')
train_df.head()