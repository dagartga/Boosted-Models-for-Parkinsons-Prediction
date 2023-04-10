
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


def preprocess_train_df(train_clin_df, train_prot_df, train_pep_df):
    
    # drop the medication column
    train_clin_df = train_clin_df.drop(columns=['upd23b_clinical_state_on_medication'])
    
    # create a column with the UniProt and Peptide name combined
    train_pep_df['peptide_uniprot'] = train_pep_df['Peptide'] + '_'+ train_pep_df['UniProt']

    # create a table with the visit_id as the index and the proteins or peptides as the feature and the abundance as the values
    train_prot_pivot = train_prot_df.pivot(index='visit_id', values='NPX', columns='UniProt')
    train_pep_pivot = train_pep_df.pivot(index='visit_id', values='PeptideAbundance', columns='peptide_uniprot')

    # combine the two tables on the visit_id
    full_prot_train_df = train_prot_pivot.join(train_pep_pivot)

    # fill nan with 0 for this first round
    full_prot_train_df = full_prot_train_df.fillna(0)

    full_train_df = train_clin_df.merge(full_prot_train_df, how='inner', left_on='visit_id', right_on='visit_id')
    full_train_df = full_train_df.sample(frac=1).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(full_train_df))))

    updrs = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']

    for target in updrs:
        
        to_remove = [updr for updr in updrs if updr != target]
        
        temp_train_df = full_train_df.drop(to_remove, axis=1)
        temp_train_df = temp_train_df.dropna()
        
        temp_train_df.loc[:, "bins"] = pd.cut(temp_train_df[target], bins=num_bins, labels=False)

        temp_train_df.to_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1//data/processed/train_{target}.csv', index=False)

if __name__ == '__main__':
    
    train_clin_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_clinical_data.csv')
    train_prot_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_proteins.csv')
    train_pep_df = pd.read_csv('~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_peptides.csv')
    
    preprocess_train_df(train_clin_df, train_prot_df, train_pep_df)
    