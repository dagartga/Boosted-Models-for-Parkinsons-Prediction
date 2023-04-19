import pandas as pd
import numpy as np
import time
from itertools import combinations



def multiply_columns(df):
    """
    Multiply all possible combinations of columns in a Pandas dataframe together to create new features from the existing columns.
    """
    
    df_multiplied = pd.DataFrame()
    
    # Generate all combinations of column names
    col_combinations = list(combinations(df.columns, 2))
    
    # Multiply each pair of columns together and create a new column with the product
    for col1, col2 in col_combinations:
        new_col_name = col1 + '_' + col2
        df_multiplied[new_col_name] = df[col1] * df[col2]
    
    # Concatenate the original dataframe and the multiplied dataframe along the columns axis
    return pd.concat([df, df_multiplied], axis=1)


def prot_prot_interaction(df):
    
    for col in ['visit_id', 'patient_id', 'kfold']:
        if col in df.columns:
            df = df.drop(columns=col)
    
    train_prot_df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_proteins.csv')
    
    unique_prot = train_prot_df['UniProt'].unique()

    cols_to_combine = [col for col in df.columns if col in unique_prot]
    
    combined_df = multiply_columns(df[cols_to_combine])
    
    return combined_df


# def pept_prot_interaction(df, batch_size=20):
    
#     for col in ['visit_id', 'patient_id', 'kfold']:
#         if col in df.columns:
#             df = df.drop(columns=col)
            
#     train_pept_df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_peptides.csv')
#     train_prot_df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/raw/train_proteins.csv')

#     unique_pept = train_pept_df['Peptide'].unique()
#     unique_prot = train_prot_df['UniProt'].unique()
    
#     pept_dict = train_pept_df.groupby('Peptide')['UniProt'].max().to_dict()

#     combined_df = pd.DataFrame()
#     temp_df = pd.DataFrame()


#     for pept in unique_pept[:batch_size]:
#         for prot in unique_prot:
#             if pept_dict[pept] != prot:
#                 uniprot = pept_dict[pept]
#                 pept_col = f'{pept}_{uniprot}'
#                 temp_df[f'{pept}_x_{prot}'] = df[pept_col] * df[prot]
#                 combined_df = pd.concat([combined_df, temp_df], axis=1)

#     combined_df = pd.concat([df, combined_df], axis=1)
    
#     return combined_df


if __name__ == '__main__':
    
    
    df = pd.read_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1.csv')
    
    start_time = time.time()
    new_feat_df = prot_prot_interaction(df)
    
    # save the new features
    new_feat_df.to_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1_new_feats.csv')
    
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    ## This feature engineering takes too long and is memory intensive. I will have to find a way to do this in batches.
    # create the peptide-protein interaction features
    # start_time = time.time()
    # new_feat_df = pept_prot_interaction(df)
    
    # # save the new features
    # new_feat_df.to_csv(f'~/parkinsons_proj_1/parkinsons_project/parkinsons_1/data/processed/train_updrs_1_pept_prot_feats_20.csv')
    
    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")