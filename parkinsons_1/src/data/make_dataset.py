
import pandas as pd
from sklearn.model_selection import StratifiedKFold


train_clin_df = pd.read_csv('../../data/raw/train_clinical_data.csv')
train_prot_df = pd.read_csv('../../data/raw/train_proteins.csv')
train_pep_df = pd.read_csv('../../data/raw/train_peptides.csv')

test_df = pd.read_csv('../../data/raw/test.csv')
test_prot_df = pd.read_csv('../../data/raw/test_proteins.csv')
test_pep_df = pd.read_csv('../../data/raw/test_peptides.csv')

sample_submission = pd.read_csv('../../data/raw/sample_submission.csv')


def preprocess_train_df(train_clin_df: DataFrame, train_prot_df: DataFrame, train_pep_df: DataFrame) -> DataFrame:
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

    return full_train_df


def write_to_csv(df: DataFrame, filename: str) -> None:
    df.to_csv(f"../../data/interim/{filename}", index=False)
    
    
    
def stratified_kfold(df: DataFrame, target: str, n_splits: int, random_state: int) -> DataFrame:
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df[target]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_idx, 'kfold'] = fold
    return df
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    full_train_df = preprocess_train_df(train_clin_df, train_prot_df, train_pep_df)
    
    
    
    for updrs in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
        if updrs != assigned_updrs:
            full_train_df = full_train_df.drop(updrs, axis=1)
            stratified_kfold_df = stratified_kfold(full_train_df, 'target', 5, 42)
    
    write_to_csv(full_train_df, 'preprocessed_train_df.csv')

    