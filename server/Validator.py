import pandas as pd
from rdkit import Chem

# takes path to CSV file with SMILES and pIC50 columns 
# returns pandas dataframe and success message
# if failed returns None and failure info
def LoadDataframeFomCSV(path):
    try:
        df = pd.read_csv(path)
    except:
        return None, "Can't load csv file from: " + path
    try:
        base_df = df[["SMILES", "pIC50"]]
        return base_df, "Dataframe loaded successfully"
    except:
        return None, "Given csv file does not contain nessesary columns: SMILES and pIC50. Check your data for spelling mistakes."
    
# check if all SMILES are correct molecules
def Validate(df):
    return df['SMILES'].map(lambda x: Chem.MolFromSmiles(x) != None)

# use rdkit to convert to uniform SMILES notation which makes comparisons possible
def ToCanonicalSmiles(df):
    df['SMILES'] = df['SMILES'].map(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    return df

# remove invalid SMILES and duplicates from Dataframe
# returns Dataframe and array conststing of number of invalid, duplicate and empty rows
def ValidateSmiles(base_df):
    valid = Validate(base_df)
    invalid_rows = valid.index[valid == False].tolist()
    base_df = base_df[valid]
    ToCanonicalSmiles(base_df)
    duplicate_rows = base_df.index[base_df.duplicated('SMILES')].tolist()
    base_df = base_df.drop_duplicates('SMILES')
    na_rows = base_df.index[base_df['pIC50'].isna()].tolist()
    base_df = base_df.dropna()
    return base_df, [invalid_rows, duplicate_rows, na_rows]

# runs validator
# return Dataframe, success message and parsing stats as array
def Validator(path):
    df, mess = LoadDataframeFomCSV(path)
    if df is None:
        return df, mess, []
    else:
        df, stat = ValidateSmiles(df)
    return df, mess, stat