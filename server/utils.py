import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors

def LoadDatasetCSV(path):
    base_data = pd.read_csv(path)
    # TODO: change column name mol to smiles
    # TODO: add custom threshold and calculate Class from pIC50
    input_df = base_data[["mol", 'Class']]
    input_df = input_df.drop_duplicates(subset=['mol'])
    input_df = input_df.dropna()
    input_df['mol_from_smiles'] = input_df['mol'].apply(Chem.MolFromSmiles)
    return input_df

def CalculateMorganFingerprint(mol):
    mfpgen = AllChem.GetMorganGenerator(radius=2,fpSize=2048)
    fingerprint = np.array([mfpgen.GetFingerprintAsNumPy(x) for x in mol])
    fingerprint = pd.DataFrame(fingerprint, columns = ['mfp'+str(i) for i in range(fingerprint.shape[1])])
    return fingerprint

def getXMorgan(file):
    df = LoadDatasetCSV(file)
    return CalculateMorganFingerprint(df['mol_from_smiles'])
