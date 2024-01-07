import pandas as pd
import numpy as np

from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem


def CalculateMorganFingerprint(mol):
    mol = mol.apply(Chem.MolFromSmiles)
    mfpgen = AllChem.GetMorganGenerator(radius=2,fpSize=2048)
    fingerprint = np.array([mfpgen.GetFingerprintAsNumPy(x) for x in mol])
    fingerprint = pd.DataFrame(fingerprint, columns = ['mfp'+str(i) for i in range(fingerprint.shape[1])])
    return fingerprint


def CalculateDescriptors(mol):
    mol = mol.apply(Chem.MolFromSmiles)
    calc = Calculator(descriptors, ignore_3D=False)
    X_mordred = calc.pandas(mol, nproc=1)
    X_mordred = X_mordred.select_dtypes(['number'])
    #normalize
    X_mordred = (X_mordred-X_mordred.min())/(X_mordred.max()-X_mordred.min())
    #drop columns wth low std
    X_mordred = X_mordred.loc[:,X_mordred.std()>0.01]
    return X_mordred


def Pipelines(input, output, models_dict, problem, threshold):
    pass


def RunPipeline(df, modelsDict, descriptiors=["mordred", "morgan"], 
                problems=["classification"], threshold=7):
    for problem in problems:
        for descriptor in descriptiors:
            if descriptor == "mordred":
                Pipelines(CalculateDescriptors(df["SMILES"]), df["pIC50"], modelsDict, problem, threshold)
            if descriptor == "morgan":
                Pipelines(CalculateMorganFingerprint(df["SMILES"]), df["pIC50"], modelsDict, problem, threshold)
    return