import pandas as pd
import numpy as np
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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


def TrainGBT(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('grid', GridSearchCV(GradientBoostingClassifier(),
                                 param_grid={'n_estimators': [10, 100, 1000], 'learning_rate': [0.1,0.5,1.0,2.0]},
                                 cv=4,
                                 refit=True))
        ])
    model.fit(X_train, y_train)
    print("model score: " + str(model.score(X_test, y_test)))
    return model


def SaveModel(model):
    filename = 'model_GBT_pipeline.sav'
    pickle.dump(model, open(filename, 'wb'))


df = LoadDatasetCSV(sys.argv[1])
X_morgan = CalculateMorganFingerprint(df['mol_from_smiles'])
y = df["Class"]
model = TrainGBT(X_morgan, y)
SaveModel(model)