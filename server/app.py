from flask import Flask, jsonify, request
import pickle
import csv
import codecs
from flask_cors import CORS
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors

app = Flask(__name__)
CORS(app)

model = pickle.load(open('./models/model_GBT_pipeline.sav', 'rb'))
print("loaded OK")

@app.after_request
def apply_caching(response):
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/get-predictions", methods=['POST'])
def getPredictions():
    data = request.get_json()
            
    return jsonify(data)

@app.route("/upload-csv", methods=['POST'])
def uploadCsv():
    flask_file = request.files['file']
    if not flask_file:
        return 'Upload a CSV file'
    df = LoadDatasetCSV(flask_file)
    X_morgan = CalculateMorganFingerprint(df['mol_from_smiles'])
    y = df["Class"]
    app.logger.info(model.score(X_morgan, y))
    return "OK"
    # return jsonify(data)


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