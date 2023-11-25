from flask import Flask, jsonify, request
import pickle
from flask_cors import CORS
from utils import *

app = Flask(__name__)
CORS(app)

model = pickle.load(open('./models/model_GBT_pipeline.sav', 'rb'))

@app.route("/")
def isUp():
  return "UP AND RUNNING"

@app.route("/get-predictions", methods=['POST'])
def getPredictions():
  csvFile = request.files['file']
  if not csvFile:
      return 'Upload a CSV file'
  
  df = LoadDatasetCSV(csvFile)
  X_morgan = CalculateMorganFingerprint(df['mol_from_smiles'])
  app.logger.info(X_morgan)

  predictions = model.predict(X_morgan).tolist()
  result = []

  for index, prediction in enumerate(predictions):
     result.append({'mol': df.iloc[index]['mol'], 'predictedClass': prediction})

  app.logger.info(result)
  return jsonify({
    'predictions': result
  })
