import pandas as pd
import pickle
from flask import jsonify
from ml_actions.Finalized_pipeline import make_prediction

def getPredictions(database, request):
  csvFile = request.files['file']
  _id = request.form['_id']

  if not csvFile:
      return 'Upload a CSV file', 500
  
  if not _id:
      return 'No _id', 500
  
  query={
    "_id": _id
  }
  foundModel = database.model.find_one(query)
  model = pickle.loads(foundModel['pickleData'])

  dataDf = pd.read_csv(csvFile)
  predictions = make_prediction(
    model, dataDf, False, True, SMILES_column_name='mol'
  )
  result = []

  for index, prediction in enumerate(predictions):
     result.append({'mol': dataDf.iloc[index]['mol'], 'predictedClass': int(prediction)})

  return jsonify({
    'data': result
  }), 200