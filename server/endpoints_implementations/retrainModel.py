from flask import jsonify
import pickle
from ml_actions.Finalized_pipeline import generate_split_dataset, calculate_features, retrain_model

def retrainModel(database, request):
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

  if not foundModel:
     return 'Model not found', 500

  model = pickle.loads(foundModel['pickleData'])

  df = generate_split_dataset(csvFile)
  df = calculate_features(df, False, True, SMILES_column_name='mol', target_column_name='Class')
  resultDict, retrainedModel = retrain_model(model, df)

  modelToSave = {
    'name': foundModel['name'],
    'description': foundModel['description'],
    'architecture': foundModel['architecture'],
    'pickleData': pickle.dumps(retrainedModel)
  }

  content={ "$set": modelToSave }

  insertResults = database.model.update_one(query, content)

  if not insertResults.matched_count:
    return jsonify({
      "message" : "Something went wrong, while saving model"
    }), 500
  

  result = []
  for key, value in resultDict.items():
     result.append([key, value])

  return jsonify({'message': 'OK', 'data': result})
