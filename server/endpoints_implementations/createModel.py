from flask import jsonify
from uuid import uuid1
import json
import ast
import pickle
from ml_actions.Finalized_pipeline import generate_split_dataset, calculate_features,hyperparameter_search

def createModel(database, request):
  csvFile = request.files['dataFile']
  modelArchitecture = request.form['modelArchitecture']
  name = request.form['name']
  description = request.form['description']
  parameters = request.form['parameters']

  if not csvFile:
    return 'Upload a CSV file', 500
  
  if not modelArchitecture:
    return 'No model type', 500

  modelArchitecture = json.loads(modelArchitecture)
  df = generate_split_dataset(csvFile)
  df = calculate_features(df, False, True, SMILES_column_name='mol', target_column_name='Class')

  hyperparameters = {
    modelArchitecture['mlIdentifier']: {}
  }

  if parameters:
    parameters = json.loads(request.form['parameters'])
    for parameterName in parameters.keys():
      hyperparameters[modelArchitecture['mlIdentifier']][parameterName] = ast.literal_eval(parameters[parameterName]['value'])

  df, bestModel = hyperparameter_search(df, hyperparameters)
  model = {
    '_id': str(uuid1().hex),
    'name': name,
    'description': description,
    'architecture': modelArchitecture,
    'pickleData': pickle.dumps(bestModel)
  }
  result = database.model.insert_one(model)

  if not result.inserted_id:
    return jsonify({
      "message" : "Something went wrong, while saving model"
    }), 500
  # sprawdzenie czy jest regresja
    # klasyfikacja
      # (optional) sprawdzenie ic50 czy pic50
      # (optional) dodanie labels do klasyfikacji
      # liczenie featerów
      # hyperparameter searc
    # regresja
      # sprawdzenie ic50 czy pic50
      # liczenie featerów
  result = [list(df.columns.values)]

  for index, _ in df.iterrows():
     result.append(df.loc[index, :].values.flatten().tolist())

  flippedResult = []
  for columnIndex in range(len(result[:2][0])):
    flippedResult.append([result[0][columnIndex], result[1][columnIndex]])

  return jsonify({'message': 'OK', 'data': flippedResult, 'withModelInfo': True})
