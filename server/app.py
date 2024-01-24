from flask import Flask, jsonify, request
import pickle
import pandas as pd
from flask_cors import CORS
import json
from Finalized_pipeline import generate_split_dataset, calculate_features, hyperparameter_search, make_prediction, retrain_model
import ast
import pymongo
from uuid import uuid1


app = Flask(__name__)
client = pymongo.MongoClient('localhost:27017')
database = client.TaskManager

CORS(app)

@app.route("/")
def isUp():
  return "UP AND RUNNING"
  

@app.route('/model', methods=['GET'])
def getAllModels():
  models = database.model.find({})

  tempModels = []
  for model in models:
     tempModels.append({
        '_id': model['_id'],
        'name': model['name'],
        'description': model['description'],
        'architecture': model['architecture'],
     })

  return jsonify({
    'data': tempModels
  }), 200
  

@app.route('/create-model-architecture', methods=['POST'])
def createModelArchitecture():
  modelArchitectureData = dict(request.json)
  modelArchitectureData['_id'] = str(uuid1().hex) 

  result = database.modelArchitecture.insert_one(modelArchitectureData)

  if not result.inserted_id:
    return jsonify({
      "message" : "Something went wrong, while creating the model architecutre"
    }), 500
  
  return jsonify({
    "message": "Model architecture added", 
    "data": {
      "id":result.inserted_id
    }
  }), 200


@app.route('/model-architecture', methods=['GET'])
def getAllModelTypes():
  modelArchitectures = database.modelArchitecture.find({})

  return jsonify({
    'data': list(modelArchitectures)
  }), 200


@app.route("/trigger-training", methods=['POST'])
def triggerTraining():
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


  print(hyperparameters)

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

  print(result)
  return jsonify({'message': 'OK', 'data': result})



@app.route("/retrain-model", methods=['POST'])
def retrainModel():
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
  result.append(list(resultDict.keys()))
  result.append(list(resultDict.values()))

  print(result)

  print(result)
  return jsonify({'message': 'OK', 'data': result})


@app.route("/get-predictions", methods=['POST'])
def getPredictions():
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
  # print(len(dataDf.index), len(resultDf.index))

  for index, prediction in enumerate(predictions):
     result.append({'mol': dataDf.iloc[index]['mol'], 'predictedClass': int(prediction)})

  return jsonify({
    'data': result
  }), 200
