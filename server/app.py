from flask import Flask, jsonify, request
import pickle
from flask_cors import CORS
from pydantic import ValidationError
from redis_om import Migrator
from redis_om.model import NotFoundError
from Model import Model
from ModelType import ModelType
from utils import *
import json
from Validator import Validator
from Finalized_pipeline import generate_split_dataset, calculate_features, hyperparameter_search
import ast

app = Flask(__name__)
CORS(app)
# r = redis.StrictRedis(host='localhost', port=6379, db=0)

@app.route("/")
def isUp():
  return "UP AND RUNNING"

@app.route("/get-predictions", methods=['POST'])
def getPredictions():
  csvFile = request.files['file']
  uniqueName = request.form['uniqueName']

  if not csvFile:
      return 'Upload a CSV file'
  
  if not uniqueName:
      return 'No model'
  
  model = pickle.load(open('./models/' + uniqueName  + '.sav', 'rb'))
  
  df = LoadDatasetCSV(csvFile)
  X_morgan = CalculateMorganFingerprint(df['mol_from_smiles'])

  predictions = model.predict(X_morgan).tolist()
  result = []

  for index, prediction in enumerate(predictions):
     result.append({'mol': df.iloc[index]['mol'], 'predictedClass': prediction})

  return jsonify({
    'predictions': result
  })


@app.route("/create-model", methods=['POST'])
def createModel():
  try:
    newModel = Model(
      uniqueName = 'model_GBT_pipeline',
      name = 'Basic model',
      description = 'Basic model for our initial tests',
    )
    print('======PK:', newModel.pk)
    newModel.save()
    return newModel.pk

  except ValidationError as e:
    print(e)
    return "Bad request.", 400
  
@app.route('/model', methods=['GET'])
def getAllModels():
  models = Model.find().all()
  response = []

  for model in models:
    response.append({
      'name': model.name,
      'description': model.description,
      'pk': model.pk
    })

  return jsonify(response)

@app.route('/model/byid/<id>', methods=['GET'])
def getModelById(id):
  try:
    model = Model.get(id)
    return jsonify({
      'name': model.name,
      'description': model.description,
      'pk': model.pk
    })
  except NotFoundError:
    return {}
  
@app.route('/create-model-type', methods=['POST'])
def createModelType():
  data = request.json
  name = data.get('name')
  identifier = data.get('identifier')
  regression = data.get('regression')
  parameters = data.get('parameters')

  try:
    newModelType = ModelType(
      name = name,
      identifier = identifier,
      regression = regression,
      parameters = parameters
    )
    print('======PK:', newModelType.pk)
    newModelType.save()
    return newModelType.pk

  except ValidationError as e:
    print(e)
    return "Bad request.", 400


@app.route('/model-type', methods=['GET'])
def getAllModelTypes():
  modelTypes = ModelType.find().all()
  response = []

  for modelType in modelTypes:
    response.append({
      'name': modelType.name,
      'identifier': modelType.identifier,
      'regression': modelType.regression,
      'parameters': [{'name': parameter.name, 'example': parameter.example, 'type': parameter.type} for parameter in modelType.parameters],
      'pk': modelType.pk
    })

  return jsonify(response)


@app.route("/trigger-training", methods=['POST'])
def triggerTraining():
  csvFile = request.files['file']
  modelInfo = request.form['modelInfo']
  parameters = request.form['parameters']

  if not csvFile:
    return 'Upload a CSV file'
  
  if not modelInfo:
    return 'No model type'

  # df, mess, stat = Validator(csvFile)

  # if df is None:
    # return jsonify({'message': mess, 'stat': stat})
  
  modelInfo = json.loads(modelInfo)
  df = generate_split_dataset(csvFile)
  df = calculate_features(df, False, True, SMILES_column_name='mol', target_column_name='Class')

  hyperparameters = {
    modelInfo['identifier']: {}
  }

  # currently hyperparameters are ignored, because they are taking too long time
  if parameters:
    parameters = json.loads(request.form['parameters'])
    for parameterName in parameters.keys():
      hyperparameters[modelInfo['identifier']][parameterName] = ast.literal_eval(parameters[parameterName]['value'])


  print(hyperparameters)

  df = hyperparameter_search(df, hyperparameters)

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

Migrator().run()
