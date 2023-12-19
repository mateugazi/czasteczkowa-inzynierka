from flask import Flask, jsonify, request
import pickle
from flask_cors import CORS
from pydantic import ValidationError
from redis_om import Migrator
from redis_om.model import NotFoundError
from Model import Model

from utils import *

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
  
Migrator().run()