from flask import Flask, request
from flask import send_file
from flask_cors import CORS
import pymongo
from endpoints_implementations.getAllModels import getAllModels
from endpoints_implementations.createModelArchitecture import createModelArchitecture
from endpoints_implementations.getAllModelArchitectures import getAllModelArchitectures
from endpoints_implementations.createModel import createModel
from endpoints_implementations.retrainModel import retrainModel
from endpoints_implementations.getPredictions import getPredictions

app = Flask(__name__)
client = pymongo.MongoClient('localhost:27017')
database = client.TaskManager

CORS(app)

@app.route("/")
def isUp():
  return "UP AND RUNNING"
  

@app.route('/model', methods=['GET'])
def getAllModelsController():
   return getAllModels(database)
  

@app.route('/create-model-architecture', methods=['POST'])
def createModelArchitectureController():
  return createModelArchitecture(database, dict(request.json))


@app.route('/model-architecture', methods=['GET'])
def getAllModelArchitecturesController():
   return getAllModelArchitectures(database)


@app.route("/create-model", methods=['POST'])
def createModelController():
  return createModel(database, request)


@app.route("/retrain-model", methods=['POST'])
def retrainModelController():
  return retrainModel(database, request)


@app.route("/get-predictions", methods=['POST'])
def getPredictionsController():
  return getPredictions(database, request)

@app.route("/explainability-plot", methods=['GET'])
def getImage():
  return send_file('explainability_plot.png')
