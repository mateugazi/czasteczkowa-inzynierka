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
  
  X_morgan = getXMorgan(csvFile)

  return jsonify({
    'predictions': model.predict(X_morgan).tolist()
  })
