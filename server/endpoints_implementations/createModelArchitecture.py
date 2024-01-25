from flask import jsonify
from uuid import uuid1

def createModelArchitecture(database, request):
  modelArchitectureData = request
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