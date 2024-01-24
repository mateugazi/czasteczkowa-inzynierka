from flask import jsonify

def getAllModelArchitectures(database):
  modelArchitectures = database.modelArchitecture.find({})

  return jsonify({
    'data': list(modelArchitectures)
  }), 200
