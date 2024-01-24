from flask import jsonify

def getAllModels(database):
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