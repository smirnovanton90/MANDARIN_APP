import functions
import dictionaries
import json
import pickle
from flask import jsonify

with open('test.json', 'r', encoding='utf-8') as f:
    json_ = json.load(f)

#print(json_)
query_df = functions.data_transform(dictionaries.columns, json_)


model_A = pickle.load(open("model_A.pkl", "rb"))
prediction = model_A.predict(query_df)[0]

model_A = pickle.load(open("model_A.pkl", "rb"))
model_B = pickle.load(open("model_B.pkl", "rb"))
model_C = pickle.load(open("model_C.pkl", "rb"))
model_D = pickle.load(open("model_D.pkl", "rb"))
model_E = pickle.load(open("model_E.pkl", "rb"))

models = {'model_A': model_A,
          'model_B': model_B,
          'model_C': model_C,
          'model_D': model_D,
          'model_E': model_E}

result_list = {}

for key in models:
    result_list[key] = functions.predictor(models[key], query_df)

print(result_list)

#return jsonify({'prediction': result_list})