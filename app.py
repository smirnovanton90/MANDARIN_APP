import pickle

from flask import Flask, request, jsonify

import dictionaries
import functions
import sklearn

app = Flask(__name__)

model_A = pickle.load(open("model_A.pkl", "rb"))
model_B = pickle.load(open("model_B.pkl", "rb"))
model_C = pickle.load(open("model_C.pkl", "rb"))
model_D = pickle.load(open("model_D.pkl", "rb"))
model_E = pickle.load(open("model_E.pkl", "rb"))

models = {'bank_A_prob': model_A,
          'bank_B_prob': model_B,
          'bank_C_prob': model_C,
          'bank_D_prob': model_D,
          'bank_E_prob': model_E}

@app.route("/predictAll", methods = ["POST"])
def predictAll():
    json_ = request.json
    query_df = functions.data_transform(dictionaries.columns, json_)

    result_list = {}

    for key in models:
        result_list[key] = functions.predictor(models[key], query_df)

    return jsonify(result_list)

@app.route("/predictA", methods = ["POST"])
def predictA():
    json_ = request.json
    query_df = functions.data_transform(dictionaries.columns, json_)

    return jsonify({'bank_A_prob': functions.predictor(model_A, query_df)})

@app.route("/predictB", methods = ["POST"])
def predictB():
    json_ = request.json
    query_df = functions.data_transform(dictionaries.columns, json_)

    return jsonify({'bank_B_prob': functions.predictor(model_B, query_df)})

@app.route("/predictC", methods = ["POST"])
def predictC():
    json_ = request.json
    query_df = functions.data_transform(dictionaries.columns, json_)

    return jsonify({'bank_C_prob': functions.predictor(model_C, query_df)})

@app.route("/predictD", methods = ["POST"])
def predictD():
    json_ = request.json
    query_df = functions.data_transform(dictionaries.columns, json_)

    return jsonify({'bank_D_prob': functions.predictor(model_D, query_df)})

@app.route("/predictE", methods = ["POST"])
def predictE():
    json_ = request.json
    query_df = functions.data_transform(dictionaries.columns, json_)

    return jsonify({'bank_E_prob': functions.predictor(model_E, query_df)})



if __name__ == "__main__":
    app.run(debug=True)
