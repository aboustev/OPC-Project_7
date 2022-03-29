"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import pandas as pd
import sys
import joblib
from flask import Flask, request

app = Flask(__name__)

sys.path.insert(0, r'..\06_MODEL')
from parameters import PredictParams


@app.route('/api/predictfromdata/', methods=['GET'])
def predict():
    data = request.args
    top_feat = PredictParams.topfeat
    if type(data) == dict:
        df = pd.DataFrame.from_dict(data)[top_feat]
    elif type(data) == list:
        df = pd.DataFrame.from_dict(data)[top_feat]
    elif type(data) == pd.DataFrame:
        df = data[top_feat]
    else:
        print('Wrong input type, please make sure the data you input is either dict, list or pandas dataframe')
        return

    cls = joblib.load(r'..\06_MODEL\final_model.sav')
    prediction = cls.predict(df)
    return prediction


@app.route('/api/bjr/', methods=['GET'])
def hello():
    args = request.args
    return args


@app.route('/api/getdecision/', methods=['GET'])
def decision():
    args = request.args
    id = args.get('id')

    return args


if __name__ == '__main__':
    app.run(debug=True)
