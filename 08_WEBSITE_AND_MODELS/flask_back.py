"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import pandas as pd
import numpy as np
import sys
import joblib
import gzip
from flask import Flask, request

app = Flask(__name__)

sys.path.insert(0, r'..\06_MODEL')
sys.path.insert(0, r'..\07_TEMPLATES')


@app.route('/')
def main_page():
    return '<h1>Main page of Flask app</h1>'


@app.route('/api/load_gd_data/')
def data_loading_1():
    try:
        all_data = pd.read_csv(r'..\06_MODEL\all_data.csv')
    except FileNotFoundError:
        all_data = pd.read_csv('https://github.com/aboustev/OPC-Project_7/blob/main/06_MODEL/all_data.csv?raw=true')

    data_granted_loan = all_data[all_data.TARGET.values == 1]
    columns = [col for col in all_data.columns if col != "TARGET"]
    granted_described = data_granted_loan[columns].describe()

    gd_json = granted_described.to_json(
        orient='columns',
        index=True
    )

    return gd_json


@app.route('/api/load_ad_data/')
def data_loading_2():
    try:
        all_data = pd.read_csv(r'..\06_MODEL\all_data.csv')
    except FileNotFoundError:
        all_data = pd.read_csv('https://github.com/aboustev/OPC-Project_7/blob/main/06_MODEL/all_data.csv?raw=true')

    columns = [col for col in all_data.columns if col != "TARGET"]
    all_described = all_data[columns].describe()

    ad_json = all_described.to_json(
        orient='columns',
        index=True
    )
    return ad_json


@app.route('/api/load_all_data/')
def data_loading_3():
    try:
        all_data = pd.read_csv(r'..\06_MODEL\all_data.csv')
    except FileNotFoundError:
        all_data = pd.read_csv('https://github.com/aboustev/OPC-Project_7/blob/main/06_MODEL/all_data.csv?raw=true')

    all_data = all_data[all_data['TARGET'].notna()].reset_index(drop=True)
    all_data_json = all_data.to_json(
        orient='columns',
        index=True
    )
    compressed = gzip.compress(all_data_json.encode('utf-8'))
    return compressed


@app.route('/api/getdecision/', methods=['GET'])
def decision():
    try:
        all_data = pd.read_csv(r'..\06_MODEL\all_data.csv')
    except FileNotFoundError:
        all_data = pd.read_csv('https://github.com/aboustev/OPC-Project_7/blob/main/06_MODEL/all_data.csv?raw=true')

    cls = joblib.load('final_model.sav')
    imp = joblib.load('knn_inputer.sav')

    args = request.args
    id_client = int(args.get('id'))
    all_data['TARGET_PROBA'] = np.nan
    selected_data = all_data.loc[all_data['SK_ID_CURR'] == id_client].reset_index(drop=True)

    if len(selected_data) == 0:
        return 'No data found'
    else:
        if str(selected_data.TARGET[selected_data.index[0]]) == 'nan':
            top_feat = [col for col in all_data.columns if col not in ['SK_ID_CURR', 'TARGET', 'TARGET_PROBA']]
            data_fitted = imp.transform(selected_data[imp.feature_names_in_])
            data_fitted = pd.DataFrame(data_fitted, columns=imp.feature_names_in_)
            proba_pred = cls.predict_proba(data_fitted[top_feat])
            if proba_pred[0][0] >= 0.75:
                prediction = 0
            else:
                prediction = 1
            selected_data.loc[0, 'TARGET'] = prediction
            selected_data.loc[0, 'TARGET_PROBA'] = proba_pred[0][0]
    return selected_data.to_json(orient='columns')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
