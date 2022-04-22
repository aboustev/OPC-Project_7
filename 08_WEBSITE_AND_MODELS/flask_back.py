"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import pandas as pd
import sys
import joblib
import json
from flask import Flask, request, jsonify

app = Flask(__name__, template_folder=r'..\07_TEMPLATES')

sys.path.insert(0, r'..\06_MODEL')
sys.path.insert(0, r'..\07_TEMPLATES')

try:
    all_data = pd.read_csv(r'..\06_MODEL\all_data.csv')
except FileNotFoundError:
    all_data = pd.read_csv('https://github.com/aboustev/OPC-Project_7/blob/main/06_MODEL/all_data.csv?raw=true')

cls = joblib.load('final_model.sav')
imp = joblib.load('knn_inputer.sav')



@app.route('/load_data')
def frontpage():
    data_granted_loan = all_data[all_data.TARGET.values == 1]
    columns = [col for col in all_data.columns if col != "TARGET"]
    granted_described = data_granted_loan[columns].describe()
    all_described = all_data[columns].describe()
    jsonified_list = jsonify([granted_described.to_json(
        orient='columns',
        index=True
    ), all_described.to_json(
        orient='columns',
        index=True
    )])
    return jsonified_list


@app.route('/api/getdecision/', methods=['GET'])
def decision():
    args = request.args
    id_client = int(args.get('id'))
    selected_data = all_data.loc[all_data['SK_ID_CURR'] == id_client]
    if len(selected_data) == 0:
        return 'No data found'
    else:
        if str(selected_data.TARGET[selected_data.index[0]]) == 'nan':
            top_feat = [col for col in all_data.columns.tolist() if col not in ['SK_ID_CURR', 'TARGET']]
            cols_without_id = selected_data.columns[1:]
            data_fitted = imp.transform(selected_data[cols_without_id])
            data_fitted = pd.DataFrame(data_fitted, columns=cols_without_id)
            prediction = cls.predict(data_fitted[top_feat])
            selected_data.loc[selected_data.index[0], 'TARGET'] = prediction
    return selected_data.reset_index(drop=True).to_json(orient='columns')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
