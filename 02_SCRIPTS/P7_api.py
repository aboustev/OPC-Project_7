"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import pandas as pd
import sys
import joblib
from flask import Flask, request, render_template
import plotly
import plotly.express as px
import json
import highcharts
import plotly.graph_objects as go

app = Flask(__name__, template_folder=r'..\07_TEMPLATES')

sys.path.insert(0, r'..\06_MODEL')
sys.path.insert(0, r'..\07_TEMPLATES')

all_data = pd.read_csv(r'..\06_MODEL\all_data.csv')
cls = joblib.load(r'..\06_MODEL\final_model.sav')
imp = joblib.load(r'..\06_MODEL\knn_inputer.sav')


@app.route('/2')
def secpage():
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=10,
        delta={
            'reference': 100,
            'decreasing': {'color': 'red'},
            'increasing': {'color': 'green'}
        },
        domain={'x': [1, 1], 'y': [0, 1]},
        title={'text': "Speed"}))
    fig2 = go.Figure(go.Indicator(
        domain={'x': [1, 1], 'y': [1, 1]},
        value=450,
        mode="gauge+number+delta",
        title={'text': "Speed"},
        delta={'reference': 380},
        gauge={'axis': {'range': [None, 500]},
               'steps': [
                   {'range': [0, 250], 'color': "lightgray"},
                   {'range': [250, 400], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 490}},
        bar={'color': 'red'}
    ))
    graph_json_1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graph_json_2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', plot1=graph_json_1, plot2=graph_json_2)


@app.route('/')
def frontpage():
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=150,
        gauge={
            'axis': {'range': [0, 300]},
            'bar': {'color': 'red'}
            },
        domain={'x': [0, 1], 'y': [0.8, 1]},
        title={'text': "Speed"}
    ))
    graph_json_1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', plot1=graph_json_1)


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
    return selected_data.to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
