"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
import pandas as pd
import requests
from Web_func import value_from_colline, create_chart

st.set_page_config(
    page_title="Bank loans",
    page_icon="(-.-)",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
host = 'http://127.0.0.1:5000'

try:
    response = requests.get("{}/load_data".format(host))
except requests.exceptions.ConnectionError:
    host = 'https://share.streamlit.io/aboustev/opc-project_7/main/08_WEBSITE_AND_MODELS/flask_back.py'
    response = requests.get("{}/load_data".format(host))

granted_described, all_described = response.json()
granted_described = pd.read_json(granted_described)
all_described = pd.read_json(all_described)
columns = all_described.columns

st.header('Loan visualization')
with st.sidebar:
    number = st.number_input('Client ID',
                             value=int(all_described.loc['min', 'SK_ID_CURR']),
                             min_value=int(all_described.loc['min', 'SK_ID_CURR']),
                             max_value=int(all_described.loc['max', 'SK_ID_CURR']),
                             step=1)
    response = requests.get("{}/api/getdecision/?id={}".format(host, int(number)))
    client_info = pd.DataFrame.from_dict(response.json())                        

col1, col2 = st.columns([20, 20])
columns = [col for col in columns if col not in ['SK_ID_CURR', 'TARGET']]
num = 0
for col in columns:
    if col == 'AMT_CREDIT':
        color_list = ['red', 'orange', '#4ee44e', 'yellow', '#FFCC00', 'red']
    else:
        color_list = ['red', 'orange', '#FFCC00', 'yellow', '#4ee44e', 'yellow']
    chart = create_chart(all_described,
                         granted_described,
                         color_list,
                         col)
    chart.update_traces(value=client_info.loc['0', col])
    if num % 2 == 0:
        with col1:
            st.plotly_chart(chart, use_container_width=True)
    else:
        with col2:
            st.plotly_chart(chart, use_container_width=True)
    num += 1
