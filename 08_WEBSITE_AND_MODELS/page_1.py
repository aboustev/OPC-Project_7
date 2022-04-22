"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
import pandas as pd
import requests
from Web_func import value_from_colline, create_chart

def detailed_page_front(host, granted_described, all_described, columns, client_id=None):
    with st.sidebar:
        if client_id:
            id_default = client_id
        else:
            id_default = int(all_described.loc['min', 'SK_ID_CURR'])
        number = st.number_input('Client ID',
                                 value=id_default,
                                 min_value=int(all_described.loc['min', 'SK_ID_CURR']),
                                 max_value=int(all_described.loc['max', 'SK_ID_CURR']),
                                 step=1)
        if number:
            response = requests.get("{}/api/getdecision/?id={}".format(host, int(number)))
        else:
            val = int(all_described.loc['min', 'SK_ID_CURR'])
            response = requests.get("{}/api/getdecision/?id={}".format(host, val))
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
    return id_default
