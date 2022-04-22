"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
import pandas as pd
import requests
from Web_func import value_from_colline, create_chart


def semi_detailed_page_front(host, df_all, df_grant, col1, col2, client_id=None):
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


