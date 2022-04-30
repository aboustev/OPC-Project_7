"""
The purpose of this python file is to regroup every functions from the external kernel
"""
from json import JSONDecodeError

import streamlit as st
import pandas as pd
import requests
from streamlit_option_menu import option_menu
from page_0 import loan_status_page_front
from page_1 import detailed_page_front
from page_2 import semi_detailed_page_front
from page_3 import custom_simu_page_front

st.set_page_config(
    page_title="Bank loans",
    page_icon="(-.-)",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    host = "54.205.25.37:8080"
    response = requests.get("{}/load_data".format(host))
except Exception as e:
    st.write(e)
    host_txt = st.text_input('Host', value='127.0.0.1')
    port = st.number_input('Port', value=5000)
    host = "{}:{}".format(host_txt, port)
    st.write(host)
    try:
        response = requests.get("{}/load_data".format(host))
    except Exception as e:
        st.write(e)
        response = None

if response:
    df_grant_desc, df_all_desc, all_data = response.json()
    df_grant_desc = pd.read_json(df_grant_desc)
    df_all_desc = pd.read_json(df_all_desc)
    all_data = pd.read_json(all_data)
    all_columns = df_all_desc.columns

    selected = option_menu(None, ["Client loan status", "Detailed", "Semi-detailed", 'Custom simulation'],
                           icons=['bank2', "briefcase-fill", 'briefcase', 'gear'], menu_icon="cast",
                           orientation="horizontal")

    with st.sidebar:
        client_id = int(df_all_desc.loc['min', 'SK_ID_CURR'])
        client_id = st.number_input('Client ID',
                                    value=client_id,
                                    min_value=int(df_all_desc.loc['min', 'SK_ID_CURR']),
                                    max_value=int(df_all_desc.loc['max', 'SK_ID_CURR']),
                                    step=1)
    if client_id:
        response = requests.get("{}/api/getdecision/?id={}".format(host, client_id))
    else:
        val = int(all_data.loc['min', 'SK_ID_CURR'])
        response = requests.get("{}/api/getdecision/?id={}".format(host, val))
    try:
        client_info = pd.DataFrame.from_dict(response.json())

        if selected == "Client loan status":
            loan_status_page_front(0.75, client_info, client_id)
        if selected == "Detailed":
            client_id = detailed_page_front(client_info, df_grant_desc, df_all_desc, all_columns, client_id)
        if selected == "Semi-detailed":
            semi_detailed_page_front(all_data, client_info, client_id)
        if selected == "Custom simulation":
            custom_simu_page_front()
    except JSONDecodeError:
        st.write('no data found')
